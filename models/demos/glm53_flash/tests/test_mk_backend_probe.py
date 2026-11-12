# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase 1, step 1 of the persistent-stack port: the reference's WHOLE decode step (StackBackend: MkEmbed -> one
persistent program per run of same-kind layers -> MkHead with device sampling, ONE trace per step) built from our HF
checkpoint and timed on a mesh.

    GLM53_MESH_SHAPE=2,4 GLM53_FABRIC=1d GLM53_MK_LAYERS=0-8 pytest .../test_mk_backend_probe.py -s

The backend's tile-cache source is replaced by ``HFSource`` (per-layer raw HF dicts, the per-expert tensors, the
head / norm / embedding entries); ``GLM53_STACK_DSA=1`` selects the DSA decode of the sparse_dsa layers (default:
the dense MLA at the reference's 2051-position context). ``GLM53_MK_EP=1`` keeps the EP-sharded experts (needs the
tile cache today), default 0 = replicated experts (a layer subset fits).
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import optimized_decoder as OD
from models.autoports.zai_org_glm_5_3_flash.tt.config import load_hf_text_config
from models.autoports.zai_org_glm_5_3_flash.tt.mk_sources import HFSource
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
B = int(os.environ.get("GLM53_MK_B", "4"))  # user slots = one 4-row chunk
SEQ = int(os.environ.get("GLM53_MK_SEQ", "2048"))
POS = int(os.environ.get("GLM53_MK_POS", "0"))  # decode position of every row (ISL: the DSA indexer scores 0..POS)
N_REPLAY = int(os.environ.get("GLM53_MK_N", "20"))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
# the reference's SERVED worker L1 (pmk_layer WORKER_L1_SERVED): 16 KB more allocatable L1 than the mk_dense note's
# 1,264,512 -- the 45-layer stack sits at that floor (the sharded pool / the attached TP prefill clashed by 1 KB,
# 2026-11-05: "static circular buffers clash with L1 buffers ... 1269248 ... 1270272")
WORKER_L1 = int(os.environ.get("GLM53_MK_WORKER_L1", "1280896"))


def _layers() -> list[int]:
    spec = os.environ.get("GLM53_MK_LAYERS", "0-8")
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


@pytest.mark.timeout(5400)
@pytest.mark.parametrize(
    "device_params",
    [mesh_device_params(*MESH, trace_region_size=TRACE_MB << 20, worker_l1_size=WORKER_L1)],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mk_backend_probe(mesh_device, monkeypatch):
    from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import StackBackend

    snapshot = default_snapshot_dir()
    cfg = load_hf_text_config(str(snapshot))
    source = HFSource(snapshot, cfg)
    monkeypatch.setattr(OD, "FdTileCacheSource", lambda cache_dir, manifest: source)
    layers = _layers()
    ep = os.environ.get("GLM53_MK_EP", "0") == "1"
    t0 = time.perf_counter()
    native = os.environ.get("GLM53_MK_NATIVE_T", "0") == "1"  # B > 4 as ONE launch of B rows (else 4-row chunks)
    backend = StackBackend(mesh_device, max_batch=B, layers=layers, hf_config=cfg, topk=1, sample=False, ep_weights=ep,
                           native_t=native)  # fmt: skip
    logger.info(
        f"StackBackend built in {time.perf_counter() - t0:.0f} s: mesh {MESH}, layers {layers}, B {B}, ep {ep}, "
        f"native_t {native} (chunk {backend.chunk} x {backend.n_chunks})"
    )
    t0 = time.perf_counter()
    kv = backend.allocate_kv_cache(max_batch=B, max_seq_len=SEQ)
    io = backend.allocate_decode_io(kv)
    logger.info(f"kv cache + decode io allocated in {time.perf_counter() - t0:.0f} s (max_seq_len {SEQ})")
    assert POS + N_REPLAY + 4 < SEQ, (POS, SEQ)
    # the physical page table (row b = blocks b * bps .. ; the sequence-sharded pool maps it to every chip's local
    # table) and the rows' positions: the step decodes at POS over the (zero) cache of positions 0 .. POS - 1
    bps = kv.blocks_per_seq
    spec = backend.dsa_shard_owners(kv) if hasattr(backend, "dsa_shard_owners") else None
    if spec is None:
        backend.write(io.page_table, torch.arange(B * bps, dtype=torch.int32).reshape(B, bps))
    else:  # the sequence-sharded pool: owner-aware PHYSICAL blocks (tt/owner_blocks.py), as the generator writes them
        from models.autoports.zai_org_glm_5_3_flash.tt.owner_blocks import OwnerBlockMap

        obm = OwnerBlockMap(*spec)
        rows = [obm.set_row(s, list(range(s * bps, (s + 1) * bps)), bps) for s in range(B)]
        backend.write(io.page_table, torch.tensor(rows, dtype=torch.int32))
    backend.write(io.cur_pos, torch.full((B,), POS, dtype=torch.int32))
    t0 = time.perf_counter()
    backend.decode(io, kv, B)  # one eager step (compiles)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"first eager step {time.perf_counter() - t0:.1f} s")
    if os.environ.get("GLM53_MLA_PROBE") == "1":  # the DSA layers' MLA program marks (chip 0) after the eager step
        from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MKMLA

        for i, e in enumerate(backend.seq):
            mk = getattr(e.get("mla"), "mk", None)
            if mk is None or getattr(mk, "probe", None) is None:
                continue
            try:
                pt = ttnn.to_torch(ttnn.get_device_tensors(mk.probe)[0])
                marks = MKMLA.decode_probe(pt, len(mk.readers(backend.chunk).assign))  # readers(t): per row count
                logger.info(f"MK_BACKEND_MLA_MARKS: layer {layers[i]}: {marks}")
            except Exception as ex:  # noqa: BLE001
                logger.info(f"MK_BACKEND_MLA_MARKS: layer {layers[i]}: unreadable ({ex})")
            # every chip's shard phase marks (tp_dsa_shard.shard_probe; the chip's own clock, us after its root's
            # attention start): the first DSA layer only -- the AG window's skew vs fabric split needs all 32 chips
            if hasattr(e.get("mla"), "shard_probe") and not getattr(backend, "_shard_probe_done", False):
                backend._shard_probe_done = True
                try:
                    for rc, m in e["mla"].shard_probe().items():
                        logger.info(f"MK_BACKEND_SHARD_PROBE: layer {layers[i]} chip {rc}: {m}")
                except Exception as ex:  # noqa: BLE001
                    logger.info(f"MK_BACKEND_SHARD_PROBE: layer {layers[i]}: unreadable ({ex})")
    if os.environ.get("GLM53_STACK_PROBE") == "1":  # the stacks' per-phase probe marks (chip 0) after the eager step
        for r, stk in kv.stacks[0]:
            if stk is None or not hasattr(stk, "probe_marks") or getattr(stk, "probe", None) is None:
                continue
            for j, i in enumerate(r):
                try:
                    _, marks = stk.probe_marks(j)
                    logger.info(f"MK_BACKEND_MARKS: layer {layers[i]} ({backend.seq[i]['sparse']}): {marks}")
                except Exception as e:  # noqa: BLE001
                    logger.info(f"MK_BACKEND_MARKS: layer {layers[i]}: unreadable ({e})")
    t0 = time.perf_counter()
    tr = backend.capture_decode(io, kv, B)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"decode trace captured in {time.perf_counter() - t0:.1f} s")
    backend.replay(tr)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(N_REPLAY):
        backend.replay(tr)
    ttnn.synchronize_device(mesh_device)
    step_us = (time.perf_counter() - t0) / N_REPLAY * 1e6
    logger.info(
        f"MK_BACKEND_PROBE: mesh {MESH} layers {len(layers)} ({layers[0]}-{layers[-1]}) B {B} native_t {native} pos {POS} "
        f"dsa_shard {os.environ.get('GLM53_DSA_SHARD', '0')}: traced decode step "
        f"{step_us:.1f} us = {step_us / len(layers):.1f} us per layer"
    )
    assert step_us > 0
    # every chip's shard marks of the LAST traced replay (the eager step's marks carry the host's per-device dispatch
    # skew, up to ~650 us over 32 chips): the first DSA layer
    if os.environ.get("GLM53_MLA_PROBE") == "1":
        for i, e in enumerate(backend.seq):
            if not hasattr(e.get("mla"), "shard_probe"):
                continue
            try:
                for rc, m in e["mla"].shard_probe().items():
                    logger.info(f"MK_BACKEND_SHARD_PROBE_TRACED: layer {layers[i]} chip {rc}: {m}")
            except Exception as ex:  # noqa: BLE001
                logger.info(f"MK_BACKEND_SHARD_PROBE_TRACED: layer {layers[i]}: unreadable ({ex})")
            break
    prompt = os.environ.get(
        "GLM53_MK_PROMPT", ""
    )  # the text, or @path of a file holding it (launchers split on spaces)
    if prompt.startswith("@"):
        with open(prompt[1:]) as f:
            prompt = f.read().strip()
    if prompt:  # coherence: the chat prompt fed token by token on every slot (fresh state), then greedy decode
        from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

        tok = load_tokenizer(snapshot)
        ids = tok.encode_chat([{"role": "user", "content": prompt}])
        gen = int(os.environ.get("GLM53_MK_GEN", "24"))
        assert len(ids) + gen < SEQ, (len(ids), gen, SEQ)
        # the SAME cache, io and decode trace (a second cache's L1 tensors lower the L1 floor under the DSA readers'
        # CBs on the 32-chip replica): every slot reset, positions back to 0, the prompt tokens written between replays
        backend.reset_all(kv)
        backend.write(io.cur_pos, torch.zeros(B, dtype=torch.int32))
        t0 = time.perf_counter()
        out = []
        for t in ids:
            backend.write(io.tokens, torch.full((B,), int(t), dtype=torch.int32))
            backend.replay(tr)
        sampled = backend.read(io.sampled)
        out.append(int(sampled[0]))
        for _ in range(gen - 1):
            backend.replay(tr)  # the device sampler feeds io.tokens
            sampled = backend.read(io.sampled)
            out.append(int(sampled[0]))
            if tok.is_stop_token(out[-1]):
                break
        rows_equal = bool((sampled == sampled[0]).all())
        text = tok.decode([i for i in out if 0 <= i < tok.vocab_size])
        logger.info(
            f"MK_BACKEND_PROMPT: mesh {MESH} layers {len(layers)} B {B}: {len(ids)} prompt + {len(out)} generated "
            f"tokens in {time.perf_counter() - t0:.1f} s; rows equal {rows_equal}; ids {out}"
        )
        logger.info(f"MK_BACKEND_TEXT: {text!r}")
