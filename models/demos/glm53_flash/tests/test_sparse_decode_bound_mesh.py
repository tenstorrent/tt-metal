# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The sparse-decode live bound (``GLM53_SPARSE_DECODE_BOUND``, ``tt/dsa.py`` + ``custom_ops/sparse_bound.py``) on the
rows-4-5 (2,4) block: the traced decode's scorer and segmented top-k loop to the step's live pool bound instead of the
pool capacity the model was built with, and the result must be bit-identical to the capacity-wide graph.

4 layers (0-3, layer 3 = DSA), max_seq_len GLM53_BOUND_MAX_SEQ_LEN (131072: a capacity of 33 024 pools per user, far
above the live pools of a 3000-token prompt), one user (slot 0 of row 0) prefilled with a GLM53_BOUND_LEN-token ramp
prompt in 1024-token chunks (the KDA SP prefill L1 window; never 800-1200-row slabs), then the decode flow -- the eager
step at position LEN, the capture at LEN + 1 (its warm-up decodes LEN + 1 too), N replays at LEN + 2 .. -- with the
bound OFF; the KDA states are reset and the user re-prefilled (the prefill's last-token logits must reproduce the
first prefill's: the decode baseline is a function of the same state), every DSA layer's ``sparse_bound`` is flipped
ON and the same flow runs again: the greedy ids and the logits of every step (eager and replays) must be bit-identical,
the replay walls are logged in tok/s. GLM53_BOUND_MAX_SEQ_LEN=524288 builds 131 328 pools per user = 3 segments
(``TopkRows.segments_for``): the two-stage top-k of the served 1 M geometry (classic body) under the bound.

    GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-3 GLM53_MODEL_B=4 GLM53_MODEL_TRACE_MB=512 \\
    pytest models/demos/glm53_flash/tests/test_sparse_decode_bound_mesh.py -s
    GLM53_BOUND_MAX_SEQ_LEN=524288 ... (the segmented top-k)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.sparse_bound import live_bound_pools, phantom_mode
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
B = int(os.environ.get("GLM53_MODEL_B", "4"))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
LEN = int(os.environ.get("GLM53_BOUND_LEN", "3000"))
MAX_SEQ_LEN = int(os.environ.get("GLM53_BOUND_MAX_SEQ_LEN", "131072"))
PREFILL_CHUNK = int(os.environ.get("GLM53_BOUND_CHUNK", "1024"))
N_REPLAYS = int(os.environ.get("GLM53_BOUND_REPLAYS", "6"))  # 6 crosses one 4-token pool boundary (3004 -> pool 751)
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)


def _layer_range() -> list[int]:
    spec = os.environ.get("GLM53_MODEL_LAYERS", "0-3")
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


def _ramp(n: int) -> torch.Tensor:
    """vllm bench "random" prompt: a ramp of consecutive vocab ids."""
    return (torch.arange(n, dtype=torch.long) + 12345) % 150000


def _dsa_layers(model: GLM53MeshModel) -> list:
    """The ``TtDSA`` instances of the model (the layers that carry the ``sparse_bound`` knob)."""
    return [dec.layer.attn for dec in model.layers if not dec.is_kda and hasattr(dec.layer.attn, "sparse_bound")]


def _to_host_logits(model: GLM53MeshModel, logits: ttnn.Tensor) -> torch.Tensor:
    return model.mesh.to_host(logits, rows=2 if model.mesh.rows > 1 else None).float().clone()


def _prefill(model: GLM53MeshModel, prompt: torch.Tensor, tag: str) -> tuple[int, torch.Tensor]:
    """The prompt into user 0 in PREFILL_CHUNK-token chunks; (first greedy token, last-token logits on the host)."""
    t0 = time.perf_counter()
    streams, logits_last, owner_row = model.prefill_chunked(prompt, user=0, chunk=PREFILL_CHUNK)
    ttnn.synchronize_device(model.device)
    first = model.first_token(logits_last, owner_row)
    lh = model.mesh.to_host(logits_last).float().flatten().clone()
    ttnn.deallocate(streams)
    ttnn.deallocate(logits_last)
    logger.info(f"BOUND {tag}: prefill of {LEN} tokens in {time.perf_counter() - t0:.1f} s, first token {first}")
    return first, lh


def _decode_flow(model: GLM53MeshModel, ids: torch.Tensor, pos0: int, tag: str) -> list[dict]:
    """The eager step at ``pos0``, the capture at ``pos0 + 1`` (its warm-up decodes ``pos0 + 1`` too) and N_REPLAYS
    replays at ``pos0 + 2 ..`` for slot 0 of every row (the other slots idle at position 0): per step the greedy ids
    [U] and the logits [1,1,U,V] on the host, plus the replay wall. The same (position, token) sequence in both passes,
    so the device state evolves identically and the outputs are comparable bit for bit."""
    U = model.total_users
    cfg = model.cfg
    pos = [pos0 if u % B == 0 else 0 for u in range(U)]
    steps = []
    # the eager step (the first pass allocates the persistent decode inputs here, as warmup_decode does)
    if model.ids_dev is None:
        model.ids_dev = model.ids_to_device(ids)
    else:
        model.load_ids(ids)
    model.set_positions(pos)
    t0 = time.perf_counter()
    outs = model.decode_step(model.ids_dev, model.positions)
    got = model.ids_to_host(outs[2]).clone()
    lh = _to_host_logits(model, outs[1])
    for t in outs:
        if t is not None:
            ttnn.deallocate(t)
    ttnn.synchronize_device(model.device)
    logger.info(
        f"BOUND {tag}: eager step at position {pos0} in {time.perf_counter() - t0:.2f} s, slot-0 ids {got[::B].tolist()}"
    )
    steps.append({"kind": "eager", "position": pos0, "ids": got, "logits": lh, "wall": None})
    # the capture (warm-up + capture at pos0 + 1), then the replays
    trace_id, outs = model.capture(ids, [p + 1 if p else 0 for p in pos])
    for k in range(N_REPLAYS):
        p_k = pos0 + 2 + k
        model.load_ids(ids)
        model.set_positions([p + 2 + k if p else 0 for p in pos])
        t0 = time.perf_counter()
        model.replay(trace_id)
        got = model.ids_to_host(outs[2]).clone()
        wall = time.perf_counter() - t0
        lh = _to_host_logits(model, outs[1])
        valid = max((p_k + 1) // cfg.index_kpool, cfg.index_select_pools)
        logger.info(
            f"BOUND {tag}: replay {k} at position {p_k} (valid {valid} pools, live bound {live_bound_pools([valid])}): "
            f"{wall * 1e3:.2f} ms = {U / wall:.1f} tok/s aggregate ({1 / wall:.1f} tok/s/user), slot-0 ids {got[::B].tolist()}"
        )
        steps.append({"kind": "replay", "position": p_k, "ids": got, "logits": lh, "wall": wall})
    ttnn.release_trace(model.device, trace_id)
    ttnn.synchronize_device(model.device)
    return steps


@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_sparse_decode_bound_bit_identical(mesh_device):
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        _layer_range(),
        max_batch=B,
        max_seq_len=MAX_SEQ_LEN,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    dsa = _dsa_layers(model)
    assert dsa, "no DSA layer in the layer range (layer 3 is the first DSA layer)"
    assert all(not a.sparse_bound for a in dsa), "the test starts from the knob's default (off)"
    dsa_dec = [dec for dec in model.layers if not dec.is_kda][0]
    W = dsa_dec.cache.index.pools_per_user  # the pool capacity per user (TtIndexCache: rounded to 256)
    segments = dsa[0].topk_rows.segments_for(W)
    logger.info(
        f"BOUND: max_seq_len {MAX_SEQ_LEN}, pools per user {W}, top-k segments {segments} (phantom mode "
        f"{phantom_mode()}), prompt {LEN} tokens (valid {max((LEN + 1) // cfg.index_kpool, cfg.index_select_pools)} "
        f"pools at the first decode)"
    )
    assert LEN > dsa[0].dense_exact_max_pos, "the prompt must put slot 0 on the sparse (gather) path"
    prompt = _ramp(LEN)
    ids = torch.full((model.total_users,), 1000, dtype=torch.long)

    # pass 1: the capacity-wide graph (knob off)
    first_a, logits_a = _prefill(model, prompt, "off")
    ref = _decode_flow(model, ids, LEN, "off")

    # the same state again: reset the KDA recurrent / conv states, re-prefill the user (the DSA cache rows [0, LEN)
    # and the pooled-key index of slot 0 are rewritten; the rows the decode steps wrote beyond are rewritten by the
    # second pass's own steps before they are read)
    model.reset_kda_states()
    first_b, logits_b = _prefill(model, prompt, "on ")
    assert first_a == first_b and torch.equal(logits_a, logits_b), (
        f"re-prefill does not reproduce the first prefill (first token {first_a} vs {first_b}, max |dlogits| "
        f"{(logits_a - logits_b).abs().max().item():.4g}): the decode baseline is not comparable"
    )

    # pass 2: the live-bound graph (knob on in every DSA layer; the eager step compiles the bounded programs, the
    # capture that follows records them)
    for a in dsa:
        a.sparse_bound = True
    got = _decode_flow(model, ids, LEN, "on ")

    assert len(ref) == len(got)
    mismatches = []
    for r, g in zip(ref, got):
        assert (r["kind"], r["position"]) == (g["kind"], g["position"])
        same_ids = torch.equal(r["ids"], g["ids"])
        same_logits = torch.equal(r["logits"], g["logits"])
        dl = (r["logits"] - g["logits"]).abs().max().item()
        logger.info(
            f"BOUND compare {r['kind']} at {r['position']}: ids {'equal' if same_ids else 'DIFFERENT'}, logits "
            f"{'bit-identical' if same_logits else f'max |diff| {dl:.4g}'}"
        )
        if not (same_ids and same_logits):
            mismatches.append((r["kind"], r["position"], same_ids, dl))
    walls_ref = [s["wall"] for s in ref if s["wall"]]
    walls_got = [s["wall"] for s in got if s["wall"]]
    U = model.total_users
    logger.info(
        f"BOUND replay wall: off {min(walls_ref) * 1e3:.2f} ms ({U / min(walls_ref):.1f} tok/s aggregate) vs on "
        f"{min(walls_got) * 1e3:.2f} ms ({U / min(walls_got):.1f} tok/s aggregate), best of {N_REPLAYS}"
    )
    assert not mismatches, f"live bound changed the decode: {mismatches}"
    logger.info(f"BOUND PASS: {len(ref)} steps bit-identical with the live bound (max_seq_len {MAX_SEQ_LEN})")
