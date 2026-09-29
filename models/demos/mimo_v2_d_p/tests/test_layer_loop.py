# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Error accumulation over depth and context: the real layers 0..5 run as a deep chain, L0 -> (L1 .. L5) x
MIMO_LOOP_R, chunked prefill over MIMO_LOOP_SEQ tokens of a real prompt, against the same chain through the HF
modules in fp32 (full-sequence forward).

The looped chain is not the real model (layers 6..47 are replaced by repeats of 1..5), but every step is a real
MiMo layer with real weights, so a numerical bias that compounds with depth (norm gain, drift of the residual
stream, routing flips feeding back) shows up as a trend over the steps. Every step is its own virtual layer with
its own KV-cache slot, so later chunks attend to that step's history exactly as a real layer would.

Per step: PCC, relative L2 error and norm ratio ||tt|| / ||ref|| of the hidden state over the whole sequence and
of the last chunk alone (the longest context), the same for the step's own update (x_out - x_in), and PCC /
relative L2 error / norm ratio of the K and V the step wrote to its cache (read back, natural token order).

    MIMO_LOOP_R=6 MIMO_LOOP_SEQ=2048 MIMO_LOOP_CHUNK=2048 scripts/run_safe_pytest.sh .../tests/test_layer_loop.py
    MIMO_LOOP_R=1 MIMO_LOOP_SEQ=65536 MIMO_LOOP_CHUNK=4096 MIMO_LOOP_PROMPT=<long text file> ...
The HF chain is cached under the golden dir (loop_R{R}_S{seq}_{prompt}.pt).
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.mimo_v2_d_p.reference import hf
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.weights import global_state, layer_state
from models.demos.mimo_v2_d_p.tests.golden import GOLDEN_DIR
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS
from models.demos.mimo_v2_d_p.tt.attention.attention import kv_heads_for_col
from models.demos.mimo_v2_d_p.tt.attention.kv_cache import allocate_kv_cache
from models.demos.mimo_v2_d_p.tt.model import TtMiMoModel, block_cyclic_index
from models.demos.mimo_v2_d_p.tt.options import MiMoRuntimeOptions
from models.demos.mimo_v2_d_p.tt.rope import rope_perm

R = int(os.environ.get("MIMO_LOOP_R", "6"))
CHUNK = int(os.environ.get("MIMO_LOOP_CHUNK", "2048"))
SEQ = int(os.environ.get("MIMO_LOOP_SEQ", str(CHUNK)))
PROMPT = os.environ.get("MIMO_LOOP_PROMPT")  # a long text file for long contexts (default: tests/prompt.txt, repeated)


def schedule(r):
    return [0] + [i for _ in range(r) for i in range(1, 6)]


def stats(a, b):
    """(pcc, rel L2 error, norm ratio) of a against the reference b."""
    a, b = a.flatten().double(), b.flatten().double()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return pcc, ((a - b).norm() / b.norm()).item(), (a.norm() / b.norm()).item()


def fmt(s):
    return f"{s[0]:.5f} {s[1]:.4f} {s[2]:.4f}"


@torch.no_grad()
def hf_chain(cfg, ids, order):
    """[steps + 1, SEQ, H] bf16 hidden states and per step (K post-rope, V) bf16, full-sequence HF forward."""
    tag = Path(PROMPT).stem if PROMPT else "prompt"
    path = GOLDEN_DIR / f"loop_R{R}_S{SEQ}_{tag}.pt"
    if path.exists():
        return torch.load(path)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))  # physical cores: SMT siblings share the FMA units
    hcfg = hf.hf_config()
    # per-block causal / window key ranges, no dense [S, S] mask (matches the blocked eager golden to ~2e-5 relative;
    # 64K-token SWA layers go from ~15-20 min to about a minute)
    hf.use_fast_attention()
    mods = {}
    x = global_state()["embed_tokens.weight"][ids][None].float()
    hidden, kv = [x[0].bfloat16()], []
    for step, i in enumerate(order):
        if i not in mods:
            mods[i] = hf.decoder_layer(i, layer_state(i, cfg), hcfg)
        spec = cfg.layer_attn(i)
        x = hf.run_layer(mods[i], x, spec.window is not None, hcfg, window=spec.window, dense_mask=False)
        k, v = hf.KV_CAPTURE.pop(i)
        kv.append((k.bfloat16(), v.bfloat16()))
        hidden.append(x[0].bfloat16())
        logger.info(f"HF step {step} (L{i}) done, rms {x.pow(2).mean().sqrt().item():.3f}")
    out = {"ids": ids, "order": order, "prompt": PROMPT or "tests/prompt.txt", "hidden": torch.stack(hidden), "kv": kv}
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out


def read_kv(mesh_device, cfg, cache, b, layer, max_seq):
    """K [1, n_kv, SEQ, 192] (Meta-rope order), V [1, n_kv, SEQ, 128] of cache slot b, natural token order."""
    spec = cfg.layer_attn(layer)
    sp, tp = tuple(mesh_device.shape)
    pos = blockcyclic_positions(sp, CHUNK, max_seq)

    def heads(tensor, d_keep):
        d = tensor.shape[3]
        sl = ttnn.slice(
            tensor, [b, 0, 0, 0], [b + 1, cache.n_kv_local, tensor.shape[2], d], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        dts = ttnn.get_device_tensors(sl)
        out = [None] * spec.n_kv
        for c in range(tp):
            dev = torch.cat([ttnn.to_torch(dts[r * tp + c]).float()[0] for r in range(sp)], dim=1)
            nat = torch.empty_like(dev)
            nat[:, pos] = dev
            for hl, g in enumerate(kv_heads_for_col(c, tp, spec.n_q, spec.n_kv)):
                if out[g] is None:
                    out[g] = nat[hl, :SEQ, :d_keep]
        sl.deallocate(True)
        return torch.stack(out)[None]

    return heads(cache.k, spec.head_dim), heads(cache.v, spec.v_head_dim)


@pytest.mark.timeout(21600)
@MESH_PARAMS
def test_layer_loop(mesh_device, device_params):
    assert SEQ % CHUNK == 0, (SEQ, CHUNK)
    cfg = MiMoTextConfig.from_json()
    order = schedule(R)
    ids = hf.tokenize_prompt(SEQ, path=PROMPT)
    max_seq = max(SEQ, 2 * CHUNK)  # the ring SDPA's chunked path needs a cache longer than the chunk
    model = TtMiMoModel(
        mesh_device,
        cfg,
        lambda i: layer_state(i, cfg),
        fabric_config=device_params["fabric_config"],
        max_seq_len=max_seq,
        chunk_size=CHUNK,
        layers=list(range(6)),
        global_state=global_state,
        allocate_kv=False,
        options=MiMoRuntimeOptions.from_env(),
    )
    # one cache slot per step (virtual layer), per attention type
    slot, n_type = [], {}
    for i in order:
        t = cfg.layer_type(i)
        slot.append(n_type.get(t, 0))
        n_type[t] = n_type.get(t, 0) + 1
    kv = {
        t: allocate_kv_cache(mesh_device, num_layers=n, max_seq_len=max_seq, **model.kv_geometry(t))
        for t, n in n_type.items()
    }

    got = torch.zeros(len(order) + 1, SEQ, cfg.hidden_size, dtype=torch.bfloat16)
    for c in range(SEQ // CHUNK):
        kv_actual = c * CHUNK
        idx = block_cyclic_index(kv_actual, model.sp, model.chunk_local) - kv_actual
        x = model.embed_device(model.tokens_to_device(ids[kv_actual : kv_actual + CHUNK][idx]))
        got[0, kv_actual : kv_actual + CHUNK] = model.gather_hidden(x, kv_actual).bfloat16()
        for s, i in enumerate(order):
            layer = model.layers[i]
            y = layer(
                x, model.rope[layer.kind], model.trans_mat, kv[layer.kind], cache_layer=slot[s], kv_actual=kv_actual
            )
            x.deallocate(True)
            x = y
            got[s + 1, kv_actual : kv_actual + CHUNK] = model.gather_hidden(x, kv_actual).bfloat16()
        x.deallocate(True)
        logger.info(f"device chunk {c + 1}/{SEQ // CHUNK} done")
    dev_kv = [read_kv(mesh_device, cfg, kv[cfg.layer_type(i)], slot[s], i, max_seq) for s, i in enumerate(order)]
    # device results next to the HF chain (same layout), for offline analysis without rerunning either side
    tl = "_twolevel" if os.environ.get("TT_METAL_SDPA_RING_TWO_LEVEL") == "1" else ""
    tag = Path(PROMPT).stem if PROMPT else "prompt"
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ids": ids,
            "order": order,
            "chunk": CHUNK,
            "hidden": got,
            "kv": [(k.bfloat16(), v.bfloat16()) for k, v in dev_kv],
        },
        GOLDEN_DIR / f"loop_R{R}_S{SEQ}_{tag}_device_C{CHUNK}{tl}.pt",
    )

    g = hf_chain(cfg, ids, order)
    ref = g["hidden"]
    two_level = os.environ.get("TT_METAL_SDPA_RING_TWO_LEVEL") == "1"
    logger.info(
        f"loop R={R} seq={SEQ} chunk={CHUNK} prompt={PROMPT or 'tests/prompt.txt'} SDPA two-level={two_level}: "
        f"{len(order)} steps"
    )
    logger.info(
        "step layer | hidden all: PCC relL2 norm | hidden last chunk: PCC relL2 norm | update: PCC relL2 norm | "
        "K: PCC relL2 norm | V: PCC relL2 norm"
    )
    last = slice(SEQ - CHUNK, SEQ)
    worst = {"hidden": 1.0, "K": 1.0, "V": 1.0}
    for s, i in enumerate(order, start=1):
        spec = cfg.layer_attn(i)
        a, b = got[s].float(), ref[s].float()
        h, hl = stats(a, b), stats(a[last], b[last])
        u = stats(a - got[s - 1].float(), b - ref[s - 1].float())
        k_ref, v_ref = (t.float() for t in g["kv"][s - 1])
        k_dev, v_dev = dev_kv[s - 1]
        ks = stats(k_dev, k_ref[..., rope_perm(spec.head_dim, spec.rope_dim)])
        vs = stats(v_dev, v_ref)
        worst["hidden"], worst["K"], worst["V"] = (
            min(worst["hidden"], h[0]),
            min(worst["K"], ks[0]),
            min(worst["V"], vs[0]),
        )
        logger.info(
            f"LOOP {s:3d} L{i} {spec.kind[:4]} {'moe' if cfg.is_moe(i) else 'dns'} | {fmt(h)} | {fmt(hl)} | {fmt(u)} | "
            f"{fmt(ks)} | {fmt(vs)}"
        )
    logger.info(f"worst PCC over {len(order)} steps: {', '.join(f'{k} {v:.5f}' for k, v in worst.items())}")
    assert worst["hidden"] > 0.98 and worst["K"] > 0.98 and worst["V"] > 0.98, worst
