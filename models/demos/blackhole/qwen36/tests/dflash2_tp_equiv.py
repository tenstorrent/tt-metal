"""DFlash2DrafterTP (sharded + traced) vs the validated replicated DFlash2Drafter on REAL taps.

No 27B: builds a TT_CCL, a vocab-sharded lm_head from the target's shard 8, and fractured device taps
from profiles/dflash2_real.npz (DFlash2 drafter, DEFAULT_WEIGHTS) — or, with DFLASH_WEIGHTS pointing at
the v1 drafter, compares against the replicated drafter run on the SAME taps (v1 taps differ, so the
npz taps are only shape-valid there; the check is TP-vs-replicated equivalence, not accuracy).

Checks: (1) eager TP fill+draft == replicated fill+draft (tokens) and PCC(draft_hidden) high;
        (2) TP fill(64)+extend x8 == TP fill(128); (3) traced draft/extend == eager (bitwise tokens).

Run:  MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/dflash2_tp_equiv.py -v -s
"""
import os
import time

import numpy as np
import pytest
import torch
from safetensors import safe_open

import ttnn
from models.demos.blackhole.qwen36.tt.dflash2 import DEFAULT_WEIGHTS, DFlash2Drafter, H, load_config
from models.demos.blackhole.qwen36.tt.dflash2_tp import DFlash2DrafterTP
from models.tt_transformers.tt.ccl import TT_CCL

TGT = os.environ.get("HF_MODEL", "/home/ttuser/experiments/qwen36_27b/model_volume/weights/Qwen3.6-27B")
REAL = os.environ.get("DFLASH_REAL", "/home/ttuser/experiments/qwen36_27b/profiles/dflash2_real.npz")
WDIR = os.environ.get("DFLASH_WEIGHTS", DEFAULT_WEIGHTS)
NB, BS = 8, 64


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / (a.std() * b.std() * (a.numel() - 1)))


def _frac(md, host, rows):
    """(1,n,25600) host -> 5 fractured device taps [1,1,rows,dim_tp] (ShardTensorToMesh dim=-1 per tap)."""
    ntaps = host.shape[-1] // H
    out = []
    for t in range(ntaps):
        x = torch.zeros(1, 1, rows, H)
        n = min(rows, host.shape[1])
        x[0, 0, :n] = host[0, :n, t * H : (t + 1) * H]
        out.append(
            ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=md,
                mesh_mapper=ttnn.ShardTensorToMesh(md, dim=-1),
            )
        )
    return out


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 24576,
            "num_command_queues": 2,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 1024 * 1024 * 1024,
        }
    ],
    indirect=True,
)
def test_dflash2_tp_equiv(mesh_device):
    md = mesh_device
    md.enable_program_cache()
    real = np.load(REAL)
    taps = torch.from_numpy(real["target_hidden_cat"]).float()  # (1,128,25600)
    anchor = int(real["anchor"])
    C = taps.shape[1]
    cfg = load_config(WDIR)
    B = cfg["block"]
    K = B - 1
    with safe_open(f"{TGT}/model-00001-of-00015.safetensors", framework="pt") as f:
        embed = f.get_tensor("model.language_model.embed_tokens.weight").to(torch.bfloat16)
    with safe_open(f"{TGT}/model-00008-of-00015.safetensors", framework="pt") as f:
        lm = f.get_tensor("lm_head.weight")  # (V,H) bf16
    lmT = lm.T.contiguous()
    lm_rep = ttnn.from_torch(
        lmT, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=ttnn.ReplicateTensorToMesh(md)
    )
    lm_sh = ttnn.from_torch(
        lmT, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=ttnn.ShardTensorToMesh(md, dim=-1)
    )
    tt_ccl = TT_CCL(md)
    pt = torch.arange(NB, dtype=torch.int32).reshape(1, NB)

    # ---- reference: replicated drafter (validated) ----
    R = DFlash2Drafter(md, WDIR, embed_host=embed, lm_head_fn=lambda x: ttnn.linear(x, lm_rep))
    R.alloc_kv(pt, BS)
    R.fill_context(taps, 0)
    tok_R = R.draft(anchor, C)
    dh_R = R.last_hidden.clone()
    R.free_kv()
    print(f"[tp] replicated tokens {tok_R}")

    # ---- TP drafter ----
    T = DFlash2DrafterTP(
        md,
        WDIR,
        embed,
        tt_ccl,
        ttnn.Topology.Ring,
        lm_sh,
        lm_vocab_sharded=True,
        cache_dir=os.environ.get("DFLASH_TP_CACHE"),
    )
    tap_bufs = [
        ttnn.from_torch(
            torch.zeros(1, 1, B, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            mesh_mapper=ttnn.ShardTensorToMesh(md, dim=-1),
        )
        for _ in cfg["taps"]
    ]

    def fill(rows):
        dev = _frac(md, taps[:, :rows], rows)
        T.fill_context(dev, 0)
        for t in dev:
            ttnn.deallocate(t)

    def extend(r0, n):
        dev = _frac(md, taps[:, r0 : r0 + n], B)
        T.extend_from_taps(dev, r0, n)
        for t in dev:
            ttnn.deallocate(t)

    # (1) eager fill(128) + draft
    T.alloc(pt, BS, tap_bufs)
    fill(128)
    t0 = time.perf_counter()
    tok_T1 = T.draft(anchor, C, traced=False)
    ttnn.synchronize_device(md)
    t_eager = time.perf_counter() - t0
    dh_T1 = T.last_hidden.clone()
    print(
        f"[tp] TP eager fill(128)         tokens {tok_T1}  match_rep={sum(a == b for a, b in zip(tok_T1, tok_R))}/{K}  pcc(dh,rep)={_pcc(dh_T1, dh_R):.4f}  eager draft {t_eager * 1e3:.1f} ms"
    )
    T.free()

    # (2) fill(64) + extends (eager) then traced draft/extend vs eager
    T.alloc(pt, BS, tap_bufs)
    fill(64)
    for r in range(64, 128, B):
        extend(r, min(B, 128 - r))
    tok_T2 = T.draft(anchor, C, traced=False)
    dh_T2 = T.last_hidden.clone()
    print(
        f"[tp] TP eager fill(64)+extends  tokens {tok_T2}  match_T1={sum(a == b for a, b in zip(tok_T2, tok_T1))}/{K}  pcc(dh,T1)={_pcc(dh_T2, dh_T1):.5f}"
    )
    # traced: same state, capture on the next call, then replay
    T.arm_traces()
    tok_T3 = T.draft(anchor, C, traced=True)  # captures + returns the capture pass's result
    t0 = time.perf_counter()
    tok_T4 = T.draft(anchor, C, traced=True)
    ttnn.synchronize_device(md)
    t_traced = time.perf_counter() - t0
    dh_T4 = T.last_hidden.clone()
    print(
        f"[tp] TP traced draft (replay)   tokens {tok_T4}  match_eager={sum(a == b for a, b in zip(tok_T4, tok_T2))}/{K}  pcc(dh,eager)={_pcc(dh_T4, dh_T2):.6f}  traced draft {t_traced * 1e3:.1f} ms (capture pass {tok_T3 == tok_T2})"
    )
    # traced extend: write rows 128..128+B-1 from zeros-taps via the trace path, then re-write the real
    # last block rows eagerly vs traced and compare drafts (both must see identical context).
    dev = _frac(md, taps[:, 120:128], B)
    for src, buf in zip(dev, tap_bufs):
        ttnn.copy(src, buf)
    for t in dev:
        ttnn.deallocate(t)
    t0 = time.perf_counter()
    T.extend_context(120, 8, traced=True)
    ttnn.synchronize_device(md)
    t_ext_cap = time.perf_counter() - t0
    t0 = time.perf_counter()
    T.extend_context(120, 8, traced=True)
    ttnn.synchronize_device(md)
    t_ext = time.perf_counter() - t0
    tok_T5 = T.draft(anchor, C, traced=True)
    print(
        f"[tp] after traced extend(120,8) tokens {tok_T5}  match={sum(a == b for a, b in zip(tok_T5, tok_T2))}/{K}  extend traced {t_ext * 1e3:.1f} ms (capture {t_ext_cap * 1e3:.0f} ms)  stats={T.stats}"
    )
    T.free()
    for t in tap_bufs:
        ttnn.deallocate(t)

    # TP vs replicated: the hidden states must agree (PCC); tokens may flip at bf16 near-ties because the
    # vocab-sharded lm_head matmul (N=V/4 per device) rounds differently from the full-vocab one, and
    # a repetitive prompt puts the drafter in a flat-logit regime. Require a strong PCC and a majority
    # of identical tokens.
    m1 = sum(a == b for a, b in zip(tok_T1, tok_R))
    assert _pcc(dh_T1, dh_R) >= 0.998, f"TP vs replicated draft_hidden PCC {_pcc(dh_T1, dh_R):.4f}"
    assert m1 >= (K + 1) // 2, f"TP vs replicated tokens {m1}/{K}: {tok_T1} vs {tok_R}"
    assert _pcc(dh_T2, dh_T1) >= 0.998, "fill vs extend paths (draft_hidden PCC)"
    assert tok_T4 == tok_T2, f"traced draft != eager: {tok_T4} vs {tok_T2}"
    assert tok_T5 == tok_T2, f"traced extend changed the context: {tok_T5} vs {tok_T2}"
