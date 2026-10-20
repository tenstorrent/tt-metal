# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.13 measurements for the realistic defaults (FP32_ACC=needed, DENSE_WEIGHTS=bfp8, WINDOW_SDPA=source). Not a gate.

test_fp32_acc_matmuls  every dense projection of the model at its per-chip shape on the run's mesh (the layer's real
                       weights in the format DENSE_WEIGHTS picks, random inputs at the model's input dtype, the
                       fidelity the model uses there), with fp32 DEST on and off: the mean synced wall time of N calls
                       (replicated on every chip) and the error of chip 0's output vs float64 on the same inputs and
                       the same (bfp8-rounded) weights: rel L2, bias <d, ref> / |ref|^2, worst row. ``floor`` = the
                       bf16 rounding of the float64 result alone (what a bf16 output costs anyway); ``bfp8`` = what
                       the bfp8 weights move the float64 result (DENSE_WEIGHTS' own error).
test_window_sdpa       the window attention's sparse_sdpa (the model's start-class-128 index lists, layer 0's sink,
                       random q / table with score spread x1 and x4) as ttnn.bringup.sparse_sdpa(high_precision) at the
                       reference kernel's dtypes (today), and stock ttnn.transformer.sparse_sdpa with fp32 DEST on and
                       off: time and error vs float64.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tt/perf_precision.py
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.bringup import hooks
from models.demos.deepseek_v41_flash_d_p.tt.common import bfp8_roundtrip
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

N_CALLS = 20
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _rep(mesh, t, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _chip0(t) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).double()


def _err(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    got, ref = got.reshape(-1, ref.shape[-1]).double(), ref.reshape(-1, ref.shape[-1]).double()
    d = got - ref
    rel = (d.norm() / ref.norm()).item()
    bias = ((d * ref).sum() / (ref * ref).sum()).item()
    row = (d.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)).max().item()
    return rel, bias, row


def _timed(mesh, fn, n=N_CALLS) -> float:
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    for _ in range(n):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    return (time.time() - t0) / n * 1e3


def _cfg(mesh, fidelity: str, fp32: bool):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=False,
    )


def _sites(ck, cfg, layer: int, rows: int, tp: int) -> list[tuple]:
    """(name, x dtype, out dtype, fidelity setting, W [K, N] per chip (as uploaded: W^T slices), fp8 weight?, batch)"""
    p = f"layers.{layer}."
    wf = lambda n: ck.weight(p + n, torch.float32)  # noqa: E731
    H = cfg.dim
    out = []
    qa = torch.cat([wf("attn.wq_a.weight"), wf("attn.wkv.weight")], 0).t()  # [H, 1792], K split over TP
    out.append(("q_a [wq_a|wkv]", ttnn.bfloat16, ttnn.bfloat16, "ATTN_FIDELITY", qa[: H // tp], True, 1))
    wq_b = wf("attn.wq_b.weight").t()  # [1280, heads * 512], N split over TP
    out.append(("attn.wq_b", ttnn.bfloat16, ttnn.float32, "ATTN_FIDELITY", wq_b[:, : wq_b.shape[1] // tp], True, 1))
    G, rank = cfg.o_groups, cfg.o_lora_rank
    wo_a = wf("attn.wo_a.weight").view(G, rank, -1).transpose(1, 2)  # [G, hpg * 512, rank], groups over TP
    out.append(("attn.wo_a", ttnn.bfloat16, ttnn.float32, "ATTN_FIDELITY", wo_a[: G // tp], True, G // tp))
    wo_b = wf("attn.wo_b.weight").t()  # [G rank, H], K split over TP
    out.append(("attn.wo_b", ttnn.bfloat16, ttnn.bfloat16, "ATTN_FIDELITY", wo_b[: wo_b.shape[0] // tp], True, 1))
    if cfg.is_index_source(layer):
        iq = wf("attn.indexer.wq_b.weight").t()  # [1280, nh * hd], replicated (rows a chip: rows / TP)
        out.append(("indexer.wq_b", ttnn.bfloat16, ttnn.float32, "ATTN_FIDELITY", iq, True, 1))
        wp = wf("attn.indexer.weights_proj.weight").t()  # [H, nh] bf16 as stored, K split
        out.append(("indexer.weights_proj", ttnn.float32, ttnn.float32, "ATTN_FIDELITY", wp[: H // tp], False, 1))
    if cfg.is_kv_source(layer):
        blocks = [wf("attn.compressor.wkv.weight")]
        if cfg.ratio(layer) > 1:
            blocks.append(wf("attn.compressor.wgate.weight"))
        cw = torch.cat(blocks, 0).t()
        out.append(("compressor [wkv|wgate]", ttnn.float32, ttnn.float32, "ATTN_FIDELITY", cw[: H // tp], False, 1))
    if cfg.is_engram(layer):
        ew = wf("engram.wkv.weight").t()  # [K, 5H], N (each chip's blocks) split over TP
        out.append(("engram.wkv", ttnn.bfloat16, ttnn.float32, "MATMUL_FIDELITY", ew[:, : ew.shape[1] // tp], True, 1))
    gw = ck.get(p + "ffn.gate.weight").float().t()  # [H, E] bf16, replicated
    out.append(("router gate", ttnn.bfloat16, ttnn.float32, "MATMUL_FIDELITY", gw, False, 1))
    return out


@pytest.mark.timeout(3600)
@mesh_parametrize
@pytest.mark.parametrize("layer", [0, 1, 2, 20], ids=["L00", "L01", "L02", "L20"])
def test_fp32_acc_matmuls(mesh_device, layer):
    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    mesh = mesh_device
    sp, tp = mesh.shape[0], mesh.shape[1]
    rows = hooks._chunk(S) // sp
    dense = settings.get("DENSE_WEIGHTS")
    g = torch.Generator().manual_seed(layer)
    for name, xdt, odt, fid_key, w, fp8, batch in _sites(ck, cfg, layer, rows, tp):
        fid = settings.get(fid_key)
        r = rows // tp if name == "indexer.wq_b" else rows
        w = w.float().contiguous()
        k = w.shape[-2]
        wdt = ttnn.bfloat8_b if (fp8 and dense == "bfp8") else ttnn.bfloat16
        wq = bfp8_roundtrip(w).reshape(w.shape) if wdt == ttnn.bfloat8_b else w.bfloat16().float()
        shape = [1, batch, r, k] if batch > 1 else [1, 1, r, k]
        x = torch.randn(*shape, generator=g)
        xh = x.bfloat16().float() if xdt == ttnn.bfloat16 else x
        wd = wq.reshape(1, batch, k, -1) if batch > 1 else wq.reshape(1, 1, k, -1)
        ref = (xh.double() @ wd.double()).squeeze(0)
        ref_exact = (xh.double() @ w.reshape(wd.shape).double()).squeeze(0)
        floor = _err(ref.float().bfloat16(), ref)[0]
        werr = _err(ref, ref_exact)
        xd = _rep(mesh, xh, xdt)
        wdev = _rep(mesh, w.reshape(wd.shape), wdt)
        mm = ttnn.matmul if batch > 1 else ttnn.linear
        res = {}
        for fp32 in (True, False):
            ck_ = _cfg(mesh, fid, fp32)

            def fn():
                return mm(xd, wdev, dtype=odt, compute_kernel_config=ck_, memory_config=DRAM)

            o = fn()
            got = _chip0(o).squeeze(0)
            ttnn.deallocate(o)
            res[fp32] = (_timed(mesh, fn), _err(got, ref))
        (t1, e1), (t0, e0) = res[True], res[False]
        print(
            f"FP32ACC L{layer:02d} {name:24s} [{r}, {k}] x [{k}, {w.shape[-1]}]{' x' + str(batch) if batch > 1 else ''} "
            f"w={str(wdt).split('.')[-1]} {fid}: fp32 {t1:.3f} ms rel {e1[0]:.5f} bias {e1[1]:+.5f} row {e1[2]:.4f} | "
            f"bf16 {t0:.3f} ms rel {e0[0]:.5f} bias {e0[1]:+.5f} row {e0[2]:.4f} | floor(bf16 out) {floor:.5f} "
            f"bfp8 weights rel {werr[0]:.5f} bias {werr[1]:+.5f}"
        )
        for t in (xd, wdev):
            ttnn.deallocate(t)


def _sdpa_ref(q, table, idx, sink):
    """float64 window attention with sink: q [Hh, R, D], table [T, D], idx [R, 128] (-1 = none), sink [Hh]."""
    q, table = q.double(), table.double()
    out = torch.zeros_like(q)
    valid = idx >= 0
    kk = table[idx.clamp(min=0)]  # [R, 128, D]
    s = torch.einsum("hrd,rjd->hrj", q, kk)
    s = s.masked_fill(~valid.unsqueeze(0), float("-inf"))
    m = torch.maximum(s.amax(dim=-1), sink.double().view(-1, 1))
    e = torch.exp(s - m.unsqueeze(-1))
    den = e.sum(-1) + torch.exp(sink.double().view(-1, 1) - m)
    out = torch.einsum("hrj,rjd->hrd", e / den.unsqueeze(-1), kk)
    return out


@pytest.mark.timeout(3600)
@mesh_parametrize
@pytest.mark.parametrize("spread", [1.0, 4.0], ids=["x1", "x4"])
def test_window_sdpa(mesh_device, spread):
    from models.demos.deepseek_v41_flash_d_p.tt.attention import SDPA_K_CHUNK, WINDOW, WindowTables

    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    mesh = mesh_device
    chunk = hooks._chunk(S)
    tables = WindowTables(mesh, chunk)
    geo = tables.geo
    D, Hh = 512, cfg.n_heads
    g = torch.Generator().manual_seed(7)
    q = (torch.randn(1, Hh, geo.q_rows, D, generator=g) * spread / D**0.5).bfloat16()
    table = torch.randn(1, 1, WINDOW + chunk, D, generator=g).bfloat16()
    sink = ck.weight("layers.0.attn.attn_sink", torch.float32).reshape(1, 1, 1, Hh)
    qd = _rep(mesh, q, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    td = _rep(mesh, table, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    sd = _rep(mesh, sink, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    index = tables.index[WINDOW]
    idx = ttnn.to_torch(ttnn.get_device_tensors(index)[0]).reshape(geo.q_rows, WINDOW).to(torch.int64)
    idx = torch.where(idx == 0xFFFFFFFF, -1, idx) if idx.max() > 2**31 else idx
    ref = _sdpa_ref(q[0].float(), table[0, 0].float(), idx, sink.reshape(-1).bfloat16().float())
    fid = settings.get("ATTN_FIDELITY")
    variants = {
        "fork high_precision (today)": lambda: ttnn.bringup.sparse_sdpa(
            qd,
            td,
            index,
            D,
            kv_format=ttnn.bringup.SparseKVFormat.BF16,
            scale=1.0,
            k_chunk_size=SDPA_K_CHUNK,
            compute_kernel_config=_cfg(mesh, fid, True),
            attention_sink=sd,
            high_precision=True,
            fp32_scores=True,
            bf16_probs=True,
        ),
    }
    for fp32 in (True, False):
        variants[f"stock fp32_dest={fp32}"] = lambda c=_cfg(mesh, fid, fp32): ttnn.transformer.sparse_sdpa(
            qd,
            td,
            index,
            D,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=1.0,
            k_chunk_size=SDPA_K_CHUNK,
            compute_kernel_config=c,
            attention_sink=sd,
        )
    for name, fn in variants.items():
        o = fn()
        got = _chip0(o)[0]
        ttnn.deallocate(o)
        e = _err(got, ref)
        print(
            f"WINDOW_SDPA spread x{spread:g} {name:28s} {fid}: {_timed(mesh, fn):.3f} ms rel {e[0]:.5f} "
            f"bias {e[1]:+.5f} row {e[2]:.4f}"
        )
