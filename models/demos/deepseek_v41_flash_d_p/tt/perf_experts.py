# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Timed loop of the routed experts (tt/experts.py:TtExperts) of one layer at the run's chunk, every EXPERTS_IMPL.

Not a gate: a measurement for BREADCRUMBS / the EXPERTS_IMPL description. Random q-able inputs (normal x, top-6
routing with sum-1.5 weights), 2 warm-up calls, then the mean wall time of N_CALLS synchronized calls (the forward
has no host sync, so this is the device time of one layer per chunk).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tt/perf_experts.py
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.bringup import hooks
from models.demos.deepseek_v41_flash_d_p.tt import experts as E
from models.demos.deepseek_v41_flash_d_p.tt.layout import row_split_to_device
from models.demos.deepseek_v41_flash_d_p.tt.settings import TABLE

N_CALLS = 3
LAYER = 0


@pytest.mark.timeout(3600)
@mesh_parametrize
def test_time_experts(mesh_device):
    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(chunk, cfg.dim, generator=g)
    sc = torch.rand(chunk, cfg.n_experts, generator=g)
    top = sc.topk(cfg.topk_experts, dim=-1)
    dense = torch.zeros(chunk, cfg.n_experts)
    dense.scatter_(1, top.indices, 1.5 * top.values / top.values.sum(-1, keepdim=True))
    outs = {}
    for impl in TABLE["EXPERTS_IMPL"].choices:
        t0 = time.time()
        mod = E.build_experts(mesh_device, ck, cfg, LAYER, chunk, impl=impl)
        load = time.time() - t0
        xd = row_split_to_device(mesh_device, x)
        rd = row_split_to_device(mesh_device, dense)
        for _ in range(2):
            ttnn.deallocate(mod(xd, rd))
        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        for _ in range(N_CALLS):
            ttnn.deallocate(mod(xd, rd))
        ttnn.synchronize_device(mesh_device)
        dt = (time.time() - t0) / N_CALLS
        od = mod(xd, rd)
        outs[impl] = ttnn.to_torch(ttnn.get_device_tensors(od)[0])
        print(f"EXPERTS_TIME impl={impl} layer={LAYER} chunk={chunk} per_call_ms={dt * 1e3:.1f} load_s={load:.1f}")
        for t in (od, xd, rd):
            ttnn.deallocate(t)
        mod.free()
        del mod
    a, b = outs["bfp8"], outs["fp4_decode"]
    print(f"EXPERTS_SAME bfp8 vs fp4_decode bit-identical (chip 0): {bool(torch.equal(a, b))}")
    if "bfp4" in outs:
        d = (outs["bfp4"].double() - a.double()).norm() / a.double().norm()
        print(f"EXPERTS_BFP4 rel vs bfp8 (chip 0, random inputs): {d.item():.4g}")


def _bfp8_roundtrip(t: torch.Tensor) -> torch.Tensor:
    """t through ttnn's host bfloat8_b packing and back (no device)."""
    return ttnn.to_torch(ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()


@pytest.mark.timeout(3600)
def test_bfp8_check_matches_ttnn():
    """tt/experts._bfp8_exact agrees with ttnn's own bfloat8_b packing: True on every routed expert matrix (W^T) of the
    run's layers, and on a crafted 7-bit grid; False (and the roundtrip differs) on 8-bit / random values."""
    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    g = torch.Generator().manual_seed(0)
    grid = (torch.randint(-127, 128, (64, 256), generator=g).float()) * 2.0 ** torch.randint(
        -8, 8, (64, 1), generator=g
    )
    off = grid.clone()
    off[:, 0::16], off[:, 1::16] = 1.0, 1.0 + 2.0**-7  # 8 significant bits under the group's exponent (bf16-exact)
    rnd = torch.randn(64, 256, generator=g)
    for t, want in ((grid, True), (off, False), (rnd, False)):
        t = t.to(torch.bfloat16).float()
        same = bool(torch.equal(_bfp8_roundtrip(t), t))
        assert E._bfp8_exact(t) == want == same, (want, same)
    layers = [int(i) for i in S.layers()]
    n = bad_t = 0
    for layer in layers:
        for e in range(cfg.n_experts):
            p = ck.expert(layer, e)
            for nm in E.NAMES:
                w = E.fp4_dequant(*p[nm])  # [N, K] as stored: what tt/experts.py uploads
                ok = E._bfp8_exact(w)
                if e < 4:  # the packing itself on a few experts per layer (slow on host)
                    assert ok == bool(torch.equal(_bfp8_roundtrip(w), w)), (layer, e, nm)
                assert ok, (layer, e, nm)
                bad_t += int(not E._bfp8_exact(w.t().contiguous()))
                n += 1
    print(f"BFP8_EXACT layers={layers} matrices={n} all exact as stored [N, K]; not exact as W^T: {bad_t}")


def _bfp4_roundtrip(t: torch.Tensor) -> torch.Tensor:
    """t through ttnn's host bfloat4_b packing and back (no device)."""
    return ttnn.to_torch(ttnn.from_torch(t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()


@pytest.mark.timeout(3600)
def test_bfp4_round_matches_ttnn():
    """tt/experts._bfp4_round == ttnn's own bfloat4_b packing (random, ties, saturation, a few expert matrices), then
    per run layer over every expert: the fraction of values bfloat4_b changes and the worst per-matrix relative change
    ||q(W) - W|| / ||W|| (the numbers EXPERTS_IMPL=bfp4 logs at load)."""
    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(64, 256, generator=g) * 2.0 ** torch.randint(-8, 8, (64, 1), generator=g)
    ties = torch.tensor([6.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7.5, 0.25, 0.75, -0.5, -1.5, -2.5, 3.0, 1.0, 0.0])
    sat = torch.tensor([15.0, 14.0, 13.0, 1.0] + [0.0] * 12)
    for t in (rnd, ties.repeat(32, 16), sat.repeat(32, 16)):
        t = t.to(torch.bfloat16).float()
        assert torch.equal(E._bfp4_round(t), _bfp4_roundtrip(t)), "bfp4 emulation != ttnn packing"
    layers = [int(i) for i in S.layers()]
    for layer in layers:
        changed = total = 0
        worst = (0.0, None)
        for e in range(cfg.n_experts):
            p = ck.expert(layer, e)
            for nm in E.NAMES:
                w = E.fp4_dequant(*p[nm])  # [N, K] as stored: what tt/experts.py uploads
                if e < 2 and layer == layers[0]:
                    assert torch.equal(E._bfp4_round(w), _bfp4_roundtrip(w)), (layer, e, nm)
                n_ch, rel = E._bfp4_change(w)
                changed, total = changed + n_ch, total + w.numel()
                if float(rel) > worst[0]:
                    worst = (float(rel), (e, nm))
        print(
            f"BFP4_CHANGE layer={layer} experts={cfg.n_experts} values_changed={100 * changed / total:.2f} % "
            f"worst_matrix_rel={worst[0]:.4g} at (expert, matrix)={worst[1]}"
        )


@mesh_parametrize
def test_transpose_b_bfp4(mesh_device):
    """ttnn.linear(x, W bfp4 [N, K], transpose_b=True) == ttnn.linear(x, q4(W)^T bf16 [K, N]) bit for bit
    (q4 = _bfp4_round: the device multiplies exactly the values the load-time stats describe), HiFi4 + fp32 acc."""
    S = spec()
    ck = hooks._checkpoint(S)
    p = ck.expert(LAYER, 3)
    w = {nm: E.fp4_dequant(*p[nm]) for nm in E.NAMES}
    cfg = E.compute_config(mesh_device)
    g = torch.Generator().manual_seed(1)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    for nm, wn in (("w1 | w3", torch.cat([w["w1"], w["w3"]])), ("w2", w["w2"])):
        n_, k_ = wn.shape
        xh = torch.randn(2048, k_, generator=g).to(torch.bfloat16)
        xd = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        wb = ttnn.from_torch(wn, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        wq = E._bfp4_round(wn)
        wt = ttnn.from_torch(
            wq.t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep
        )
        kw = dict(dtype=ttnn.float32, compute_kernel_config=cfg)
        a = ttnn.to_torch(ttnn.get_device_tensors(ttnn.linear(xd, wb, transpose_b=True, **kw))[0])
        b = ttnn.to_torch(ttnn.get_device_tensors(ttnn.linear(xd, wt, **kw))[0])
        ref = xh.double() @ wn.double().t()
        rel = ((a.double() - ref).norm() / ref.norm()).item()
        print(f"TRANSPOSE_B_BFP4 {nm}: equal={bool(torch.equal(a, b))} rel_vs_exact_W={rel:.4g}")
        assert torch.equal(a, b), nm


@mesh_parametrize
def test_transpose_b_exact(mesh_device):
    """ttnn.linear(x, W bfp8 [N, K], transpose_b=True) == ttnn.linear(x, W^T bf16 [K, N]) bit for bit on one expert's
    w1 | w3 and w2 (the transpose happens in the matmul, not as a lossy bfp8 -> bfp8 transpose), HiFi4 + fp32 acc."""
    S = spec()
    ck = hooks._checkpoint(S)
    p = ck.expert(LAYER, 3)  # expert 3: its w2^T is not bfp8-exact, its w2 is
    w = {nm: E.fp4_dequant(*p[nm]) for nm in E.NAMES}
    cfg = E.compute_config(mesh_device)
    g = torch.Generator().manual_seed(1)
    for nm, wn in (("w1 | w3", torch.cat([w["w1"], w["w3"]])), ("w2", w["w2"])):
        n_, k_ = wn.shape
        xh = torch.randn(2048, k_, generator=g).to(torch.bfloat16)
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        xd = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        wb = ttnn.from_torch(wn, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        wt = ttnn.from_torch(
            wn.t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep
        )
        kw = dict(dtype=ttnn.float32, compute_kernel_config=cfg)
        a = ttnn.linear(xd, wb, transpose_b=True, **kw)
        b = ttnn.linear(xd, wt, **kw)
        ttnn.synchronize_device(mesh_device)
        times = {}
        for tag, fn in (
            ("bfp8_nk_transpose_b", lambda: ttnn.linear(xd, wb, transpose_b=True, **kw)),
            ("bf16_kn", lambda: ttnn.linear(xd, wt, **kw)),
        ):
            t0 = time.time()
            for _ in range(5):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(mesh_device)
            times[tag] = (time.time() - t0) / 5 * 1e3
        at = ttnn.to_torch(ttnn.get_device_tensors(a)[0])
        bt = ttnn.to_torch(ttnn.get_device_tensors(b)[0])
        ref = xh.double() @ wn.double().t()
        rel = ((at.double() - ref).norm() / ref.norm()).item()
        print(
            f"TRANSPOSE_B {nm}: equal={bool(torch.equal(at, bt))} max|a-b|={(at - bt).abs().max().item():.3g} "
            f"rel_vs_fp64={rel:.3g} ms={times}"
        )
        assert torch.equal(at, bt), nm
