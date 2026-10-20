# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Timed loop of the routed experts (tt/experts.py:TtExperts) of one layer at the run's chunk, every EXPERTS_IMPL.

Not a gate: a measurement for BREADCRUMBS / the EXPERTS_IMPL description. Random q-able inputs (normal x, top-6
routing with sum-1.5 weights), 2 warm-up calls, then the mean wall time of N_CALLS synchronized calls (the forward
has no host sync, so this is the device time of one layer per chunk).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tt/perf_experts.py

test_unified_vs_loop: EXPERTS_KERNEL=unified vs loop on the same inputs (P.2: layer 0, 2x2, chunk 2048: bfp8 317.7 vs
20.4 ms, bfp4 315.1 vs 20.0 ms per call; unified vs loop rel 6e-5).

test_weights_bfp4: EXPERTS_IMPL bfp8 vs bfp4 on the default path (P.10: flat + weighted combine, bf16 sums, glm;
layers 0 / 24, 2x2, chunk 2048: 4.47 vs 3.47 ms per call; vs float64 bfp8 rel 0.020, bfp4 0.18-0.20, bfp4 vs
float64 with bfp4-rounded weights 0.020).
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
N_REF = 128  # tokens of test_weighted_vs_separate's float64 reference


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


@pytest.mark.timeout(3600)
@mesh_parametrize
@pytest.mark.parametrize("impl", ["bfp8", "bfp4"])
def test_unified_vs_loop(mesh_device, impl, monkeypatch):
    """EXPERTS_KERNEL=unified (one fused unified_routed_expert_moe) vs loop on the same random inputs: rel / PCC of
    the outputs (both device paths; the fp8 q(h) turns GEMM-order differences into whole e4m3 steps, DS41-1) and the
    mean time per call (N_CALLS synchronized calls after 2 warm-ups)."""
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
    for kernel in ("loop", "unified"):
        monkeypatch.setenv("DS41_EXPERTS_KERNEL", kernel)
        mod = E.build_experts(mesh_device, ck, cfg, LAYER, chunk, impl=impl)
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
        outs[kernel] = torch.cat([ttnn.to_torch(t).float().reshape(-1) for t in ttnn.get_device_tensors(od)])
        print(f"EXPERTS_KERNEL_TIME impl={impl} kernel={kernel} layer={LAYER} chunk={chunk} per_call_ms={dt * 1e3:.2f}")
        for t in (od, xd, rd):
            ttnn.deallocate(t)
        mod.free()
        del mod
    a, b = outs["loop"].double(), outs["unified"].double()
    rel = ((b - a).norm() / a.norm()).item()
    ac, bc = a - a.mean(), b - b.mean()
    pcc = (ac @ bc / (ac.norm() * bc.norm())).item()
    print(
        f"EXPERTS_KERNEL_CMP impl={impl} unified vs loop: rel {rel:.5f} pcc {pcc:.7f} finite {bool(torch.isfinite(b).all())}"
    )
    assert bool(torch.isfinite(b).all()) and pcc > 0.999, (rel, pcc)


@pytest.mark.timeout(3600)
@mesh_parametrize
@pytest.mark.parametrize("impl", ["bfp8", "bfp4"])
@pytest.mark.parametrize("kernels", ["flat,unified", "flat,unified,loop"], ids=["fu", "ful"])
def test_kernels(mesh_device, impl, kernels, monkeypatch):
    """EXPERTS_KERNEL A/B (P.3: flat vs unified vs loop) on the same random inputs, layer LAYER at the run's chunk: the
    mean time of a whole experts call (dispatch .. reduce, no host sync) and of its kernel part alone (the fused op or
    the loop, timed with a device sync before and after it), plus rel / PCC of every kernel's output vs the first."""
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
    for kernel in kernels.split(","):
        monkeypatch.setenv("DS41_EXPERTS_KERNEL", kernel)
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
        name = {"flat": "_flat", "unified": "_unified", "loop": "_loop"}[kernel]
        orig, spent = getattr(mod, name), []

        def timed(*a, _orig=orig, _spent=spent):
            ttnn.synchronize_device(mesh_device)
            t = time.time()
            out = _orig(*a)
            ttnn.synchronize_device(mesh_device)
            _spent.append(time.time() - t)
            return out

        setattr(mod, name, timed)
        for _ in range(N_CALLS):
            ttnn.deallocate(mod(xd, rd))
        k_ms = 1e3 * sum(spent) / len(spent)
        od = mod(xd, rd)
        outs[kernel] = torch.cat([ttnn.to_torch(t).float().reshape(-1) for t in ttnn.get_device_tensors(od)])
        print(
            f"EXPERTS_KERNELS impl={impl} kernel={kernel} layer={LAYER} chunk={chunk} call_ms={dt * 1e3:.2f} "
            f"kernel_part_ms={k_ms:.2f} load_s={load:.1f}"
        )
        for t in (od, xd, rd):
            ttnn.deallocate(t)
        mod.free()
        del mod
    first = kernels.split(",")[0]
    a = outs[first].double()
    for kernel, o in outs.items():
        b = o.double()
        rel = ((b - a).norm() / a.norm()).item()
        ac, bc = a - a.mean(), b - b.mean()
        pcc = (ac @ bc / (ac.norm() * bc.norm())).item()
        print(f"EXPERTS_KERNELS_CMP impl={impl} {kernel} vs {first}: rel {rel:.5f} pcc {pcc:.7f}")
        assert bool(torch.isfinite(b).all()) and pcc > 0.999, (kernel, rel, pcc)


def _weighted_params() -> dict:
    """The run's device params with the fabric payload EXPERTS_COMBINE=weighted needs (bringup/hooks.py), so that
    both paths run on one mesh."""
    from models.demos.common.bringup.testing import harness
    from models.demos.deepseek_v41_flash_d_p.tt.weighted_combine import FABRIC_PAYLOAD

    return harness.device_params(hooks._WeightedPayloadSpec(spec(), FABRIC_PAYLOAD))


def _host(mesh, t) -> torch.Tensor:
    from models.demos.deepseek_v41_flash_d_p.tt.layout import hidden_split_to_host

    return hidden_split_to_host(mesh, t).double()


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_weighted_params()], indirect=True, ids=["box_payload"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
@pytest.mark.parametrize("combines", ["separate,weighted", "weighted"], ids=["ab", "w"])
@pytest.mark.parametrize("prof", [False, True], ids=["time", "ops"])
def test_weighted_vs_separate(mesh_device, combines, prof):
    """EXPERTS_COMBINE A/B (P.8) on the same random inputs (layer LAYER, the run's chunk, the current NUMERICS /
    FFN_NORM_DTYPE / EXPERTS_PRECISION / CCL_IMPL settings): the mean wall time of a whole experts call (q(x) ..
    reduce over the row, synchronized: the device time of one layer's experts per chunk), weighted vs separate rel
    error / PCC / worst row, and the determinism check (weighted twice on the same inputs: elements that differ).
    With NUMERICS=official separate rounds q(r h) to fp8 before w2 and weighted cannot (the weight moves to the
    combine), so the rel error there includes that rounding; with NUMERICS=native both run the same math.
    prof (ids "ops"; run with the device profiler env, testing/profiler.py PROFILER_ENV): one more call per path in the
    profiler's op mode, the device time per op printed (EXPERTS_COMBINE_OPS)."""
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(chunk, cfg.dim, generator=g)
    sc = torch.rand(chunk, cfg.n_experts, generator=g)
    top = sc.topk(cfg.topk_experts, dim=-1)
    dense = torch.zeros(chunk, cfg.n_experts)
    dense.scatter_(1, top.indices, 1.5 * top.values / top.values.sum(-1, keepdim=True))
    xdt = ttnn.bfloat16 if settings.get("FFN_NORM_DTYPE") == "bf16" else ttnn.float32
    tag = " ".join(f"{k}={settings.get(k)}" for k in ("NUMERICS", "FFN_NORM_DTYPE", "EXPERTS_PRECISION", "CCL_IMPL"))
    outs, again = {}, None
    for combine in combines.split(","):  # (separate first: its programs are built before weighted's)
        t0 = time.time()
        mod = E.build_experts(mesh_device, ck, cfg, LAYER, chunk, combine=combine)
        load = time.time() - t0
        xd = row_split_to_device(mesh_device, x, dtype=xdt)
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
        outs[combine] = _host(mesh_device, od)
        ttnn.deallocate(od)
        if combine == "weighted":
            od = mod(xd, rd)
            again = _host(mesh_device, od)
            ttnn.deallocate(od)
        if prof:
            from models.demos.common.bringup.testing import profiler

            profiler.enable(mesh_device, ops=True)
            profiler.signpost(f"experts_{combine}")
            ttnn.deallocate(mod(xd, rd))
            profiler.signpost("end")
            res = profiler.result()
            profiler.disable()
            tot = 0.0
            for key, rows_ in res["op_ns"].items():
                if not key.startswith("experts_"):
                    continue
                for r_ in rows_:
                    tot += r_["ns"]
                    print(
                        f"EXPERTS_COMBINE_OPS {combine} {r_['op'][:60]:60s} calls {r_['calls']:3d} "
                        f"{r_['ns'] / 1e3:9.1f} us  {r_['shape'][:70]}"
                    )
            print(f"EXPERTS_COMBINE_OPS {combine} total {tot / 1e3:.1f} us (device, slowest chip per op)")
        print(
            f"EXPERTS_COMBINE_TIME combine={combine} layer={LAYER} chunk={chunk} call_ms={dt * 1e3:.2f} load_s={load:.1f} {tag}"
        )
        for t in (xd, rd):
            ttnn.deallocate(t)
        mod.free()
        del mod
    # float64 reference on N_REF tokens: y = sum_k w_k w2(silu(min(w1 q, 10)) clamp(w3 q, +-10)), q = x as the device
    # reads it (bf16 cast natively, fp8 q-dq officially), no rounding of h (the exact math both paths approximate)
    sel = torch.arange(0, chunk, chunk // N_REF)
    xq = x[sel].double()
    if xdt == ttnn.bfloat16:
        xq = xq.to(torch.bfloat16).double()
    if settings.get("NUMERICS") == "official":
        from models.demos.deepseek_v41_flash_d_p.reference.kernels import act_quant_dequant

        xq = act_quant_dequant(xq.float(), 32).double()
    else:
        xq = xq.to(torch.bfloat16).double()
    ref = torch.zeros(len(sel), cfg.dim, dtype=torch.float64)
    ds = dense[sel].double()
    lim = float(cfg.swiglu_limit)
    for e in sorted(set(torch.nonzero(ds)[:, 1].tolist())):
        rows = torch.nonzero(ds[:, e]).flatten()
        p = ck.expert(LAYER, e)
        w1, w3, w2 = ((wround or (lambda t: t))(E.fp4_dequant(*p[nm])).double() for nm in E.NAMES)
        g_, u_ = xq[rows] @ w1.T, xq[rows] @ w3.T
        h = torch.nn.functional.silu(g_.clamp(max=lim)) * u_.clamp(-lim, lim)
        ref[rows] += ds[rows, e : e + 1] * (h @ w2.T)
    for name, o in outs.items():
        d = o[sel] - ref
        rel = (d.norm() / ref.norm()).item()
        bias = ((d * ref).sum() / (ref * ref).sum()).item()
        rr = o[sel].norm(dim=1) / ref.norm(dim=1)
        print(
            f"EXPERTS_COMBINE_REF {name} vs float64 ({len(sel)} tokens): rel {rel:.6f} bias {bias:+.6f} row_norm_ratio "
            f"mean {rr.mean().item():.6f} [{rr.min().item():.5f}, {rr.max().item():.5f}] {tag}"
        )
    b = outs["weighted"]
    nd = int((again != b).sum().item())
    dmax = ((again - b).abs().max() / b.abs().max()).item()
    print(
        f"EXPERTS_COMBINE_DETERMINISM weighted twice: {nd} of {b.numel()} elements differ (max |diff| / max|y| "
        f"{dmax:.3g}) {tag}"
    )
    assert bool(torch.isfinite(b).all())
    if "separate" not in outs:
        return
    a = outs["separate"]
    rel = ((b - a).norm() / a.norm()).item()
    ac, bc = a - a.mean(), b - b.mean()
    pcc = (ac.reshape(-1) @ bc.reshape(-1) / (ac.norm() * bc.norm())).item()
    row = ((b - a).norm(dim=1) / a.norm(dim=1).clamp_min(1e-30)).max().item()
    ratio = b.norm(dim=1) / a.norm(dim=1).clamp_min(1e-30)
    print(
        f"EXPERTS_COMBINE_CMP weighted vs separate: rel {rel:.6f} pcc {pcc:.8f} worst_row {row:.5f} "
        f"row_norm_ratio [{ratio.min().item():.5f}, {ratio.max().item():.5f}] {tag}"
    )
    assert pcc > 0.999, (rel, pcc)


def _wc_setup(mesh_device, route, S, **wc_kw):
    """test_weighted_combine_routes' harness: fake-weight flat expert, WeightedCombine(**wc_kw), the routing forced
    by ``route``, gathered x / ids / weights. Returns (wc, fe, xg, ids_g, wts_g)."""
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

    from models.demos.deepseek_v41_flash_d_p.tt.weighted_combine import WeightedCombine

    S_ = spec()
    cfg = hooks._config(S_)
    R, C = tuple(mesh_device.shape)
    E, K, H, I = cfg.n_experts, cfg.topk_experts, cfg.dim, cfg.inter
    epc = E // (R * C)
    Eg = R * epc
    gids = [[r * epc + le for le in range(epc)] for r in range(R) for c in range(C)]
    fe = FlatRoutedExpert(
        mesh_device,
        "fake",
        m=max(256, R * S),
        H=H,
        I=I,
        gids=gids,
        n_global=Eg,
        wdtype="bf8",
        act="clamped_silu",
        pin=1,
        weights_nk=True,
        routed=False,
    )
    wc = WeightedCombine(
        mesh_device,
        dict(fe.plan),
        H=H,
        I=I,
        epc=epc,
        K=K,
        S=S,
        out_dtype=ttnn.float32,
        x_bf16=False,
        h_bf16=False,
        **wc_kw,
    )
    g = torch.Generator().manual_seed(1)
    ids = torch.zeros(R, S, K, dtype=torch.int64)  # per mesh row r, its tokens' global expert ids
    for r in range(R):
        for t in range(S):
            for k in range(K):
                c = k % C  # half the slots in each column
                if route == "local":
                    rr = r
                elif route == "remote":
                    rr = (r + 1) % R
                else:
                    rr = int(torch.randint(0, R, (1,), generator=g))
                ids[r, t, k] = c * Eg + rr * epc + (t * K + k) % epc  # (distinct experts within a token)
    w = torch.rand(R, S, K, generator=g) + 0.5
    idx_d = ttnn.from_torch(
        ids.reshape(1, R * S, K).unsqueeze(0).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(2, None)),
    )
    w_d = ttnn.from_torch(
        w.reshape(1, 1, R * S, K),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(2, None)),
    )
    x_d = ttnn.from_torch(
        (torch.randn(1, 1, R * S, H, generator=g) * 0.3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(2, None)),
    )
    xg = wc.gather_x(x_d)
    ids_g, wts_g = wc.routing(idx_d, w_d)
    return wc, fe, xg, ids_g, wts_g


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [_weighted_params()], indirect=True, ids=["box_payload"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
@pytest.mark.parametrize("route", ["local", "remote", "random"])
@pytest.mark.parametrize("S", [64, 256, 448, 468, 512])
@pytest.mark.parametrize("arena_kb", [0, 1252], ids=["arena_auto", "arena1252"])
def test_weighted_combine_routes(mesh_device, route, S, arena_kb):
    """Debug harness of tt/weighted_combine.py alone (no model inputs): fake (uninitialized) expert weights, S tokens per
    chip, the routing forced so that every row stays on its home chip (local), goes to the other chip of the column
    (remote) or is random; the call must finish, then the combine's sums are checked against the y rows it read only
    when ``route`` is local... (fake weights: only completion and finiteness of the rows that got experts)."""
    wc, fe, xg, ids_g, wts_g = _wc_setup(mesh_device, route, S, arena_bytes=arena_kb * 1024 or None)
    t0 = time.time()
    out = wc(fe, xg, ids_g, wts_g, 4)
    ttnn.synchronize_device(mesh_device)
    print(f"WC_ROUTES route={route} S={S} passes={wc.P} first call {time.time() - t0:.2f} s")
    o = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    print(f"WC_ROUTES route={route} S={S} out {tuple(o.shape)} finite {bool(torch.isfinite(o).all())}")


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [_weighted_params()], indirect=True, ids=["box_payload"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
@pytest.mark.parametrize("route", ["local", "remote", "random"])
@pytest.mark.parametrize("S", [1024])
def test_weighted_combine_routes_bf16(mesh_device, route, S):
    """COMBINE_ACC=bf16 (P.9): the 2x2 chunk's 1024 home tokens per chip in ONE pass (bf16 sums, 9 accumulators),
    every row local / remote / random (fake weights: the call must finish, in one pass)."""
    wc, fe, xg, ids_g, wts_g = _wc_setup(mesh_device, route, S, acc_dtype=ttnn.bfloat16)
    assert wc.P == 1, f"bf16 sums: {S} home tokens in {wc.P} passes"
    t0 = time.time()
    out = wc(fe, xg, ids_g, wts_g, 4)
    ttnn.synchronize_device(mesh_device)
    print(f"WC_ROUTES_BF16 route={route} S={S} passes={wc.P} x {wc.S_pass} first call {time.time() - t0:.2f} s")
    o = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    print(f"WC_ROUTES_BF16 route={route} S={S} out {tuple(o.shape)} finite {bool(torch.isfinite(o).all())}")


def _ref64(ck, cfg, layer, x, dense, sel, xdt, numerics, wround=None):
    """float64 y = sum_k w_k w2(silu(min(w1 q, lim)) clamp(w3 q, +-lim)) on the rows ``sel``, q = x as the device reads it
    (bf16 cast natively, fp8 q-dq officially), no rounding of h. wround: the stored [N, K] weights as the device holds
    them (None: the checkpoint's exact fp4 values; E._bfp4_round: bfloat4_b's)."""
    xq = x[sel].double()
    if xdt == ttnn.bfloat16:
        xq = xq.to(torch.bfloat16).double()
    if numerics == "official":
        from models.demos.deepseek_v41_flash_d_p.reference.kernels import act_quant_dequant

        xq = act_quant_dequant(xq.float(), 32).double()
    else:
        xq = xq.to(torch.bfloat16).double()
    ref = torch.zeros(len(sel), cfg.dim, dtype=torch.float64)
    ds = dense[sel].double()
    lim = float(cfg.swiglu_limit)
    for e in sorted(set(torch.nonzero(ds)[:, 1].tolist())):
        rows = torch.nonzero(ds[:, e]).flatten()
        p = ck.expert(layer, e)
        w1, w3, w2 = ((wround or (lambda t: t))(E.fp4_dequant(*p[nm])).double() for nm in E.NAMES)
        g_, u_ = xq[rows] @ w1.T, xq[rows] @ w3.T
        h = torch.nn.functional.silu(g_.clamp(max=lim)) * u_.clamp(-lim, lim)
        ref[rows] += ds[rows, e : e + 1] * (h @ w2.T)
    return ref


def _cmp(a, b):
    d = a - b
    rel = (d.norm() / b.norm()).item()
    bias = ((d * b).sum() / (b * b).sum()).item()
    row = (d.norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)).max().item()
    return rel, bias, row


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_weighted_params()], indirect=True, ids=["box_payload"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
@pytest.mark.parametrize("layer", [0, 24], ids=["L00", "L24"])
@pytest.mark.parametrize("prof", [False, True], ids=["time", "ops"])
def test_combine_acc(mesh_device, layer, prof):
    """COMBINE_ACC A/B (P.9) on the same random inputs (EXPERTS_COMBINE=weighted, the current NUMERICS /
    FFN_NORM_DTYPE / EXPERTS_PRECISION / CCL_IMPL settings): fp32 sums in passes vs bf16 sums in one pass. Per side the
    pass count, the mean wall time of a whole experts call (q(x) .. the weighted combine's partial, synchronized: the
    device time of one layer's experts per chunk) and, with prof (ids "ops"; the profiler env of testing/profiler.py),
    the device time per op of one more call. Precision: each vs a float64 reference on N_REF tokens, bf16 vs fp32 on
    every row, and bf16 twice on the same inputs (the fixed order: bit-identical)."""
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(chunk, cfg.dim, generator=g)
    sc = torch.rand(chunk, cfg.n_experts, generator=g)
    top = sc.topk(cfg.topk_experts, dim=-1)
    dense = torch.zeros(chunk, cfg.n_experts)
    dense.scatter_(1, top.indices, 1.5 * top.values / top.values.sum(-1, keepdim=True))
    xdt = ttnn.bfloat16 if settings.get("FFN_NORM_DTYPE") == "bf16" else ttnn.float32
    tag = " ".join(f"{k}={settings.get(k)}" for k in ("NUMERICS", "FFN_NORM_DTYPE", "EXPERTS_PRECISION", "CCL_IMPL"))
    outs, again = {}, None
    for acc in ("fp32", "bf16"):
        t0 = time.time()
        mod = E.build_experts(mesh_device, ck, cfg, layer, chunk, combine="weighted", combine_acc=acc)
        load = time.time() - t0
        P = mod.wc.P
        if acc == "bf16":
            assert P == 1, f"COMBINE_ACC=bf16 ran {P} passes"
        xd = row_split_to_device(mesh_device, x, dtype=xdt)
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
        outs[acc] = _host(mesh_device, od)
        ttnn.deallocate(od)
        if acc == "bf16":
            od = mod(xd, rd)
            again = _host(mesh_device, od)
            ttnn.deallocate(od)
        if prof:
            from models.demos.common.bringup.testing import profiler

            profiler.enable(mesh_device, ops=True)
            profiler.signpost(f"experts_{acc}")
            ttnn.deallocate(mod(xd, rd))
            profiler.signpost("end")
            res = profiler.result()
            profiler.disable()
            tot = 0.0
            for key, rows_ in res["op_ns"].items():
                if not key.startswith("experts_"):
                    continue
                for r_ in rows_:
                    tot += r_["ns"]
                    print(
                        f"COMBINE_ACC_OPS L{layer:02d} {acc} {r_['op'][:60]:60s} calls {r_['calls']:3d} "
                        f"{r_['ns'] / 1e3:9.1f} us  {r_['shape'][:70]}"
                    )
            print(f"COMBINE_ACC_OPS L{layer:02d} {acc} total {tot / 1e3:.1f} us (device, slowest chip per op)")
        print(
            f"COMBINE_ACC_TIME L{layer:02d} acc={acc} passes={P} x {mod.wc.S_pass} chunk={chunk} "
            f"call_ms={dt * 1e3:.2f} load_s={load:.1f} {tag}"
        )
        for t in (xd, rd):
            ttnn.deallocate(t)
        mod.free()
        del mod
    sel = torch.arange(0, chunk, chunk // N_REF)
    ref = _ref64(ck, cfg, layer, x, dense, sel, xdt, settings.get("NUMERICS"))
    for acc, o in outs.items():
        rel, bias, row = _cmp(o[sel], ref)
        print(
            f"COMBINE_ACC_REF L{layer:02d} {acc} vs float64 ({len(sel)} tokens): rel {rel:.6f} bias {bias:+.6f} "
            f"worst_row {row:.5f} {tag}"
        )
    rel, bias, row = _cmp(outs["bf16"], outs["fp32"])
    print(
        f"COMBINE_ACC_CMP L{layer:02d} bf16 vs fp32 (all {chunk} rows): rel {rel:.6f} bias {bias:+.6f} worst_row {row:.5f}"
    )
    b = outs["bf16"]
    nd = int((again != b).sum().item())
    print(f"COMBINE_ACC_DETERMINISM L{layer:02d} bf16 twice: {nd} of {b.numel()} elements differ")
    assert bool(torch.isfinite(b).all())
    assert nd == 0, f"bf16 sums not bit-identical between calls ({nd} values)"


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [_weighted_params()], indirect=True, ids=["box_payload"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
@pytest.mark.parametrize("fid", [0, 4], ids=["LoFi", "HiFi4"])
@pytest.mark.parametrize("passes", [1, 2, 3])
@pytest.mark.parametrize("S", [512, 1024])
def test_weighted_passes_cost(mesh_device, S, passes, fid):
    """What a pass costs: S home tokens per chip (fake weights, random routing) in passes (1 where it fits), the mean
    wall time of N_CALLS synchronized calls (the flat expert streams every weight once a pass). S=1024 is the model's
    2x2 chunk (2048): T(3) - T(2) = one pass's fixed cost, T(2) minus that = what one pass would take."""
    from ttnn.bringup.combine_rf.sparse_combine_rf import accumulator_bytes

    from models.demos.deepseek_v41_flash_d_p.tt.weighted_combine import ARENA_MAX

    R = tuple(mesh_device.shape)[0]
    if accumulator_bytes(R, 9, hooks._config(spec()).dim, -(-S // passes), out_dtype=ttnn.float32) > ARENA_MAX:
        pytest.skip(f"{S} home tokens do not fit {passes} pass(es)")
    wc, fe, xg, ids_g, wts_g = _wc_setup(mesh_device, "random", S, passes=passes)
    assert wc.P == passes, (wc.P, passes)
    for _ in range(2):
        ttnn.deallocate(wc(fe, xg, ids_g, wts_g, fid))
    ttnn.synchronize_device(mesh_device)
    t0 = time.time()
    for _ in range(N_CALLS):
        ttnn.deallocate(wc(fe, xg, ids_g, wts_g, fid))
    ttnn.synchronize_device(mesh_device)
    dt = (time.time() - t0) / N_CALLS
    print(f"WC_PASSES S={S} fid={fid} passes={wc.P} x {wc.S_pass} call_ms={dt * 1e3:.3f}")


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [_weighted_params()], indirect=True, ids=["box_payload"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
def test_weighted_refuses_small_arena(mesh_device):
    """sparse_combine_rf refuses an arena that cannot hold its accumulators' fp32 sums (S = 512 home tokens, 9
    accumulators: 1140 KB of sums each, in the flat expert's 1156 KB arena) with the numbers, instead of building."""
    wc, fe, xg, ids_g, wts_g = _wc_setup(mesh_device, "local", 512, arena_bytes=1156 * 1024)
    with pytest.raises(ValueError, match="home tokens over 9 accumulators") as e:
        wc(fe, xg, ids_g, wts_g, 4)
    print(f"WC_REFUSE {e.value}")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_weighted_params()], indirect=True, ids=["box_payload"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
@pytest.mark.parametrize("layer", [0, 24], ids=["L00", "L24"])
@pytest.mark.parametrize("prof", [False, True], ids=["time", "ops"])
def test_weights_bfp4(mesh_device, layer, prof):
    """EXPERTS_IMPL A/B (P.10) on the same random inputs, the current default path otherwise (EXPERTS_KERNEL /
    EXPERTS_COMBINE / COMBINE_ACC / EXPERTS_PRECISION / NUMERICS / FFN_NORM_DTYPE / CCL_IMPL settings): the routed
    experts' weights bfp8 (exact) vs bfp4 (the checkpoint's fp4 width, rule 7 / F117). Per side the flat plan and the
    weighted combine's passes (logged: the plan may change with the smaller weights), the mean wall time of a whole
    synchronized experts call (one layer's experts per chunk) and, with prof (ids "ops"; the profiler env of
    testing/profiler.py), the device time per op of one more call. Precision: each vs a float64 reference with the
    checkpoint's exact fp4 weights on N_REF tokens, bfp4 vs bfp8 on every row, bfp4 twice (bit-identical)."""
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(chunk, cfg.dim, generator=g)
    sc = torch.rand(chunk, cfg.n_experts, generator=g)
    top = sc.topk(cfg.topk_experts, dim=-1)
    dense = torch.zeros(chunk, cfg.n_experts)
    dense.scatter_(1, top.indices, 1.5 * top.values / top.values.sum(-1, keepdim=True))
    xdt = ttnn.bfloat16 if settings.get("FFN_NORM_DTYPE") == "bf16" else ttnn.float32
    keys = ("EXPERTS_KERNEL", "EXPERTS_COMBINE", "COMBINE_ACC", "EXPERTS_PRECISION", "NUMERICS", "FFN_NORM_DTYPE")
    tag = " ".join(f"{k}={settings.get(k)}" for k in keys + ("CCL_IMPL",))
    outs, again = {}, None
    for impl in ("bfp8", "bfp4"):
        t0 = time.time()
        mod = E.build_experts(mesh_device, ck, cfg, layer, chunk, impl=impl)
        load = time.time() - t0
        plan = ""
        if getattr(mod, "fe", None) is not None:
            p = mod.fe.plan
            plan = f"plan nu={p['nu']} np={p['np']} mt={p['mt']} gu={len(p['gu'])} down={p['nd']} arena={p['arena_tiles'] * 2} KB"
        if getattr(mod, "wc", None) is not None:
            plan += f" passes={mod.wc.P} x {mod.wc.S_pass} accumulators={mod.wc.A} free_cells={len(mod.wc.cells)}"
        print(f"BFP4_PLAN L{layer:02d} impl={impl} {plan}")
        xd = row_split_to_device(mesh_device, x, dtype=xdt)
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
        outs[impl] = _host(mesh_device, od)
        ttnn.deallocate(od)
        if impl == "bfp4":
            od = mod(xd, rd)
            again = _host(mesh_device, od)
            ttnn.deallocate(od)
        if prof:
            from models.demos.common.bringup.testing import profiler

            profiler.enable(mesh_device, ops=True)
            profiler.signpost(f"experts_{impl}")
            ttnn.deallocate(mod(xd, rd))
            profiler.signpost("end")
            res = profiler.result()
            profiler.disable()
            tot = 0.0
            for key, rows_ in res["op_ns"].items():
                if not key.startswith("experts_"):
                    continue
                for r_ in rows_:
                    tot += r_["ns"]
                    print(
                        f"BFP4_OPS L{layer:02d} {impl} {r_['op'][:60]:60s} calls {r_['calls']:3d} "
                        f"{r_['ns'] / 1e3:9.1f} us  {r_['shape'][:70]}"
                    )
            print(f"BFP4_OPS L{layer:02d} {impl} total {tot / 1e3:.1f} us (device, slowest chip per op)")
        print(f"BFP4_TIME L{layer:02d} impl={impl} chunk={chunk} call_ms={dt * 1e3:.2f} load_s={load:.1f} {tag}")
        for t in (xd, rd):
            ttnn.deallocate(t)
        mod.free()
        del mod
    sel = torch.arange(0, chunk, chunk // N_REF)
    ref = _ref64(ck, cfg, layer, x, dense, sel, xdt, settings.get("NUMERICS"))
    for impl, o in outs.items():
        rel, bias, row = _cmp(o[sel], ref)
        print(
            f"BFP4_REF L{layer:02d} {impl} vs float64 ({len(sel)} tokens, exact fp4 weights): rel {rel:.6f} "
            f"bias {bias:+.6f} worst_row {row:.5f} {tag}"
        )
    # bfp4 against what its weights hold (the same float64 math with bfloat4_b-rounded weights): the device error of
    # the bfp4 path alone, comparable with bfp8 vs the exact weights
    ref4 = _ref64(ck, cfg, layer, x, dense, sel, xdt, settings.get("NUMERICS"), wround=E._bfp4_round)
    rel, bias, row = _cmp(outs["bfp4"][sel], ref4)
    print(
        f"BFP4_REF L{layer:02d} bfp4 vs float64 with bfp4-rounded weights ({len(sel)} tokens): rel {rel:.6f} "
        f"bias {bias:+.6f} worst_row {row:.5f}"
    )
    rel, bias, row = _cmp(ref4, ref)
    print(
        f"BFP4_REF L{layer:02d} float64: bfp4-rounded vs exact weights: rel {rel:.6f} bias {bias:+.6f} worst_row {row:.5f}"
    )
    rel, bias, row = _cmp(outs["bfp4"], outs["bfp8"])
    print(f"BFP4_CMP L{layer:02d} bfp4 vs bfp8 (all {chunk} rows): rel {rel:.6f} bias {bias:+.6f} worst_row {row:.5f}")
    b = outs["bfp4"]
    nd = int((again != b).sum().item())
    print(f"BFP4_DETERMINISM L{layer:02d} bfp4 twice: {nd} of {b.numel()} elements differ")
    assert bool(torch.isfinite(b).all())
    assert nd == 0, f"bfp4 experts not bit-identical between calls ({nd} values)"
