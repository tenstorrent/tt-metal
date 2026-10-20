# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""SHARED_EXPERT_IMPL A/B (P.12): the shared expert as today's ttnn.linear chain vs the flat routed expert (always-on
slices, bfp8 weights), on one layer at the run's chunk with the layer's real weights and random inputs.

Not a gate: a measurement for BREADCRUMBS / the SHARED_EXPERT_IMPL description. Per side, with the current settings
otherwise (NUMERICS, EXPERTS_PRECISION, FFN_NORM_DTYPE, CCL_IMPL, EXPERTS_COMBINE ...):
  shared   the mean wall time of N_CALLS synchronized shared-expert partials (the model's shared_expert step: the
           forward has no host sync, so this is its device time per layer per chunk); flat: with its own row-major
           conversion of x ("own") and reading the routed experts' gathered q(x) by token index ("rows", the model's
           path under EXPERTS_COMBINE=weighted)
  moe      the mean wall time of the MoE block as the model runs it (router -> experts.partial -> shared.partial ->
           moe_add -> the one reduce-scatter over the row), synchronized
  error    shared_out (partial reduced over the row) vs float64 with the checkpoint's exact fp8 weights on N_REF rows
           (rel L2, bias <out - ref, ref> / |ref|^2, worst row); flat also vs float64 with the bfp8-rounded weights
           (ttnn's host packing: the device path apart from the weight format); the weights' bfp8 statistics
  ops      (prof: the device profiler env of testing/profiler.py) the device time per op of one more partial

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tt/perf_shared_expert.py -k time
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import spec
from models.demos.deepseek_v41_flash_d_p.bringup import hooks
from models.demos.deepseek_v41_flash_d_p.tt import shared_expert as SE
from models.demos.deepseek_v41_flash_d_p.tt.layout import hidden_split_to_host, row_split_to_device
from models.demos.deepseek_v41_flash_d_p.tt.perf_experts import _weighted_params

N_CALLS = 5
N_REF = 256  # rows of the float64 reference (spread over both mesh rows)


def _bfp8_roundtrip(t: torch.Tensor) -> torch.Tensor:
    """t ([N, K] as stored: the bfp8 groups of 16 run along K, as in the op's weights_nk layout) through ttnn's host
    bfloat8_b packing and back (no device)."""
    return ttnn.to_torch(ttnn.from_torch(t.float(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()


def _ref64(x, w1, w3, w2, lim, numerics):
    """float64 shared_out = w2 q(silu(min(w1 q(x), lim)) clamp(w3 q(x), +-lim)); q = the device's input cast (bf16)
    natively, the fp8 act quant-dequant officially (then also of h)."""
    from models.demos.deepseek_v41_flash_d_p.reference.kernels import act_quant_dequant

    xq = x.to(torch.bfloat16).double()
    if numerics == "official":
        xq = act_quant_dequant(xq.float(), 32).double()
    g, u = xq @ w1.double().T, xq @ w3.double().T
    h = torch.nn.functional.silu(g.clamp(max=lim)) * u.clamp(-lim, lim)
    if numerics == "official":
        h = act_quant_dequant(h.float(), 32).double()
    return h @ w2.double().T


def _err(got, ref) -> str:
    d = got - ref
    rel = (d.norm() / ref.norm()).item()
    bias = ((d * ref).sum() / (ref * ref).sum()).item()
    row = (d.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)).max().item()
    return f"rel={rel:.5f} bias={bias:+.5f} worst_row={row:.4f}"


def _timed(mesh, fn, n=N_CALLS) -> float:
    for _ in range(2):
        fn()
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    for _ in range(n):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.time() - t0) / n * 1e3


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_weighted_params()], indirect=True, ids=["box_payload"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
@pytest.mark.parametrize("layer", [0, 24], ids=["L00", "L24"])
@pytest.mark.parametrize("prof", [False, True], ids=["time", "ops"])
def test_shared_flat_vs_linear(mesh_device, layer, prof):
    from models.demos.deepseek_v41_flash_d_p.tt import ccl
    from models.demos.deepseek_v41_flash_d_p.tt.experts import build_experts
    from models.demos.deepseek_v41_flash_d_p.tt.router import build_router
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    mesh = mesh_device
    R = mesh.shape[0]
    numerics = settings.get("NUMERICS")
    xdt = ttnn.bfloat16 if settings.get("FFN_NORM_DTYPE") == "bf16" else ttnn.float32
    keys = ("NUMERICS", "EXPERTS_PRECISION", "MATMUL_FIDELITY", "FFN_NORM_DTYPE", "CCL_IMPL", "EXPERTS_COMBINE")
    tag = " ".join(f"{k}={settings.get(k)}" for k in keys)
    g = torch.Generator().manual_seed(layer)
    x = torch.randn(chunk, cfg.dim, generator=g)
    p = f"layers.{layer}.ffn.shared_experts."
    w = {n: ck.weight(p + n + ".weight", torch.float32) for n in ("w1", "w3", "w2")}
    lim = float(cfg.swiglu_limit)
    sel = torch.cat([torch.arange(N_REF // 2), chunk // 2 + torch.arange(N_REF // 2)])
    ref = _ref64(x[sel], w["w1"], w["w3"], w["w2"], lim, numerics)
    wq = {n: _bfp8_roundtrip(t) for n, t in w.items()}
    for n in ("w1", "w3", "w2"):
        d = wq[n] - w[n]
        print(
            f"SHARED_BFP8 L{layer:02d} {n} {tuple(w[n].shape)} values changed {(d != 0).float().mean().item():.4%} "
            f"||dW||/||W|| {(d.norm() / w[n].norm()).item():.4g}"
        )
    ref8 = _ref64(x[sel], wq["w1"], wq["w3"], wq["w2"], lim, numerics)
    print(f"SHARED_REF L{layer:02d} float64 bfp8-rounded weights vs exact: {_err(ref8, ref)}")

    xd = row_split_to_device(mesh, x, dtype=xdt)
    router = build_router(mesh, ck, cfg, layer, chunk)
    experts = build_experts(mesh, ck, cfg, layer, chunk)
    outs = {}
    for impl in ("linear", "flat"):
        t0 = time.time()
        se = SE.build_shared_expert(mesh, ck, cfg, layer, chunk=chunk, impl=impl)
        load = time.time() - t0
        rows_ok = impl == "flat" and se.reads_rows and experts.weighted
        experts.keep_rows = rows_ok

        def moe():
            dense, idx, wts = router(xd)
            a = experts.partial(xd, dense, idx)
            b = se.partial(xd, rows=experts.take_rows())
            sdt = ccl.partial_dtype(mesh) if experts.sum_mode == "gather_add" else ttnn.float32
            s_ = ttnn.add(a, b, dtype=sdt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out = experts.reduce(s_)
            for t in {id(t): t for t in (dense, idx, wts, a, b, s_)}.values():
                ttnn.deallocate(t)
            ttnn.deallocate(out)

        sides = {"own": lambda: ttnn.deallocate(se.partial(xd))}
        if rows_ok:
            # (the routed experts' gathered rows, as the weighted path hands them over; made once, re-used per call)
            rows = experts.wc.gather_x(xd if xdt == ttnn.bfloat16 else ttnn.typecast(xd, ttnn.bfloat16))

            def with_rows():
                r2 = ttnn.clone(rows, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(se.partial(xd, rows=r2))

            clone_ms = _timed(mesh, lambda: ttnn.deallocate(ttnn.clone(rows, memory_config=ttnn.DRAM_MEMORY_CONFIG)))
            sides["rows"] = with_rows
        for name, fn in sides.items():
            ms = _timed(mesh, fn)
            extra = ""
            if name == "rows":
                extra = f" (minus the rows clone {clone_ms:.3f} ms: {ms - clone_ms:.3f})"
            print(f"SHARED_TIME L{layer:02d} impl={impl} x={name} partial_ms={ms:.3f}{extra} load_s={load:.1f} {tag}")

        def routed_only():
            dense, idx, wts = router(xd)
            a = experts.partial(xd, dense, idx)
            r_ = experts.take_rows()
            out = experts.reduce(a)
            for t in {id(t): t for t in (dense, idx, wts, a)}.values():
                ttnn.deallocate(t)
            for t in (r_, out):
                if t is not None:
                    ttnn.deallocate(t)

        moe_ms = _timed(mesh, moe)
        routed_ms = _timed(mesh, routed_only)
        print(
            f"SHARED_MOE L{layer:02d} impl={impl} moe_block_ms={moe_ms:.3f} (router..reduce, synced; without the shared "
            f"expert / moe_add {routed_ms:.3f}) {tag}"
        )
        sides["moe"] = moe
        od = se(xd)
        got = hidden_split_to_host(mesh, od).double()
        ttnn.deallocate(od)
        outs[impl] = got
        print(f"SHARED_ERR L{layer:02d} impl={impl} vs float64 exact weights: {_err(got[sel], ref)}")
        if impl == "flat":
            print(f"SHARED_ERR L{layer:02d} impl=flat vs float64 bfp8-rounded weights: {_err(got[sel], ref8)}")
            if rows_ok:  # (the token-index path equals the own-x path)
                r2 = ttnn.clone(rows, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                pa, pb = se.partial(xd, rows=r2), se.partial(xd)
                ta, tb = (ttnn.to_torch(ttnn.get_device_tensors(t_)[0]) for t_ in (pa, pb))
                print(f"SHARED_ROWS L{layer:02d} rows vs own x bit-identical (chip 0): {bool(torch.equal(ta, tb))}")
                for t_ in (pa, pb):
                    ttnn.deallocate(t_)
            od2 = se(xd)
            again = hidden_split_to_host(mesh, od2).double()
            ttnn.deallocate(od2)
            print(f"SHARED_DET L{layer:02d} flat twice: {int((again != got).sum().item())} values differ")
        if prof:
            from models.demos.common.bringup.testing import profiler

            profiler.enable(mesh, ops=True)
            for name, fn in sides.items():
                profiler.signpost(f"shared_{impl}_{name}")
                fn()
            profiler.signpost("end")
            res = profiler.result()
            profiler.disable()
            for key, rows_ in res["op_ns"].items():
                if not key.startswith("shared_"):
                    continue
                tot = 0.0
                for r_ in rows_:
                    tot += r_["ns"]
                    print(
                        f"SHARED_OPS L{layer:02d} {key} {r_['op'][:60]:60s} calls {r_['calls']:3d} "
                        f"{r_['ns'] / 1e3:9.1f} us  {r_['shape'][:70]}"
                    )
                print(f"SHARED_OPS L{layer:02d} {key} total {tot / 1e3:.1f} us (device, slowest chip per op)")
        if rows_ok:
            ttnn.deallocate(rows)
        experts.keep_rows = False
        del se
    print(f"SHARED_CMP L{layer:02d} flat vs linear: {_err(outs['flat'][sel], outs['linear'][sel])}")
    assert bool(torch.isfinite(outs["flat"]).all())
    ttnn.deallocate(xd)
    experts.free()


VARIANTS = {  # flat_opts of TtSharedExpert (comparison only): the setting's own choice is "setting"
    "setting": {},
    "lofi_x16_h16": dict(x_bf16=True, h_bf16=True, fidelity=0),
    "lofi_x8_h16": dict(x_bf16=False, h_bf16=True, fidelity=0),
    "lofi_x16_h8": dict(x_bf16=True, h_bf16=False, fidelity=0),
    "lofi_x8_h8_srnd": dict(x_bf16=False, h_bf16=False, fidelity=0, stochastic=True),
    "hifi2_x16_h16": dict(x_bf16=True, h_bf16=True, fidelity=2),
    "hifi4_x16_h16": dict(x_bf16=True, h_bf16=True, fidelity=4),
}


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_weighted_params()], indirect=True, ids=["box_payload"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
@pytest.mark.parametrize("layer", [0, 24], ids=["L00", "L24"])
def test_flat_variants(mesh_device, layer):
    """The flat shared expert's tile formats / fidelity / packer rounding (own x path): partial time and shared_out
    vs float64 (exact and bfp8-rounded weights), to pick what SHARED_EXPERT_IMPL=flat ships."""
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    mesh = mesh_device
    numerics = settings.get("NUMERICS")
    xdt = ttnn.bfloat16 if settings.get("FFN_NORM_DTYPE") == "bf16" else ttnn.float32
    g = torch.Generator().manual_seed(layer)
    x = torch.randn(chunk, cfg.dim, generator=g)
    p = f"layers.{layer}.ffn.shared_experts."
    w = {n: ck.weight(p + n + ".weight", torch.float32) for n in ("w1", "w3", "w2")}
    lim = float(cfg.swiglu_limit)
    sel = torch.cat([torch.arange(N_REF // 2), chunk // 2 + torch.arange(N_REF // 2)])
    ref = _ref64(x[sel], w["w1"], w["w3"], w["w2"], lim, numerics)
    wq = {n: _bfp8_roundtrip(t) for n, t in w.items()}
    ref8 = _ref64(x[sel], wq["w1"], wq["w3"], wq["w2"], lim, numerics)
    xd = row_split_to_device(mesh, x, dtype=xdt)
    for name, opts in VARIANTS.items():
        se = SE.build_shared_expert(mesh, ck, cfg, layer, chunk=chunk, impl="flat", flat_opts=opts)
        ms = _timed(mesh, lambda: ttnn.deallocate(se.partial(xd)))
        od = se(xd)
        got = hidden_split_to_host(mesh, od).double()[sel]
        ttnn.deallocate(od)
        print(
            f"SHARED_VAR L{layer:02d} {name:18s} partial_ms={ms:.3f} vs exact: {_err(got, ref)} | "
            f"vs bfp8-rounded: {_err(got, ref8)}"
        )
        del se
    ttnn.deallocate(xd)
