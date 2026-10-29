# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.9 (DS41-132): the shared expert's x / h format in the block MoE's fused head (MOE_HEAD=fused), GEMM_INPUTS.

Not a gate: a measurement for BREADCRUMBS. One layer's MoE block (tt/moe_block_adapter.py, real weights, random x),
the setting as the process has it (DS41_GEMM_INPUTS=bf16: the engine's x / h tiles bf16, the path before FIX.9;
bfp8: bfloat8_b, the reference's fp8 act quant of both):
  SE_ERR   the fused head's shared expert output (the owners' rows) vs float64 with the checkpoint's exact fp8 weights:
           "native" (x and h unrounded: the reference every check compares against) and "q8" (the released code's fp8
           act quant-dequant of x and h, reference/kernels.py act_quant_dequant, per 32): rel L2, bias, worst row
  (SE_PACK_PRECISE: the bfp8 h packed in one rounding; the HEAD line prints it)
  SE_DET   the head twice on the same x: values that differ
  SE_TIME  the head call (FusedHead: x copy, the typecast, the one program) synced wall, and with the device profiler
           (ops=True) the device time per op of the head calls (the program shows as "(other)": run_prepared)

    DS41_GEMM_INPUTS=bf16 PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tt/perf_se_activations.py
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import spec
from models.demos.deepseek_v41_flash_d_p.bringup import hooks
from models.demos.deepseek_v41_flash_d_p.reference import kernels as K
from models.demos.deepseek_v41_flash_d_p.tt.perf_experts import _weighted_params

N_CALLS = 10


def _ref64(x, w1, w3, w2, lim, q8):
    q = (lambda t: K.act_quant_dequant(t.float(), 32).double()) if q8 else (lambda t: t.double())
    xq = q(x)
    g, u = xq @ w1.double().T, xq @ w3.double().T
    h = torch.nn.functional.silu(g.clamp(max=lim)) * u.clamp(-lim, lim)
    return q(h) @ w2.double().T


def _err(got, ref) -> str:
    d = got - ref
    rel = (d.norm() / ref.norm()).item()
    bias = ((d * ref).sum() / (ref * ref).sum()).item()
    row = (d.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)).max().item()
    return f"rel={rel:.5f} bias={bias:+.5f} worst_row={row:.4f}"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_weighted_params()], indirect=True, ids=["box_payload"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
@pytest.mark.parametrize("layer", [0, 24], ids=["L00", "L24"])
def test_se_activations(mesh_device, layer):
    from models.demos.deepseek_v41_flash_d_p.tt.moe_block_adapter import TtMoEBlockLayer
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    mesh = mesh_device
    R, C = tuple(mesh.shape)
    tag = f"GEMM_INPUTS={settings.get('GEMM_INPUTS')} MOE_HEAD={settings.get('MOE_HEAD')}"
    assert settings.get("MOE_IMPL") == "block" and settings.get("MOE_HEAD") == "fused", tag
    x = torch.randn(chunk, cfg.dim, generator=torch.Generator().manual_seed(layer)).to(torch.bfloat16)
    xd = ttnn.from_torch(
        x.view(1, 1, chunk, cfg.dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(2, 3)),
    )
    moe = TtMoEBlockLayer(mesh, ck, cfg, layer, chunk)
    y = moe(xd)  # (builds the block, lays out this layer's head weights)
    ttnn.deallocate(y)
    blk = moe.block()
    fh = blk.fused
    assert fh is not None, "the fused head did not build"
    print(
        f"SE_HEAD L{layer:02d} {tag} engine x {fh.head.eng.x_dtype} h {fh.head.eng.h_dtype} pack_precise {fh.head.eng.pack_precise} arena "
        f"{fh.head.eng.arena_bytes // 1024} KB{blk.fused.describe()}"
    )

    def head():
        se, out = fh(xd)
        return se

    def owners(se) -> torch.Tensor:  # (rs_dim -2: chip d = r C + c holds tokens [d S/N, (d + 1) S/N) at full width)
        parts = [ttnn.to_torch(t).reshape(-1, cfg.dim).double() for t in ttnn.get_device_tensors(se)]
        return torch.cat(parts)

    got = owners(head())
    again = owners(head())
    print(f"SE_DET L{layer:02d} {tag} head twice: {int((again != got).sum().item())} values differ")
    p = f"layers.{layer}.ffn.shared_experts."
    w = {n: ck.weight(p + n + ".weight", torch.float32) for n in ("w1", "w3", "w2")}
    lim = float(cfg.swiglu_limit)
    sel = torch.cat([torch.arange(0, chunk, 7)])  # (every 7th token: every chip's rows)
    for name, q8 in (("native", False), ("q8", True)):
        ref = _ref64(x[sel].float(), w["w1"], w["w3"], w["w2"], lim, q8)
        print(f"SE_ERR L{layer:02d} {tag} vs float64 {name}: {_err(got[sel], ref)}")
    for _ in range(3):
        head()
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(N_CALLS):
        t0 = time.perf_counter()
        head()
        ttnn.synchronize_device(mesh)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    print(f"SE_TIME L{layer:02d} {tag} head call synced wall median {ts[len(ts) // 2]:.3f} ms (min {ts[0]:.3f})")
    from models.demos.common.bringup.testing import profiler

    profiler.enable(mesh, ops=True, rt=True)  # (F155: the program real-time profiler)
    profiler.signpost("head")
    for _ in range(N_CALLS):
        head()
    profiler.signpost("end")
    profiler.finish_rt()
    res = profiler.result()
    profiler.disable()
    print(
        f"SE_OPS L{layer:02d} {tag} section 'head': {res['kernel_ns'].get('head', 0.0) / 1e3 / N_CALLS:.1f} us a call "
        f"(slowest chip), {res['programs'].get('head', 0) // N_CALLS} programs a call"
    )
    for key, rows_ in res["op_ns"].items():
        if not key.startswith("head"):
            continue
        tot = 0.0
        for r_ in rows_:
            tot += r_["ns"]
            print(
                f"SE_OPS L{layer:02d} {tag} {r_['op'][:50]:50s} calls {r_['calls']:3d} "
                f"{r_['ns'] / 1e3 / N_CALLS:9.1f} us a call  {r_['shape'][:60]}"
            )
        print(f"SE_OPS L{layer:02d} {tag} total {tot / 1e3 / N_CALLS:.1f} us a head call (device, slowest chip per op)")
    ttnn.deallocate(xd)
