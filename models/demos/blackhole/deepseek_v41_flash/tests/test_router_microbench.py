# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Result (2026-10-14, .44, T=16): router matmul 22.5 us on every core grid (1x4..8x8: compute-bound; HiFi2/LoFi are 18 us but not bit-exact), add_act 6.4, add_bias 6.0,
select 11.2 us. Folding the bias add into the select kernel is bit-exact but 16 us SLOWER (soft-float adds on the scalar core): rejected.

Traced micro-benchmark of the decode router chain (fused mode: matmul + softplus/sqrt add + bias add + router_select), T=16 rows per device:
matmul core grids and fusing the two adds. Prints RT lines: name = us per call (20 calls in one trace)."""

import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.router_select import router_select

T, K_DIM, E, TOPK = 16, 5120, 384, 6
REP = 20


def traced_us(mesh, fn, n=30):
    def body():
        for _ in range(REP):
            fn()

    body()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(3):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t) / n / REP
    ttnn.release_trace(mesh, tid)
    return dt * 1e6


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "trace_region_size": 100_000_000}, id="noring")],
    indirect=True,
)
@torch.no_grad()
def test_router_microbench(mesh_device):
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)
    mk = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(
        t,
        device=md,
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    x = mk(torch.randn(1, 1, T, K_DIM).to(torch.bfloat16))
    w = mk((torch.randn(1, 1, K_DIM, E) * 0.02).to(torch.bfloat16))
    bias = mk(torch.randn(1, 1, 1, E), ttnn.float32)
    zero = mk(torch.zeros(1, 1, 1, E), ttnn.float32)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    L1 = ttnn.L1_MEMORY_CONFIG
    U, UT = ttnn.UnaryWithParam, ttnn.UnaryOpType
    res = {}
    outs = {}

    def mm(grid, **kw):
        return ttnn.matmul(
            x,
            w,
            compute_kernel_config=ckc,
            dtype=ttnn.float32,
            core_grid=ttnn.CoreGrid(y=grid[0], x=grid[1]),
            memory_config=L1,
            **kw,
        )

    for grid in [(1, 8), (1, 12), (1, 6), (1, 4), (2, 6), (3, 4), (2, 8), (4, 8), (8, 8)]:
        try:
            res[f"mm_grid{grid[0]}x{grid[1]}"] = traced_us(md, lambda: mm(grid))
            outs[grid] = ttnn.to_torch(ttnn.get_device_tensors(mm(grid))[0])
        except Exception as e:  # noqa: BLE001
            res[f"mm_grid{grid}"] = f"ERR {str(e)[:150]}"
    ref = outs.get((1, 8))
    for g, o in outs.items():
        if ref is not None:
            res[f"mm_grid{g[0]}x{g[1]}_bitexact_vs_1x8"] = bool(torch.equal(o, ref))

    logits = mm((1, 8))
    acts = [U(UT.SOFTPLUS, 1.0, 20.0), U(UT.SQRT)]
    res["add_act"] = traced_us(md, lambda: ttnn.add(logits, zero, input_tensor_a_activations=acts, memory_config=L1))
    score = ttnn.add(logits, zero, input_tensor_a_activations=acts, memory_config=L1)
    res["add_bias"] = traced_us(md, lambda: ttnn.add(score, bias, memory_config=L1))
    rank = ttnn.add(score, bias, memory_config=L1)
    res["select"] = traced_us(md, lambda: router_select(rank, score, TOPK, 1e-20, 2.5))
    # score + rank from one op? matmul with fused activation chain is not available; try add with both outputs separately is 2 ops

    # math fidelity / accumulation variants of the matmul (grid 1x8)
    base_out = ttnn.to_torch(ttnn.get_device_tensors(mm((1, 8)))[0])
    for name, fid, fp32acc in [
        ("HiFi2", ttnn.MathFidelity.HiFi2, True),
        ("LoFi", ttnn.MathFidelity.LoFi, True),
        ("HiFi4_nofp32acc", ttnn.MathFidelity.HiFi4, False),
    ]:
        ck2 = ttnn.init_device_compute_kernel_config(
            md.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32acc, packer_l1_acc=False
        )
        f = lambda: ttnn.matmul(
            x, w, compute_kernel_config=ck2, dtype=ttnn.float32, core_grid=ttnn.CoreGrid(y=1, x=8), memory_config=L1
        )
        try:
            res[f"mm_{name}"] = traced_us(md, f)
            o = ttnn.to_torch(ttnn.get_device_tensors(f())[0])
            res[f"mm_{name}_bitexact"] = bool(torch.equal(o, base_out))
            res[f"mm_{name}_maxabs"] = float((o - base_out).abs().max())
        except Exception as e:  # noqa: BLE001
            res[f"mm_{name}"] = f"ERR {str(e)[:150]}"
    for k, v in res.items():
        print(
            f"RT {k} = {v if isinstance(v, (str, bool)) else f'{v:.2f} us' if abs(v) > 0.5 else f'{v:.3g}'}", flush=True
        )
