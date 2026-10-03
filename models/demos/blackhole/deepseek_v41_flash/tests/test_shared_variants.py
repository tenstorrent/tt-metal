# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared-expert variants: timing inside a long trace ((t(3)-t(1))/2) and PCC vs the current implementation (layer 2 weights)."""

import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer
from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert import DSV41SharedExpert

D, N, T = 5120, 2304, 4


def chain_ms(md, fn):
    def run(k):
        def f():
            for _ in range(k):
                fn()

        fn()
        ttnn.synchronize_device(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        f()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        t = time.perf_counter()
        for _ in range(20):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        r = (time.perf_counter() - t) / 20 * 1e3
        ttnn.release_trace(md, tid)
        return r

    return (run(3) - run(1)) / 2


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 200_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_shared_variants(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    w = load_moe_layer(2, experts=[0])  # only the shared expert is needed
    sid = next(iter(w["shared_w0"]))
    w0, w1, w2 = w["shared_w0"][sid], w["shared_w1"][sid], w["shared_w2"][sid]  # [1,1,in,out]
    cur = DSV41SharedExpert(md, w0, w1, w2)
    h_host = torch.randn(1, 1, T, D)
    rep = ttnn.ReplicateTensorToMesh(md)
    h = ttnn.from_torch(
        h_host.to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    dev0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    ref = dev0(cur.forward(h))
    ckc = cur.ckc
    ckc2 = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    up = lambda t, dt=ttnn.bfloat8_b, mp=rep: ttnn.from_torch(
        t.contiguous(),
        device=md,
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mp,
    )
    lim = cur.limit
    res = {}

    def report(name, fn, check=True):
        out = fn()
        p = pcc(dev0(out), ref) if check else float("nan")
        res[name] = (chain_ms(md, fn), p)
        print(f"SHV {name:48s} {res[name][0]:.3f} ms   PCC vs current {p:.5f}", flush=True)

    report("current (3 linears fp32 out, 4 eltwise)", lambda: cur.forward(h))

    # fused gate|up weight: one matmul, slice
    w01 = up(torch.cat([w0, w1], dim=-1))

    def fused():
        gu = ttnn.linear(h, w01, dtype=ttnn.float32, compute_kernel_config=ckc)
        gate, upv = gu[:, :, :, :N], gu[:, :, :, N:]
        upv = ttnn.clamp(upv, min=-lim, max=lim)
        act = ttnn.multiply(ttnn.silu(ttnn.minimum(gate, lim)), upv)
        return ttnn.linear(act, cur.w2, dtype=ttnn.float32, compute_kernel_config=ckc)

    report("fused gate|up (2 linears)", fused)

    def fused_hifi2():
        gu = ttnn.linear(h, w01, dtype=ttnn.float32, compute_kernel_config=ckc2)
        gate, upv = gu[:, :, :, :N], gu[:, :, :, N:]
        upv = ttnn.clamp(upv, min=-lim, max=lim)
        act = ttnn.multiply(ttnn.silu(ttnn.minimum(gate, lim)), upv)
        return ttnn.linear(act, cur.w2, dtype=ttnn.float32, compute_kernel_config=ckc2)

    report("fused gate|up, HiFi2+packer L1 acc", fused_hifi2)

    silu_act = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)]

    def fused2():  # one clamp over the whole gate|up (gate also gets a -10 floor), silu fused into the multiply
        gu = ttnn.clamp(ttnn.linear(h, w01, dtype=ttnn.float32, compute_kernel_config=ckc), min=-lim, max=lim)
        act = ttnn.multiply(gu[:, :, :, :N], gu[:, :, :, N:], input_tensor_a_activations=silu_act)
        return ttnn.linear(act, cur.w2, dtype=ttnn.float32, compute_kernel_config=ckc)

    report("fused + single clamp + silu in multiply", fused2)

    def fused3():  # exact clamps, silu fused into the multiply
        gu = ttnn.linear(h, w01, dtype=ttnn.float32, compute_kernel_config=ckc)
        gate, upv = ttnn.minimum(gu[:, :, :, :N], lim), ttnn.clamp(gu[:, :, :, N:], min=-lim, max=lim)
        act = ttnn.multiply(gate, upv, input_tensor_a_activations=silu_act)
        return ttnn.linear(act, cur.w2, dtype=ttnn.float32, compute_kernel_config=ckc)

    report("fused + exact clamps + silu in multiply", fused3)

    def fused_bf16():  # bf16 linear outputs (half the bytes on every elementwise op)
        gu = ttnn.linear(h, w01, dtype=ttnn.bfloat16, compute_kernel_config=ckc)
        gate, upv = ttnn.minimum(gu[:, :, :, :N], lim), ttnn.clamp(gu[:, :, :, N:], min=-lim, max=lim)
        act = ttnn.multiply(gate, upv, input_tensor_a_activations=silu_act, dtype=ttnn.bfloat16)
        return ttnn.linear(act, cur.w2, dtype=ttnn.float32, compute_kernel_config=ckc)

    report("fused + exact clamps + silu fused + bf16 mid", fused_bf16)

    def fused_cg():  # explicit core grid on both matmuls
        gu = ttnn.linear(h, w01, dtype=ttnn.float32, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=8, x=8))
        gate, upv = ttnn.minimum(gu[:, :, :, :N], lim), ttnn.clamp(gu[:, :, :, N:], min=-lim, max=lim)
        act = ttnn.multiply(gate, upv, input_tensor_a_activations=silu_act)
        return ttnn.linear(
            act, cur.w2, dtype=ttnn.float32, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=8, x=8)
        )

    try:
        report("fused + exact clamps + silu fused + core_grid 8x8", fused_cg)
    except Exception as e:
        print("SHV core_grid variant failed:", str(e)[:160], flush=True)
