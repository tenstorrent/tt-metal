# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe: decode-sized projection matmuls ([1,1,4,K] x [K,N]) under different weight dtypes / core grids / program
configs, timed as the per-call cost inside a trace ((t(3 calls) - t(1 call)) / 2). Prints 'MM ...' lines."""

import math
import os
import time

import pytest
import torch

import ttnn


def traced_ms(mesh, fn, n=20):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(3):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t) / n
    ttnn.release_trace(mesh, tid)
    return dt * 1e3


def chain_ms(mesh, fn):
    def rep(k):
        def f():
            for _ in range(k):
                fn()

        return f

    return (traced_ms(mesh, rep(3)) - traced_ms(mesh, rep(1))) / 2


def cfg_1d(Kt, Nt, max_cores=64, gx_max=8):
    pcn = math.ceil(Nt / max_cores)
    n = math.ceil(Nt / pcn)
    gx, gy = min(n, gx_max), math.ceil(n / gx_max)
    sw = max(d for d in range(1, min(pcn, 4) + 1) if pcn % d == 0)
    ibw = max(d for d in range(1, 9) if Kt % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=sw,
        per_core_M=1,
        per_core_N=pcn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_probe_matmul(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(md)
    shapes = [(5120, 1792), (5120, 1280), (5120, 512), (1280, 4096), (4096, 1024), (1024, 5120)]
    if os.environ.get("PROBE_SHAPES"):
        shapes = [shapes[int(i)] for i in os.environ["PROBE_SHAPES"].split(",")]
    ckcs = {
        "HiFi4f32": ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
        "HiFi2f32": ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    }
    for K, N in shapes:
        x = ttnn.from_torch(
            torch.randn(1, 1, 4, K).to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        for wdt_name, wdt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)):
            w = ttnn.from_torch(
                (torch.randn(1, 1, K, N) * 0.02).to(torch.bfloat16),
                device=md,
                dtype=wdt,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            variants = {
                "default": dict(),
                "cg2x8": dict(core_grid=ttnn.CoreGrid(y=2, x=8)),
                "cg4x8": dict(core_grid=ttnn.CoreGrid(y=4, x=8)),
                "cg8x8": dict(core_grid=ttnn.CoreGrid(y=8, x=8)),
                "1d64": dict(program_config=cfg_1d(K // 32, N // 32)),
                "1d32": dict(program_config=cfg_1d(K // 32, N // 32, max_cores=32)),
            }
            for ck_name, ck in ckcs.items():
                if ck_name == "HiFi2f32" and wdt_name == "bf16":
                    continue
                for vn, kw in variants.items():
                    try:
                        ms = chain_ms(md, lambda: ttnn.linear(x, w, compute_kernel_config=ck, **kw))
                        print(f"MM K={K} N={N} w={wdt_name} {ck_name} {vn}: {ms * 1e3:.1f} us", flush=True)
                    except Exception as e:  # unsupported combination
                        print(f"MM K={K} N={N} w={wdt_name} {ck_name} {vn}: FAIL {str(e)[:80]!r}", flush=True)
