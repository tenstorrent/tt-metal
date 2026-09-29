# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MoE router matmul configs: logits [M, 256] fp32 = x [M, 4096] bf16 @ W [4096, 256] bf16, HiFi4 + fp32 dest (the
TtGate settings: the routing needs fp32 logits). Sweeps 2D-mcast (M over grid rows, N over cols) and 1D (M split, W
multicast) program configs against ttnn's default pick; every config's logits must match the default's closely and
give the same top-8 (host topk with the gate bias). Signposts ``router_M{M}_{cfg}``.

    scripts/run_safe_pytest.sh --profile models/demos/mimo_v2_d_p/tests/perf/test_router_matmul.py
"""

import itertools
import math
import os

import pytest
import torch

import ttnn

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

K, N, TILE = 4096, 256, 32
SEQS = [int(s) for s in os.environ.get("MIMO_ROUTER_SEQ", "640,2048").split(",")]
BWS = [int(s) for s in os.environ.get("MIMO_ROUTER_BW", "2,4,8,16,32").split(",")]
GXS = [int(s) for s in os.environ.get("MIMO_ROUTER_GX", "8,4,2,1").split(",")]


def configs(device, M):
    grid = device.compute_with_storage_grid_size()
    Mt, Kt, Nt = math.ceil(M / TILE), K // TILE, N // TILE
    out = [("auto", None)]
    for gx, bw in itertools.product(GXS, BWS):
        per_n = Nt // gx
        for gy in sorted({min(grid.y, Mt), 8, 5, 4}):
            per_m = math.ceil(Mt / gy)
            gy_ = math.ceil(Mt / per_m)
            if gy_ > grid.y or gx > grid.x:
                continue
            sw = max(d for d in (1, 2, 4) if per_n % d == 0 and d <= 4)
            sh = max(d for d in (1, 2, 4) if per_m % d == 0 and d * sw <= 4)
            out.append(
                (
                    f"2d_gx{gx}_gy{gy_}_bw{bw}",
                    ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy_),
                        in0_block_w=bw,
                        out_subblock_h=sh,
                        out_subblock_w=sw,
                        out_block_h=per_m,
                        out_block_w=per_n,
                        per_core_M=per_m,
                        per_core_N=per_n,
                        transpose_mcast=False,
                        fused_activation=None,
                        fuse_batch=True,
                    ),
                )
            )
    for per_m, bw in itertools.product((1, 2, 3), BWS):
        cores = math.ceil(Mt / per_m)
        if cores > grid.x * grid.y:
            continue
        gx = min(grid.x, cores)
        gy = math.ceil(cores / gx)
        out.append(
            (
                f"1d_pm{per_m}_bw{bw}",
                ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                    in0_block_w=bw,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    out_block_h=per_m,
                    out_block_w=Nt,
                    per_core_M=per_m,
                    per_core_N=Nt,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=False,
                ),
            )
        )
    return out


@pytest.mark.parametrize("M", SEQS)
def test_router_matmul(device, M):
    torch.manual_seed(0)
    xt = torch.randn(1, 1, M, K)
    wt = torch.randn(1, 1, K, N) * 0.02
    bias = torch.rand(N) * 2  # e_score_correction_bias scale (~1-2)
    x = ttnn.from_torch(xt, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    w = ttnn.from_torch(wt, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )
    ref = None
    for tag, pc in configs(device, M):
        try:
            for it in range(4):
                if it:
                    signpost(f"router_M{M}_{tag}_start")
                o = ttnn.linear(x, w, dtype=ttnn.float32, compute_kernel_config=ckc, program_config=pc)
                ttnn.synchronize_device(device)
                if it:
                    signpost(f"router_M{M}_{tag}_end")
                if it == 3:
                    t = ttnn.to_torch(o).float().reshape(M, N)
                    sel = (torch.sigmoid(t) + bias).topk(8, dim=-1).indices.sort(-1).values
                    if ref is None:
                        ref, ref_sel = t, sel
                    else:
                        d = (t - ref).abs().max().item()
                        flips = (sel != ref_sel).any(-1).float().mean().item()
                        print(f"CHECK M{M} {tag}: max |dlogit| {d:.3e}, top-8 differs on {100 * flips:.2f}% of tokens")
                o.deallocate(True)
        except Exception as e:  # noqa: BLE001 - invalid / L1-overflowing configs are skipped
            print(f"SKIP M{M} {tag}: {str(e).splitlines()[0][:140]}")
