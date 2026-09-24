# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standalone S512 LayerNorm (with residual add) at the model shapes.

Runs the stock interleaved ttnn.layer_norm as the model calls it (bf8 L1 input and
residual, bf16 gamma/beta, HiFi2 with fp32 DEST, bf8 L1 output) and prints the
accuracy against a float64 reference and the traced time per call.

  python models/demos/wormhole/bge_m3/tests/perf/layernorm_s512.py --batch 8 \
      [--legacy-reduction] [--legacy-rsqrt] [--welford] [--balanced [--grid-x X --grid-y Y]]

--balanced also runs custom_ops/balanced_layernorm and compares it with stock (bitwise
and in float64). --grid-x/--grid-y set its grid (default: the device grid).
"""

import argparse
import time

import torch

import ttnn

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--legacy-reduction", action="store_true")
parser.add_argument("--legacy-rsqrt", action="store_true")
parser.add_argument("--welford", action="store_true")
parser.add_argument("--iters", type=int, default=200)
parser.add_argument("--balanced", action="store_true")
parser.add_argument("--grid-x", type=int, default=None)
parser.add_argument("--grid-y", type=int, default=None)
parser.add_argument("--no-timing", action="store_true")
args = parser.parse_args()

SEQ, HIDDEN, EPS = 512, 1024, 1e-5
B = args.batch
device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=20_000_000)
torch.manual_seed(0)
mem = ttnn.L1_MEMORY_CONFIG

x_h = torch.randn((B, 1, SEQ, HIDDEN), dtype=torch.bfloat16)
r_h = torch.randn((B, 1, SEQ, HIDDEN), dtype=torch.bfloat16) * 4
g_h = 1 + 0.1 * torch.randn((HIDDEN,), dtype=torch.bfloat16)
b_h = 0.1 * torch.randn((HIDDEN,), dtype=torch.bfloat16)


def dev(t, dtype=ttnn.bfloat8_b, memory_config=mem):
    return ttnn.from_torch(t, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)


x, r = dev(x_h), dev(r_h)


# The model keeps gamma/beta as row-major bf16 [1, 1, W/32, 32] in DRAM (norm.py).
def gb(t):
    return ttnn.from_torch(
        t.reshape(1, 1, HIDDEN // 32, 32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


gamma, beta = gb(g_h), gb(b_h)

xs = ttnn.to_torch(x).double() + ttnn.to_torch(r).double()
ref = torch.nn.functional.layer_norm(xs, (HIDDEN,), g_h.double(), b_h.double(), EPS)

ckc = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)
pcfg = ttnn.LayerNormDefaultProgramConfig(
    legacy_reduction=args.legacy_reduction, legacy_rsqrt=args.legacy_rsqrt, use_welford=args.welford
)


def stock():
    return ttnn.layer_norm(
        x,
        epsilon=EPS,
        weight=gamma,
        bias=beta,
        residual_input_tensor=r,
        program_config=pcfg,
        memory_config=mem,
        compute_kernel_config=ckc,
    )


def report(name, out):
    t = ttnn.to_torch(out).double()
    cos = torch.nn.functional.cosine_similarity(t.flatten(), ref.flatten(), dim=0).item()
    print("ACC %-8s cos64 %.9f maxabs %.5f" % (name, cos, (t - ref).abs().max().item()), flush=True)
    return t


def traced_us(fn):
    for _ in range(3):
        ttnn.deallocate(fn())
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(20):
        ttnn.deallocate(fn())
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    start = time.perf_counter()
    for _ in range(args.iters // 20):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - start) * 1e6 / (args.iters // 20 * 20)
    ttnn.release_trace(device, tid)
    return us


out = stock()
s_t = report("stock", out)
ttnn.deallocate(out)
if args.balanced:
    from models.demos.wormhole.bge_m3.tt.custom_ops.balanced_layernorm import (
        BalancedLayerNormPlan,
        bge_balanced_layernorm,
    )

    g = device.compute_with_storage_grid_size()
    grid = (args.grid_x or int(g.x), args.grid_y or int(g.y))
    plan = BalancedLayerNormPlan(rows=B * SEQ // 32, num_cores=grid[0] * grid[1])
    print(
        "PLAN grid %dx%d rows/core max %d" % (grid[0], grid[1], max(plan.core_rows(0)[1], 1)),
        flush=True,
    )

    def balanced():
        return bge_balanced_layernorm(x, r, gamma, beta, eps=EPS, memory_config=mem, grid=grid)

    out = balanced()
    b_t = report("balanced", out)
    ttnn.deallocate(out)
    cos = torch.nn.functional.cosine_similarity(b_t.flatten(), s_t.flatten(), dim=0).item()
    diff = (b_t - s_t).abs()
    bad_rows = sorted({int(i) // 32 for i in torch.nonzero(diff.reshape(-1, HIDDEN).amax(-1))[:, 0].tolist()})
    print(
        "ACC balanced-vs-stock cos64 %.9f maxabs %.5f bit-identical %s rows-differing %d %s"
        % (cos, diff.max().item(), torch.equal(b_t, s_t), len(bad_rows), bad_rows[:12]),
        flush=True,
    )
    if not args.no_timing:
        print("TIME balanced %.2f us/call" % traced_us(balanced), flush=True)
if not args.no_timing:
    print("TIME stock %.2f us/call" % traced_us(stock), flush=True)
ttnn.close_mesh_device(device)
