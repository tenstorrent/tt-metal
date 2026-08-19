# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm on device: out = x / sqrt(mean(x^2) + eps) * weight.

Needed no new library work -- it is the ops from phases 1-6 rearranged. What it adds over
the attention kernel is the OTHER broadcast axis: Cols for the per-row reciprocal RMS and
Rows for the per-feature weight, both in one kernel.

Two gates, and the second is the one that carries information:

  max absolute error vs torch

  with weight == 1, every output ROW must have RMS 1.0. That is the definition of the op
  rather than a tolerance, and it is exactly what a scale error survives: getting the mean
  scaler wrong -- 1 instead of 1/N, which turns a mean into a sum -- leaves the output
  perfectly correlated with the truth and off by sqrt(N).

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_rmsnorm.py
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import make_cb, single_core, unified_program

KERNEL = "unified_kernels/rmsnorm.cpp"
TILE = 32
CB = dict(x=0, w=1, eps=2, inv_n=3, sq=4, mean=5, rsqrt=6, normed=7, out=16)
EPS = 1.0e-2  # large enough to be representable in bfloat16 and to matter


def bf16_pair(v):
    bits = int(torch.tensor([v], dtype=torch.bfloat16).view(torch.uint16)[0])
    return (bits << 16) | bits


def run(device, ht, wt, unit_weight=False, seed=0):
    torch.manual_seed(seed)
    H, W = ht * TILE, wt * TILE

    x = (0.5 + torch.rand([H, W])).to(torch.bfloat16)  # away from zero, so RMS is well posed
    if unit_weight:
        w_row = torch.ones([W])
    else:
        w_row = 0.5 + torch.rand([W])
    w_row = w_row.to(torch.bfloat16)

    # The weight is a row: one value per feature column, meaningful in row 0 of each tile.
    w_t = torch.zeros([TILE, W], dtype=torch.bfloat16)
    w_t[0, :] = w_row

    dram = ttnn.DRAM_MEMORY_CONFIG

    def to_dev(t):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        )

    tx, tw = to_dev(x), to_dev(w_t)
    tout = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, H, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    core_ranges, cores = single_core()
    ct_args = [ht, wt]
    for t in (tx, tw, tout):
        ct_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rt_args = [tx.buffer_address(), tw.buffer_address(), tout.buffer_address(), bf16_pair(EPS)]

    cbs = [
        make_cb(CB["x"], core_ranges, num_pages=ht * wt),
        make_cb(CB["w"], core_ranges, num_pages=wt),
        make_cb(CB["eps"], core_ranges, num_pages=1),
        make_cb(CB["inv_n"], core_ranges, num_pages=1),
        make_cb(CB["sq"], core_ranges, num_pages=ht * wt),
        make_cb(CB["mean"], core_ranges, num_pages=ht),
        make_cb(CB["rsqrt"], core_ranges, num_pages=ht),
        make_cb(CB["normed"], core_ranges, num_pages=ht * wt),
        make_cb(CB["out"], core_ranges, num_pages=ht * wt),
    ]

    program = unified_program(
        kernel_source=KERNEL,
        core_ranges=core_ranges,
        cores=cores,
        cbs=cbs,
        compile_time_args=ct_args,
        runtime_args=rt_args,
    )
    out = ttnn.generic_op([tx, tw, tout], program)
    got = ttnn.to_torch(out).to(torch.float32)[0, 0]

    xf = x.to(torch.float32)
    ms = (xf * xf).mean(dim=-1, keepdim=True)
    want = xf / torch.sqrt(ms + EPS) * w_row.to(torch.float32).unsqueeze(0)
    return got, want


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--abs-err", type=float, default=0.05)
    p.add_argument("--rms-tol", type=float, default=0.02)
    args = p.parse_args(argv)

    cases = [(1, 1), (2, 2), (2, 4), (4, 2), (1, 4)]

    device = ttnn.open_device(device_id=0)
    failed = []
    try:
        for ht, wt in cases:
            got, want = run(device, ht, wt)
            e = (got - want).abs().max().item()
            ok = e <= args.abs_err
            logger.info(f"H={ht * TILE:3d} W={wt * TILE:3d}  max|err|={e:.5f}  {'ok' if ok else 'FAIL'}")
            if not ok:
                failed.append((ht, wt))

        # weight == 1: every output row must have RMS 1. eps shifts it very slightly, so the
        # target is the exact value including eps rather than a bare 1.0.
        for ht, wt in cases:
            got, want = run(device, ht, wt, unit_weight=True)
            rms = got.pow(2).mean(dim=-1).sqrt()
            target = want.pow(2).mean(dim=-1).sqrt()
            spread = (rms - target).abs().max().item()
            ok = spread <= args.rms_tol
            logger.info(
                f"H={ht * TILE:3d} W={wt * TILE:3d}  weight=1  row RMS in "
                f"[{rms.min():.4f}, {rms.max():.4f}] vs {target.mean():.4f}  "
                f"max dev={spread:.5f}  {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failed.append(("rms", ht, wt))
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
