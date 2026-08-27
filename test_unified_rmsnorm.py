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
from unified_harness import core_block, dfb, run_unified_spec, single_core, split_evenly, unified_program_spec

KERNEL = "unified_kernels/rmsnorm.cpp"
TILE = 32
EPS = 1.0e-2  # large enough to be representable in bfloat16 and to matter


def bf16_pair(v):
    bits = int(torch.tensor([v], dtype=torch.bfloat16).view(torch.uint16)[0])
    return (bits << 16) | bits


def run(device, ht, wt, unit_weight=False, seed=0, chunk=None, cores=1):
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

    # `chunk` is the rows-per-pass; ht is the whole tensor. Rows are independent, so this is
    # both the L1 bound and the unit of work across cores. Default: the whole thing at once,
    # which is what this kernel used to have to do.
    hc = ht if chunk is None else chunk
    assert ht % hc == 0, "the chunk height must divide the tensor's"
    nchunks = ht // hc
    ncores = min(cores, nchunks)
    core_ranges, cores = core_block(ncores)
    shares = split_evenly(nchunks, ncores)
    named_ct_args = [("ht", hc), ("wt", wt)]

    dfbs = [
        dfb("x", hc * wt),
        dfb("w", wt),
        dfb("eps", 1),
        dfb("inv_n", 1),
        dfb("sq", hc * wt),
        dfb("mean", hc),
        dfb("rsqrt", hc),
        dfb("normed", hc * wt),
        dfb("out", hc * wt),
    ]

    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=dfbs,
        named_compile_time_args=named_ct_args,
        tensors={"x": tx, "w": tw, "out": tout},
        runtime_arg_names=["eps_bits", "chunk_begin", "chunk_count"],
    )
    run_unified_spec(
        device,
        spec,
        {"x": tx, "w": tw, "out": tout},
        runtime_args={
            "eps_bits": bf16_pair(EPS),
            "chunk_begin": {c: b for c, (b, _) in zip(cores, shares)},
            "chunk_count": {c: n for c, (_, n) in zip(cores, shares)},
        },
        nodes=cores,
    )
    out = tout
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

        # Row chunking. Rows are normalised independently, so a chunk height is a pure
        # decomposition and every one has to give the SAME answer, not merely a close one.
        for ht, wt in ((8, 4), (4, 8)):
            ref = None
            for chunk in (ht, ht // 2, ht // 4, 1):
                if ht % chunk:
                    continue
                got, want = run(device, ht, wt, chunk=chunk)[:2]
                spread = 0.0 if ref is None else (got - ref).abs().max().item()
                if ref is None:
                    ref = got
                ok = spread == 0.0
                logger.info(
                    f"H={ht * TILE} W={wt * TILE} chunk={chunk}: vs-whole={spread:.6f}  {'ok' if ok else 'FAIL'}"
                )
                if not ok:
                    failed.append(f"chunk-{ht}-{wt}-{chunk}")

        # The shape that could not be done at all before chunking: S=512 by d_model 2048 is
        # 1024 tiles, and this kernel holds four blocks of them at once (x, sq, normed, out),
        # so the whole thing resident would want 8MB. chunk=2 is 1MB; chunk=4 does not fit.
        #
        # Gated on relative L2, not max|err|: the abs bound above is calibrated for narrow
        # rows and does not scale with the row width. Going from 128-wide rows to 2048-wide
        # takes max|err| from 0.030 to 0.051 while rel-L2 only moves 0.0052 -> 0.0069, which
        # is ordinary bf16 accumulation over 16x the terms and not a chunking artifact --
        # chunk=1 and chunk=2 agree to the bit.
        for ht, wt, chunk in ((16, 64, 1), (16, 64, 2), (8, 64, 1)):
            got, want = run(device, ht, wt, chunk=chunk)[:2]
            rel = ((got - want).norm() / want.norm()).item()
            ok = rel <= 0.01
            logger.info(
                f"H={ht * TILE} W={wt * TILE} chunk={chunk} ({chunk * wt} tiles resident): "
                f"rel-L2={rel:.5f}  {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failed.append(f"wide-{ht}-{wt}-{chunk}")

        # Chunks partitioned across cores.
        for ht, wt, chunk, ncores in ((8, 4, 1, 4), (8, 4, 2, 2), (16, 64, 1, 8)):
            got, want = run(device, ht, wt, chunk=chunk, cores=ncores)[:2]
            rel = ((got - want).norm() / want.norm()).item()
            ok = rel <= 0.01
            logger.info(
                f"H={ht * TILE} W={wt * TILE} chunk={chunk} cores={ncores}: rel-L2={rel:.5f}  "
                f"{'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failed.append(f"mc-{ht}-{wt}-{ncores}")

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
