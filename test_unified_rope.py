# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Rotary position embedding on device: out = x * cos + (x @ M) * sin.

No library changes -- the rotation is a matmul with kt_dim == 1, which makes it per-tile,
and the rest is one four-leaf SFPU tree. See unified_kernels/rope.cpp.

The reference does NOT use M. It applies the rotation directly --
rotated[..., 2i] = -x[..., 2i+1], rotated[..., 2i+1] = x[..., 2i] -- so the matrix the
device multiplies by is checked against the permutation it is supposed to encode, rather
than against itself.

Two independent gates:

  max absolute error against the direct reference.

  NORM PRESERVATION, per pair. A rotation by any angle preserves the length of each
  (x[2i], x[2i+1]) pair, so sqrt(out[2i]^2 + out[2i+1]^2) must equal sqrt(x[2i]^2 +
  x[2i+1]^2) for every pair and every position. That is a property of the op rather than of
  this reference, and it fails loudly if cos/sin are mispaired, if the sign of the rotation
  is wrong, or if the matmul is not per-tile -- all things an error-versus-reference check
  can hide when both sides are built from the same misunderstanding.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_rope.py
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import core_block, dfb, run_unified_spec, split_evenly, unified_program_spec

KERNEL = "unified_kernels/rope.cpp"
TILE = 32


def trans_mat():
    """ttnn's rotation tile: M[2i][2i+1] = +1, M[2i+1][2i] = -1, so x @ M sends the pair
    (x[2i], x[2i+1]) to (-x[2i+1], x[2i])."""
    m = torch.zeros([TILE, TILE])
    m[torch.arange(0, TILE, 2), torch.arange(1, TILE, 2)] = 1.0
    m[torch.arange(1, TILE, 2), torch.arange(0, TILE, 2)] = -1.0
    return m


def rotate_pairs(x):
    """The permutation M encodes, applied directly -- an independent reference."""
    r = torch.empty_like(x)
    r[..., 0::2] = -x[..., 1::2]
    r[..., 1::2] = x[..., 0::2]
    return r


def cos_sin(seq, dim, theta=10000.0):
    """Interleaved RoPE angles, each duplicated across its pair."""
    half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = torch.outer(torch.arange(seq, dtype=torch.float32), freqs)  # seq x half
    c = torch.repeat_interleave(torch.cos(ang), 2, dim=-1)
    s = torch.repeat_interleave(torch.sin(ang), 2, dim=-1)
    return c, s


def run(device, seq_t, dim_t, chunk, seed=0, cores=1):
    torch.manual_seed(seed)
    S, D = seq_t * TILE, dim_t * TILE
    total_tiles = seq_t * dim_t
    assert total_tiles % chunk == 0, "chunk must divide the tile count"

    x = (torch.rand([S, D]) - 0.5).to(torch.bfloat16)
    c, s = cos_sin(S, D)
    cos_t, sin_t = c.to(torch.bfloat16), s.to(torch.bfloat16)
    m = trans_mat().to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG

    def to_dev(t):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        )

    tx, tc, ts, tm = to_dev(x), to_dev(cos_t), to_dev(sin_t), to_dev(m)
    tout = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, S, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    # Chunks are the unit of work and the rotation is per-tile, so they share nothing:
    # splitting them needs no reduction and no ordering. Every core still reads the one
    # rotation tile, which is 2KB and read once per core.
    nchunks = total_tiles // chunk
    ncores = min(cores, nchunks)
    core_ranges, core_list = core_block(ncores)
    shares = split_evenly(nchunks, ncores)

    named_ct_args = [("chunk", chunk), ("num_chunks", nchunks)]

    dfbs = [
        dfb("x", chunk),
        dfb("cos", chunk),
        dfb("sin", chunk),
        dfb("m", 1),
        dfb("rot", chunk),
        dfb("out", chunk),
    ]

    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=dfbs,
        named_compile_time_args=named_ct_args,
        tensors={"x": tx, "cos": tc, "sin": ts, "m": tm, "out": tout},
        runtime_arg_names=["chunk_begin", "chunk_count"],
    )
    run_unified_spec(
        device,
        spec,
        {"x": tx, "cos": tc, "sin": ts, "m": tm, "out": tout},
        runtime_args={
            "chunk_begin": {c: b for c, (b, _) in zip(core_list, shares)},
            "chunk_count": {c: n for c, (_, n) in zip(core_list, shares)},
        },
    )
    out = tout
    got = ttnn.to_torch(out).to(torch.float32)[0, 0]

    xf = x.to(torch.float32)
    want = xf * c + rotate_pairs(xf) * s
    return got, want, xf


def pair_norm(t):
    """sqrt(t[2i]^2 + t[2i+1]^2) for every pair."""
    return (t[..., 0::2].pow(2) + t[..., 1::2].pow(2)).sqrt()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--abs-err", type=float, default=0.02)
    p.add_argument("--norm-tol", type=float, default=0.02)
    args = p.parse_args(argv)

    # (seq tiles, dim tiles, tiles per chunk). The chunk is the matmul's rt_dim, so it is
    # capped at the 8-tile DST budget.
    cases = [(1, 1, 1), (2, 2, 2), (2, 2, 4), (4, 4, 8), (2, 4, 8), (4, 1, 2)]

    device = ttnn.open_device(device_id=0)
    failed = []
    try:
        for seq_t, dim_t, chunk in cases:
            got, want, xf = run(device, seq_t, dim_t, chunk)
            e = (got - want).abs().max().item()
            # A rotation preserves each pair's length.
            dev = (pair_norm(got) - pair_norm(xf)).abs().max().item()
            ok = e <= args.abs_err and dev <= args.norm_tol
            logger.info(
                f"S={seq_t * TILE:3d} D={dim_t * TILE:3d} chunk={chunk}  max|err|={e:.5f}  "
                f"pair-norm dev={dev:.5f}  {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failed.append((seq_t, dim_t, chunk))

        # Partition invariance, checked EXACTLY: chunks share nothing, so splitting them is
        # a decomposition and not an approximation, and anything other than a bit-identical
        # result means a core read or wrote outside its range. 12 chunks over 5 cores is the
        # uneven split, which is where an off-by-one in the range arithmetic lives.
        one = run(device, 4, 3, 1)[0]
        for n in (2, 5, 12):
            many = run(device, 4, 3, 1, cores=n)[0]
            diff = (many - one).abs().max().item()
            ok = diff == 0.0
            logger.info(f"12 chunks over {n:2d} cores vs 1: max|diff| = {diff:.6f}   {'ok' if ok else 'FAIL'}")
            if not ok:
                failed.append(f"cores-{n}")
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
