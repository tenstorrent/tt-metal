# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sweep real model attention shapes through the unified flash kernel.

Per HEAD, on ONE core, which is what this kernel does today -- there is no GQA head
mapping and no multi-core partitioning yet (phase 11). A full layer is n_heads of
these, spread over the grid, so the numbers here are the per-head unit of that.

Causal prefill is triangular, and this measures it as such rather than guessing. The
kernel takes S_q queries against S_k keys and its mask already handles S_q < S_k
("query i sees keys up to i + (S_k - S_q)"), which is exactly one q-chunk of a causal
prefill. So a whole head is the SUM over q-chunks, each with its own key extent:
chunk i attends to (i+1)*S_q keys. Reporting only the last chunk would overstate the
per-chunk cost by about 2x.

sq is no longer set by a rule. Single-shot matmuls walk row bands, so an output block
wider than the 8-tile DST budget is fine, and what runs out instead is circular-buffer
space -- q, o, pv and scores all scale with sq. The ceiling is therefore probed rather
than derived: 8 for 64- and 96-wide heads, 6 for 128, 4 for 256, against the 4, 2, 2 and
1 the old DST rule forced.

But the largest sq is NOT the fastest, because causal work grows with the q-chunk. A
prefill in chunks of Q computes S*(S+Q)/2 tiles of scores, so a coarser chunk computes
more of the triangle it is going to throw away. Fewer launches pull one way and wasted
masked work the other, and which wins depends on the shape. So sq is SEARCHED, and the
table reports the one that won.

    python bench_models.py --seq 512
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_bench import bench

TILE = 32

# The smallest published config of each family, since this is one chip. head_dim is what
# the kernel actually sees; n_heads and n_kv are recorded to size a full layer.
# sq=1 is not a candidate: it is what the old DST rule forced on wide heads, and banding
# exists precisely so nothing has to run that narrow. Measuring it costs the most launches
# of any candidate for the least likely win.
SQ_CANDIDATES = (2, 4, 6, 8)

MODELS = [
    # name                      head_dim  n_heads  n_kv
    ("Llama 3.2 1B", 64, 32, 8),
    ("Qwen2.5 0.5B", 64, 14, 2),
    ("TinyLlama 1.1B", 64, 32, 4),
    ("Phi-3 mini 3.8B", 96, 32, 32),
    ("Qwen3 0.6B", 128, 16, 8),
    ("Llama 3.2 3B", 128, 24, 8),
    ("Gemma 3 1B", 256, 4, 1),
]


def largest_divisor(extent, cap):
    """Biggest sk <= cap that divides extent, so the chunk count comes out whole."""
    for sk in range(min(cap, extent), 0, -1):
        if extent % sk == 0:
            return sk
    return 1


_SQ_CACHE = {}


def probe_sq(device, flash, dt, fid):
    """Largest sq this head width can allocate and compute. L1-bound, so measured."""
    if dt in _SQ_CACHE:
        return _SQ_CACHE[dt]
    for sq in (16, 8, 6, 4, 2, 1):
        try:
            got, want = flash.run(device, sq, min(8, sq), dt, 1, True, fidelity=fid)
            if (got - want).abs().max().item() / want.abs().max().item() < 0.08:
                _SQ_CACHE[dt] = sq
                return sq
        except Exception:  # noqa: BLE001 - out of L1 for this sq, try a smaller one
            continue
    _SQ_CACHE[dt] = 0
    return 0


def plan(head_dim, seq):
    """(dt, k_tiles) for this shape, or None if it cannot be expressed at all."""
    if head_dim % TILE or seq % TILE:
        return None
    return head_dim // TILE, seq // TILE


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, nargs="+", default=[512])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--skip-ttnn", action="store_true")
    p.add_argument(
        "--fidelity",
        choices=["hifi2", "hifi4"],
        default="hifi2",
        help="hifi2 = HiFi2 + approx, which is what the ttnn reference is pinned to. "
        "hifi4 is metal's exact default and is NOT comparable to that reference.",
    )
    args = p.parse_args(argv)

    import test_unified_flash as flash

    # Both sides at the same settings or the comparison means nothing.
    fid = {"math_fidelity": ttnn.MathFidelity.HiFi2, "math_approx_mode": True} if args.fidelity == "hifi2" else None

    device = ttnn.open_device(device_id=0)
    rows = []
    try:
        for seq in args.seq:
            for name, head_dim, n_heads, n_kv in MODELS:
                got = plan(head_dim, seq)
                if got is None:
                    rows.append((name, head_dim, seq, None, None, None, "shape not expressible"))
                    continue
                dt, k_tiles = got
                sq_max = min(probe_sq(device, flash, dt, fid), k_tiles)
                if sq_max == 0:
                    rows.append((name, head_dim, seq, None, None, None, "no sq fits in L1"))
                    continue

                # Search sq rather than maximise it: more query rows per launch means
                # fewer launches, but more of the causal triangle computed and then
                # masked away. Which wins depends on the shape, so measure both.
                best = None
                for sq in (c for c in SQ_CANDIDATES if c <= sq_max and k_tiles % c == 0):
                    total = 0.0
                    for i in range(k_tiles // sq):
                        extent = (i + 1) * sq  # chunk i sees only the keys up to its own rows
                        sk = largest_divisor(extent, min(8, extent))
                        st = bench(
                            device,
                            lambda s=sq, e=extent, c=extent // sk: flash.run(device, s, e, dt, c, True, fidelity=fid),
                            iters=args.iters,
                            warmup=2,
                            match="flash_attention.cpp",
                        )
                        total += st["median_us"]
                    if best is None or total < best[1]:
                        best = (sq, total, k_tiles // sq)
                sq, total, nchunks = best
                # Heads are independent, so a layer is not n_heads serial passes: it is
                # however many rounds the grid needs.
                rounds = -(-n_heads // 64)
                per_layer = total * rounds
                rows.append((name, head_dim, seq, sq, total, per_layer, f"{nchunks} q-chunks, sq<={sq_max}"))
                logger.info(
                    f"{name} d={head_dim} S={seq}: per head {total:.1f}us, "
                    f"layer on a 64-core grid {per_layer:.0f}us ({rounds} round(s) of {n_heads} heads)"
                )

        ttnn_ref = {}
        if not args.skip_ttnn:
            for seq in args.seq:
                for head_dim in sorted({m[1] for m in MODELS}):
                    q = torch.randn([1, 1, seq, head_dim], dtype=torch.bfloat16)
                    tq, tk, tv = (
                        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                        for x in (q, q.clone(), q.clone())
                    )
                    grid = ttnn.CoreCoord(1, 1)
                    chunk = min(128, seq)
                    pc = ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=grid, q_chunk_size=chunk, k_chunk_size=chunk
                    )
                    ckc = ttnn.init_device_compute_kernel_config(
                        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True
                    )
                    try:
                        st = bench(
                            device,
                            lambda: ttnn.transformer.scaled_dot_product_attention(
                                tq, tk, tv, is_causal=True, program_config=pc, compute_kernel_config=ckc
                            ),
                            iters=args.iters,
                            warmup=2,
                            match="sdpa",
                        )
                        ttnn_ref[(seq, head_dim)] = st["median_us"]
                    except Exception as exc:  # noqa: BLE001 - report and continue the sweep
                        logger.warning(f"ttnn SDPA d={head_dim} S={seq}: {type(exc).__name__}")
    finally:
        ttnn.close_device(device)

    logger.info("")
    logger.info(f"per HEAD, one core, causal prefill summed over q-chunks ({args.fidelity} both sides):")
    logger.info(
        f"  {'model':18s} {'d':>4s} {'S':>6s} {'sq':>3s} {'ours/head':>10s} {'ttnn/head':>10s} {'ratio':>6s}  note"
    )
    for name, head_dim, seq, sq, total, per_layer, note in rows:
        ref = ttnn_ref.get((seq, head_dim))
        if total is None:
            logger.info(f"  {name:18s} {head_dim:4d} {seq:6d} {'-':>3s} {'-':>10s} {'-':>10s} {'-':>6s}  {note}")
            continue
        r = f"{total / ref:.2f}x" if ref else "-"
        rs = f"{ref:8.1f}us" if ref else "-"
        logger.info(
            f"  {name:18s} {head_dim:4d} {seq:6d} {sq:3d} {total:8.1f}us {rs:>10s} {r:>6s}  "
            f"{note}, layer/grid {per_layer:.0f}us"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
