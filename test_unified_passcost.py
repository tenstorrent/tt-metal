# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Price one compute pass, by op, against a zero-math control.

Not a test of anything interesting -- the point is the SLOPE, which is why every
mode here has a trivial reference. Runtime against the number of passes cancels
every fixed cost (program launch, first load, last store), because those are paid
once whatever the pass count is, so nothing has to be modelled or guessed at.

  copy    the control: one L1 round trip and its CB handshake, no math
  bcast   shape-preserving, so chainable; minus the copy slope it is bcast's math
  reduce  collapses, so it cannot be chained and is swept by shape instead

Measured on one Wormhole core, the copy control comes out at ~0.10us per tile per
pass while an exact SFPU op costs 0.67-1.19us per tile, which is what said the
plumbing was never the thing to optimise. See unified_llama_prefill.md.

    python test_unified_passcost.py
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import dfb, run_unified_spec, single_core, unified_program_spec

KERNEL = "unified_kernels/passcost.cpp"

TILE = 32

DEFINE = {"copy": None, "bcast": "PC_BCAST", "matmul": "PC_MATMUL", "alt": "PC_ALT", "reduce": "PC_REDUCE"}
# name -> defines, for the FPU-vs-SFPU binary comparison.
BIN = {
    f"{unit}_{op}": [("PC_BIN", "1")]
    + ([("PC_FPU", "1")] if unit == "fpu" else [])
    + ([(f"PC_OP_{op.upper()}", "1")] if op != "add" else [])
    for unit in ("fpu", "sfpu")
    for op in ("add", "sub", "mul")
}


def run(device, mode="copy", passes=1, rows=1, cols=8, seed=0, buffering=2, fidelity=None):
    """Returns (got, want). For reduce, `want` is the row fold and got is column 0."""
    assert mode in DEFINE or mode in BIN
    if mode == "reduce":
        assert passes == 1, "a reduction cannot be chained"
    if mode == "matmul":
        assert rows == cols, "a chained matmul has to be square"

    # Chosen so every partial sum is EXACTLY representable, which lets the checks below
    # gate on equality instead of on a tolerance. bfloat16 spacing on [1, 2) is 2^-7, so
    # values built as 1 + k*2^-7 and addends as j*2^-7 add without rounding as long as
    # the running sum stays under 2: a < 1.5 plus at most 8 passes of at most 8 steps
    # tops out at 1.5 + 0.5. Anything sloppier drifts about one LSB per pass, one-sided,
    # which no fixed tolerance can tell apart from a real broadcast bug.
    torch.manual_seed(seed)
    lsb = 2.0**-7
    a = (1.0 + torch.randint(0, 64, [1, 1, rows * TILE, cols * TILE]) * lsb).to(torch.bfloat16)
    # Distinct per row, so a broadcast that picked up the wrong row cannot pass.
    step = (1 + torch.arange(rows * TILE) % 8).reshape(1, 1, -1, 1) * lsb
    v = step.expand(1, 1, rows * TILE, TILE).to(torch.bfloat16).contiguous()
    if mode in BIN:
        # Full shape, and exact: 1 + k*2^-7 times/plus/minus a small multiple of 2^-7
        # stays representable, so the binary checks gate on equality like the rest.
        # Small for add and sub, so the running value never leaves [0.5, 2) where a
        # multiple of 2^-7 is representable -- an rhs near 1.0 pushes the sum past 2,
        # into 2^-6 spacing, and the check then fails by one LSB on both units.
        v = (torch.randint(1, 5, [1, 1, rows * TILE, cols * TILE]) * lsb).to(torch.bfloat16)
        if mode.endswith("mul"):
            v = torch.ones([1, 1, rows * TILE, cols * TILE]).to(torch.bfloat16)
    if mode == "matmul":
        # The identity: every product but one is a zero, so the chain is exact while the
        # FPU still does the full inner product.
        v = torch.eye(rows * TILE).reshape(1, 1, rows * TILE, rows * TILE).to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)

    out_cols = 1 if mode == "reduce" else cols
    out_shape = [1, 1, rows * TILE, out_cols * TILE]
    tout = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    core_ranges, cores = single_core()

    two_operand = mode in ("bcast", "matmul", "alt") or mode in BIN
    tensors = [ta, tv, tout] if two_operand else [ta, tout]
    named_ct_args = [("rows", rows), ("cols", cols)]

    # The vector CB holds the broadcast operand, or the 1x1 reduce scaler.
    vec_pages = rows if mode in ("bcast", "alt") else (rows * cols if mode == "matmul" or mode in BIN else 1)
    dfbs = [
        dfb("in", buffering * rows * cols),
        dfb("vec", buffering * vec_pages),
        dfb("out", buffering * rows * out_cols),
    ]
    # One scratch buffer per intermediate pass, whether or not this shape uses them all --
    # the kernel declares all seven Storages unconditionally.
    dfbs += [dfb(f"s{i}", buffering * rows * cols) for i in range(1, 8)]

    defines = [("PASSES", str(passes))]
    if mode in BIN:
        defines.extend(BIN[mode])
    elif DEFINE[mode]:
        defines.append((DEFINE[mode], "1"))

    # `vec` is bound on every shape: the kernel names tensor::vec whether or not the pass
    # chain reads it.
    bound = {"in": ta, "vec": tv, "out": tout}
    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=dfbs,
        tensors=bound,
        named_compile_time_args=named_ct_args,
        defines=defines,
        name="passcost",
        **(fidelity or {}),
    )
    run_unified_spec(device, spec, bound)
    out = tout
    got = ttnn.to_torch(out).to(torch.float32)
    af, vf = a.to(torch.float32), v.to(torch.float32)

    if mode == "reduce":
        # reduce<Cols> folds every column into one, so the answer is column 0 and the
        # rest of the tile is the packer's zeroing contract.
        return got[0, 0, :, 0], af.max(dim=3).values[0, 0]
    if mode == "bcast":
        # vec[r] applies to every column of row r, once per pass.
        return got[0, 0], (af + passes * vf[:, :, :, 0:1])[0, 0]
    if mode in BIN:
        op = mode.split("_")[1]
        acc = af
        for _ in range(passes):
            acc = acc - vf if op == "sub" else (acc * vf if op == "mul" else acc + vf)
        return got[0, 0], acc[0, 0]
    if mode == "matmul":
        return got[0, 0], af[0, 0]  # a @ I @ ... @ I == a
    if mode == "alt":
        # Odd passes add the vector, even passes copy: ceil(passes / 2) additions.
        return got[0, 0], (af + ((passes + 1) // 2) * vf[:, :, :, 0:1])[0, 0]
    return got[0, 0], af[0, 0]


def slope(points):
    """Least-squares slope of us against the swept integer, in us per unit."""
    n = len(points)
    xs = [float(x) for x, _ in points]
    ys = [y for _, y in points]
    mx, my = sum(xs) / n, sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--max-passes", type=int, default=8)
    p.add_argument("--tiles", type=int, default=8, help="block width for the pass sweeps")
    args = p.parse_args(argv)

    from unified_bench import bench

    device = ttnn.open_device(device_id=0)
    failed = []
    try:

        def measure(fn):
            return bench(device, fn, iters=20, warmup=3, match="passcost.cpp")["median_us"]

        def check(name, got, want):
            # Equality, not a tolerance: the inputs are built so that nothing rounds.
            diff = (got - want).abs().max().item()
            if diff != 0.0:
                logger.error(f"{name}: max |got - want| = {diff} (expected exact)")
                failed.append(name)

        # ---- pass sweeps: copy (the control) and bcast, same shape, same tile traffic
        sweeps = {}
        for mode in ("copy", "bcast", "alt"):
            points = []
            for n in range(1, args.max_passes + 1):
                check(f"{mode} passes={n}", *run(device, mode, passes=n, cols=args.tiles))
                points.append((n, measure(lambda m=mode, n=n: run(device, m, passes=n, cols=args.tiles))))
            sweeps[mode] = points

        # ---- matmul: square so it chains, at 2x2 tiles to stay inside the DST budget
        mm, mm_n = [], 2
        for n in range(1, args.max_passes + 1):
            check(f"matmul passes={n}", *run(device, "matmul", passes=n, rows=mm_n, cols=mm_n))
            mm.append((n, measure(lambda n=n: run(device, "matmul", passes=n, rows=mm_n, cols=mm_n))))

        # ---- reduce: swept by shape, since it cannot be chained
        red_cols, red_rows = [], []
        for c in range(1, args.max_passes + 1):
            check(f"reduce cols={c}", *run(device, "reduce", rows=1, cols=c))
            red_cols.append((c, measure(lambda c=c: run(device, "reduce", rows=1, cols=c))))
        for r in range(1, args.max_passes + 1):
            check(f"reduce rows={r}", *run(device, "reduce", rows=r, cols=1))
            red_rows.append((r, measure(lambda r=r: run(device, "reduce", rows=r, cols=1))))
    finally:
        ttnn.close_device(device)

    t = args.tiles
    logger.info(f"pass sweeps over {t} tiles (slope = cost of one pass):")
    for mode in ("copy", "bcast", "alt"):
        for n, us in sweeps[mode]:
            logger.info(f"  {mode:6s} passes={n}  median={us:7.2f}us")
        s = slope(sweeps[mode])
        logger.info(f"  {mode:6s} slope = {s:.3f}us/pass ({s / t:.4f}us per tile per pass)")

    copy_s, bcast_s, alt_s = (slope(sweeps[m]) for m in ("copy", "bcast", "alt"))
    d = bcast_s - copy_s
    logger.info(f"broadcast math = bcast - copy = {d:.3f}us/pass ({d / t:.4f}us/tile), plumbing cancels")
    # alt runs the same two kinds in the same proportion, so anything above their mean
    # is what CHANGING kind costs -- which a homogeneous chain never pays.
    mean = (copy_s + bcast_s) / 2
    logger.info(
        f"switching kinds = alt - mean(copy, bcast) = {alt_s - mean:+.3f}us/pass ({(alt_s - mean) / t:+.4f}us/tile)"
    )

    for n, us in mm:
        logger.info(f"  matmul {mm_n}x{mm_n} passes={n}  median={us:7.2f}us")
    mm_s = slope(mm)
    # Each pass is mm_n*mm_n output tiles, each an inner product over mm_n tiles.
    macs = mm_n**3
    logger.info(f"  matmul slope = {mm_s:.3f}us/pass over {mm_n**2} output tiles, {macs} tile-MACs")
    logger.info(f"  => {mm_s / mm_n**2:.3f}us per output tile, {mm_s / macs:.3f}us per tile-MAC")
    logger.info(f"  a copy of the same {mm_n**2} tiles costs {slope(sweeps['copy']) / t * mm_n**2:.3f}us")

    logger.info("reduce<Cols>, one pass, swept by shape:")
    for c, us in red_cols:
        logger.info(f"  rows=1 cols={c}  ({c} in -> 1 out)  median={us:7.2f}us")
    per_in = slope(red_cols)
    logger.info(f"  slope = {per_in:.3f}us per INPUT tile (unpack + accumulate, no per-tile pack)")
    for r, us in red_rows:
        logger.info(f"  rows={r} cols=1  ({r} in -> {r} out)  median={us:7.2f}us")
    per_both = slope(red_rows)
    logger.info(f"  slope = {per_both:.3f}us per (input + output) tile (adds acquire and pack)")
    # The rows sweep grows input and output together, so what it adds over the cols
    # sweep is the per-OUTPUT-tile part on its own -- which is the price of the one
    # tile_regs_acquire per output tile that Strategy<ReduceFusion> still does.
    logger.info(f"  => per OUTPUT tile alone = {per_both - per_in:.3f}us (the acquire and pack)")

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
