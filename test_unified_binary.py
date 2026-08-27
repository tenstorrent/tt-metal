# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the unified binary ops on device: add, sub, mul, div, and a mixed chain.

One kernel source (unified_kernels/binary.cpp) compiled for all five baby RISC-V
threads via three KernelDescriptors. See unified_harness.py.

Two gates, both needed:

  1. Max RELATIVE error against torch. PCC is reported but does not gate -- it is
     invariant to a global scale, so it passes bugs (measured: see
     test_unified_unary.py, where a dropped chain link still scored 0.9958).

  2. For sub and div, the SWAPPED reference must FAIL. a-b and b-a differ only in
     sign, so a test that only checked `a-b` would pass an implementation that
     computed `b-a` if it were also comparing against the wrong reference. This
     makes operand order an explicit, independently failing assertion rather than
     something inferred from a single number.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_binary.py
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import core_block, dfb, run_unified_spec, split_evenly, unified_program_spec

KERNEL = "unified_kernels/binary.cpp"


# name -> (kernel define, reference, commutative, max relative error)
#
# mul's limit is looser than the others and that is a hardware choice, not slack.
# add, sub and mul all dispatch to the FPU now (see expr::kind_of), and for add and sub
# that is free -- as accurate or better than the SFPU forms. The FPU multiply is not:
# 0.01023 against the SFPU's 0.00380, for 3.4x the speed. The second phase in main()
# re-runs these three with the FPU disabled and holds THAT path to 0.01, so loosening
# here cannot hide a regression in the accurate path -- and the gap itself is asserted
# rather than described.
#
# The chain gets its own, looser tolerance, and the reason is arithmetic rather
# than slack: (a + b) - a cancels, which amplifies bfloat16's relative error by
# |a + b| / |b|. Over the input range below that factor reaches (2 + 0.5) / 0.5 = 5,
# so 5 * 2^-8 = 0.020 before the mul and div round again. Measured: 0.025. The
# single ops sit at 0.004, which is 2^-8 itself -- the format's own floor.
OPS = {
    "add": (None, lambda a, b: a + b, True, 0.01),
    "sub": ("BN_SUB", lambda a, b: a - b, False, 0.01),
    "mul": ("BN_MUL", lambda a, b: a * b, True, 0.015),  # FPU; the SFPU form is held to 0.01 below
    "div": ("BN_DIV", lambda a, b: a / b, False, 0.01),
    # An elementwise max, which the online softmax of a flash attention folds its running
    # row maxima with. Commutative, so no swapped-reference check applies.
    "max": ("BN_MAX", lambda a, b: torch.maximum(a, b), True, 0.01),
    # ((a + b) - a) * b / a  ==  b*b/a
    "chain": ("BN_CHAIN", lambda a, b: b * b / a, False, 0.05),
    # SwiGLU's core. silu is x*sigmoid(x), one SFPU op on device.
    "silu_mul": ("BN_SILU_MUL", lambda a, b: torch.nn.functional.silu(a) * b, False, 0.02),
}


def run(device, op, num_blocks=1, tiles_per_block=1, seed=0, force_sfpu=False, cores=1):
    num_tiles = num_blocks * tiles_per_block
    shape = [1, num_tiles, 32, 32]

    torch.manual_seed(seed)
    # Strictly positive and away from zero, so div is well conditioned and the
    # swapped reference is a genuinely different tensor rather than a near-tie.
    a = (0.5 + 1.5 * torch.rand(shape)).to(torch.bfloat16)
    b = (0.5 + 1.5 * torch.rand(shape)).to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    tout = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    # Blocks are the unit of work and they share nothing: block b touches its own pages of
    # both inputs and of the output, so splitting them needs no reduction and no ordering.
    ncores = min(cores, num_blocks)
    core_ranges, core_list = core_block(ncores)
    shares = split_evenly(num_blocks, ncores)

    named_ct_args = [("num_blocks", num_blocks), ("tiles_per_block", tiles_per_block)]

    dfbs = [
        dfb("in0", 2 * tiles_per_block),
        dfb("in1", 2 * tiles_per_block),
        dfb("out", 2 * tiles_per_block),
    ]

    define, reference, commutative, _tol = OPS[op]
    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=dfbs,
        named_compile_time_args=named_ct_args,
        tensors={"in0": ta, "in1": tb, "out": tout},
        runtime_arg_names=["block_begin", "block_count"],
        defines=([(define, "1")] if define else []) + ([("TT_UNIFIED_NO_FPU_ELTWISE", "1")] if force_sfpu else []),
    )

    logger.info(
        f"running unified binary: op={op} num_blocks={num_blocks} tiles_per_block={tiles_per_block} cores={ncores}"
    )
    run_unified_spec(
        device,
        spec,
        {"in0": ta, "in1": tb, "out": tout},
        runtime_args={
            # Per core: its slice of the block range. Named, so a launcher that supplied
            # one and not the other is an error from metal rather than a garbage bound.
            "block_begin": {c: b for c, (b, _) in zip(core_list, shares)},
            "block_count": {c: n for c, (_, n) in zip(core_list, shares)},
        },
    )
    out = tout

    got = ttnn.to_torch(out).to(torch.float32)
    af, bf = a.to(torch.float32), b.to(torch.float32)
    want = reference(af, bf)
    swapped = None if commutative else reference(bf, af)
    return got, want, swapped


def pcc(got, want):
    g, w = got.flatten(), want.flatten()
    if torch.equal(g, w):
        return 1.0
    return torch.corrcoef(torch.stack([g, w]))[0, 1].item()


def max_rel_err(got, want):
    return ((got - want).abs() / want.abs().clamp(min=1e-6)).max().item()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=list(OPS) + ["all"], default="all")
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--tiles-per-block", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pcc", type=float, default=0.99)
    p.add_argument("--rel-err", type=float, default=None, help="override the per-op relative-error limit")
    args = p.parse_args(argv)

    ops = list(OPS) if args.op == "all" else [args.op]
    results = {}

    device = ttnn.open_device(device_id=0)
    try:
        for op in ops:
            results[op] = run(device, op, args.num_blocks, args.tiles_per_block, args.seed)
        # The three ops that exist on both units, run again on the SFPU. This is what
        # keeps mul's looser limit above honest: the accurate path is still gated at
        # 0.01, so a regression there cannot hide behind the FPU's allowance.
        sfpu = {
            op: run(device, op, args.num_blocks, args.tiles_per_block, args.seed, force_sfpu=True)
            for op in ops
            if op in ("add", "sub", "mul")
        }
        # Partition invariance. Blocks share nothing, so splitting them is not an
        # approximation and the check is EXACT rather than toleranced -- a core reading or
        # writing outside its range would show as a difference, and nothing else would.
        # 16 blocks over 1, 3 and 16 cores also covers the uneven split, which is where an
        # off-by-one in the range arithmetic lives.
        partitioned = {}
        for op in ("add", "silu_mul"):
            if op not in ops:
                continue
            one = run(device, op, 16, 4, args.seed)[0]
            partitioned[op] = [(n, one, run(device, op, 16, 4, args.seed, cores=n)[0]) for n in (3, 8, 16)]
    finally:
        ttnn.close_device(device)

    failed = []
    for op in ops:
        got, want, swapped = results[op]
        tol = args.rel_err if args.rel_err is not None else OPS[op][3]
        measured, rel = pcc(got, want), max_rel_err(got, want)
        ok = measured >= args.pcc and rel <= tol
        line = f"{op:6s} PCC = {measured:>9.6f}   max rel err = {rel:.5f} (<= {tol})"
        if swapped is not None:
            # Operand order: the swapped reference must NOT match.
            rel_swapped = max_rel_err(got, swapped)
            order_ok = rel_swapped > tol
            line += f"   swapped rel err = {rel_swapped:.3f} ({'rejected' if order_ok else 'MATCHES'})"
            ok = ok and order_ok
        logger.info(f"{line}   {'ok' if ok else 'FAIL'}")
        if not ok:
            failed.append(op)

    for op, (got, want, _) in sfpu.items():
        rel = max_rel_err(got, want)
        ok = rel <= 0.01
        logger.info(f"{op:6s} on the SFPU: max rel err = {rel:.5f} (<= 0.01)   {'ok' if ok else 'FAIL'}")
        if not ok:
            failed.append(f"{op}-sfpu")
    # The FPU multiply really is the less accurate one. Asserting the ORDER, not just
    # the bounds, is what would catch the two implementations being silently swapped.
    if "mul" in sfpu and "mul" in results:
        fpu_err = max_rel_err(results["mul"][0], results["mul"][1])
        sfpu_err = max_rel_err(sfpu["mul"][0], sfpu["mul"][1])
        logger.info(f"mul    FPU {fpu_err:.5f} vs SFPU {sfpu_err:.5f}  (FPU is expected to be the looser)")
        if not fpu_err > sfpu_err:
            logger.error("the FPU multiply is no longer the less accurate one -- has dispatch changed?")
            failed.append("mul-order")

    for op, runs in partitioned.items():
        for n, one, many in runs:
            diff = (many - one).abs().max().item()
            ok = diff == 0.0
            logger.info(f"{op:6s} 16 blocks over {n:2d} cores vs 1: max|diff| = {diff:.6f}   {'ok' if ok else 'FAIL'}")
            if not ok:
                failed.append(f"{op}-cores-{n}")

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
