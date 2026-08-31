# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The custom_compute escape hatch: a pass written against the raw compute API.

unified_kernels/custom_compute.cpp computes a - b by hand -- its own reserve, its own DST
bracketing, its own pack and push -- on blocks the unified model waits and pops. Subtraction
is the point: it is not commutative, so a routine handed the two dataflow-buffer ids in the
wrong order gives a wrong answer rather than the same one.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_custom_compute.py
"""

import sys

import torch
from loguru import logger

import ttnn
from unified_harness import dfb, run_unified_spec, single_core, unified_program_spec

KERNEL = "unified_kernels/custom_compute.cpp"
TILE = 32


def run(device, tiles=4, seed=0):
    torch.manual_seed(seed)
    shape = [1, tiles, TILE, TILE]
    a = (torch.rand(shape) - 0.5).to(torch.bfloat16)
    b = (torch.rand(shape) - 0.5).to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG

    def to_dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)

    ta, tb = to_dev(a), to_dev(b)
    tout = to_dev(torch.full(shape, float("nan")).to(torch.bfloat16))

    core_ranges, cores = single_core()

    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=[
            dfb("a", tiles),
            dfb("b", tiles),
            dfb("out", tiles),
        ],
        named_compile_time_args=[("tiles", tiles)],
        tensors={"a": ta, "b": tb, "out": tout},
    )

    run_unified_spec(device, spec, {"a": ta, "b": tb, "out": tout})
    out = tout
    got = ttnn.to_torch(out).to(torch.float32)

    # Both references, because ONE of them cannot check what matters.
    #
    # a - b is checked to a TOLERANCE, not exactly. Two attempts at exactness both failed
    # and each was informative: against the exact fp32 difference the error is 0.001953,
    # and against that difference rounded to bfloat16 it is 0.003906 -- so the device is
    # neither, it computes at the FPU's own precision and the pack narrows. One bfloat16
    # ULP at these magnitudes is 0.0039, which is where the gate sits.
    #
    # b - a is checked to be REJECTED, and that is the operand-order test. A tolerance
    # alone could not make it: swap the two buffer ids and the answer is wrong by twice
    # the operand magnitude, which no rounding gate would admit but which a loose one
    # might. Asserting the swapped form does NOT match is the sharp version, and it is
    # what test_unified_binary does for its non-commutative ops.
    af, bf = a.to(torch.float32), b.to(torch.float32)
    return got, af - bf, bf - af


def main():
    device = ttnn.open_device(device_id=0)
    failed = []
    try:
        for tiles in (1, 4, 8):
            got, want, swapped = run(device, tiles)
            err = (got - want).abs().max().item()
            swapped_err = (got - swapped).abs().max().item()
            ok = err <= 0.004 and swapped_err > 0.1
            logger.info(
                f"  custom_compute a - b, {tiles} tiles: max|error| = {err:.6f}, "
                f"swapped = {swapped_err:.4f} ({'rejected' if swapped_err > 0.1 else 'MATCHES'})   "
                f"{'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failed.append(f"tiles-{tiles}")
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
