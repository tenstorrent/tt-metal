# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Misuse that MUST be refused. Each case is a bug the API used to accept silently.

Every other suite here checks that correct code gives correct answers. This one checks
that incorrect code is stopped, which is a different claim and needs its own tests:
a protection nobody has watched fail is not known to work.

Each case runs in a SUBPROCESS, for two reasons. A tripped device assert aborts the
process rather than raising, so it cannot be caught in-process; and a case that leaves
the device stopped must not take the rest of the suite with it.

See unified_api_hazards.md for the catalogue these come from.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_negative.py
"""

import os
import subprocess
import sys
import textwrap

from loguru import logger

# A circular buffer one page smaller than the block the kernel declares. Nothing connects
# the host's page count to the kernel's Shape, so before the check in Storage's constructor
# this waited on cb_reserve_back forever -- no assert, no output, and the device needed
# tt-smi -r. Verified: without the check it hangs, and it hangs under the watcher too.
UNDERSIZED_CB = """
import sys; sys.path.insert(0, ".")
import torch, ttnn
from unified_harness import make_cb, unified_program
import example_reduce as er
torch.manual_seed(0)
rows = er.NUM_CORES * er.IN_HT * er.TILE
x = ((torch.rand([rows, er.IN_WT * er.TILE]) - 0.5) / rows**0.5).to(torch.bfloat16)
d = ttnn.open_device(device_id=0)
try:
    tx = er.to_device(d, x)
    tout = er.to_device(d, torch.full([er.TILE, er.IN_WT * er.TILE], float("nan")))
    cr = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, er.NUM_CORES - 1))])
    cores = [ttnn.CoreCoord(0, y) for y in range(er.NUM_CORES)]
    cta = []
    for t in (tx, tout):
        cta.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    prog = unified_program(kernel_source=er.KERNEL, core_ranges=cr, cores=cores,
        cbs=[make_cb(er.CB_IN, cr, num_pages=er.IN_HT * er.IN_WT - 1),
             make_cb(er.CB_SCALER, cr, num_pages=1),
             make_cb(er.CB_PARTIAL, cr, num_pages=er.IN_WT),
             make_cb(er.CB_GATHERED, cr, num_pages=er.NUM_CORES * er.IN_WT),
             make_cb(er.CB_OUT, cr, num_pages=er.IN_WT)],
        compile_time_args=cta, runtime_args=[t.buffer_address() for t in (tx, tout)])
    ttnn.generic_op([tx, tout], prog)
    print("ACCEPTED")
finally:
    ttnn.close_device(d)
"""

# Twelve cores laid out row-major are eight in row 0 and four in row 1, whose bounding box
# is 2x8 = sixteen. reduction_tree.cpp barriers with the no-region synchronize_cores(),
# which addresses that box -- so it would wait on four cores that were never launched.
NON_RECTANGULAR_BARRIER = """
import sys; sys.path.insert(0, ".")
import torch, ttnn
from unified_harness import core_block, make_cb, unified_program
import test_unified_reduction as rt
cr, cores = core_block(12)
d = ttnn.open_device(device_id=0)
try:
    ht, wt, grid_h = 2, 2, 2
    dram = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(torch.zeros([1, 1, 4 * ht * 32, wt * 32], dtype=torch.bfloat16),
                         dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d, memory_config=dram)
    tout = ttnn.from_torch(torch.zeros([1, 1, 4 * 32, wt * 32], dtype=torch.bfloat16),
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d, memory_config=dram)
    cta = [1, ht, wt, grid_h]
    for t in (ta, ta, tout):
        cta.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    prog = unified_program(kernel_source="unified_kernels/reduction_tree.cpp",
        core_ranges=cr, cores=cores,
        cbs=[make_cb(rt.CB_IN0, cr, num_pages=ht * wt), make_cb(rt.CB_TMP0, cr, num_pages=wt),
             make_cb(rt.CB_TMP1, cr, num_pages=wt * grid_h), make_cb(rt.CB_SCALER, cr, num_pages=1),
             make_cb(rt.CB_OUT, cr, num_pages=wt)],
        compile_time_args=cta,
        runtime_args=[ta.buffer_address(), ta.buffer_address(), tout.buffer_address()])
    ttnn.generic_op([ta, ta, tout], prog)
    print("ACCEPTED")
finally:
    ttnn.close_device(d)
"""

# A launcher that passes one runtime argument too few -- the mistake that hung this device
# three times. The kernel then reads a loop bound from a slot nobody filled and spins. The
# harness appends a sentinel after the last argument and the kernel names the count it
# expects, so a short list puts the sentinel somewhere else and the check fires.
SHORT_RUNTIME_ARGS = """
import sys; sys.path.insert(0, ".")
import torch, ttnn
from unified_harness import core_block, make_cb, unified_program
import test_unified_binary as bn
d = ttnn.open_device(device_id=0)
try:
    shape = [1, 4, 32, 32]
    dram = ttnn.DRAM_MEMORY_CONFIG
    mk = lambda: ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=d, memory_config=dram)
    ta, tb, tout = mk(), mk(), mk()
    cr, cores = core_block(1)
    cta = []
    for t in (ta, tb, tout):
        cta.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    prog = unified_program(
        kernel_source=bn.KERNEL, core_ranges=cr, cores=cores,
        cbs=[make_cb(bn.CB_IN0, cr, num_pages=8), make_cb(bn.CB_IN1, cr, num_pages=8),
             make_cb(bn.CB_OUT, cr, num_pages=8)],
        compile_time_args=cta,
        named_compile_time_args=[("num_blocks", 2), ("tiles_per_block", 2)],
        runtime_args=[ta.buffer_address(), tb.buffer_address(), tout.buffer_address(), 0])
    ttnn.generic_op([ta, tb, tout], prog)
    print("ACCEPTED")
finally:
    ttnn.close_device(d)
"""


# (name, source, needs_watcher, what the refusal must say)
#
# needs_watcher marks a RUNTIME check. Device asserts are compiled out unless the watcher
# or LIGHTWEIGHT_KERNEL_ASSERTS is on, and only the watcher reports them to the host --
# lightweight halts the RISC, which the host cannot tell from a hang. A compile-time
# refusal needs neither.
CASES = [
    ("circular buffer smaller than the block", UNDERSIZED_CB, True, "tripped an assert"),
    ("no-region barrier on a non-rectangular grid", NON_RECTANGULAR_BARRIER, False, "static assertion failed"),
    # "tripped assert" rather than "tripped an assert": the watcher's per-core detail line
    # ("NCRISC tripped an assert on line N") goes to the watcher log, and what reaches the
    # host here is its summary. Matching what it actually prints, not what it might.
    ("one runtime argument too few", SHORT_RUNTIME_ARGS, True, "tripped assert"),
]

TIMEOUT_S = 240


def run_case(source, needs_watcher):
    env = dict(os.environ)
    if needs_watcher:
        env["TT_METAL_WATCHER"] = "5"
    try:
        p = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            env=env,
        )
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() + (e.stderr or b"").decode()
        return None, out


def main():
    failed = []
    for name, source, needs_watcher, expected in CASES:
        code, output = run_case(source, needs_watcher)

        if code is None:
            # The whole point of these checks: a timeout means the misuse still hangs.
            verdict, ok = "HUNG -- not refused", False
        elif "ACCEPTED" in output:
            verdict, ok = "ACCEPTED -- not refused", False
        elif expected not in output:
            verdict, ok = f"refused, but not by the expected check ({expected!r} absent)", False
        else:
            verdict, ok = f"refused: {expected}", True

        logger.info(f"  {name}: {verdict}   {'ok' if ok else 'FAIL'}")
        if not ok:
            failed.append(name)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
