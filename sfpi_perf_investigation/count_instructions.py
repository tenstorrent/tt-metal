#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Count the Tensix instructions each probe function executes per DEST row.

sfpi drives the SFPU through the replay buffer, so the number of instructions in
the emitted stream is not the number executed. This walks the stream and models
the buffer:

  TTREPLAY start,len,_,1   records the next `len` Tensix instructions into slots
                           [start, start+len) and executes them once
  TTREPLAY start,len,0,0   executes slots [start, start+len) again

Total executed, divided by the 8 unrolled loop iterations, is the cost per DEST
row -- the quantity PerfRunType.MATH_ISOLATE measures.

Usage: count_instructions.py <file.s> [file.s ...]
"""
import os
import re
import sys

# A Tensix instruction reaches the SFPU either as a bare mnemonic or, when the
# address is not a compile-time constant, as a store to the instruction buffer
# that the assembler annotates with the mnemonic it encodes.
TENSIX = re.compile(r"^\s*(?:sw\s+\S+\s*0\(\S+\)\s*#\s*\d+:(SFP\w+)|(SFP\w+|TTINCRWC)\b)")
REPLAY = re.compile(r"^\s*TTREPLAY\s+(\d+),\s*(\d+),\s*(\d+),\s*(\d+)")
FUNC = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*$")
TAIL = re.compile(r"^\s*tail\s+(\S+)")

UNROLL = 8


def analyse(path):
    order, results = [], {}
    cur = None
    for line in open(path):
        m = FUNC.match(line)
        if m:
            cur = m.group(1)
            results[cur] = {"executed": 0, "recording": 0}
            order.append(cur)
            continue
        if cur is None:
            continue
        st = results[cur]

        m = TAIL.match(line)
        if m:
            # The compiler tail-merged two probes: they compile identically.
            st["alias"] = m.group(1)
            continue

        m = REPLAY.match(line)
        if m:
            _start, length, _exec, record = (int(x) for x in m.groups())
            if record:
                st["recording"] = length
            else:
                st["recording"] = 0
                st["executed"] += length
            continue

        if TENSIX.match(line):
            # While recording, an instruction is both emitted and executed once;
            # its replays are counted at the TTREPLAY above.
            st["executed"] += 1
            st["recording"] = max(0, st["recording"] - 1)
    return order, results


for path in sys.argv[1:]:
    order, results = analyse(path)
    print(f"=== {os.path.basename(path)} ===")
    for fn in order:
        st = results[fn]
        if "alias" in st:
            print(f"  {fn:28s} identical codegen to {st['alias']}")
        elif st["executed"]:
            print(f"  {fn:28s} {st['executed'] / UNROLL:6.2f} instr/row" f"   (total {st['executed']})")
    print()
