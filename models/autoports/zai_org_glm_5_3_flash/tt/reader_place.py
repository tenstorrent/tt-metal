# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRAM-bank reader placement for a mesh program on Blackhole (host only: no ttnn import; mk_readers.reader_assignments
calls it under GLM53_MK_READER_PLACE, unset = the port's placement, unchanged).

Every chip of a mesh program runs its readers on the SAME logical cores, but the port takes ttnn's bank -> worker
assignment of the mesh's reference chip alone (MeshDevice.get_optimal_dram_bank_to_logical_worker_assignment(noc);
tt_metal/distributed/mesh_device.cpp: "exact only when the mesh is homogeneously harvested"). On Blackhole a bank's
optimal worker is the first active column EAST of the bank's DRAM column (tt_metal/common/core_assignment.cpp: dram x
+ 1, past harvested columns), because the weight streamer reads on NOC 0 (+x then +y, a torus) and a read's data comes
back along it. A logical column holds a different physical column on chips harvested differently: a reader the
reference chip sees just east of the east DRAM column (physical x 9) sits WEST of it on a chip with fewer harvested
west columns, and its data wraps around the torus (~15 x hops; Zoe Carver's regen D1: 15-18-hop readers made a third
of a Galaxy's chips 7-8 us late per MoE round, -425 us / token once placed east). The port's extra readers (index >= 1)
take the nearest free core by LOGICAL Manhattan distance, ties to the lower x then the lower y: the west / north
neighbour, against NOC 0's return direction (a north neighbour's data wraps ~11 y hops).

Modes, per bank b (opts = every chip's optimal logical worker per bank, {bank: (x, y)} each):
  "row"  the base core = (max over the chips of opt[b].x, opt[b].y): the first logical column east of the bank's DRAM
         column on EVERY chip (logical order is physical order on each chip); extra readers along the base's row at
         +x, then the rows below (+y), then the rows above; west of the base column only when nothing else is left.
         With 2 readers per bank this is the reference fork's MESH_ROW_PLUS_X pattern (tt/mk_dense.rect_placement: "chip
         0: readers at x = 0, 1, 6, 7").
  "col"  the same base; extra readers down the base's column (+y), then the columns east of it.
The list order is the port's: index 0 of every bank in bank order, then index 1, ... (consumers key on (bank, index)).
"""

from __future__ import annotations

MODES = ("row", "col")


def mesh_base(opts: list, banks: list) -> dict:
    """{bank: (x, y)}: per bank the largest optimal x and the largest optimal y over the chips -- east of the DRAM
    column and not north of the optimal row on any chip, so north() is 0 for the base cores by construction (rows are
    not harvested on Blackhole: every chip agrees on y unless its DRAM endpoints differ)"""
    out = {}
    for b in banks:
        xs = [int(o[b][0]) for o in opts]
        ys = [int(o[b][1]) for o in opts]
        out[b] = (max(xs), max(ys))
    return out


def walk(base: tuple, grid: tuple, mode: str) -> list:
    """the cells a bank's readers take in order from its base (the base first)"""
    bx, by = base
    gx, gy = grid
    if not (0 <= bx < gx and 0 <= by < gy):
        raise ValueError(f"reader placement: base {base} outside the {gx} x {gy} grid")
    below, above = list(range(by, gy)), list(range(by - 1, -1, -1))
    if mode == "row":
        east = [(x, y) for y in below + above for x in range(bx, gx)]
    elif mode == "col":  # down the columns east of the DRAM column first; rows above the base only after those
        east = [(x, y) for ys in (below, above) for x in range(bx, gx) for y in ys]
    else:
        raise ValueError(f"GLM53_MK_READER_PLACE={mode!r}: one of {MODES}")
    west = [(x, y) for x in range(bx - 1, -1, -1) for y in below + above]
    return east + west


def plan(opts: list, grid: tuple, readers: int, mode: str) -> list:
    """[((x, y), bank, index)] for `readers` readers per bank (see the module docstring); opts[0] is the reference
    chip. Raises when the grid has fewer free cells than readers."""
    banks = sorted(int(b) for b in opts[0])
    base = mesh_base(opts, banks)
    paths = {b: walk(base[b], grid, mode) for b in banks}
    pos = {b: 0 for b in banks}
    taken, out = set(), []
    for idx in range(readers):
        for b in banks:
            p = paths[b]
            while pos[b] < len(p) and p[pos[b]] in taken:
                pos[b] += 1
            if pos[b] >= len(p):
                raise ValueError(f"reader placement {mode}: no free core for bank {b} reader {idx}")
            cell = p[pos[b]]
            taken.add(cell)
            out.append((cell, b, idx))
    return out


def wrapped(assign: list, opts: list) -> list:
    """per chip: how many readers sit WEST of their bank's optimal column on that chip (x < opt[b].x: NOC 0 read data
    wraps around the torus to reach them) -- 0 everywhere for "row" / "col" by construction"""
    return [sum(1 for (x, _), b, _ in assign if int(x) < int(o[int(b)][0])) for o in opts]


def north(assign: list, opts: list) -> list:
    """per chip: readers in a row ABOVE their bank's optimal row (y < opt[b].y: the data wraps in y on NOC 0)"""
    return [sum(1 for (_, y), b, _ in assign if int(y) < int(o[int(b)][1])) for o in opts]
