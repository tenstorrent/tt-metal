# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14b: a host-side guard against overlapping circular buffers (2026-10-20 wm7 / pw4 root cause).

tt-metal places program CBs per UNIQUE CoreRange (tt_metal/impl/program/program.cpp allocate_circular_buffers: a CB's
address is the max region end of the allocators whose range EQUALS one of its ranges; intersecting allocators of other
ranges are only advanced when the CB is not below their end). A CB whose ranges are rectangles merged across core
groups (the stack's argument CB over readers + owners + root) can therefore land below another CB's end on some cores:
the native-t serving stack put its argument rows at 0x124000 inside the readers' third weight ring [0x121480, 0x143480)
and the weight stream overwrote them (a misaligned table address -> watcher NOC error). `overlaps(cbs)` replays that
placement exactly and returns the overlapping (core, CB, CB) triples; `check(cbs)` raises on any."""

from __future__ import annotations


def _r(r) -> tuple:
    return (int(r.start.x), int(r.start.y), int(r.end.x), int(r.end.y))


def _hit(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def placement(cbs, base: int = 0) -> list:
    """[(address, size, ranges, indices)] of `cbs` (CBDescriptors, program order) as tt-metal allocates them"""
    descs = [([_r(r) for r in cb.core_ranges.ranges()], int(cb.total_size),
              tuple(int(f.buffer_index) for f in cb.format_descriptors)) for cb in cbs]  # fmt: skip
    allocs = []  # [range, regions]
    for rs, _, _ in descs:
        for r in rs:
            if not any(a[0] == r for a in allocs):
                allocs.append([r, []])
    end = lambda a: a[1][-1][1] if a[1] else 0  # noqa: E731
    out = []
    for rs, size, idx in descs:
        addr = base
        for r in rs:
            addr = max([addr] + [end(a) for a in allocs if a[0] == r])
        addr = -(-addr // 64) * 64
        for r in rs:
            for a in allocs:
                if _hit(a[0], r) and not (a[0] != r and addr < end(a)):
                    if not a[1]:
                        a[1].append([base, base])
                    if addr == a[1][-1][1]:
                        a[1][-1][1] += size
                    elif addr > a[1][-1][1]:
                        a[1].append([addr, addr + size])
        out.append((addr, size, rs, idx))
    return out


def overlaps(cbs, base: int = 0) -> list:
    per_core = {}
    for addr, size, rs, idx in placement(cbs, base):
        for r in rs:
            for x in range(r[0], r[2] + 1):
                for y in range(r[1], r[3] + 1):
                    per_core.setdefault((x, y), []).append((addr, addr + size, idx))
    bad = []
    for c, lst in per_core.items():
        lst.sort()
        bad += [(c, i0, i1) for (_, a1, i0), (b0, _, i1) in zip(lst, lst[1:]) if b0 < a1]
    return bad


def check(cbs, what: str = "program"):
    bad = overlaps(cbs)
    if bad:
        pairs = sorted({(i0, i1) for _, i0, i1 in bad})
        raise RuntimeError(f"cb_guard: {what}: tt-metal's per-range CB placement overlaps CBs {pairs} on "
                           f"{len(bad)} core(s), e.g. {bad[0][0]}")  # fmt: skip
    return cbs
