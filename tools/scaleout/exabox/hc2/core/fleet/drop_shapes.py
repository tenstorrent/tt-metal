#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""H3-shaped tray-drop loads.

A quad 3 host's tray 2 (41:00.0-48:00.0) lost its 12 V as a whole tray under H3 2 of 2 times and under plain run.sh 0 of 9.
Both H3 drops came 5-8 min into a hot phase, with each tray swinging on its own between ~1.2 kW (busy clock: CCL,
DRAM, PCIe) and the 2.55 kW sensor top every 2-8 s, out of phase with the others; run.sh loads every tray in lockstep
in minute-long chunks between clock-checked idle quiets.

A shape says, at each moment of the load, which of chunk_load.py's parts run on which tray: "hi" (matmul) or "lo"
(the busy-clock CCL + DRAM + PCIe mix) per tray. chunk_load.py --shape runs each part per launch on only the trays
whose state lists it (tray_lanes.only). The chunk list and quiet come from here too (repro.sh: FLEET_BUSBAR_CHUNK_MIN,
FLEET_GAP_S). Shapes never touch a power limit; the load runs at the box's own TDP.

  python3 drop_shapes.py list            # the shapes, in try order: name<TAB>chunk min<TAB>gap s<TAB>what
  python3 drop_shapes.py env <shape>     # FLEET_* settings repro.sh runs it with
  python3 drop_shapes.py trace <shape> <tray> [seconds] [seed]   # the per-tray state each second (tests, dry runs)
"""
import random
import sys

TRAYS = (1, 2, 3, 4)
FULL = ("heavy", "h3", "mm", "sfpu", "dram", "pcie")

# name: what it does, the evidence, chunk minutes, forced quiet after each chunk (s), hi parts, lo parts, chunk_load.py
# argument overrides. Try order: the shape the notes point to first, the cold start (run.sh-like, H3 never dropped
# there) last.
SHAPES = {
    "h3-replay": dict(
        what="each tray on its own: matmul+CCL ~1-3 s, busy-clock CCL+DRAM+PCIe ~1.5-5 s, seeded random phase per tray",
        why="both H3 drops: trays swing 1.2<->2.55 kW out of phase every 2-8 s for minutes; lost 5-8 min into it",
        chunks="12",
        gap=10,
        hi=("heavy", "h3", "mm", "sfpu"),
        lo=("h3", "dram", "pcie"),
        args={},
    ),
    "h3-cadence": dict(
        what="all trays together: matmul+CCL 2 s, then busy-clock CCL+DRAM+PCIe 4-8 s (seeded), repeated",
        why="H3's step period at busy clock, without the cross-tray imbalance",
        chunks="12",
        gap=10,
        hi=("heavy", "h3", "mm", "sfpu"),
        lo=("h3", "dram", "pcie"),
        args={},
    ),
    "imbalance": dict(
        what="the named tray at matmul while the other three stay at busy-clock CCL+DRAM+PCIe, swapped every 3 s",
        why="drop 1: tray 2 high while 2 or more other trays were low 23% of the time",
        chunks="12",
        gap=10,
        hi=("heavy", "mm", "sfpu"),
        lo=("h3", "dram", "pcie"),
        args={},
    ),
    "weights": dict(
        what="matmul on all trays with 64 MB x4 DRAM streams and a 16 MB PCIe write per chip every launch",
        why="H3 reads weights and latents from DRAM and moves them over PCIe during compute",
        chunks="12",
        gap=10,
        hi=("heavy", "dram", "pcie"),
        lo=("heavy", "dram", "pcie"),
        args=dict(dram_mb=64, dram_reps=4, pcie_mb=16, pcie_every=1),
    ),
    "cold-full": dict(
        what="clock-checked idle, then straight to the full mix on all trays, 2 min each, 6 times",
        why="control for the cold step: H3 never dropped there, run.sh's chunks start this way",
        chunks="2,2,2,2,2,2",
        gap=60,
        hi=FULL,
        lo=FULL,
        args={},
    ),
}
ORDER = tuple(SHAPES)


class Shape:
    """The per-tray hi/lo state at t seconds into the load, for the trays loaded (target: the tray named by the BDF)."""

    def __init__(self, name, trays=TRAYS, target=2, seed=3290):
        if name not in SHAPES:
            raise ValueError(f"--shape takes {', '.join(ORDER)} (got '{name}')")
        self.name, self.s = name, SHAPES[name]
        self.trays, self.target = tuple(trays or TRAYS), int(target)
        if self.target not in self.trays:
            raise ValueError(f"shape {name}: tray {self.target} is not loaded (trays {','.join(map(str, self.trays))})")
        self.rng = {t: random.Random(seed * 10 + t) for t in (0,) + self.trays}  # 0: the box (h3-cadence)
        self.seg = {t: [] for t in self.trays}  # per tray: [(end s, state)], grown lazily

    def _grow(self, key, t):
        """Segments for one tray (h3-replay) or the box (key 0) until past t."""
        seg = self.seg.setdefault(key, [])
        rng = self.rng[key]
        if not seg:  # a random phase: start hi or lo, part way through
            st = rng.choice(("hi", "lo"))
            seg.append((rng.uniform(0.2, 3.0), st))
        while seg[-1][0] <= t:
            end, st = seg[-1]
            nxt = "lo" if st == "hi" else "hi"
            if self.name == "h3-replay":
                d = rng.uniform(1.0, 3.0) if nxt == "hi" else rng.uniform(1.5, 5.0)
            else:  # h3-cadence
                d = 2.0 if nxt == "hi" else rng.uniform(4.0, 8.0)
            seg.append((end + d, nxt))
            if len(seg) > 4096:
                del seg[:2048]
        for end, st in seg:
            if t < end:
                return st
        return seg[-1][1]

    def state(self, t):
        """{tray: 'hi' | 'lo'} at t s."""
        n = self.name
        if n == "h3-replay":
            return {tr: self._grow(tr, t) for tr in self.trays}
        if n == "h3-cadence":
            st = self._grow(0, t)
            return {tr: st for tr in self.trays}
        if n == "imbalance":
            on = int(t // 3.0) % 2 == 0
            return {tr: ("hi" if (tr == self.target) == on else "lo") for tr in self.trays}
        return {tr: "hi" for tr in self.trays}  # weights, cold-full: one state

    def parts(self, t, parts=FULL):
        """{part: None (every loaded tray) or a tuple of trays (some only)} for the parts that run on any tray at t s."""
        st = self.state(t)
        out = {}
        for p in parts:
            on = tuple(tr for tr in self.trays if p in self.s[st[tr]])
            if on:
                out[p] = None if len(on) == len(self.trays) else on
        return out

    def per_tray(self):
        """True when the shape runs a part on some trays only (needs chunk_load.py --trays: one submesh per tray)."""
        return self.name in ("h3-replay", "imbalance")


def env(name):
    """The FLEET_* settings repro.sh's tray-drop way runs a shape with (plain run.sh, bus-bar stage mix only)."""
    s = SHAPES[name]
    return (
        f"FLEET_BUSBAR_SHAPE={name} FLEET_BUSBAR_ORDER=mix FLEET_BUSBAR_CHUNK_MIN={s['chunks']} FLEET_GAP_S={s['gap']} "
        "FLEET_BUSBAR_RW_AFTER=0 FLEET_BUSBAR_HEAVY_N=1 FLEET_BUSBAR_PCIE_EVERY=1 FLEET_LOAD_TRAYS=1,2,3,4"
    )


def minutes(name):
    s = SHAPES[name]
    ch = [float(x) for x in s["chunks"].split(",")]
    return sum(ch) + len(ch) * s["gap"] / 60.0


def main(argv):
    if argv and argv[0] == "list":
        for n in ORDER:
            print(f"{n}\t{SHAPES[n]['chunks']}\t{SHAPES[n]['gap']}\t{SHAPES[n]['what']}")
        return 0
    if len(argv) == 2 and argv[0] == "env" and argv[1] in SHAPES:
        print(env(argv[1]))
        return 0
    if len(argv) >= 3 and argv[0] == "trace" and argv[1] in SHAPES:
        sh = Shape(argv[1], target=int(argv[2]), seed=int(argv[4]) if len(argv) > 4 else 3290)
        for t in range(int(argv[3]) if len(argv) > 3 else 30):
            st = sh.state(float(t))
            print(t, " ".join(f"{tr}:{st[tr]}" for tr in sh.trays))
        return 0
    print(
        f"usage: drop_shapes.py list | env <shape> | trace <shape> <tray> [s] [seed]; shapes: {', '.join(ORDER)}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
