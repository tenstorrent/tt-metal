# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""EP-4 with weights on Y: speed of light per component (WORKLOG "EP-4 with weights on Y (LoudBox)"). n tokens a chip,
raw 512 GB/s DRAM, 608 TFLOP/s, 2 links a direction (25 GB/s each on the Galaxy; the LoudBox's from argv[1], default
50). P(chip holds >= 1 of a token's 12 experts) = 1 - (3/4)^12; dispatch a drop-off tree, combine unicast partials;
bf16 rows of 7,680 B; expert weights 4.15 MB (bfp4); expert DRAM 12 n rows read + 12 n written; pre-sum 12 n read + the
own partial; home pass 3.87 n partials + shared + output; on the ring the opposite chip's traffic split over both
directions. Usage: python ep4_targets.py [LoudBox GB/s a link direction]."""
import itertools
import sys

H, I, E, K, SH = 3840, 640, 640, 12, 2944
Rb = H * 2
L = 25e9
NL = 2
LLB = float(sys.argv[1]) * 1e9 if len(sys.argv) > 1 else 50e9
D = 512e9
F = 608e12
EXP_B = 4.15e6
q = (3 / 4) ** K
p = 1 - q


def route(s, d, ring, v=0):
    if not ring:
        st = 1 if d > s else -1
        return list(range(s, d + st, st))
    f = (d - s) % 4
    if f == 2:
        return [s, (s + 1) % 4, (s + 2) % 4] if v == 0 else [s, (s - 1) % 4, (s - 2) % 4]
    if f == 1:
        return [s, (s + 1) % 4]
    return [s, (s - 1) % 4]


def links_load(ring, mode):
    load = {}

    def add(a, b, w):
        load[(a, b)] = load.get((a, b), 0) + w

    for s in range(4):
        others = [c for c in range(4) if c != s]
        for hit in itertools.product([0, 1], repeat=3):
            pr = 1
            for h in hit:
                pr *= p if h else q
            dests = [o for o, h in zip(others, hit) if h]
            if not dests:
                continue
            variants = [(1.0, 0)] if not ring else [(0.5, 0), (0.5, 1)]
            for w, v in variants:
                paths = [route(s, d, ring, v) for d in dests]
                if mode == "unicast":
                    for path in paths:
                        for a, b in zip(path, path[1:]):
                            add(a, b, pr * w)
                else:
                    used = {(a, b) for path in paths for a, b in zip(path, path[1:])}
                    for a, b in used:
                        add(a, b, pr * w)
    return load


def tx(n, ring, mode, lk=None):
    ld = links_load(ring, mode)
    m = max(ld.values())
    return m * n * Rb / (NL * (lk or L)) * 1e3, m


if __name__ == "__main__":
    for name, T_glx in [("302K", 302373), ("588K", 588369), ("874K", 874365), ("1M", 1000000)]:
        n = T_glx / 32
        disp_r = tx(n, True, "tree")
        disp_l = tx(n, False, "tree", LLB)
        comb_r = tx(n, True, "unicast")
        comb_l = tx(n, False, "unicast", LLB)
        rows = K * n
        math = rows * 2 * 3 * H * I / F * 1e3
        dram_exp = (2 * rows * Rb) / D * 1e3
        st_glx = 140 * EXP_B / (4 * L) * 1e3
        st_lb = 80 * EXP_B / (2 * LLB) * 1e3
        presum = (rows * Rb + p * n * Rb) / D * 1e3
        home = ((4 * p) * n * Rb + 2 * n * Rb) / D * 1e3
        shared = n * 2 * 3 * H * SH / F * 1e3
        print(f"{name}: n/chip={n:,.0f}  LoudBox T={8 * n:,.0f}")
        print(
            f"  dispatch X  ring {disp_r[0]:.2f} ms (max {disp_r[1]:.2f} rows/tok/link-dir)  line {disp_l[0]:.2f} ms ({disp_l[1]:.2f})"
        )
        print(f"  combine  X  ring {comb_r[0]:.2f} ms ({comb_r[1]:.2f})  line {comb_l[0]:.2f} ms ({comb_l[1]:.2f})")
        print(f"  experts: stream Y glx {st_glx:.2f} / LB {st_lb:.2f}; math {math:.2f}; DRAM rows {dram_exp:.2f}")
        print(f"  presum DRAM {presum:.2f}; home pass DRAM {home:.2f}; shared math {shared:.2f}")
        S_lb = max(st_lb, math, dram_exp)
        S_g = max(st_glx, math, dram_exp)
        for lab, S, C in [("glx", S_g, comb_r[0]), ("LB", S_lb, comb_l[0])]:
            x = min(1, (S / (3 * C)) ** 0.5)
            sent = min(1, x**3 + (1 - x) * S / C)
            print(
                f"  overlap {lab}: S {S:.2f} C {C:.2f} x* {x:.2f} hidden {sent * 100:.0f}% tail {(1 - sent) * C:.2f} ms"
            )
        seq_lb = max(disp_l[0], shared) + S_lb + max(presum, comb_l[0]) + home
        print(f"  LB in-sequence ~{seq_lb:.1f} ms (disp||shared + experts + max(presum,comb) + home; head excluded)")
