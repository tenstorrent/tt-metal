#!/usr/bin/env python3
"""Draw the waveform pictures used on the repro pages as SVG files.

usage: make_figures.py <run.cyc> <out_prefix> [cycle]
Writes <out_prefix>.cycles.svg (90 cycles from `cycle`, default: 12 cycles before the math core's first L1 read
in the pack loop, or the middle of the loop if there is none) and <out_prefix>.tiles.svg (DEST waits per tile).
Colors: blue DEST read granted, red waiting, green L1 write accepted or instruction accepted,
grey pack RISC-V blocked on a full instruction queue, orange the math core's L1 read.
"""
import sys

rows = [l.split() for l in open(sys.argv[1])]
keys, rows = rows[0], rows[1:]
col = {k: i for i, k in enumerate(keys)}


def v(r, k):
    return r[col[k]]


def bit(s, b):
    return s[len(s) - 1 - b] == "1"


COLORS = {
    "grant": "#2f5d8a",
    "wait": "#a63c3c",
    "ok": "#2c7a4b",
    "stall": "#8a929a",
    "read": "#b7791f",
}

busy = [i for i, r in enumerate(rows) if v(r, "rden") not in ("0000", "?")]
spans, start, prev = [], busy[0], busy[0]
for i in busy[1:]:
    if i - prev > 500:
        spans.append((start, prev))
        start = i
    prev = i
spans.append((start, prev))
c0, c1 = max(spans, key=lambda s: s[1] - s[0])
reads = [i for i in range(c0, c1) if v(rows[i], "t1_l1_rden") == "1"]
at = (
    int(sys.argv[3])
    if len(sys.argv) > 3
    else (reads[0] - 12 if reads else (c0 + c1) // 2)
)


def svg(lines, n, cw, rh=20, lw=190, ticks=10, tick_label=lambda i: f"+{i}"):
    W, H = lw + n * cw + 10, len(lines) * rh + 26
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    ]
    for i in range(0, n + 1, ticks):
        x = lw + i * cw
        o.append(
            f'<line x1="{x}" y1="14" x2="{x}" y2="{H - 4}" stroke="#dcdfe2"/><text x="{x + 2}" y="10" font-size="10" fill="#5b636c">{tick_label(i)}</text>'
        )
    for r, (label, cells) in enumerate(lines):
        y = 18 + r * rh
        o.append(
            f'<text x="4" y="{y + rh * 0.62}" font-size="12" fill="#1d2228">{label}</text>'
        )
        for i, c in enumerate(cells):
            if c:
                o.append(
                    f'<rect x="{lw + i * cw + 0.5}" y="{y + 3}" width="{cw - 1}" height="{rh - 7}" fill="{COLORS[c]}"/>'
                )
    o.append("</svg>")
    return "\n".join(o)


span = range(at, at + 90)
lines = [
    (
        "math core: L1 read",
        ["read" if v(rows[i], "t1_l1_rden") == "1" else None for i in span],
    )
]
for b in range(4):
    lines.append(
        (
            f"packer {b}: L1 write",
            [
                (
                    "ok"
                    if v(rows[i], f"p{b}_l1_wren") == "1"
                    and v(rows[i], f"p{b}_l1_ready") == "1"
                    else ("wait" if v(rows[i], f"p{b}_l1_wren") == "1" else None)
                )
                for i in span
            ],
        )
    )
for b in range(4):
    lines.append(
        (
            f"packer {b}: DEST read",
            [
                (
                    "grant"
                    if bit(v(rows[i], "rden"), b) and bit(v(rows[i], "ready"), b)
                    else ("wait" if bit(v(rows[i], "rden"), b) else None)
                )
                for i in span
            ],
        )
    )
lines.append(
    (
        "pack RISC-V: instr. accepted",
        [
            (
                "ok"
                if v(rows[i], "t2_ibuf_wren") == "1"
                and v(rows[i], "t2_ibuf_ready") == "1"
                else None
            )
            for i in span
        ],
    )
)
lines.append(
    (
        "pack RISC-V: queue full",
        [
            (
                "stall"
                if v(rows[i], "t2_ibuf_wren") == "1"
                and v(rows[i], "t2_ibuf_ready") == "0"
                else None
            )
            for i in span
        ],
    )
)
open(sys.argv[2] + ".cycles.svg", "w").write(
    svg(lines, len(span), 9, tick_label=lambda i: str(at + i))
)

g0, waits = 0, {}
for i in range(c0, c1 + 1):
    r = rows[i]
    if bit(v(r, "rden"), 0) and bit(v(r, "ready"), 0):
        g0 += 1
    t = (g0 - 1) // 32
    waits[t] = waits.get(t, 0) + sum(
        bit(v(r, "rden"), b) and not bit(v(r, "ready"), b) for b in (1, 2, 3)
    )
cells = [None if waits.get(t, 0) == 0 else "wait" for t in range(64)]
open(sys.argv[2] + ".tiles.svg", "w").write(
    svg(
        [("packers 1-3 wait (red)", cells)],
        64,
        9,
        lw=150,
        ticks=8,
        tick_label=lambda i: f"tile {i}",
    )
)
print(
    "wrote", sys.argv[2] + ".cycles.svg", sys.argv[2] + ".tiles.svg", "from cycle", at
)
