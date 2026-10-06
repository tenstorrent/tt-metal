#!/usr/bin/env python3
"""Print the numbers on the repro pages from one run's .cyc file (to_cycles.py output).

usage: analyze_waveform.py <run.cyc>
"""
import sys

rows = [l.split() for l in open(sys.argv[1])]
keys, rows = rows[0], rows[1:]
col = {k: i for i, k in enumerate(keys)}


def v(r, k):
    return r[col[k]]


def bit(s, b):
    return s[len(s) - 1 - b] == "1"


# The measured pack loop: the longest stretch with DEST read requests and no gap longer than 500 cycles.
busy = [i for i, r in enumerate(rows) if v(r, "rden") not in ("0000", "?")]
spans, start, prev = [], busy[0], busy[0]
for i in busy[1:]:
    if i - prev > 500:
        spans.append((start, prev))
        start = i
    prev = i
spans.append((start, prev))
c0, c1 = max(spans, key=lambda s: s[1] - s[0])
win = range(c0, c1 + 1)
print(f"pack loop: cycles {c0} to {c1} ({c1 - c0 + 1} cycles)")

print("\nDEST reads per packer over the loop: cycles requesting, reads granted")
for b in range(4):
    req = sum(bit(v(rows[i], "rden"), b) for i in win)
    gnt = sum(bit(v(rows[i], "rden"), b) and bit(v(rows[i], "ready"), b) for i in win)
    print(f"  packer {b}: {req} cycles requesting, {gnt} granted")

# Tiles: packer 0 is granted 32 DEST reads per tile.
tile, g0, waits = [], 0, {}
for i in win:
    r = rows[i]
    if bit(v(r, "rden"), 0) and bit(v(r, "ready"), 0):
        g0 += 1
    t = (g0 - 1) // 32
    tile.append(t)
    for b in (1, 2, 3):
        if bit(v(r, "rden"), b) and not bit(v(r, "ready"), b):
            waits.setdefault(t, [0, 0, 0])[b - 1] += 1
print("\nwait cycles of packers 1/2/3 per tile, tiles 0 to 39:")
print("  " + " ".join("%d/%d/%d" % tuple(waits.get(t, [0, 0, 0])) for t in range(40)))

print("\nmath core (TRISC1) L1 accesses during the loop:")
probe = []
for k, i in enumerate(win):
    r = rows[i]
    if v(r, "t1_l1_rden") == "1" or v(r, "t1_l1_wren") == "1":
        addr = int(v(r, "t1_l1_addr"), 2) * 16
        print(
            f'  cycle {i} (tile {tile[k]}): {"write" if v(r, "t1_l1_wren") == "1" else "read"} at 0x{addr:x}'
        )
        probe.append(i)
for p in probe:
    print(
        f"\naround the math read at cycle {p}: packer L1 writes (W accepted, w waiting, . none), L1 port accepts"
    )
    for k in ["p0", "p1", "p2", "p3"]:
        s = "".join(
            (
                "W"
                if v(rows[i], f"{k}_l1_wren") == "1"
                and v(rows[i], f"{k}_l1_ready") == "1"
                else ("w" if v(rows[i], f"{k}_l1_wren") == "1" else ".")
            )
            for i in range(p - 3, p + 13)
        )
        print(f"  {k} L1 write   {s}")
    print(
        "  L1 port accepts "
        + "".join(v(rows[i], "l1_arbiter_packet_accept") for i in range(p - 3, p + 13))
        .replace("11", "2")
        .replace("01", "1")
        .replace("10", "1")
        .replace("00", "0")
    )
    print(
        "                  "
        + "".join("^" if i == p else " " for i in range(p - 3, p + 13))
    )

acc = sum(
    v(rows[i], "t2_ibuf_wren") == "1" and v(rows[i], "t2_ibuf_ready") == "1"
    for i in win
)
stall = sum(
    v(rows[i], "t2_ibuf_wren") == "1" and v(rows[i], "t2_ibuf_ready") == "0"
    for i in win
)
print(
    f"\npack RISC-V (TRISC2) instruction pushes: {acc} accepted, {stall} cycles blocked on a full queue"
)
