#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Link-data patterns and their bitwise check (a data bit stuck at 0, a flipped bit in link traffic).

Each sending chip gets its own bf16 pattern, built on the host from (seed, chip index): walking ones and walking zeros
over the sign and mantissa bits, then seeded random words. Every exponent is in [120, 130]: 127 and 128 are bit
complements, so every data bit is driven both ways, and every value is a normal finite number (no zero, NaN, Inf or
denormal), so a device ne() equals a bitwise compare. Every received copy is compared bit for bit with this host-built
value. A 16x16 face row is 16 words = 32 B, so a bit's place in a 32 B chunk, (column % 16) * 16 + bit, is the same in
tile and row-major order; the report names it ("pos").

  link_pattern.py summary <rows.jsonl> <devnode A> <name A> <devnode B> <name B>
      fabric_repro.py's linkdata rows -> one line per direction, "A -> B: N words wrong, bit 3 1->0 x12 (pos 3,19)"
"""
import json, sys

try:   # fmt() and summary() run under a plain python3 (fleet.sh's report lines); the rest needs numpy
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

EXP_LO, EXP_HI = 120, 130
MB = [0, 1, 2, 3, 4, 5, 6, 15]          # sign and mantissa bits
MBMASK = sum(1 << b for b in MB)       # 0x807F
BLOCK = 256                            # rows built per chip; tiled over the rest


def _block(seed, chip, w):
    rng = np.random.default_rng([int(seed), int(chip)])
    b = np.empty((BLOCK, w), dtype=np.uint16)
    lane = np.arange(w) % 16
    for r in range(32):                # rows 0-15 walking ones, 16-31 walking zeros, shifted per row and chip
        m = (lane + r + chip) % 8
        e = np.where((lane + r // 8) % 2 == 0, 127, 128).astype(np.uint16) << 7
        one = (np.uint16(1) << np.array(MB, dtype=np.uint16)[m]).astype(np.uint16)
        b[r] = e | (one if r < 16 else (MBMASK & ~one).astype(np.uint16))
    n = (BLOCK - 32, w)
    sign = rng.integers(0, 2, n, dtype=np.uint16) << 15
    exp = rng.integers(EXP_LO, EXP_HI + 1, n, dtype=np.uint16) << 7
    man = rng.integers(0, 128, n, dtype=np.uint16)
    b[32:] = sign | exp | man
    return b


def pattern(seed, chip, h, w):
    """uint16 [h, w]: chip's bf16 bit pattern."""
    b = _block(seed, chip, w)
    return np.tile(b, (-(-h // BLOCK), 1))[:h] if h > BLOCK else b[:h].copy()


def to_torch(u16):
    """uint16 numpy -> torch bfloat16, same bits."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(u16).view(np.int16)).view(torch.bfloat16)


def from_torch(t):
    """torch bfloat16 -> uint16 numpy [-1, last dim], same bits."""
    import torch
    t = t.contiguous()
    return t.view(torch.int16).numpy().view(np.uint16).reshape(-1, t.shape[-1])


def diff(got, exp, chunk=1024):
    """Bitwise compare of two uint16 [h, w] arrays. None when equal, else words wrong, the first wrong (row, col) and
    per 32 B position the count of 1->0 and 0->1 flips ({pos: n})."""
    got = got.reshape(-1, exp.shape[-1]); exp = exp.reshape(got.shape)
    words, first = 0, None
    c10 = np.zeros((16, 16), dtype=np.int64); c01 = np.zeros((16, 16), dtype=np.int64)   # [lane, bit]
    for r0 in range(0, exp.shape[0], chunk):
        e = exp[r0:r0 + chunk]; x = got[r0:r0 + chunk] ^ e
        if not x.any():
            continue
        nz = np.flatnonzero(x)
        words += nz.size
        if first is None:
            first = [int(r0 + nz[0] // x.shape[1]), int(nz[0] % x.shape[1])]
        if x.shape[1] % 16 == 0:
            xr, er = x.reshape(-1, 16), e.reshape(-1, 16)
            lane_of = None
        else:                          # odd width: lane from the column, on the wrong words only
            xr, er = x.reshape(-1)[nz][:, None], e.reshape(-1)[nz][:, None]
            lane_of = (nz % x.shape[1]) % 16
        for bit in range(16):
            f = (xr >> bit) & 1
            if not f.any():
                continue
            d = f & ((er >> bit) & 1)  # was 1, read 0
            if lane_of is None:
                c10[:, bit] += d.sum(axis=0, dtype=np.int64); c01[:, bit] += (f - d).sum(axis=0, dtype=np.int64)
            else:
                np.add.at(c10[:, bit], lane_of, d[:, 0].astype(np.int64)); np.add.at(c01[:, bit], lane_of, (f - d)[:, 0].astype(np.int64))
    if not words:
        return None
    pos = lambda c: {int(l * 16 + b): int(c[l, b]) for l, b in zip(*np.nonzero(c))}
    return {"words": words, "first": first, "f10": pos(c10), "f01": pos(c01)}


def fmt(d, top=6):
    """'N words wrong, bit 3 1->0 x12 (pos 3,19), ...; first at row r col c'."""
    if not d:
        return "clean"
    by = {}
    for key, s in (("f10", "1->0"), ("f01", "0->1")):
        for p, n in d[key].items():
            k = (int(p) % 16, s); c, ps = by.get(k, (0, []))
            by[k] = (c + n, ps + [int(p)])
    parts = [f"bit {b} {s} x{c} (pos {','.join(str(p) for p in sorted(ps)[:4])}{',...' if len(ps) > 4 else ''})"
             for (b, s), (c, ps) in sorted(by.items(), key=lambda kv: -kv[1][0])[:top]]
    more = f", +{len(by) - top} more" if len(by) > top else ""
    first = d.get("first")
    return f"{d['words']} words wrong, {', '.join(parts)}{more}" + (f"; first at row {first[0]} col {first[1]}" if first else "")


def summary(path, ia, na, ib, nb):
    """fabric_repro rows -> 'A -> B: ...; B -> A: ...' over every linkdata row (sender != receiver), merged."""
    names = {str(ia): na, str(ib): nb}; acc = {}
    local = []
    for l in open(path, errors="replace"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get("event") != "linkdata":
            continue
        s, v = str(r.get("send_node", r.get("send"))), str(r.get("recv_node", r.get("recv")))
        if s == v:
            local.append(f"{names.get(v, v)} local copy: {r.get('detail', '')}")
            continue
        a = acc.setdefault((s, v), {"words": 0, "launches": [], "f10": {}, "f01": {}, "first": r.get("first")})
        a["words"] += int(r.get("words", 0)); a["launches"].append(r.get("i"))
        for k in ("f10", "f01"):
            for p, n in (r.get(k) or {}).items():
                a[k][p] = a[k].get(p, 0) + int(n)
    out = [f"{names.get(s, s)} -> {names.get(v, v)}: {fmt(a)} (launch {','.join(str(i) for i in a['launches'][:3])})"
           for (s, v), a in sorted(acc.items())]
    return "; ".join(out + local)


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "summary":
        print(summary(*sys.argv[2:]))
    else:
        sys.exit(__doc__)
