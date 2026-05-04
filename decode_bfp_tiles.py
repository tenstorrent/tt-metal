#!/usr/bin/env python3
"""Decode per-K cb_in1 tile bytes from dprint.txt using TT-NN's bfp encoding.

Matches the encoding in models/demos/deepseek_v3_b1/compressed_tensor/tile_utils.py.

Each line in dprint.txt:
  13:(x=7,y=4):TR0: [exp_i=0 K=k N=2 fmt=F off=OFF sz=SZ]: w0 w1 w2 ...

Tile layout (32x32 logical, 4 faces of 16x16):
  - 64 bytes shared exponents (1 per face-row, IEEE-754 biased 8-bit).
  - mantissa+sign block:
      bfp8 (fmt=3): 1 byte/element, 1024 bytes total
      bfp4 (fmt=2): 2 elements/byte, 512 bytes total
      bfp2 (fmt=1): 4 elements/byte, 256 bytes total

Within a byte for bfp4/bfp2: low nibble/bits = lower element index.

Float reconstruction per element (matches ttnn HW unpack):
  code = (sign << mant_bits) | mantissa
  shift_cnt[man], man_shifted[man] = lookup table (see _build_table)
  exp_out = shared_exp - shift_cnt (zero if man==0)
  u32 = (sign << 31) | (exp_out << 23) | (man_shifted << (23 - mant_bits))
  value = float32 from u32 bits
"""
from __future__ import annotations

import math
import re
import struct
import sys
from collections import defaultdict


def words_to_bytes(words: list[int]) -> bytes:
    return b"".join(struct.pack("<I", w) for w in words)


def _build_table(mant_bits: int) -> tuple[list[int], list[int]]:
    mask = (1 << mant_bits) - 1
    shift_cnt = [0] * (mask + 1)
    man_shifted = [0] * (mask + 1)
    for man in range(1, mask + 1):
        msb_pos = int(math.floor(math.log2(man)))
        shift = (mant_bits - 1) - msb_pos
        shift_cnt[man] = shift
        man_shifted[man] = (man << (shift + 1)) & mask
    return shift_cnt, man_shifted


def _u32_to_f32(u32: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u32 & 0xFFFFFFFF))[0]


def _decode_one(code: int, mant_bits: int, shared_exp: int, shift_cnt: list[int], man_shifted: list[int]) -> float:
    mask = (1 << mant_bits) - 1
    sign = (code >> mant_bits) & 0x1
    man = code & mask
    if man == 0:
        # +0 / -0 in IEEE-754; treat as 0.
        return 0.0
    msh = man_shifted[man]
    exp_out = (shared_exp - shift_cnt[man]) & 0xFF
    u32 = (sign << 31) | (exp_out << 23) | (msh << (23 - mant_bits))
    return _u32_to_f32(u32)


def decode_bfp(fmt: int, raw: bytes) -> list[float]:
    """Decode a tile's bytes to 1024 float values in face-major × row-major × col order."""
    if fmt == 0 or len(raw) == 0:
        return [0.0] * 1024

    fmt_to_mb = {1: 1, 2: 3, 3: 7}
    mant_bits = fmt_to_mb[fmt]
    shift_cnt, man_shifted = _build_table(mant_bits)
    exps = raw[:64]
    mants = raw[64:]
    out = [0.0] * 1024

    if mant_bits == 7:  # bfp8
        for face in range(4):
            for row in range(16):
                se = exps[face * 16 + row]
                row_off = (face * 16 + row) * 16
                base = face * 256 + row * 16
                for col in range(16):
                    code = mants[row_off + col]
                    out[base + col] = _decode_one(code, mant_bits, se, shift_cnt, man_shifted)
    elif mant_bits == 3:  # bfp4
        for face in range(4):
            for row in range(16):
                se = exps[face * 16 + row]
                row_off = (face * 16 + row) * 8
                base = face * 256 + row * 16
                for byte_idx in range(8):
                    b = mants[row_off + byte_idx]
                    for nib_idx in range(2):
                        code = (b >> (4 * nib_idx)) & 0xF
                        out[base + byte_idx * 2 + nib_idx] = _decode_one(code, mant_bits, se, shift_cnt, man_shifted)
    elif mant_bits == 1:  # bfp2
        for face in range(4):
            for row in range(16):
                se = exps[face * 16 + row]
                row_off = (face * 16 + row) * 4
                base = face * 256 + row * 16
                for byte_idx in range(4):
                    b = mants[row_off + byte_idx]
                    for elem_idx in range(4):
                        code = (b >> (2 * elem_idx)) & 0x3
                        out[base + byte_idx * 4 + elem_idx] = _decode_one(code, mant_bits, se, shift_cnt, man_shifted)
    return out


_LINE_RE = re.compile(
    r"^(?P<dev>\d+):\([^)]*\):TR\d:\s*\[exp_i=0 K=(?P<k>\d+) N=2 fmt=(?P<fmt>\d+) "
    r"off=(?P<off>[0-9a-fA-F]+) sz=(?P<sz>\d+)\]:\s*(?P<words>.*)$"
)

_CB_IN0_RE = re.compile(r"^(?P<dev>\d+):\([^)]*\):TR\d:\s*\[exp_i=0\]\s+cb_in0\s+K=(?P<k>\d+):\s*(?P<words>.*)$")

_POST_HDR_RE = re.compile(r"^(?P<dev>\d+):\([^)]*\):TR1:\s*\[exp_i=0 ng=0 POST-FINALIZE N=2\]:\s*$")
_POST_ROW_RE = re.compile(r"^(?P<dev>\d+):\([^)]*\):TR1:\s+(?P<vals>(\s*-?\d+\.\d+){8,})\s*$")


def bf16_from_u16(u16: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (u16 & 0xFFFF) << 16))[0]


def decode_cb_in0_words(words: list[int]) -> list[float]:
    """Each uint32 word holds 2 bf16: low 16 bits = even-index, high 16 bits = odd-index."""
    out: list[float] = []
    for w in words:
        out.append(bf16_from_u16(w & 0xFFFF))
        out.append(bf16_from_u16((w >> 16) & 0xFFFF))
    return out


def parse_dprint(path: str):
    with open(path) as f:
        for line in f:
            m = _LINE_RE.match(line.rstrip())
            if not m:
                continue
            words = [int(w, 16) for w in m.group("words").split() if w]
            # Note: the kernel emits all integers in HEX mode (HEX() persists in DPRINT
            # state), so sz, K, fmt, off, dev are all hex-encoded in the dump.
            yield (
                int(m.group("dev")),
                int(m.group("k"), 16),
                int(m.group("fmt"), 16),
                int(m.group("off"), 16),
                int(m.group("sz"), 16),
                words,
            )


def parse_post_finalize(path: str) -> dict[int, list[float]]:
    """Returns {dev: [floats...]} for POST-FINALIZE N=2 dst dumps.

    dprint output is interleaved across devices: after dev D emits its
    POST-FINALIZE header, other devs' lines may appear before dev D's float
    rows arrive. So track an *active set* of devices and accept float rows
    from any active dev regardless of order.
    """
    out: dict[int, list[float]] = defaultdict(list)
    active: set[int] = set()
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            m = _POST_HDR_RE.match(line)
            if m:
                active.add(int(m.group("dev")))
                continue
            m = _POST_ROW_RE.match(line)
            if m:
                dev = int(m.group("dev"))
                if dev in active:
                    out[dev].extend(float(v) for v in m.group("vals").split())
    return out


def summarize(values: list[float]) -> str:
    nz = [v for v in values if v != 0.0]
    if not nz:
        return "all-zero"
    return (
        f"min={min(nz):+.4f} max={max(nz):+.4f} " f"|max|={max(abs(v) for v in nz):.4g} " f"nz={len(nz)}/{len(values)}"
    )


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "/data/yugao/tt-metal/dprint.txt"
    by_k: dict[int, list[tuple[int, int, list[float]]]] = defaultdict(list)
    for dev, k, fmt, off, sz, words in parse_dprint(path):
        raw = words_to_bytes(words)
        if len(raw) != sz:
            print(f"WARN dev{dev} K={k}: dumped {len(raw)} bytes but sz={sz}")
        decoded = decode_bfp(fmt, raw)
        by_k[k].append((dev, fmt, decoded))

    for k in sorted(by_k):
        print(f"=== K={k} ===")
        for dev, fmt, decoded in sorted(by_k[k]):
            head = " ".join(f"{v:+.4g}" for v in decoded[:8])
            print(f"  dev{dev:>2} fmt={fmt} {summarize(decoded)} head=[{head}]")

    post = parse_post_finalize(path)
    if post:
        print("\n=== POST-FINALIZE dst[N=2] (output) ===")
        for dev in sorted(post):
            vals = post[dev]
            mx = max(abs(v) for v in vals) if vals else 0.0
            wild = sum(1 for v in vals if abs(v) > 1e10)
            flag = "  *** WILD ***" if wild > 0 else ""
            head = " ".join(f"{v:+.4g}" for v in vals[:8])
            print(f"  dev{dev:>2} |max|={mx:.4g} wild={wild}/{len(vals)} head=[{head}]{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
