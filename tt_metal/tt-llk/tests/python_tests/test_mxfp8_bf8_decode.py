# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
MXFP8 (E4M3 / E5M2) stored as remapped bfloat8_b, restored on device by the SFPU decode
    out = U - (b << 17) + (g(b) << 23),  g(b) = 2b - popcount(b)
(b = the unpacked mantissa bits below the MXFP8 mantissa), checked against the exact MXFP8 value for every e8m0.

Host encoding: magnitude code c -> 7-bit bfp8 magnitude m = ENC[c] (inverse of the decode on the integers 0..127),
stored value sign * m * 2^(e8m0 - 127 - C), C = 9 (E4M3) / 16 (E5M2); raw shared exponent e8m0 + 6 - C.

Stimuli: for each e8m0 in 0..254, every finite signed code once (0 once), padded with +0 to 256 = 16 blocks.
255 * 16 blocks + 16 zero blocks = 64 tiles. Every block lies inside one e8m0 (one MXFP8 scale group).

Variants:
  packed           the harness packer (pack_bfp8_b) on the stored floats, codes in ascending order per e8m0
  packed_shuffled  same, codes shuffled within each e8m0 (seeded) so blocks have varied maxima
  direct           raw bytes: shared exponent e8m0 + 6 - C, magnitude ENC[c], -0 written as +0 (the host rule)
"""

import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.pack import pack_bfp8_b
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TILE_COUNT, TemplateParameter


@dataclass
class MXFP8_FORMAT(TemplateParameter):
    fmt: str = "e4m3"

    def convert_to_cpp(self) -> str:
        return "#define MXFP8_E5M2" if self.fmt == "e5m2" else ""


TILES = 64
N = TILES * 1024
BF16_MAX = (2 - 2.0**-7) * 2.0**127
MIN_NORMAL = 2.0**-126
OUT_DIR = os.environ.get("MXFP8_BF8_OUT", "/tmp/mxfp8_bf8")
MASK32 = 0xFFFFFFFF

FMT = {
    # name: (mantissa bits, bias)
    "e4m3": (3, 7),
    "e5m2": (2, 15),
}


def _value(fmt, c):
    mb, bias = FMT[fmt]
    e, f = c >> mb, c & ((1 << mb) - 1)
    if (fmt == "e4m3" and c == 127) or (fmt == "e5m2" and e == 31):
        return None
    if e == 0:
        return f * 2.0 ** (1 - bias - mb)
    return ((1 << mb) + f) * 2.0 ** (e - bias - mb)


def _g(b):
    return 2 * b - bin(b).count("1")


def _decode_formula(u, fmt):
    bbits = 6 - FMT[fmt][0]
    b = (u >> 17) & ((1 << bbits) - 1)
    return (u - (b << 17) + (_g(b) << 23)) & MASK32


def _encoding(fmt):
    """code -> m: invert the decode on float(m), m = 0..127 (q = 0)."""
    mb, bias = FMT[fmt]
    C = bias - 1 + mb
    by_value = {}
    for m in range(128):
        u = int(np.array([m], dtype=np.float32).view(np.uint32)[0])
        out = np.array([_decode_formula(u, fmt)], dtype=np.uint32).view(np.float32)[0]
        by_value.setdefault(float(out) * 2.0**-C, []).append(m)
    enc = {}
    for c in range(128):
        v = _value(fmt, c)
        if v is not None:
            ms = by_value[v]
            assert len(ms) == 1
            enc[c] = ms[0]
    assert len(set(enc.values())) == len(enc)
    return enc, C


def _stimuli(fmt, shuffle):
    rnd = random.Random(1234)
    codes = [c for c in range(128) if _value(fmt, c) is not None]
    signed = [(c, s) for c in codes for s in ((1, -1) if c else (1,))]
    assert len(signed) <= 256
    code, sgn, e8 = [], [], []
    for e in range(255):
        grp = list(signed)
        if shuffle:
            rnd.shuffle(grp)
        grp += [(0, 1)] * (256 - len(grp))
        code += [c for c, _ in grp]
        sgn += [s for _, s in grp]
        e8 += [e] * 256
    pad = N - len(code)
    return (
        np.array(code + [0] * pad),
        np.array(sgn + [1] * pad),
        np.array(e8 + [-1] * pad),
    )


def _direct_bytes(m, sgn, e8, C):
    """bfloat8_b tile images: 64 shared exponents, then 1024 bytes sign << 7 | magnitude."""
    out = bytearray()
    for t in range(TILES):
        sl = slice(t * 1024, (t + 1) * 1024)
        tm, ts, tile_e8 = m[sl], sgn[sl], e8[sl]
        exps = []
        for b in range(64):
            e = tile_e8[b * 16]
            exps.append(0 if e < 0 else int(np.clip(e + 6 - C, 0, 255)))
        sign = np.where((ts < 0) & (tm != 0), 1, 0).astype(np.uint8)
        out += bytes(exps) + bytes(((sign << 7) | tm.astype(np.uint8)).tolist())
    return bytes(out)


FORMATS = [InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Float16_b)]


@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    variant=["packed", "packed_shuffled", "direct"],
    mxfp8=["e4m3", "e5m2"],
)
def test_mxfp8_bf8_decode(formats, dest_acc, variant, mxfp8):
    enc, C = _encoding(mxfp8)
    code, sgn, e8 = _stimuli(mxfp8, shuffle=variant == "packed_shuffled")
    valid = e8 >= 0
    vals = np.array([_value(mxfp8, int(c)) for c in code], dtype=np.float64)
    m = np.array([enc[int(c)] for c in code], dtype=np.int64)
    s_exp = e8.astype(np.float64) - 127
    exact = np.where(valid, sgn * vals * np.exp2(s_exp), 0.0)
    stored = np.where(valid, sgn * m * np.exp2(s_exp - C), 0.0)

    if variant.startswith("packed"):
        st = torch.from_numpy(stored.astype(np.float32))
        packed = b"".join(
            bytes(pack_bfp8_b(st[t * 1024 : (t + 1) * 1024])) for t in range(TILES)
        )
        storable = np.ones(N, dtype=bool)
    else:
        packed = _direct_bytes(m, sgn, e8, C)
        storable = (e8 + 6 - C) >= 0
    assert len(packed) == TILES * 1088

    configuration = TestConfig(
        "sources/mxfp8_bf8_decode_test.cpp",
        formats,
        templates=[MXFP8_FORMAT(mxfp8)],
        runtimes=[TILE_COUNT(TILES)],
        variant_stimuli=StimuliConfig(
            packed,
            formats.input_format,
            torch.zeros(1024, dtype=torch.bfloat16),
            DataFormat.Float16_b,
            formats.output_format,
            tile_count_A=TILES,
            tile_count_B=1,
            tile_count_res=TILES + 1,  # the extra tile carries the timing words
        ),
        dest_acc=dest_acc,
    )
    res = configuration.run().result
    full = torch.tensor(res, dtype=torch.bfloat16)
    assert full.numel() >= N + 8, full.numel()
    timing_halves = (
        full[N : N + 8].view(torch.int16).numpy().view(np.uint16).astype(np.uint32)
    )
    timing_words = (timing_halves[0::2] | (timing_halves[1::2] << 16)).tolist()
    got = full[:N]
    got_bits = got.view(torch.int16).numpy().view(np.uint16)
    got64 = got.double().numpy()

    # exactness domain: exact value a normal bf16 (or 0) and the stored bfp8 value a normal fp32 (or 0)
    in_range = (exact == 0) | (
        (np.abs(exact) >= MIN_NORMAL) & (np.abs(exact) <= BF16_MAX)
    )
    stored_normal = (stored == 0) | (np.abs(stored) >= MIN_NORMAL)
    check = valid & storable
    dom = check & in_range & stored_normal
    exact_bits = (
        torch.from_numpy(exact)
        .to(torch.bfloat16)
        .view(torch.int16)
        .numpy()
        .view(np.uint16)
    )
    value_ok = got64 == exact
    bits_ok = got_bits == exact_bits

    def by_e8(mask):
        return {
            int(e): int(c) for e, c in zip(*np.unique(e8[mask], return_counts=True))
        }

    summary = dict(
        format=mxfp8, variant=variant, dest_acc=dest_acc.name, checked=int(check.sum())
    )
    decode_cycles, empty_cycles, tile_cnt, marker = timing_words
    summary["timing"] = dict(
        marker_ok=marker == 0x4D584638,
        tiles=tile_cnt,
        net_cycles_per_tile=(decode_cycles - empty_cycles) / TILES,
    )
    summary["domain_checked"] = int(dom.sum())
    summary["domain_value_mismatch"] = int((dom & ~value_ok).sum())
    summary["domain_bit_mismatch"] = int((dom & ~bits_ok).sum())
    summary["domain_value_mismatch_by_e8m0"] = by_e8(dom & ~value_ok)
    # outside the domain: zero in a tiny block (harness packer quirk), stored denormal, exact out of bf16 range
    zero_leak = check & (exact == 0) & ~value_ok
    summary["zero_mismatch_by_e8m0"] = by_e8(zero_leak)
    summary["stored_denormal_mismatch_by_e8m0"] = by_e8(
        check & in_range & ~stored_normal & ~value_ok
    )
    summary["exact_out_of_range_mismatch_by_e8m0"] = by_e8(
        check & ~in_range & ~value_ok
    )
    examples = []
    for j in np.nonzero(check & ~value_ok)[0][:12]:
        examples.append(
            dict(
                e8m0=int(e8[j]),
                code=int(code[j]),
                sign=int(sgn[j]),
                m=int(m[j]),
                exact=float(exact[j]),
                got=float(got64[j]),
                got_bits=f"0x{int(got_bits[j]):04x}",
            )
        )
    summary["examples"] = examples
    summary["padding_nonzero"] = int((~valid & (got64 != 0)).sum())

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/{mxfp8}-{variant}-{dest_acc.name}.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

    assert summary["padding_nonzero"] == 0, summary
    # a zero in a block whose shared exponent is <= 8 is a harness-packer artifact (pack.py ORs the implicit bit)
    domain = dom & ~(
        (exact == 0) & np.isin(variant, ["packed", "packed_shuffled"]) & (e8 < 30)
    )
    assert int((domain & ~value_ok).sum()) == 0, summary
