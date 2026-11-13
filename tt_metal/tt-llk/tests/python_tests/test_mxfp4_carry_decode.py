# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
MXFP4 stored as relabelled bfloat4_b, restored on device by the SFPU carry decode
bits += 3 * (bits & 0x00200000), checked against the exact MXFP4 value for every e8m0 scale.

Stimuli: for each e8m0 in 0..254 and each block maximum (e2m1 index 0..7, both signs), one 16-element
bfloat4_b block holding that maximum plus every signed e2m1 code no larger than it. 255 * 16 blocks,
padded with zero blocks to 64 tiles. Each block is one face row and lies inside one MXFP4 scale group,
as in the along-K layout.

Variants (how the bfloat4_b bytes are made):
  packed     the harness packer on the relabelled values sign * L[i] * 2^(e8m0 - 128),
             L = [0, 1, 2, 3, 4, 6, 5, 7] (normalized, like ttnn's packer)
  direct     the MXFP4 nibble with magnitude indices 5 and 6 swapped, -0 written as 0x0, shared exponent
             e8m0 + 1 (not normalized: a block's largest mantissa may be below 4). This is the host rule.
  direct_raw like direct but -0 keeps its sign bit (nibble 0x8): checks that it decodes to -inf, which is
             why the host rule writes -0 as 0x0
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.pack import pack_bfp4_b
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TILE_COUNT, TemplateParameter


@dataclass
class MXFP4_DECODER(TemplateParameter):
    """The ckernel_sfpu_mxfp4_decode.h decoder: sfpi (the six-instruction sfpi loop), macro (3-issue
    SFPLOADMACRO), macro2 (2-load SFPLOADMACRO) or macro2_replay (macro2 from the replay buffer, as the
    Blaze K3 decode runs it). The SFPLOADMACRO ones are silicon only; ttsim lacks the instruction.
    """

    decoder: str = "sfpi"

    def convert_to_cpp(self) -> str:
        return f"#define MXFP4_DECODER {DECODERS.index(self.decoder)}"


DECODERS = ["sfpi", "macro", "macro2", "macro2_replay"]


ON_SIMULATOR = bool(os.environ.get("TT_METAL_SIMULATOR")) and not os.environ.get(
    "MXFP4_FORCE_MACRO"
)

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
L = np.array([0, 1, 2, 3, 4, 6, 5, 7], dtype=np.int64)
TILES = 64
N = TILES * 1024
FP32_MAX = (2 - 2.0**-23) * 2.0**127
BF16_MAX = (2 - 2.0**-7) * 2.0**127
BF16_MIN_NORMAL = 2.0**-126
OUT_DIR = os.environ.get("MXFP4_CARRY_OUT", "/tmp/mxfp4_carry")


def _relabel(i):
    return i ^ (3 * ((i >> 2) & ((i >> 1) ^ i) & 1))


assert all(_relabel(i) == L[i] for i in range(8))


def _stimuli():
    idx, sgn, e8 = [], [], []
    for e in range(255):
        for top in range(8):
            for top_sign in (1, -1):
                codes = [(top, top_sign)] + [
                    (i, s) for i in range(top + 1) for s in (1, -1)
                ]
                codes = (codes + [(0, 1)] * 16)[:16]
                idx += [c[0] for c in codes]
                sgn += [c[1] for c in codes]
                e8 += [e] * 16
    pad = N - len(idx)
    idx = np.array(idx + [0] * pad)
    sgn = np.array(sgn + [1] * pad)
    e8 = np.array(e8 + [-1] * pad)  # -1: padding block
    return idx, sgn, e8


def _direct_bytes(idx, sgn, e8, keep_neg_zero):
    """bfloat4_b tile images: 64 shared exponents, then 512 bytes of nibbles (even element in the low nibble)."""
    out = bytearray()
    for t in range(TILES):
        sl = slice(t * 1024, (t + 1) * 1024)
        ti, ts, tile_e8 = idx[sl], sgn[sl], e8[sl]
        exps = []
        for b in range(64):
            e = tile_e8[b * 16]
            exps.append(0 if e < 0 else e + 1)
        mag = np.array([_relabel(int(i)) for i in ti], dtype=np.uint8)
        sign = (ts < 0).astype(np.uint8)
        if not keep_neg_zero:
            sign = np.where(mag == 0, 0, sign).astype(np.uint8)
        nib = (sign << 3) | mag
        nib = nib.reshape(-1, 2)
        out += bytes(exps) + bytes(
            (nib[:, 0] | (nib[:, 1] << 4)).astype(np.uint8).tolist()
        )
    return bytes(out)


FORMATS = [InputOutputFormat(DataFormat.Bfp4_b, DataFormat.Float16_b)]


@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    variant=["packed", "direct", "direct_raw"],
    decoder=DECODERS,
)
def test_mxfp4_carry_decode(formats, dest_acc, variant, decoder):
    if decoder.startswith("macro") and ON_SIMULATOR:
        pytest.skip("ttsim does not implement SFPLOADMACRO")
    idx, sgn, e8 = _stimuli()
    valid = e8 >= 0
    exact = np.where(valid, sgn * E2M1[idx] * np.exp2(e8.astype(np.float64) - 127), 0.0)
    stored = np.where(valid, sgn * L[idx] * np.exp2(e8.astype(np.float64) - 128), 0.0)

    # blocks the float path cannot hold (a stored value above the fp32 maximum, e8m0 = 254 with codes >= 2.0)
    unstorable = (np.abs(stored).reshape(-1, 16) > FP32_MAX).any(-1).repeat(16)

    if variant == "packed":
        st = torch.from_numpy(np.where(unstorable, 0.0, stored).astype(np.float32))
        packed = b"".join(
            bytes(pack_bfp4_b(st[t * 1024 : (t + 1) * 1024])) for t in range(TILES)
        )
        skipped = unstorable
    else:
        packed = _direct_bytes(idx, sgn, e8, keep_neg_zero=variant == "direct_raw")
        skipped = np.zeros(N, dtype=bool)
    assert len(packed) == TILES * 576

    configuration = TestConfig(
        "sources/mxfp4_carry_decode_test.cpp",
        formats,
        templates=[MXFP4_DECODER(decoder)],
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

    in_range = (exact == 0) | (
        (np.abs(exact) >= BF16_MIN_NORMAL) & (np.abs(exact) <= BF16_MAX)
    )
    check = valid & ~skipped
    exact_bits = (
        torch.from_numpy(exact)
        .to(torch.bfloat16)
        .view(torch.int16)
        .numpy()
        .view(np.uint16)
    )
    value_ok = (got64 == exact) | (np.isnan(got64) & np.isnan(exact))
    bits_ok = got_bits == exact_bits

    summary = dict(
        variant=variant,
        dest_acc=dest_acc.name,
        decoder=decoder,
        checked=int(check.sum()),
    )
    decode_cycles, empty_cycles, tile_cnt, marker = timing_words
    summary["timing"] = dict(
        marker_ok=marker == 0x4D584650,
        tiles=tile_cnt,
        decode_cycles_per_tile=decode_cycles / TILES,
        bracket_cycles_per_tile=empty_cycles / TILES,
        net_cycles_per_tile=(decode_cycles - empty_cycles) / TILES,
    )
    summary["in_range_checked"] = int((check & in_range).sum())
    summary["in_range_value_mismatch"] = int((check & in_range & ~value_ok).sum())
    summary["in_range_bit_mismatch"] = int((check & in_range & ~bits_ok).sum())
    summary["out_of_range_value_mismatch_by_e8m0"] = {
        int(e): int(c)
        for e, c in zip(
            *np.unique(e8[check & ~in_range & ~value_ok], return_counts=True)
        )
    }
    summary["in_range_value_mismatch_by_e8m0"] = {
        int(e): int(c)
        for e, c in zip(
            *np.unique(e8[check & in_range & ~value_ok], return_counts=True)
        )
    }
    neg_zero = check & (exact == 0) & (sgn < 0)
    summary["neg_zero_outputs"] = sorted({str(v) for v in got64[neg_zero].tolist()})
    examples = []
    for j in np.nonzero(check & ~value_ok)[0][:12]:
        examples.append(
            dict(
                e8m0=int(e8[j]),
                code=f"{'-' if sgn[j] < 0 else '+'}{E2M1[idx[j]]}",
                exact=float(exact[j]),
                got=float(got64[j]),
                got_bits=f"0x{int(got_bits[j]):04x}",
            )
        )
    summary["examples"] = examples
    # padding blocks must come back as zero
    summary["padding_nonzero"] = int((~valid & (got64 != 0)).sum())

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/{decoder}-{variant}-{dest_acc.name}.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

    assert summary["padding_nonzero"] == 0, summary
    if variant == "direct_raw":
        # A sign-only nibble (0x8) decodes to -inf, so the host must write MXFP4 -0 as 0x0.
        bad = check & in_range & ~value_ok
        assert np.array_equal(bad, neg_zero), summary
        assert np.all(np.isneginf(got64[neg_zero])), summary
    else:
        assert summary["in_range_value_mismatch"] == 0, summary
