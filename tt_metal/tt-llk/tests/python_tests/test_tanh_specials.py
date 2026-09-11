# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The 257 bfloat16 patterns the exhaustive ULP sweep cannot reach.

`StimuliSpec.ulp_sweep` enumerates the 65,279 distinct *finite* values and
dedupes -0.0 against +0.0, so 2^16 - 65,279 = 257 bit patterns never reach the
kernel: 2 infinities, 254 NaNs, and negative zero.

That matters for this kernel specifically. Segment 5 of the SFPLUT is the
constant pair (A=0, B=1), evaluated as `A*|x| + B` -- so an infinite input makes
it compute `0 * inf + 1`, which is NaN in IEEE arithmetic. Whether the hardware
actually does that is not something to reason about; a pre-scaled variant of this
same table was measured returning NaN for 120 large *finite* inputs for exactly
this reason.

Reference: tanh(+-inf) = +-1, tanh(NaN) = NaN, tanh(+-0.0) = +-0.0 (sign kept).
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    generate_input_dim,
)


def _bf16_bits_to_f32(bits: int) -> float:
    """A bfloat16 bit pattern widened to float32 (bf16 is fp32's top 16 bits)."""
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


# exp field all ones -> 2 infinities + 254 NaNs, both signs
POS_SPECIALS = [_bf16_bits_to_f32(b) for b in range(0x7F80, 0x8000)]  # 128
NEG_SPECIALS = [_bf16_bits_to_f32(b) for b in range(0xFF80, 0x10000)]  # 128
ZEROS = [0.0, -0.0]


def _run(values_by_face, out_fmt, dest_acc):
    formats = InputOutputFormat(DataFormat.Float16_b, out_fmt)
    dims = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]]
    spec = StimuliSpec.custom_faces(values_by_face)
    src_A, tc_A, src_B, tc_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims,
        spec_A=spec,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims,
    )
    nb, ntb = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        dims,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )
    cfg = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(dims, dims),
            APPROX_MODE(ApproximationMode.Yes),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(False),
            MATH_OP(mathop=MathOperation.Tanh),
        ],
        runtimes=[TILE_COUNT(tc_A), NUM_BLOCKS(nb), NUM_TILES_IN_BLOCK(ntb)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tc_A,
            tile_count_B=tc_B,
            tile_count_res=tc_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )
    hw = torch.tensor(cfg.run().result, dtype=format_dict[formats.output_format])
    return src_A.to(torch.float32).numpy(), hw.to(torch.float32).numpy()


@pytest.mark.parametrize(
    "out_fmt,dest_acc",
    [
        (DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float32, DestAccumulation.Yes),
    ],
    ids=["bf16-out", "fp32-out"],
)
def test_tanh_specials(out_fmt, dest_acc):
    x, hw = _run({0: POS_SPECIALS, 1: NEG_SPECIALS, 2: ZEROS}, out_fmt, dest_acc)

    def classify(v):
        if np.isnan(v):
            return "NaN"
        if np.isinf(v):
            return "+inf" if v > 0 else "-inf"
        if v == 0.0:
            return "+0.0" if np.signbit(v) == 0 else "-0.0"
        return "%.9g" % v

    groups = {}
    for xv, hv in zip(x, hw):
        if np.isnan(xv):
            key = "NaN" + ("(-)" if np.signbit(xv) else "(+)")
        elif np.isinf(xv):
            key = "+inf" if xv > 0 else "-inf"
        elif xv == 0.0:
            key = "-0.0" if np.signbit(xv) else "+0.0"
        else:
            continue
        groups.setdefault(key, {}).setdefault(classify(hv), 0)
        groups[key][classify(hv)] += 1

    print("\n=== %s out, dest_acc=%s ===" % (out_fmt.name, dest_acc.name))
    print(
        "%-10s %8s   %-14s %-14s %s"
        % ("input", "count", "IEEE tanh", "recorded", "hardware returned")
    )

    # What the hardware actually does today, measured on a Wormhole n150 and
    # byte-for-byte identical on main's 3-entry table -- so these are properties of
    # the SFPLUT path, not of any particular coefficient table.
    #
    #   +-inf -> NaN on the fp32 path: segment 5 is the constant pair (A=0, B=1) and
    #            the hardware evaluates A*|x| + B, so an infinite input computes
    #            0 * inf + 1. IEEE tanh(+-inf) is +-1.
    #   -0.0  -> +0.0: the sign of zero is not carried through.
    #   On the bf16 path a NaN packs out as +inf and the infinities stay infinite --
    #   a NaN only survives to L1 on an fp32-end-to-end pipeline, so that arm says
    #   nothing about the kernel.
    #
    # Asserting the divergences on purpose: they predate this table, fixing them costs
    # instructions on every datum, and until that trade is made deliberately this test
    # is here to catch anyone changing them by accident.
    RECORDED = {
        DataFormat.Float32: {
            "+inf": "NaN",
            "-inf": "NaN",
            "NaN(+)": "NaN",
            "+0.0": "+0.0",
            "-0.0": "+0.0",
        },
        DataFormat.Float16_b: {
            "+inf": "+inf",
            "-inf": "-inf",
            "NaN(+)": "+inf",
            "+0.0": "+0.0",
            "-0.0": "+0.0",
        },
    }[out_fmt]
    IEEE = {
        "+inf": "1",
        "-inf": "-1",
        "NaN(+)": "NaN",
        "NaN(-)": "NaN",
        "+0.0": "+0.0",
        "-0.0": "-0.0",
    }

    failures = []
    for key in ("+inf", "-inf", "NaN(+)", "NaN(-)", "+0.0", "-0.0"):
        if key not in groups:
            continue
        got = groups[key]
        total = sum(got.values())
        shown = ", ".join("%s x%d" % (k, v) for k, v in sorted(got.items()))
        want = RECORDED.get(key)
        uniform = list(got) == [want] if want else False
        note = "OK" if uniform else "<-- CHANGED"
        if want is not None and not uniform:
            failures.append("%s: recorded %s, got %s" % (key, want, shown))
        print(
            "%-10s %8d   %-14s %-14s %s   %s"
            % (key, total, IEEE[key], want or "-", shown, note)
        )

    assert not failures, (
        "behaviour on the non-finite / signed-zero patterns changed:\n  "
        + "\n  ".join(failures)
        + "\nThese were identical on main. If the change is deliberate, update RECORDED."
    )
