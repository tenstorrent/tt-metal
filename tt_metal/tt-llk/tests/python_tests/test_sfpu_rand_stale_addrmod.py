# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Regression test: rand stores the whole tile even when another op's init (topk) last programmed
the dest address-mode slot it stores under; with a 16-bit Dest only the top half of each face's rows
carry draws (32-bit stores over a 16-bit Dest), so the checks cover those rows."""

import struct

import pytest
import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    VectorMode,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    RAND_RANGE,
    RAND_SEED,
    VECTOR_MODE,
)
from helpers.tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    FACE_C_DIM,
    MAX_FACE_ELEMENTS,
    MAX_FACE_R_DIM,
    MAX_TILE_ELEMENTS,
)

pytestmark = [skip_for_quasar]

# Draw interval [from, from + scale] as fp32 bits, and the seed; passed to the kernel as templates.
RAND_FROM_BITS = 0x00000000  # 0.0f
RAND_SCALE_BITS = 0x3F800000  # 1.0f
PRNG_SEED = 7919
DRAW_LOW = struct.unpack("<f", struct.pack("<I", RAND_FROM_BITS))[0]
DRAW_HIGH = DRAW_LOW + struct.unpack("<f", struct.pack("<I", RAND_SCALE_BITS))[0]
SENTINEL = 2.0  # exact in every format; outside the draw interval
assert not (
    DRAW_LOW <= SENTINEL <= DRAW_HIGH
), "the sentinel must lie outside the draw interval"
# Rows of each face that hold a draw with a 16-bit Dest: a 32-bit store at row r lands its high half
# at 16-bit row 2 * (r & ~7) + (r & 7), so the top half of each face.
DRAW_ROWS_PER_FACE_16B_DEST = MAX_FACE_R_DIM // 2
# At least 512 draws from a 31-bit grid; even in bfloat16 far more than a quarter are distinct.
MIN_DISTINCT = 128
SURVIVORS_SHOWN = 8

FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True)


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_rand_stale_addrmod(formats, dest_acc):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip(
            "16-bit Dest cannot hold Float32 draws exactly; the Float16_b rows cover 16-bit Dest"
        )

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM],
        spec_A=StimuliSpec.constant(SENTINEL),
        spec_B=StimuliSpec.constant(SENTINEL),
    )

    configuration = TestConfig(
        "sources/sfpu_rand_stale_addrmod_test.cpp",
        formats,
        templates=[
            APPROX_MODE(ApproximationMode.No),
            VECTOR_MODE(VectorMode.RC),
            RAND_RANGE(rand_from_bits=RAND_FROM_BITS, rand_scale_bits=RAND_SCALE_BITS),
            RAND_SEED(rand_seed=PRNG_SEED),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result[:MAX_TILE_ELEMENTS]
    tile = (
        torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
        .flatten()
        .to(torch.float32)
    )

    # Elements that can hold a draw: all of them with a 32-bit Dest; the top rows of each face with
    # a 16-bit Dest (face-major tile layout: element = face * MAX_FACE_ELEMENTS + row * FACE_C_DIM + column).
    rows = torch.arange(MAX_TILE_ELEMENTS) % MAX_FACE_ELEMENTS // FACE_C_DIM
    draw_rows = (
        torch.ones(MAX_TILE_ELEMENTS, dtype=torch.bool)
        if dest_acc == DestAccumulation.Yes
        else rows < DRAW_ROWS_PER_FACE_16B_DEST
    )
    draw_elements = draw_rows.nonzero().flatten()
    draws = tile[draw_rows]

    sentinel_count = int((draws == SENTINEL).sum().item())
    survivors = draw_elements[
        (draws == SENTINEL).nonzero().flatten()[:SURVIVORS_SHOWN]
    ].tolist()
    assert sentinel_count == 0, (
        f"{sentinel_count} of {draws.numel()} draw elements still hold the sentinel after rand: rand "
        f"stored under the address mode the topk init left (first survivors at elements {survivors})"
    )
    in_range = (draws >= DRAW_LOW) & (draws <= DRAW_HIGH)  # False for NaN as well
    out_of_range = draws[~in_range]
    assert bool(
        in_range.all()
    ), f"draws outside [{DRAW_LOW}, {DRAW_HIGH}] or non-finite: {out_of_range[:SURVIVORS_SHOWN].tolist()}"
    distinct = int(torch.unique(draws).numel())
    assert (
        distinct >= MIN_DISTINCT
    ), f"only {distinct} distinct values among the draws; not a rand tile"
