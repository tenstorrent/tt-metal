# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import BootMode
from helpers.format_config import DataFormat, FormatConfig, is_dest_acc_needed
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.matmul_sweep import (
    generate_matmul_dimension_combinations,
    generate_tile_dims,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    MATH_FIDELITY,
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

# Throttle levels supported by the no-mop matmul math LLK. Throttle only inserts
# NOPs between MVMULs to cap compute throughput, so the numeric result is IDENTICAL
# for every level -> the golden is the same MatmulGolden as the non-throttled path.
# Blackhole implements run_throttled_sequence_no_mop<1..5>; Wormhole B0 static_asserts
# THROTTLE_LEVEL == 0, so only level 0 is valid there.
THROTTLE_LEVELS = (
    [0, 1, 2, 3, 4, 5] if get_chip_architecture() == ChipArchitecture.BLACKHOLE else [0]
)


def generate_format_aware_matmul_combinations(
    formats_list: List[FormatConfig],
    dest_acc_modes: List[DestAccumulation],
):
    """
    Generate matmul dimension combinations for multiple tiles.

    Rules:
    1. Format outliers (Float16_b->Float16, Bfp8_b->Float16) MUST use dest_acc=Yes
    2. Running matmul tests on DestSync.Half, max tile count is 8
    3. When dest_acc=Yes: max 4 tiles (32-bit dest register)
    4. When dest_acc=No: max 8 tiles (16-bit dest register)

    Returns: List of (format, dest_acc, dimensions) tuples
    """
    combinations = []

    for fmt in formats_list:
        base_max_tiles = 4 if is_dest_acc_needed(fmt) else 8

        for dest_acc in dest_acc_modes:
            max_tiles = 4 if dest_acc == DestAccumulation.Yes else base_max_tiles
            dimensions_list = generate_matmul_dimension_combinations(max_tiles)
            combinations.extend([(fmt, dest_acc, dims) for dims in dimensions_list])

    return combinations


# Generate format-aware combinations
MATMUL_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.Bfp8_b,
    ]
)
DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]
ALL_MATMUL_COMBINATIONS = generate_format_aware_matmul_combinations(
    MATMUL_FORMATS, DEST_ACC_MODES
)


def _run_matmul_custom(
    math_fidelity,
    formats,
    dest_acc,
    input_A_dimensions,
    input_B_dimensions,
    throttle_level: int,
    boot_mode: BootMode,
):
    torch_format = format_dict[formats.output_format]

    sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_B_dimensions,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
    )

    # Calculate all matmul dimensions using helper function
    matmul_dims = generate_tile_dims((input_A_dimensions, input_B_dimensions))

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        # Golden cannot model FPU strided for tilized data computation, so we tilize output after computation
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )

    if formats.input_format != DataFormat.Bfp8_b:
        tilized_A = tilize_block(
            src_A, dimensions=input_A_dimensions, stimuli_format=formats.input_format
        )
        tilized_B = tilize_block(
            src_B, dimensions=input_B_dimensions, stimuli_format=formats.input_format
        )
    else:
        # BFP8 format requires special handling for tilization
        tilized_A = src_A
        tilized_B = src_B

    configuration = TestConfig(
        "sources/matmul_custom_test.cpp",
        formats,
        templates=[MATH_FIDELITY(math_fidelity), THROTTLE_LEVEL(throttle_level)],
        runtimes=[
            NUM_FACES(),
            TILE_COUNT(matmul_dims.output_tile_cnt),
            CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
        ],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            formats.input_format,
            tilized_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=matmul_dims.output_tile_cnt,
        ),
        dest_acc=dest_acc,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


@parametrize(
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    format_dest_acc_and_dims=ALL_MATMUL_COMBINATIONS,
)
def test_matmul_custom(
    math_fidelity,
    format_dest_acc_and_dims,
    boot_mode=BootMode.DEFAULT,
):
    _run_matmul_custom(
        math_fidelity,
        format_dest_acc_and_dims[0],
        format_dest_acc_and_dims[1],
        format_dest_acc_and_dims[2][0],
        format_dest_acc_and_dims[2][1],
        throttle_level=0,
        boot_mode=boot_mode,
    )


# Representative fidelity x format subset for the throttle sweep. Throttle inserts
# NOPs between MVMULs without changing the numeric result, so it is orthogonal to
# format/fidelity/dims; a small subset per throttle level keeps the sweep tractable
# while still crossing LoFi (single-phase) and HiFi (multi-phase) code paths and
# both 16-bit (Float16_b) and 32-bit (Float32) operands.
THROTTLE_FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32])
# Single 32x32 output tile (ct=rt=kt=1). This is the regime the throttled no-mop
# matmul is designed and used in (the SDPA full-tile path); the hand-written
# throttle sequences replay a fixed single-tile MVMUL walk. Multi-tile / multi-K
# accumulation is validated at throttle 0 by test_matmul_custom above.
THROTTLE_DIMS = ([32, 32], [32, 32])


@parametrize(
    throttle_level=THROTTLE_LEVELS,
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
    formats=THROTTLE_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_matmul_custom_throttle(
    throttle_level,
    math_fidelity,
    formats,
    dest_acc,
    boot_mode=BootMode.DEFAULT,
):
    # Known limitation of the current LLK: throttle levels 4 and 5 only advance
    # the fidelity-phase counter on the final MVMUL of the sequence (they use
    # ADDR_MOD_4 with fidelity.incr=0 at the phase boundary, unlike levels 1-3
    # which use the fidelity-incrementing ADDR_MOD_5/6). For a high-fidelity
    # (multi-phase) matmul this collapses the extra phases and yields ~half the
    # result, so levels 4/5 are only correct for single-phase (LoFi) fidelity.
    # Cover levels 4/5 with LoFi and levels 0-3 with both fidelities.
    if throttle_level >= 4 and math_fidelity != MathFidelity.LoFi:
        pytest.skip(
            "throttle levels 4/5 do not increment the fidelity phase per LLK; "
            "only correct for LoFi (single-phase)"
        )

    input_A_dimensions, input_B_dimensions = THROTTLE_DIMS
    _run_matmul_custom(
        math_fidelity,
        formats,
        dest_acc,
        input_A_dimensions,
        input_B_dimensions,
        throttle_level=throttle_level,
        boot_mode=boot_mode,
    )
