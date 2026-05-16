# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import BinarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import InputOutputFormat
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    SFPU_INT_OP,
    SFPU_TILE_INDICES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test

_SRC0_IDX, _SRC1_IDX, _DST_IDX = 0, 1, 0


def _run_sfpu_binary_quasar(
    data_format,
    dest_acc,
    src0_idx,
    src1_idx,
    dst_idx,
    mathop,
    sfpu_int_op="",
    clamp_inputs=None,
):
    """Shared driver for all binary SFPU tests on Quasar.

    Args:
        sfpu_int_op: passed to SFPU_INT_OP template parameter ("MUL", "GT",
                     "LT", "LE", "GE", or "" for the default add path).
        clamp_inputs: if set, clamp src_A to (-clamp_inputs, clamp_inputs) to
                      avoid overflow.
    """
    num_tiles_needed = max(src0_idx, src1_idx, dst_idx) + 1
    formats = InputOutputFormat(input_format=data_format, output_format=data_format)

    input_dimensions = [num_tiles_needed * 32, 32]

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=data_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=data_format,
        negative_values=True,
        input_dimensions_B=input_dimensions,
        sfpu=False,
        full_2sc_int_range=True,
    )

    if clamp_inputs is not None:
        src_A = torch.clamp(src_A, -clamp_inputs, clamp_inputs)
        src_B = torch.clamp(src_B, -clamp_inputs, clamp_inputs)

    num_faces = 4

    elements_per_tile = 1024  # 4 faces * 16 rows * 16 cols
    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_full = generate_golden(
        mathop,
        src_A,
        src0_idx,
        src1_idx,
        dst_idx,
        32,  # num_iterations: 32 rows = 1 full tile
        input_dimensions,
        data_format,
    ).flatten()
    dst_start = dst_idx * elements_per_tile
    golden_tensor = golden_full[dst_start : dst_start + elements_per_tile]

    tile_count_res = 1

    templates = [
        MATH_OP(mathop=mathop),
        IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
        UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
        DEST_SYNC(),
    ]
    if sfpu_int_op:
        templates.insert(1, SFPU_INT_OP(sfpu_int_op))

    configuration = TestConfig(
        "sources/quasar/sfpu_binary_quasar_test.cpp",
        formats,
        templates=templates,
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            SFPU_TILE_INDICES(src0_idx, src1_idx, dst_idx),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            data_format,
            src_B,
            data_format,
            data_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_count_res,
            num_faces=num_faces,
            twos_complement=data_format.is_integer(),
        ),
        unpack_to_dest=True,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[data_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, data_format
    ), "Assert against golden failed"


# ---------------------------------------------------------------------------
# ADD (Int32 + Float16_b)
# ---------------------------------------------------------------------------
@pytest.mark.quasar
@pytest.mark.parametrize(
    "data_format, dest_acc",
    [
        (DataFormat.Int32, DestAccumulation.Yes),
        # (DataFormat.Float16_b, DestAccumulation.Yes), // TODO pgardner: Add back through ckernel_sfpu_add.h
        # (DataFormat.Float16_b, DestAccumulation.No),
    ],
)
def test_sfpu_binary_add_quasar(data_format, dest_acc):
    """Test binary SFPU ADD on Quasar architecture."""
    _run_sfpu_binary_quasar(
        data_format,
        dest_acc,
        _SRC0_IDX,
        _SRC1_IDX,
        _DST_IDX,
        mathop=MathOperation.SfpuElwadd,
    )


# ---------------------------------------------------------------------------
# MUL_INT (Int32 only — Float16_b requires bf16 SFPU mul kernel from PR2)
# ---------------------------------------------------------------------------
@pytest.mark.quasar
@pytest.mark.parametrize(
    "data_format, dest_acc",
    [
        (DataFormat.Int32, DestAccumulation.Yes),
        # (DataFormat.Float16_b, DestAccumulation.Yes), // TODO pgardner: Add back through ckernel_sfpu_add.h THIS MAY HAVE A BUG??
        # (DataFormat.Float16_b, DestAccumulation.No),
    ],
)
def test_sfpu_binary_mul_quasar(data_format, dest_acc):
    """Test binary SFPU MUL_INT on Quasar architecture."""
    _run_sfpu_binary_quasar(
        data_format,
        dest_acc,
        _SRC0_IDX,
        _SRC1_IDX,
        _DST_IDX,
        mathop=MathOperation.SfpuElwmulInt,
        sfpu_int_op="MUL",
        clamp_inputs=1000,
    )


# ---------------------------------------------------------------------------
# Integer comparisons (GT, LT, LE, GE) — parametrized over comp_op
# ---------------------------------------------------------------------------
_COMP_OPS = [
    ("GT", MathOperation.SfpuGtInt),
    ("LT", MathOperation.SfpuLtInt),
    ("LE", MathOperation.SfpuLeInt),
    ("GE", MathOperation.SfpuGeInt),
]


@pytest.mark.quasar
@pytest.mark.parametrize("comp_op, mathop", _COMP_OPS, ids=[op for op, _ in _COMP_OPS])
@pytest.mark.parametrize(
    "data_format, dest_acc",
    [
        (DataFormat.Int32, DestAccumulation.Yes),
    ],
)
def test_sfpu_binary_comp_int_quasar(comp_op, mathop, data_format, dest_acc):
    """Test binary SFPU integer comparison (GT/LT/LE/GE) on Quasar."""
    _run_sfpu_binary_quasar(
        data_format,
        dest_acc,
        _SRC0_IDX,
        _SRC1_IDX,
        _DST_IDX,
        mathop=mathop,
        sfpu_int_op=comp_op,
    )
