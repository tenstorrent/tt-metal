# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sum every input tile into one scalar with the mul_reduce_scalar reduce tail on Quasar."""

import pytest
import torch
from helpers.constraints import get_valid_math_fidelities
from helpers.data_format_inference import dest_acc_for
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_variant_parameters import (
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.tile_constants import MAX_TILE_ELEMENTS
from helpers.tile_shape import construct_tile_shape
from helpers.utils import tolerances

FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
    # Float16 keeps its 10 bit mantissa in SrcA, so its HiFi pools run more than one phase
    InputOutputFormat(DataFormat.Float16, DataFormat.Float16),
    InputOutputFormat(DataFormat.Float16, DataFormat.Float32),
]

# Each shape puts its faces in DEST differently. Faces of 1, 2 and 4 rows take the SrcB staging path.
ALL_TILE_DIMENSIONS = [[32, 32], [16, 32], [16, 16], [8, 32], [4, 32], [2, 32], [1, 32]]
FP32_DEST_TILE_DIMENSIONS = [[32, 32], [8, 32], [1, 32]]
FLOAT16_TILE_DIMENSIONS = [[32, 32], [8, 32]]

# Poison tiles after the input catch a read past the last tile
POISON_TILES = 2
POISON_VALUE = 100.0


def _tile_dimensions_for_format(formats):
    if formats.input_format == DataFormat.Float16:
        return FLOAT16_TILE_DIMENSIONS
    if dest_acc_for(formats.output_format) == DestAccumulation.Yes:
        return FP32_DEST_TILE_DIMENSIONS
    return ALL_TILE_DIMENSIONS


def _num_tiles_for_format(formats):
    # One tile, and a full DEST half section
    return (
        [1, 4]
        if dest_acc_for(formats.output_format) == DestAccumulation.Yes
        else [1, 8]
    )


@pytest.mark.quasar
@parametrize(
    formats=FORMATS,
    math_fidelity=lambda formats: get_valid_math_fidelities(formats),
    num_tiles=_num_tiles_for_format,
    tile_dimensions=_tile_dimensions_for_format,
    run_types=[[PerfRunType.L1_TO_L1]],
    # Two passes, so the second one uses the other half of DEST
    loop_factor=[2],
)
def test_sum_reduce_scalar_quasar(
    formats,
    math_fidelity,
    num_tiles,
    tile_dimensions,
    run_types,
    loop_factor,
    *,
    is_perf=False,
    perf_report=None,
):
    tile_shape = construct_tile_shape(tile_dimensions)
    elements_per_tile = tile_shape.total_tile_size()
    dest_acc = dest_acc_for(formats.output_format)
    input_dimensions = [num_tiles * tile_dimensions[0], tile_dimensions[1]]

    src_A, tile_cnt_A, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
        spec_A=StimuliSpec.uniform(low=0.0, high=1.0),
    )
    src_B = torch.zeros_like(src_A)
    golden_scalar = float(src_A.to(torch.float32).sum().item())

    poison = torch.full(
        (POISON_TILES * MAX_TILE_ELEMENTS,), POISON_VALUE, dtype=src_A.dtype
    )

    test_config_kwargs = {
        "test_name": "sources/quasar/sum_reduce_scalar_quasar_test.cpp",
        "formats": formats,
        "templates": [
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(DestSync.Half),
        ],
        "runtimes": [
            TILE_COUNT(num_tiles),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            TEST_FACE_DIMS(face_r_dim=tile_shape.face_r_dim),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            torch.cat([src_A, poison]),
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A + len(poison) // elements_per_tile,
            tile_count_B=tile_cnt_A,
            tile_count_res=1,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        "dest_acc": dest_acc,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result
    assert (
        len(res_from_L1) == elements_per_tile
    ), f"Expected one {elements_per_tile}-element output tile, got {len(res_from_L1)}"

    device_scalar = float(res_from_L1[0])
    tol = tolerances[formats.output_format]
    rtol, atol = tol.rtol, tol.atol
    if math_fidelity != MathFidelity.LoFi:
        # The bound the Metal test uses above LoFi
        rtol, atol = 0.01, 0.01
    assert abs(device_scalar - golden_scalar) <= max(rtol * abs(golden_scalar), atol), (
        f"sum_reduce_scalar mismatch: device={device_scalar} golden={golden_scalar} "
        f"(num_tiles={num_tiles}, tile={tile_dimensions}, fidelity={math_fidelity.name})"
    )
