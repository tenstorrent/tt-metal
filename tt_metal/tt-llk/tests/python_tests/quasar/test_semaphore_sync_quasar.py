# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.data_format_inference import unpacks_to_dest
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    DataCopyGolden,
    ReduceGapoolGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestSync,
    MathFidelity,
    MathOperation,
    ReduceDimension,
    ReducePool,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
    quasar_mx_smoke,
    runtime,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test, result_tensor

# (DEST sections, tiles per section): every section is one semaphore handoff. Four tiles per section
# fits the smallest section (32-bit DEST, SyncHalf).
SEMAPHORE_SECTION_SHAPES = [(4, 1), (1, 4), (4, 4), (8, 1), (8, 2)]


# Two flows under the semaphore scheme. 16-bit and MX inputs run a reduce through SrcA / SrcB with the math-pack
# semaphore pair. 32-bit inputs (dest_acc=Yes only: the source registers are 19-bit wide) unpack straight into DEST and
# run the full unpack-to-dest protocol, UNPACK_PACK / UNPACK_MATH / MATH_PACK, with the kernel calling the
# synchronization primitives itself around the sync-free placer, math-forward and pack calls; output = input.
@pytest.mark.quasar
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
        ],
    )
    + quasar_mx_smoke(DataFormat.MxFp4, DataFormat.Float16_b)
    + input_output_formats([DataFormat.Float32, DataFormat.Int32], same=True),
    dest_acc=get_valid_dest_accumulation_modes,
    dest_sync=[DestSync.Full, DestSync.Half],
    # Unpack-to-dest only: with no sections, only the semaphore setup runs, and every thread must still finish.
    zero_tiles=lambda formats: (
        [False, True] if formats.input_format.is_32_bit() else [False]
    ),
    # With zero tiles every section shape runs the same empty kernel, so one shape covers it.
    sections=runtime(
        lambda zero_tiles: [(1, 1)] if zero_tiles else SEMAPHORE_SECTION_SHAPES
    ),
)
def test_semaphore_sync_quasar(
    formats,
    dest_acc,
    dest_sync,
    zero_tiles,
    sections,
):
    num_sections, tiles_per_section = sections

    pool_type = ReducePool.Sum
    reduce_dim = ReduceDimension.Row
    mathop = MathOperation.ReduceRow
    math_fidelity = MathFidelity.LoFi

    input_dimensions = [32 * num_sections * tiles_per_section, 32]

    src_A, tile_cnt, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # result in srcA should be multiplied by 1 for pool_type = sum
    src_B = torch.full((1024,), 1)

    unpack_to_dest = unpacks_to_dest(formats.input_format, dest_acc)
    if unpack_to_dest:
        # Datacopy through DEST: the unpacker writes the tiles, math only forwards the semaphores, pack reads them back.
        generate_golden = get_golden_generator(DataCopyGolden)
        golden_tensor = generate_golden(
            src_A,
            formats.output_format,
            num_faces=4,
            input_dimensions=input_dimensions,
            input_format=formats.input_format,
            face_r_dim=16,
            tile_shape=construct_tile_shape((32, 32)),
        )
    else:
        generate_golden = get_golden_generator(ReduceGapoolGolden)
        golden_tensor = generate_golden(
            src_A,
            src_B,
            formats.output_format,
            reduce_dim,
            math_fidelity,
            tile_cnt,
            input_format=formats.input_format,
            dest_acc=dest_acc,
        )

    configuration = TestConfig(
        "sources/quasar/semaphore_sync_quasar_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop, pool_type=pool_type),
            UNPACKER_ENGINE_SEL(),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            TILE_COUNT(0 if zero_tiles else tile_cnt),
            NUM_TILES_IN_BLOCK(tiles_per_section),
            TEST_FACE_DIMS(),
            NUM_FACES(),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=1,
            tile_count_res=tile_cnt,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    if zero_tiles:
        return

    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
