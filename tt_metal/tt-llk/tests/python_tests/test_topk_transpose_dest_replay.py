# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Regression test for the topk replay-buffer contract on Wormhole.

topk records anywhere in the math thread's replay buffer, including the slots transpose_dest
recorded at init. The kernel transposes tile 0, runs a topk local sort, re-runs the
transpose_dest init and transposes tile 1; both tiles must match the golden bit for bit.

Omitting the re-init trips the LLK_ASSERT in _llk_math_transpose_dest_. That variant is not
swept: an LLK assert leaves the core halted, and conftest reports it as a failure.
"""

import torch
from conftest import skip_for_blackhole, skip_for_quasar
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    DataCopyGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, Transpose
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import MATH_TRANSPOSE_FACES, UNPACK_TRANS_FACES
from helpers.utils import result_tensor

# The ownership tag and its asserts are Wormhole-only.
pytestmark = [skip_for_blackhole, skip_for_quasar]

# Two tiles: one transposed before topk, one after.
INPUT_DIMENSIONS = [32, 64]


@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    math_transpose_faces=[Transpose.Yes, Transpose.No],
)
def test_topk_transpose_dest_replay(formats, math_transpose_faces):
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=INPUT_DIMENSIONS,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=INPUT_DIMENSIONS,
    )

    if TestConfig.BUILD_MODE != BuildMode.PRODUCE:
        datacopy_tensor = get_golden_generator(DataCopyGolden)(
            src_A,
            formats.output_format,
            num_faces=4,
            input_dimensions=INPUT_DIMENSIONS,
        )
        t_matrix = get_golden_generator(TransposeGolden)
        golden_tensor = t_matrix.transpose_faces_multi_tile(
            datacopy_tensor,
            formats.output_format,
            num_tiles=tile_cnt_A,
            tilize=False,
            input_dimensions=INPUT_DIMENSIONS,
        )
        golden_tensor = t_matrix.transpose_within_faces_multi_tile(
            golden_tensor,
            formats.output_format,
            num_tiles=tile_cnt_A,
            untilize=False,
            input_dimensions=INPUT_DIMENSIONS,
        )
    else:
        golden_tensor = []

    configuration = TestConfig(
        "sources/topk_transpose_dest_replay_test.cpp",
        formats,
        templates=[MATH_TRANSPOSE_FACES(math_transpose_faces)],
        runtimes=[
            # Without math face transposition, the unpacker transposes the faces.
            UNPACK_TRANS_FACES(
                Transpose.Yes if math_transpose_faces == Transpose.No else Transpose.No
            ),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )

    res_from_L1 = configuration.run().result
    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    assert torch.equal(
        res_tensor, golden_tensor
    ), "transpose after topk does not match golden: the transpose_dest re-init did not restore its replay recording"
