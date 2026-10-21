# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# T0 probe for the dest-semaphore design (not for commit). See sources/quasar/sem_probe_quasar_test.cpp.

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_SYNC,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test
from test_eltwise_unary_sfpu_quasar import prepare_square_inputs


@pytest.mark.quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    input_dimensions=[[32, 32], [64, 64]],
)
def test_sem_probe_quasar(formats, input_dimensions):
    torch.manual_seed(42)
    dest_acc = DestAccumulation.No
    src_A, tile_cnt, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    src_A = prepare_square_inputs(
        src_A, src_B, formats.input_format, formats.output_format
    )
    square = get_golden_generator(UnarySFPUGolden)(
        MathOperation.Square,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )
    torch_format = format_dict[formats.output_format]
    golden = torch.cat([square.to(torch_format), torch.zeros(len(square), dtype=torch_format)])

    configuration = TestConfig(
        "sources/quasar/sem_probe_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Square),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpA),
            DEST_SYNC(DestSync.Full),
        ],
        runtimes=[TILE_COUNT(tile_cnt), NUM_FACES(4), TEST_FACE_DIMS()],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=2 * tile_cnt,
        ),
        dest_acc=dest_acc,
    )
    res = torch.tensor(configuration.run().result, dtype=torch_format)
    assert len(res) == len(golden)
    n = len(square)
    assert passed_test(golden[:n], res[:n], formats.output_format), "pass 1 (sem chain) wrong"
    assert torch.count_nonzero(res[n:]) == 0, "pass 2: ZEROACC not visible before the post"
