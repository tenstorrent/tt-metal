# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Non-YAML counterpart of the four-case BF16 Add1 fuser example."""

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathOperation,
    PerfRunType,
    UnpackerEngine,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    PERF_RUN_TYPE,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test


@pytest.mark.quasar
def test_sfpu_add1_bf16_quasar():
    """Check x + 1 on all four faces for random, negative, zero and positive inputs."""
    torch.manual_seed(42)
    data_format = DataFormat.Float16_b
    formats = InputOutputFormat(data_format, data_format)
    random_input, _, _, _ = generate_stimuli(
        stimuli_format_A=data_format,
        input_dimensions_A=[32, 32],
        stimuli_format_B=data_format,
        input_dimensions_B=[32, 32],
    )
    cases = [
        ("random", random_input),
        *[(str(value), torch.full_like(random_input, value)) for value in (-2, 0, 2)],
    ]

    for case_name, src in cases:
        configuration = TestConfig(
            "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp",
            formats,
            templates=[
                MATH_OP(mathop=MathOperation.Add1),
                APPROX_MODE(ApproximationMode.No),
                IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
                DATA_COPY_TYPE(DataCopyType.A2D),
                UNPACKER_ENGINE_SEL(UnpackerEngine.UnpA),
                DEST_SYNC(DestSync.Half),
                TYPECAST_FORMATS(),
                PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
            ],
            runtimes=[
                TILE_COUNT(1),
                NUM_FACES(4),
                TEST_FACE_DIMS(),
                DEST_INDEX(0),
                LOOP_FACTOR(1),
            ],
            variant_stimuli=StimuliConfig(
                src,
                data_format,
                torch.zeros_like(src),
                data_format,
                data_format,
                tile_count_A=1,
                tile_count_B=1,
                tile_count_res=1,
                num_faces=4,
            ),
            unpack_to_dest=False,
            dest_acc=DestAccumulation.No,
        )
        result = torch.tensor(configuration.run().result, dtype=torch.bfloat16)
        golden = (src.to(torch.float32) + 1.0).flatten()
        assert result.numel() == golden.numel(), case_name
        assert passed_test(
            golden,
            result.flatten(),
            data_format,
            custom_atol=0.05,
            custom_rtol=0.05,
            custom_pcc_threshold=0.99,
        ), f"Add1 failed for {case_name}"
