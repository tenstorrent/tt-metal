# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TEST_FACE_DIMS,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test
from loguru import logger

# Float32 whose bit pattern is exactly 0xDEADBEEF (~ -6.2598534e18).
DEADBEEF_F32 = (
    torch.tensor([0xDEADBEEF - (1 << 32)], dtype=torch.int32).view(torch.float32).item()
)

# Run 2 fill: exact in fp16 (2.5 = 0x4100)
RUN2_FILL = 2.5


def _run_unpack_to_dest(
    input_format: DataFormat,
    output_format: DataFormat,
    fill_value: float,
    disable_format_inference: bool = True,
) -> torch.Tensor:
    """One L1 -> UNP_DEST -> Dest(32-bit) -> pack -> L1 pass with constant stimuli."""
    formats = InputOutputFormat(
        input_format=input_format,
        output_format=output_format,
    )
    input_dimensions = [32, 32]  # single 32x32 tile

    spec = StimuliSpec.constant(fill_value)
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec,
        spec_B=spec,
    )

    configuration = TestConfig(
        "sources/quasar/unpack_to_dest_simple_quasar_test.cpp",
        formats,
        templates=[],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(),  # 16x16 faces
            NUM_FACES_R_DIM(),  # 2 face rows per tile
            NUM_FACES_C_DIM(),  # 2 face columns per tile
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
        ),
        unpack_to_dest=True,
        dest_acc=DestAccumulation.Yes,
        disable_format_inference=disable_format_inference,
    )

    res_from_L1 = configuration.run().result
    return torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])


@pytest.mark.quasar
def test_unpack_to_dest_simple_quasar():
    # Run 1: Float32 DEADBEEF pattern primes every 32-bit Dest word.
    res_deadbeef = _run_unpack_to_dest(
        DataFormat.Float32, DataFormat.Float32, DEADBEEF_F32
    )
    golden_deadbeef = torch.full_like(res_deadbeef, DEADBEEF_F32)
    assert passed_test(
        golden_deadbeef, res_deadbeef, DataFormat.Float32
    ), "Float32 DEADBEEF datacopy failed"

    # Run 2: Float16 (Unapck src and dst format) -> Dst 32bit container -> Pack Float32
    res_fill = _run_unpack_to_dest(DataFormat.Float16, DataFormat.Float32, RUN2_FILL)

    words, counts = torch.unique(res_fill.view(torch.int32), return_counts=True)
    for word, count in zip(words.tolist(), counts.tolist()):
        logger.info(f"Dest word read back: 0x{word & 0xFFFFFFFF:08X} x{count}")

    golden_fill = torch.full_like(res_fill, RUN2_FILL)
    assert passed_test(
        golden_fill, res_fill, DataFormat.Float32
    ), "fp16 unpack-to-dest fill did not come back as 2.5 — the logged hex words show what the packer actually read"
