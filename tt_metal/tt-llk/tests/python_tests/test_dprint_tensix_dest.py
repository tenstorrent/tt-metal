# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    Tilize,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)


@pytest.fixture(scope="module", autouse=True)
def _force_device_print_enabled():
    prev = TestConfig.DEVICE_PRINT_ENABLED
    TestConfig.DEVICE_PRINT_ENABLED = True
    yield
    TestConfig.DEVICE_PRINT_ENABLED = prev


# bf16 is the smoke test: under DestAccumulation.Yes the helper picks the
# Float32 row reader, and bf16 -> fp32 widening is bit-exact.
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
)
def test_dprint_tensix_dest(formats):
    if get_chip_architecture() == ChipArchitecture.QUASAR:
        pytest.skip("dprint_tensix_dest_reg has no Quasar build (no ckernel_debug.h)")

    formats = formats[0]
    input_dimensions = [32, 32]
    num_faces = 4
    dest_acc = DestAccumulation.Yes

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        (32, 32),
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/dprint_tensix_dest_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILIZE(Tilize.No),
        ],
        runtimes=[
            DEST_INDEX(0),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
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
            num_faces=num_faces,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    outcome = configuration.run()

    # L1 -> DEST -> L1 round-trip through the real LLK datacopy pipeline.
    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(outcome.result, dtype=torch_format)
    assert torch.equal(
        src_A.to(torch_format), res_tensor
    ), "L1 -> DEST -> L1 round-trip diverged from stimulus"

    # Decode the dp_typed_array_t records emitted by dprint_tensix_dest_reg:
    # 64 logical rows × 16 Float32 elements under fp32 dest accumulation.
    full = "".join(outcome.device_print_lines)
    assert "Tile ID = 0" in full, "dprint_tensix_dest_reg banner missing"

    expected = src_A.to(torch.float32).flatten().tolist()
    printed: list[float] = []
    for line in outcome.device_print_lines:
        # Each typed-array row prints as "0.0 1.0 ... 15.0 " after the [...] prefix.
        body = line.split("] ", 1)[-1].strip()
        toks = body.split()
        if len(toks) != 16:
            continue
        try:
            printed.extend(float(t) for t in toks)
        except ValueError:
            continue

    assert len(printed) >= len(expected), (
        f"Decoded {len(printed)} dest floats from typed-array records, "
        f"expected at least {len(expected)}"
    )
    # bf16 -> fp32 widening is bit-exact, so the first len(expected) decoded
    # values must equal the stimulus exactly.
    assert (
        printed[: len(expected)] == expected
    ), "Decoded DEST floats from dprint_tensix_dest_reg do not match the bf16-widened stimulus"
