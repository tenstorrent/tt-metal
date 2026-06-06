# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Test pack operation with various configurations.

Tests the LLK pack kernel with:
- Different data formats (Float16_b, Float16, Float32, Int32, Bfp8_b)
- Destination accumulation modes
- Variable tile dimensions
- ReLU activation
- Destination sync modes (SyncHalf for double-buffering, SyncFull for single-buffering)
"""

from dataclasses import dataclass

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.constraints import (
    get_valid_dest_accumulation_modes,
    get_valid_dest_indices,
)
from helpers.data_format_inference import infer_data_formats
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    FACES_PER_TILE,
    TILE_DIMENSIONS,
    PackGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    PackerReluType,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    RELU_CONFIG,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)
from helpers.utils import passed_test

# ---------------------------------------------------------------------------
# Config dataclass + sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True, repr=False)
class PackConfig:
    formats: InputOutputFormat
    dest_acc: DestAccumulation
    input_dimensions: tuple
    dest_sync: DestSync
    relu_type: PackerReluType
    dest_index: int

    def __repr__(self):
        f = self.formats
        return (
            f"{f.input_format.name}->{f.output_format.name}"
            f"-{self.dest_acc.name}-{self.input_dimensions[0]}x{self.input_dimensions[1]}"
            f"-{self.dest_sync.name}-{self.relu_type.name}-di{self.dest_index}"
        )


def _valid_relu_types(formats, dest_acc):
    all_relu_types = [
        PackerReluType.NoRelu,
        PackerReluType.ZeroRelu,
        PackerReluType.MinThresholdRelu,
        PackerReluType.MaxThresholdRelu,
    ]

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    try:
        data_formats = infer_data_formats(
            input_format=formats.input_format,
            output_format=formats.output_format,
            is_fp32_dest_acc_en=dest_acc,
            unpacking_to_dest=unpack_to_dest,
        )
    except ValueError:
        return []

    if (
        dest_acc == DestAccumulation.Yes
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        and not formats.input_format.is_integer()
    ):
        data_formats.pack_src = DataFormat.Float32

    if data_formats.pack_src.is_integer():
        return [
            rt
            for rt in all_relu_types
            if rt
            not in [PackerReluType.MinThresholdRelu, PackerReluType.MaxThresholdRelu]
        ]

    return all_relu_types


def _valid_dest_indices(dest_acc, dest_sync, formats, input_dimensions):
    indices = get_valid_dest_indices(dest_sync, dest_acc, formats, input_dimensions)

    tile_cnt_A = (input_dimensions[0] // TILE_DIMENSIONS[0]) * (
        input_dimensions[1] // TILE_DIMENSIONS[1]
    )

    _, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    valid = []
    for idx in indices:
        if idx == 0:
            valid.append(idx)
        else:
            adjusted = num_tiles_in_block - idx
            if adjusted > 0 and tile_cnt_A % adjusted == 0:
                valid.append(idx)

    return valid


_ALL_FORMATS = [
    f
    for f in input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.Bfp8_b,
        ]
    )
    if not (
        (f.input_format == DataFormat.Int32) ^ (f.output_format == DataFormat.Int32)
    )
]

_INPUT_DIMENSIONS = [[32, 32], [64, 64], [32, 64], [64, 32]]
_DEST_SYNCS = [DestSync.Half, DestSync.Full]


def _sweep_pack():
    """Pre-build all valid PackConfig combinations.

    Outer loops: template-affecting params (formats, dest_acc, input_dimensions, dest_sync)
    Inner loops: runtime-only params (relu_type, dest_index)
    """
    combos = []
    for fmt in _ALL_FORMATS:
        for da in get_valid_dest_accumulation_modes(fmt):
            for dims in _INPUT_DIMENSIONS:
                for ds in _DEST_SYNCS:
                    for rt in _valid_relu_types(fmt, da):
                        for di in _valid_dest_indices(da, ds, fmt, dims):
                            combos.append(
                                PackConfig(
                                    formats=fmt,
                                    dest_acc=da,
                                    input_dimensions=tuple(dims),
                                    dest_sync=ds,
                                    relu_type=rt,
                                    dest_index=di,
                                )
                            )
    return combos


ALL_PACK_CONFIGS = _sweep_pack()


# ---------------------------------------------------------------------------
# Tolerance helper
# ---------------------------------------------------------------------------


def is_relu_threshold_tolerance_issue(
    golden_tensor,
    result_tensor,
    relu_config,
    intermediate_format,
    rtol=0.01,
    atol=0.01,
):
    relu_type = PackGolden.get_relu_type(relu_config)
    threshold = PackGolden.get_relu_threshold(relu_config, intermediate_format)

    if relu_type not in [
        PackerReluType.MinThresholdRelu,
        PackerReluType.MaxThresholdRelu,
    ]:
        return False

    mismatches = ~torch.isclose(golden_tensor, result_tensor, rtol=rtol, atol=atol)

    if not mismatches.any():
        return False

    golden_near_threshold = torch.isclose(
        golden_tensor[mismatches],
        torch.full_like(golden_tensor[mismatches], threshold),
        rtol=rtol,
        atol=atol,
    )
    result_near_threshold = torch.isclose(
        result_tensor[mismatches],
        torch.full_like(result_tensor[mismatches], threshold),
        rtol=rtol,
        atol=atol,
    )

    acceptable = False
    if relu_type == PackerReluType.MinThresholdRelu:
        golden_is_zero = golden_tensor[mismatches] == 0.0
        result_is_zero = result_tensor[mismatches] == 0.0
        acceptable = (golden_is_zero & result_near_threshold) | (
            result_is_zero & golden_near_threshold
        )
    else:
        acceptable = golden_near_threshold & result_near_threshold

    return acceptable.all().item()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", ALL_PACK_CONFIGS)
def test_pack(config: PackConfig):
    formats = config.formats
    dest_acc = config.dest_acc
    input_dimensions = list(config.input_dimensions)
    dest_sync = config.dest_sync
    relu_type = config.relu_type
    dest_index = config.dest_index

    tile_cnt_A = (input_dimensions[0] // TILE_DIMENSIONS[0]) * (
        input_dimensions[1] // TILE_DIMENSIONS[1]
    )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    if dest_index != 0:
        num_tiles_in_block = num_tiles_in_block - dest_index
        num_blocks = tile_cnt_A // num_tiles_in_block

    # Construct StimuliConfig with layout info but no tensor data yet.
    # This gives TestConfig the struct layout it needs for compilation.
    stimuli = StimuliConfig(
        None,
        formats.input_format,
        None,
        formats.input_format,
        formats.output_format,
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_A,
        tile_count_res=tile_cnt_A,
    )

    configuration = TestConfig(
        "sources/pack_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            TILIZE(),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            DEST_INDEX(dest_index),
            RELU_CONFIG(0),
            NUM_FACES(num_faces=FACES_PER_TILE),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=stimuli,
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )

    configuration.prepare()
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    # Generate stimuli + golden (only reached in consumer / default mode)
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(PackGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.output_format,
        input_dimensions=input_dimensions,
    )

    data_formats = infer_data_formats(
        input_format=formats.input_format,
        output_format=formats.output_format,
        is_fp32_dest_acc_en=dest_acc,
        unpacking_to_dest=unpack_to_dest,
    )

    if (
        dest_acc == DestAccumulation.Yes
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        and not formats.input_format.is_integer()
    ):
        data_formats.pack_src = DataFormat.Float32

    tensor_average = (
        torch.mean(golden_tensor).item()
        if not formats.output_format.is_integer()
        else 0.0
    )

    relu_config = PackGolden.generate_relu_config(
        relu_type,
        relu_threshold=tensor_average,
        intermediate_format=data_formats.pack_src,
    )

    golden_tensor = PackGolden.apply_relu(
        golden_tensor,
        relu_config,
        data_formats.pack_src,
    )

    # Attach real tensor data and update relu_config runtime before execution
    stimuli.set_buffers(src_A, src_B)
    configuration.runtimes[2] = RELU_CONFIG(relu_config)
    configuration.generate_runtime_args_struct()

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    test_passed = passed_test(
        golden_tensor, res_tensor, formats.output_format, print_errors=False
    )

    if (
        not test_passed
        and relu_type
        in [
            PackerReluType.MinThresholdRelu,
            PackerReluType.MaxThresholdRelu,
        ]
        and is_relu_threshold_tolerance_issue(
            golden_tensor,
            res_tensor,
            relu_config,
            data_formats.pack_src,
        )
    ):
        test_passed = True

    assert test_passed
