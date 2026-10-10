# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Mask with the data and the mask tile at any two DEST indices of the acquired half; Float16_b
drives calculate_mask and calculate_mask_posinf, Int32 calculate_int_mask, on Blackhole also in the
one 32-row call per tile that the Int32 mask_tile issues there."""

import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, BinarySFPUGolden
from helpers.llk_params import DestAccumulation, MathOperation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import SFPU_MASK_PLACEMENT, TILE_COUNT

FLOAT_FORMATS = input_output_formats([DataFormat.Float16_b], same=True)
INT_FORMATS = input_output_formats([DataFormat.Int32], same=True)

# (data tile, mask tile) placements; the first pair is the one the in-tree callers pass.
PLACEMENTS_16BIT_DEST = [(0, 1), (1, 0), (0, 2), (0, 7), (7, 0), (3, 5), (6, 2)]
PLACEMENTS_32BIT_DEST = [(0, 1), (1, 0), (0, 3), (3, 0), (2, 1)]
ONE_CALL_FORMS = (
    [False, True] if get_chip_architecture() == ChipArchitecture.BLACKHOLE else [False]
)
# Non-zero Int32 masks, some negative and some with a zero low half that a 16-bit load would read as zero.
INT_MASK_VALUES = [1, 1 << 16, 0x7FFFFFFF, -1, -(1 << 16)]


def _placements(dest_acc):
    if dest_acc == DestAccumulation.Yes:
        return PLACEMENTS_32BIT_DEST
    return PLACEMENTS_16BIT_DEST


def _stimuli(torch_format):
    """Non-zero data (a ramp of 1..8) and a mask with an exact zero in about every third element."""
    torch.manual_seed(0)
    position = torch.arange(ELEMENTS_PER_TILE)
    data = (position % 8 + 1).to(torch_format)
    if torch_format.is_floating_point:
        mask = torch.where(position % 3 == 0, 0, 1).to(torch_format)
        # Different magnitudes on the kept side, so a passthrough of the wrong tile cannot match.
        data = (
            data.to(torch.float32) * 0.5
            + torch.randint(0, 4, (ELEMENTS_PER_TILE,)).to(torch.float32)
        ).to(torch_format)
        mask = (
            mask.to(torch.float32)
            * torch.randint(1, 5, (ELEMENTS_PER_TILE,)).to(torch.float32)
        ).to(torch_format)
    else:
        values = torch.tensor(INT_MASK_VALUES, dtype=torch.int32)
        mask = torch.where(
            position % 3 == 0, 0, values[(position // 3) % len(INT_MASK_VALUES)]
        ).to(torch_format)
    return data, mask


def _run(formats, dest_acc, data_index, mask_index, posinf, one_call=False):
    torch_format = format_dict[formats.input_format]
    data, mask = _stimuli(torch_format)

    if posinf:
        operation = MathOperation.SfpuMaskPosinf
    elif torch_format.is_floating_point:
        operation = MathOperation.SfpuMask
    else:
        operation = MathOperation.SfpuIntMask
    golden_op = BinarySFPUGolden().ops[operation]
    golden = torch.tensor(
        [float(golden_op(d, m)) for d, m in zip(data, mask)], dtype=torch.float32
    )

    configuration = TestConfig(
        "sources/sfpu_mask_test.cpp",
        formats,
        templates=[
            SFPU_MASK_PLACEMENT(
                mask_data_dst_index=data_index,
                mask_mask_dst_index=mask_index,
                mask_posinf=posinf,
                mask_one_call=one_call,
            ),
        ],
        runtimes=[TILE_COUNT(1)],
        variant_stimuli=StimuliConfig(
            data,
            formats.input_format,
            mask,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
        compile_time_formats=True,
    )

    res = torch.tensor(
        configuration.run().result, dtype=format_dict[formats.output_format]
    ).to(torch.float32)
    return res, golden


def _check(res, golden, data_index, mask_index):
    mismatch = (res != golden).nonzero().flatten().tolist()
    assert not mismatch, (
        f"mask with the data in DEST tile {data_index} and the mask in DEST tile {mask_index}: "
        f"{len(mismatch)} of {ELEMENTS_PER_TILE} elements differ from the golden, first at flat offsets "
        f"{mismatch[:16]}"
    )


@parametrize(
    formats=FLOAT_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    placement=lambda dest_acc: _placements(dest_acc),
)
def test_sfpu_mask(formats, dest_acc, placement):
    data_index, mask_index = placement
    res, golden = _run(formats, dest_acc, data_index, mask_index, posinf=False)
    _check(res, golden, data_index, mask_index)


@parametrize(
    formats=FLOAT_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    placement=lambda dest_acc: _placements(dest_acc),
)
def test_sfpu_mask_posinf(formats, dest_acc, placement):
    data_index, mask_index = placement
    res, golden = _run(formats, dest_acc, data_index, mask_index, posinf=True)
    _check(res, golden, data_index, mask_index)


@parametrize(
    formats=INT_FORMATS,
    placement=PLACEMENTS_32BIT_DEST,
    one_call=ONE_CALL_FORMS,
)
def test_sfpu_mask_int(formats, placement, one_call):
    data_index, mask_index = placement
    # Int32 runs with the 32-bit DEST only.
    res, golden = _run(
        formats,
        DestAccumulation.Yes,
        data_index,
        mask_index,
        posinf=False,
        one_call=one_call,
    )
    _check(res, golden, data_index, mask_index)
