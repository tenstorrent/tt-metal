# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dropout with a deterministic outcome: probability 0 keeps every element, INT_MAX drops every one,
with and without the eltwise binary init that add_tiles before dropout leaves, and on Blackhole also
as the one 32-row call per tile that dropout_tile issues there."""

import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import ELEMENTS_PER_TILE
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import SFPU_DROPOUT_PROBE, TILE_COUNT

FORMATS = input_output_formats([DataFormat.Float16_b], same=True)
NUM_TILES = 1
ONE_CALL_FORMS = (
    [False, True] if get_chip_architecture() == ChipArchitecture.BLACKHOLE else [False]
)

PROBABILITY_KEEP_ALL = 0
PROBABILITY_DROP_ALL = 0x7FFFFFFF
SCALE_ONE = 0x3F800000
SCALE_TWO = 0x40000000

# (probability, scale as fp32 bits)
CASES = [
    (PROBABILITY_KEEP_ALL, SCALE_ONE),
    (PROBABILITY_KEEP_ALL, SCALE_TWO),
    (PROBABILITY_DROP_ALL, SCALE_ONE),
]


def _golden(src, probability, scale_bits):
    if probability == PROBABILITY_DROP_ALL:
        return torch.zeros_like(src, dtype=torch.float32)
    scale = torch.tensor([scale_bits], dtype=torch.int32).view(torch.float32)
    return (src.to(torch.float32) * scale).to(torch.bfloat16).to(torch.float32)


def _run(formats, dest_acc, binary_init_before, one_call, probability, scale_bits):
    torch.manual_seed(0)
    torch_format = format_dict[formats.input_format]
    # Non-zero values in [1, 2): a dropped, misplaced or unscaled element shows, the doubling is exact.
    src_A = (
        torch.empty(NUM_TILES * ELEMENTS_PER_TILE, dtype=torch.float32)
        .uniform_(1.0, 2.0)
        .to(torch_format)
    )
    src_B = torch.zeros(NUM_TILES * ELEMENTS_PER_TILE, dtype=torch_format)
    golden = _golden(src_A, probability, scale_bits)

    configuration = TestConfig(
        "sources/sfpu_dropout_test.cpp",
        formats,
        templates=[
            SFPU_DROPOUT_PROBE(
                dropout_binary_init_before=binary_init_before,
                dropout_one_call=one_call,
                dropout_probability=probability,
                dropout_scale_bits=scale_bits,
            ),
        ],
        runtimes=[TILE_COUNT(NUM_TILES)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=NUM_TILES,
            tile_count_B=NUM_TILES,
            tile_count_res=NUM_TILES,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    res = torch.tensor(
        configuration.run().result, dtype=format_dict[formats.output_format]
    ).to(torch.float32)
    return res, golden


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    binary_init_before=[False, True],
    one_call=ONE_CALL_FORMS,
    case=CASES,
)
def test_sfpu_dropout(formats, dest_acc, binary_init_before, one_call, case):
    probability, scale_bits = case
    res, golden = _run(
        formats, dest_acc, binary_init_before, one_call, probability, scale_bits
    )
    mismatch = (res != golden).nonzero().flatten().tolist()
    assert not mismatch, (
        f"dropout with probability {probability:#x}, scale bits {scale_bits:#x}, "
        f"{'after' if binary_init_before else 'without'} an eltwise binary init in the section, "
        f"{'one call' if one_call else 'four calls'} per tile: "
        f"{len(mismatch)} of {res.numel()} elements differ from the golden, first at flat offsets "
        f"{mismatch[:16]}"
    )
