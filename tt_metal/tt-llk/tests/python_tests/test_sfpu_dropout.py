# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Dropout SFPU test with a deterministic outcome, before and after an eltwise binary init.

calculate_dropout keeps an element where the PRNG draw is above the probability and zeroes it
otherwise, then scales what it keeps. Probability 0 keeps every element (the output is the input
times the scale, exact in bf16 for a scale of 1.0 or 2.0) and probability INT_MAX drops every
element, so the outcome does not depend on the draw and the test needs no model of the PRNG.

The binary_init_before variant runs the eltwise binary init between the datacopy and the dropout
body, in the same DEST section, as a kernel that fuses dropout_tile after add_tiles does. On
Blackhole that init programs address-modifier slot 3 to a DEST step of 8 rows; a dropout body that
addressed DEST through that slot read and wrote the wrong rows of the tile (127 of 1024 elements
came out wrong for an identity dropout) and only passed when a datacopy init had run last. The
body addresses DEST through the SFPU slot now, and this variant is the regression test.
"""

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import ELEMENTS_PER_TILE
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import SFPU_DROPOUT_PROBE, TILE_COUNT

FORMATS = input_output_formats([DataFormat.Float16_b], same=True)

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


def _run(formats, dest_acc, binary_init_before, probability, scale_bits):
    torch.manual_seed(0)
    torch_format = format_dict[formats.input_format]
    # Non-zero everywhere and away from the bf16 overflow, so a dropped, misplaced or unscaled
    # element is visible and the doubling is exact.
    src_A = torch.empty(ELEMENTS_PER_TILE, dtype=torch.float32).uniform_(1.0, 2.0).to(torch_format)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch_format)
    golden = _golden(src_A, probability, scale_bits)

    configuration = TestConfig(
        "sources/sfpu_dropout_test.cpp",
        formats,
        templates=[
            SFPU_DROPOUT_PROBE(
                dropout_binary_init_before=binary_init_before,
                dropout_probability=probability,
                dropout_scale_bits=scale_bits,
            ),
        ],
        runtimes=[TILE_COUNT(1)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    res = torch.tensor(configuration.run().result, dtype=format_dict[formats.output_format]).to(torch.float32)
    return res, golden


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    binary_init_before=[False, True],
    case=CASES,
)
def test_sfpu_dropout(formats, dest_acc, binary_init_before, case):
    probability, scale_bits = case
    res, golden = _run(formats, dest_acc, binary_init_before, probability, scale_bits)
    mismatch = (res != golden).nonzero().flatten().tolist()
    assert not mismatch, (
        f"dropout with probability {probability:#x}, scale bits {scale_bits:#x}, "
        f"{'after' if binary_init_before else 'without'} an eltwise binary init in the section: "
        f"{len(mismatch)} of {ELEMENTS_PER_TILE} elements differ from the golden, first at flat offsets "
        f"{mismatch[:16]}"
    )
