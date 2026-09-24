# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Regression test: _topk_finalize_hi16_index_tile_ must return the u16 indices it was given.

The sweep loads each [0|u16] index word from 32-bit DEST and stores it back with the
low->high packer mode, so a UInt16 pack of the tile reads the indices. Wormhole encodes
the SFPLOAD/SFPSTORE address mode in two bits, so an out-of-range ADDR_MOD spilled into
instr_mod0 and this sweep packed garbage there. Int32 input lands in DEST bit-exactly, so
the expected output is simply the input.
"""

import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig

# Quasar has no topk SFPU implementation.
pytestmark = [skip_for_quasar]

ELEMENTS_PER_TILE = 1024

# Every u16 value is a valid index word; exercise the whole range.
INDEX_STIMULI = StimuliSpec.uniform(low=0, high=65535)


def test_topk_finalize_uint16():
    torch.manual_seed(0)
    formats = InputOutputFormat(DataFormat.Int32, DataFormat.UInt16)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[32, 32],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[32, 32],
        spec_A=INDEX_STIMULI,
        spec_B=INDEX_STIMULI,
    )

    configuration = TestConfig(
        "sources/topk_finalize_uint16_test.cpp",
        formats,
        templates=[],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
        ),
        # Int32 unpacks straight into 32-bit DEST.
        unpack_to_dest=True,
        dest_acc=DestAccumulation.Yes,
    )

    res_from_L1 = configuration.run().result[:ELEMENTS_PER_TILE]
    torch_format = format_dict[formats.output_format]

    device = torch.tensor(res_from_L1, dtype=torch_format).flatten()
    golden = src_A.flatten().to(torch_format)[:ELEMENTS_PER_TILE]

    mismatches = int((device != golden).sum().item())
    assert mismatches == 0, (
        f"{mismatches} of {ELEMENTS_PER_TILE} indices came back wrong after the finalize sweep; "
        f"first few: expected {golden[:8].tolist()}, got {device[:8].tolist()}"
    )
