# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Datacopy whose math/pack dest handshake is re-initialised between every block.

The pack thread calls ``_llk_pack_dest_init_`` directly behind the previous block's last PACR, with no
reconfig in between, and ``REINIT_DELAY`` RISC nops move that re-init relative to the pack in flight.
The re-init rewrites the packer's dest offset and address counters; if those writes were not ordered
behind the packer by the LLK itself, the previous block's last tile would be packed with the new
offset/counters and the copy would differ from its input.
"""

from dataclasses import dataclass

import torch
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    DataCopyGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
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
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TemplateParameter,
    generate_input_dim,
)
from helpers.utils import passed_test


@dataclass
class REINIT_DELAY(TemplateParameter):
    """RISC nops issued by the pack thread before each block's ``_llk_pack_dest_init_``."""

    delay: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t REINIT_DELAY = {self.delay}u;"


REINIT_FORMATS = [
    fmt
    for fmt in input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32, DataFormat.Bfp8_b]
    )
    if fmt.input_format == fmt.output_format
]


@parametrize(
    formats=REINIT_FORMATS,
    dest_acc=get_valid_dest_accumulation_modes,
    reinit_delay=[0, 2, 8, 32, 128],
    input_dimensions=[[128, 128], [64, 512]],
)
def test_pack_dest_reinit(formats, dest_acc, reinit_delay, input_dimensions):
    num_faces = 4

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(DataCopyGolden)
    golden_tensor = generate_golden(
        src_A, formats.output_format, num_faces, input_dimensions
    )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )
    assert num_blocks >= 2, "the test needs at least one re-init between blocks"

    configuration = TestConfig(
        test_name="sources/pack_dest_reinit_test.cpp",
        formats=formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            REINIT_DELAY(reinit_delay),
        ],
        runtimes=[
            DEST_INDEX(0),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            LOOP_FACTOR(1),
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
        unpack_to_dest=unpack_to_dest,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(golden_tensor, res_tensor, formats.output_format)
