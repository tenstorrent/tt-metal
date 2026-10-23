# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Block-float pack of tiles whose faces are shorter than a full face, on Wormhole.

A block-float tile stores one exponent byte per 16-datum row ahead of the mantissas, in 16-byte chunks. A tile
whose faces are shorter than FACE_R_DIM rows (8x32, 4x32, 8x16, 4x16) carries at most 16 exponents, one chunk,
and the Wormhole packer joins its faces through one packer interface for that; every other tile packs one face
per packer instruction. The pack MOP's form therefore depends on the output format, and two things must hold:

- a one-face short tile (8x16, 4x16) packs with one instruction in every format: the joining sequence packs
  one more face's worth of datums after the tile (``test_pack_one_face_bfp_guard``, guard page after the tile);
- a format reconfig that moves a two-face short tile across the block-float line rebuilds the MOP
  (``test_pack_reconfig_partial_face_bfp``: init for the Dst format, reconfig to the block-float output, back).

Each kernel fills its guard pages with a sentinel before the packer is configured; the host checks the packed
tiles against their goldens and the guard pages against the sentinel. Wormhole only: the Blackhole pack MOP
packs one instruction per face in every format, so neither property exists there.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.data_format_inference import infer_data_formats
from helpers.device_io import read_from_device
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, format_dict, format_tile_sizes
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    IN_TILE_DIMS,
    NUM_FACES,
    NUM_GUARD_TILES,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.tile_constants import FACE_C_DIM, MAX_FACE_R_DIM, calculate_tile_size_bytes
from helpers.utils import passed_test

wormhole_pack_mop = pytest.mark.skipif(
    get_chip_architecture() != ChipArchitecture.WORMHOLE,
    reason="pins the Wormhole tile pack MOP, whose form depends on the output format; "
    "the Blackhole pack MOP packs one instruction per face in every format",
)

# Written over every guard page by the kernel's pack thread; repeated in the kernel sources.
GUARD_SENTINEL_BYTE = 0xC3
GUARD_PAGES = 1

INPUT_FORMAT = DataFormat.Float16_b
BLOCK_FLOAT_OUTPUTS = [
    InputOutputFormat(INPUT_FORMAT, DataFormat.Bfp8_b),
    InputOutputFormat(INPUT_FORMAT, DataFormat.Bfp4_b),
]
DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]


def _tile_geometry(tile_dims):
    """(face_r_dim, num_faces, tile_elements) of a tile of at most one face row."""
    tile_r, tile_c = tile_dims
    assert tile_r <= MAX_FACE_R_DIM, "tiles of one face row only"
    return tile_r, tile_c // FACE_C_DIM, tile_r * tile_c


def _tiles_from_stimuli(num_tiles, tile_elements):
    """The first ``num_tiles`` dense tiles of a fresh Float16_b stimulus and the matching B operand."""
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=INPUT_FORMAT,
        input_dimensions_A=[64, 32],
        stimuli_format_B=INPUT_FORMAT,
        input_dimensions_B=[64, 32],
    )
    count = num_tiles * tile_elements
    assert count <= src_A.numel()
    return src_A[:count].clone(), src_B[:count].clone()


def _page_bytes(data_format, tile_dims):
    return calculate_tile_size_bytes(data_format, list(tile_dims), format_tile_sizes)


def _assert_guard_pages_intact(stimuli, first_guard_page, tile_dims):
    """Every guard page after the packed output tiles still holds the sentinel."""
    page_bytes = stimuli.buf_res_tile_size
    for guard in range(GUARD_PAGES):
        address = stimuli.buf_res_addr + (first_guard_page + guard) * page_bytes
        raw = read_from_device(
            TestConfig.TENSIX_LOCATION, address, num_bytes=page_bytes
        )
        clobbered = [
            offset for offset, byte in enumerate(raw) if byte != GUARD_SENTINEL_BYTE
        ]
        assert not clobbered, (
            f"the pack of a {tile_dims[0]}x{tile_dims[1]} tile wrote {len(clobbered)} bytes into guard page "
            f"{guard} (first at byte {clobbered[0]} of the {page_bytes}-byte page after the tile)"
        )


@wormhole_pack_mop
@parametrize(
    formats=BLOCK_FLOAT_OUTPUTS,
    dest_acc=DEST_ACC_MODES,
    tile_dims=[
        (8, 16),  # face_r_dim=8,  num_faces=1: partial face, one face
        (4, 16),  # face_r_dim=4,  num_faces=1: partial face, one face
        (
            16,
            16,
        ),  # face_r_dim=16, num_faces=1: full face (control: one instruction on every revision)
    ],
)
def test_pack_one_face_bfp_guard(formats, dest_acc, tile_dims):
    """A one-face tile packs to a block-float output with one instruction: the guard page after it stays."""
    face_r_dim, num_faces, tile_elements = _tile_geometry(tile_dims)
    num_tiles = 1

    src_A, src_B = _tiles_from_stimuli(num_tiles, tile_elements)
    golden = src_A.to(format_dict[formats.output_format])

    configuration = TestConfig(
        "sources/pack_one_face_bfp_guard_test.cpp",
        formats,
        runtimes=[
            TILE_COUNT(num_tiles),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(face_r_dim),
            IN_TILE_DIMS(*tile_dims),
            NUM_GUARD_TILES(GUARD_PAGES),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            INPUT_FORMAT,
            src_B,
            INPUT_FORMAT,
            formats.output_format,
            tile_count_A=num_tiles,
            tile_count_B=num_tiles,
            tile_count_res=num_tiles + GUARD_PAGES,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=list(tile_dims),
            use_dense_tile_dimensions=True,
            operand_res_tile_size=_page_bytes(formats.output_format, tile_dims),
        ),
        dest_acc=dest_acc,
    )

    # The result decode covers the guard page too (it is part of the declared result region); only the tile counts.
    res_from_L1 = configuration.run().result[: num_tiles * tile_elements]
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(golden, res_tensor, formats.output_format)

    _assert_guard_pages_intact(configuration.variant_stimuli, num_tiles, tile_dims)


@wormhole_pack_mop
@parametrize(
    formats=BLOCK_FLOAT_OUTPUTS,
    dest_acc=DEST_ACC_MODES,
    tile_dims=[
        (
            8,
            32,
        ),  # face_r_dim=8,  num_faces=2: partial faces, MOP form depends on the format
        (
            4,
            32,
        ),  # face_r_dim=4,  num_faces=2: partial faces, MOP form depends on the format
        (
            16,
            32,
        ),  # face_r_dim=16, num_faces=2: full faces (control: no rebuild on any revision)
    ],
)
def test_pack_reconfig_partial_face_bfp(formats, dest_acc, tile_dims):
    """A format reconfig from the Dst format to a block-float output, and back, packs both outputs right."""
    face_r_dim, num_faces, tile_elements = _tile_geometry(tile_dims)
    num_tiles = 3  # tile 0 and tile 2 in the Dst format (buffer_C), tile 1 in the output format (buffer_Res)
    dst_format_tiles = [0, 2]
    output_format_tile = 1

    # The first and third pack use the Dst format the inference gives the packer source (Float16_b, or Float32
    # with FP32 accumulation): the matmul's partials.
    dst_format = infer_data_formats(
        input_format=formats.input_format,
        output_format=formats.output_format,
        is_fp32_dest_acc_en=dest_acc,
        unpacking_to_dest=False,
    ).pack_src

    src_A, src_B = _tiles_from_stimuli(num_tiles, tile_elements)
    tiles = src_A.view(num_tiles, tile_elements)
    golden_output = tiles[output_format_tile].to(format_dict[formats.output_format])
    golden_dst_format = tiles[dst_format_tiles].reshape(-1).to(format_dict[dst_format])

    configuration = TestConfig(
        "sources/pack_reconfig_partial_face_bfp_test.cpp",
        formats,
        runtimes=[
            TILE_COUNT(num_tiles),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(face_r_dim),
            IN_TILE_DIMS(*tile_dims),
            NUM_GUARD_TILES(GUARD_PAGES),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            INPUT_FORMAT,
            src_B,
            INPUT_FORMAT,
            formats.output_format,
            tile_count_A=num_tiles,
            tile_count_B=num_tiles,
            tile_count_res=1 + GUARD_PAGES,
            buffer_C=torch.zeros(
                len(dst_format_tiles) * tile_elements, dtype=format_dict[dst_format]
            ),
            stimuli_C_format=dst_format,
            tile_count_C=len(dst_format_tiles),
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=list(tile_dims),
            use_dense_tile_dimensions=True,
            operand_res_tile_size=_page_bytes(formats.output_format, tile_dims),
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result[:tile_elements]
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden_output, res_tensor, formats.output_format
    ), "block-float tile after the reconfig"

    dst_format_from_L1 = configuration.variant_stimuli.collect_buffer_c_results(
        TestConfig.TENSIX_LOCATION
    )
    dst_format_tensor = torch.tensor(dst_format_from_L1, dtype=format_dict[dst_format])
    assert passed_test(
        golden_dst_format, dst_format_tensor, dst_format
    ), "Dst-format tiles before and after"

    _assert_guard_pages_intact(configuration.variant_stimuli, 1, tile_dims)
