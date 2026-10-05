# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-preservation test for the tilize teardown.

Configures the SrcA descriptor with a ``num_faces`` that differs from the tilize
operand's, runs ``init`` + ``uninit`` with no geometry reconfig between, and reads
the descriptor word back on device. That difference is the point: the other
teardown tests use the same ``num_faces`` for both, so they pass whether or not
uninit rewrites the descriptor.
"""

import pytest
from conftest import skip_for_coverage
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats, parametrize
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import NUM_FACES, TEST_FACE_DIMS

pytestmark = skip_for_coverage


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    # Tilize operand num_faces. The kernel picks a *different* pre-tilize
    # baseline (4 -> 2, otherwise 4), so every case crosses a num_faces boundary.
    num_faces=[4, 2, 1],
    # 16 = normal tile, <16 = tiny tile (also covers the Tile_x_dim restore branch).
    face_r_dim=[16, 8, 2],
)
def test_unpack_tilize_uninit_descriptor(
    formats,
    dest_acc,
    num_faces,
    face_r_dim,
):
    # BH unpack_tilize does not support num_faces=1 (LLK asserts num_faces in {2, 4}).
    # WH supports num_faces=1. Tracked in https://github.com/tenstorrent/tt-metal/issues/50707.
    if num_faces == 1 and get_chip_architecture() == ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "BH unpack_tilize does not support num_faces=1; see https://github.com/tenstorrent/tt-metal/issues/50707"
        )

    TestConfig(
        "sources/unpack_tilize_uninit_descriptor_test.cpp",
        formats,
        runtimes=[
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(face_r_dim=face_r_dim, face_c_dim=16),
        ],
        dest_acc=dest_acc,
    ).run()
