# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tiny-tile (face_r_dim < 16) cross-op restore test for `_llk_unpack_tilize_uninit_`.

This is the Phase 2 companion to ``test_unpack_tilize_uninit_restore.py``. Phase 1
covers the ``num_faces`` (tile-descriptor Z-dim) restore for full 16-row faces.
Phase 2 covers the orthogonal axis the PR actually changed on Wormhole: the
``Tile_x_dim_cntx0`` restore, which is computed as
``canonical_unpA_tile_x_dim_cntx(face_r_dim)`` and therefore only differs from the
old hardcoded ``16 | (16 << 16)`` value when ``face_r_dim < 16``.

No existing tilize test runs ``unpack_tilize`` with ``face_r_dim < 16``, so this
also serves as the first end-to-end exercise of the tiny-tile tilize path.

Flow (same C++ source as Phase 1, ``face_r_dim`` is threaded through every call):
    1. ``unpack_tilize`` of a tiny [face_r_dim, 32] operand A (num_faces=2).
    2. ``_llk_unpack_tilize_uninit_(dst, num_faces, face_r_dim)`` (restore under test).
    3. a plain ``_llk_unpack_A_`` datacopy of the tilized tile, with NO data-format
       reconfig in between.

Because there is no reconfig, the uninit is the only thing that restores
``Tile_x_dim_cntx0`` back to the tiny-tile baseline programmed by
``configure_unpack_AB``. If uninit restored the old ``16|16`` value instead of
``canonical_unpA_tile_x_dim_cntx(face_r_dim)``, the second datacopy reads the
operand with the wrong per-row datum count and the result diverges from the
tilized golden.
"""

import torch
from conftest import skip_for_blackhole
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import NUM_FACES, TEST_FACE_DIMS
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test

# Tiny tiles are always 2 horizontal faces ([face_r_dim, 32] => f0 | f1).
TINY_NUM_FACES = 2


# Blackhole's `_llk_unpack_tilize_uninit_` does honor `face_r_dim` (via its
# `TensorShape` param), but the BH test wrapper `_llk_unpack_tilize_uninit_wrapper_`
# is 2-arg (`dst, num_faces`) and drops `face_r_dim` (WH's wrapper is 3-arg), so the
# tiny-tile (face_r_dim < 16) restore branch under test here cannot be expressed on BH
# without extending that wrapper. The num_faces axis (face_r_dim=16) still runs on BH
# via test_unpack_tilize_uninit_restore.py.
@skip_for_blackhole
@parametrize(
    # Same input/output format for both ops so the second op needs NO data-format
    # reconfig — isolating the uninit restore as the only state reset.
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    # All sub-16 face row counts. face_r_dim=16 is the Phase 1 baseline and is
    # covered by test_unpack_tilize_uninit_restore.py.
    face_r_dim=[8, 4, 2, 1],
)
def test_unpack_tilize_uninit_restore_tiny(
    formats,
    dest_acc,
    face_r_dim,
):
    torch_format = format_dict[formats.output_format]
    # Tiny tile: a single [face_r_dim, 32] tile (2 horizontal faces).
    input_dimensions = [face_r_dim, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        face_r_dim=face_r_dim,
        num_faces=TINY_NUM_FACES,
    )

    # Golden: the run-1 datacopy is an identity copy of the run-0 tilized tile,
    # so a correct restore yields exactly the tilized tiny operand A. tile_dimensions
    # drives the horizontal 2-face (Nx32) tilize layout for face_r_dim < 16.
    golden_tensor = tilize(
        src_A,
        stimuli_format=formats.output_format,
        tile_dimensions=input_dimensions,
    ).to(torch_format)

    L1_to_L1_iterations = 2
    configuration = TestConfig(
        "sources/unpack_tilize_uninit_restore_test.cpp",
        formats,
        templates=[],
        runtimes=[
            NUM_FACES(TINY_NUM_FACES),
            TEST_FACE_DIMS(face_r_dim=face_r_dim, face_c_dim=16),
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
            num_faces=TINY_NUM_FACES,
            face_r_dim=face_r_dim,
            # Tiny tile is dense (num_faces * face_r_dim * 16 == face_r_dim * 32
            # row-major elements), so do NOT pad to a full 32x32 tile.
        ),
        dest_acc=dest_acc,
        L1_to_L1_iterations=L1_to_L1_iterations,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        L1_to_L1_iterations=L1_to_L1_iterations,
    ), "Assert against golden failed"
