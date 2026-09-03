# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-ownership check for `_llk_unpack_tilize_uninit_` (tt-llk#1161).

`test_unpack_tilize_uninit_restore.py` tears down with the SAME ``num_faces``
the SrcA baseline was configured with, so the old (descriptor-writing) and new
(descriptor-preserving) Wormhole teardown both leave the same value behind — it
cannot distinguish them. This test can.

The kernel (`unpack_tilize_uninit_descriptor_test.cpp`) establishes a pre-tilize
tile-descriptor Z-dim that DIFFERS from the tilize operand's ``num_faces``, runs
tilize init + uninit with no geometry reconfig in between, reads the descriptor
word back, and LLK_ASSERTs the per-arch contract on-device:

* Wormhole: the descriptor word is bit-identical across tilize+uninit. Tilize
  neither writes nor mutates it, so teardown must leave it alone. The removed
  teardown write stamped ``z_dim = tilize num_faces``, which is exactly the
  cross-operand corruption of tt-metal#45179 / #47016 — this assert fires on
  that code.
* Blackhole: tilize init does write the descriptor, so teardown must
  re-establish ``z_dim = tilize num_faces``; ``y_dim`` stays untouched.
* Wormhole only: ``Tile_x_dim_cntx0`` is tilize's to restore, so it must come back
  to the canonical ``face_r_dim``-derived value. Deliberately not asserted on
  Blackhole: ``_llk_unpack_tilize_uninit_wrapper_`` hardcodes ``MAX_FACE_R_DIM``
  there instead of threading this test's ``face_r_dim`` through, so the expected
  value would not correspond to the operand under test. The Blackhole descriptor
  assertions above are what this test exists for.

All verification is on-device (LLK_ASSERT); there is no stimuli or golden — the
register state is the deliverable, same shape as
`test_unpack_canonical_baseline.py`.
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
