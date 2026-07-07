# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Register-level restore test for `_llk_unpack_tilizeA_B_uninit_` (fused tilize-A / unpack-B).

This is the fused-path companion to `test_unpack_tilize_uninit_restore_tiny.py`
(single-operand tilize) and `test_unpack_canonical_baseline.py`. It closes the
same class of teardown gap as tt-llk#1161, but for the `tilizeA_B` path that
backs the tilize-fused reduce/matmul consumers.

Before this fix, `_llk_unpack_tilizeA_B_uninit_` restored `Tile_x_dim_cntx0` from
the hardcoded FACE_DIM_16x16 GPR (= 256 | (256 << 16)), which is correct only for
face_r_dim == 16. For tiny tiles (face_r_dim < 16) it overwrote the descriptor
with a 16-row geometry, leaking a wrong per-row datum count into the next operand.
The fix threads the operand's TensorShape so the restore matches
`canonical_unpA_tile_x_dim_cntx(face_r_dim)` — exactly the value
`configure_unpack_AB` programs.

The kernel (`unpack_tilizeAB_uninit_restore_test.cpp`) runs, unpack-thread-only:
hw_configure -> tilizeA_B_init -> tilizeA_B_uninit, then reads back
`Tile_x_dim_cntx0` (and, on Blackhole, the SrcA Y-stride) and LLK_ASSERTs each
equals the canonical baseline for the operand's face_r_dim. A broken restore (the
old FACE_DIM_16x16 value) fails the on-device assert for every tiny face_r_dim,
which surfaces here as a test failure. There is no data golden — the restored
registers are the deliverable.
"""

from conftest import skip_for_coverage
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats, parametrize
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import NUM_FACES, TEST_FACE_DIMS

pytestmark = skip_for_coverage


@parametrize(
    # Float formats the tilize unpacker can read as operand-A source. `same=True`
    # keeps the (unused for this config-only check) dst == src; the canonical SrcA
    # baseline keys off unpack_A_dst and spans the x-stride buckets Float16->2 and
    # Float32->4 either way.
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    # Tile-descriptor face count (tilizeA_B supports 1, 2, 4).
    num_faces=[4, 2, 1],
    # Tile_x_dim restore axis: 16 == regular (old FACE_DIM_16x16 was already
    # correct), < 16 == tiny (only the face_r_dim-aware restore is correct).
    face_r_dim=[16, 8, 4, 2, 1],
)
def test_unpack_tilizeAB_uninit_restore(
    formats,
    dest_acc,
    num_faces,
    face_r_dim,
):
    # All verification is on-device: the kernel LLK_ASSERTs the restored
    # Tile_x_dim_cntx0 (and BH Y-stride) against the canonical helper value. A
    # failed assert surfaces here as a test failure.
    TestConfig(
        "sources/unpack_tilizeAB_uninit_restore_test.cpp",
        formats,
        runtimes=[
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(face_r_dim=face_r_dim, face_c_dim=16),
        ],
        dest_acc=dest_acc,
    ).run()
