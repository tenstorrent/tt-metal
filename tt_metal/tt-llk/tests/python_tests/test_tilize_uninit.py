# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Register-level regressions for tt-llk#1161 / tt-metal#47016 (Wormhole tilize z-dim).

Two complementary register-level checks, because plain/uncompressed unpack drives
its face iteration from the MOP loop count (not the descriptor z-dim), so an
end-to-end PCC test on plain ops cannot observe a wrong z-dim. We therefore read
the THCON_SECx_REG0_TileDescriptor z-dim back via get_tensix_state:

1. test_tilize_uninit_restores_z_dim (tt-llk#1161):
   #45179 made _llk_unpack_tilize_uninit_ restore the descriptor z-dim to the
   operand's num_faces (the default state set by _llk_unpack_init_) instead of a
   hardcoded 4. Run a tilize-only kernel (which ends on the uninit) and assert the
   restored SEC0 z-dim == num_faces. Guards against re-hardcoding the restore to 4.

2. test_bfp8_matmul_consumer_restores_z_dim (tt-metal#47016):
   The flip side. A num_faces=2 tilize leaves SEC0 z-dim=2 (correct for #1161). A
   following Bfp8_b (BFP-compressed) matmul must NOT inherit that stale z-dim: for
   compressed operands z-dim sizes the per-tile exponent / RowStart arrays
   (NumBlobs = BlobsPerXYPlane * ZDim * WDim), so a wrong z-dim corrupts the
   exponent decode (saturated logits in the galaxy lm_head). The fix makes
   _llk_unpack_AB_matmul_init_ program each operand's z-dim from its own face
   count. Run tilize(num_faces=2) -> Bfp8_b matmul and assert both SEC0 and SEC1
   z-dim == the matmul operand's face count (4), not the stale tilize value (2).
   A plain end-to-end PCC test never exercised a BFP matmul consumer, which is why
   the original regression test missed #47016.
"""

from dataclasses import asdict

import pytest
from fuser.fuser_config_parser import FuserConfigSchema
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.logger import logger
from ttexalens.tt_exalens_lib import get_tensix_state


@pytest.mark.parametrize(
    "test_name,num_faces",
    [("tilize_only_num_faces_2", 2)],
    ids=["num_faces_2"],
)
def test_tilize_uninit_restores_z_dim(test_name, num_faces, regenerate_cpp):
    if get_chip_architecture() != ChipArchitecture.WORMHOLE:
        pytest.skip("tt-llk#1161 fix is Wormhole-specific")

    config = FuserConfigSchema.load(test_name)
    config.global_config.regenerate_cpp = regenerate_cpp

    # Builds, runs the tilize-only kernel on device, and verifies the tilize
    # output against golden. The kernel's last unpacker call is the tilize
    # uninit, so the descriptor is left in its restored state.
    config.run_regular_test()

    state = asdict(get_tensix_state(config.TENSIX_LOCATION, device_id=0))
    z_dims = [d["z_dim"] for d in state["unpack_tile_descriptor"]]
    logger.info("unpack_tile_descriptor z_dims after tilize uninit: {}", z_dims)

    assert z_dims[0] == num_faces, (
        f"_llk_unpack_tilize_uninit_ left tile-descriptor z_dim={z_dims[0]}, "
        f"expected num_faces={num_faces}. A value of 4 indicates "
        f"the pre-fix hardcoded restore regressed."
    )


@pytest.mark.parametrize(
    "test_name,mm_num_faces",
    [("tilize_num_faces_2_then_bfp8_matmul", 4)],
    ids=["bfp8_matmul_after_num_faces_2_tilize"],
)
def test_bfp8_matmul_consumer_restores_z_dim(test_name, mm_num_faces, regenerate_cpp):
    """A BFP-compressed matmul after a num_faces!=4 tilize must own its z-dim.

    Regression for tt-metal#47016: a num_faces=2 tilize leaves SEC0 z-dim=2; the
    following Bfp8_b matmul's init must reprogram both SEC0 (SrcA) and SEC1 (SrcB)
    z-dim to the matmul operand's own face count so the BFP exponent arrays are
    sized correctly. A stale z-dim (2) here is exactly the bug that corrupted the
    lm_head logits. The end-to-end PCC of this pipeline is also checked by
    test_fused.py; this asserts the underlying register state directly.
    """
    if get_chip_architecture() != ChipArchitecture.WORMHOLE:
        pytest.skip("tt-metal#47016 fix is Wormhole-specific")

    config = FuserConfigSchema.load(test_name)
    config.global_config.regenerate_cpp = regenerate_cpp

    # Builds and runs tilize(num_faces=2) -> Bfp8_b matmul on device. The final
    # unpacker call is the matmul unpack, so the descriptor reflects what the
    # matmul init programmed.
    config.run_regular_test()

    state = asdict(get_tensix_state(config.TENSIX_LOCATION, device_id=0))
    z_dims = [d["z_dim"] for d in state["unpack_tile_descriptor"]]
    logger.info("unpack_tile_descriptor z_dims after bfp8 matmul: {}", z_dims)

    # SEC0 = SrcA, SEC1 = SrcB; both matmul operands are full 4-face Bfp8_b tiles.
    assert z_dims[0] == mm_num_faces and z_dims[1] == mm_num_faces, (
        f"_llk_unpack_AB_matmul_init_ left tile-descriptor z_dims={z_dims[:2]}, "
        f"expected [{mm_num_faces}, {mm_num_faces}]. A SEC0 z_dim of 2 means the "
        f"Bfp8_b matmul inherited the stale z-dim from the preceding num_faces=2 "
        f"tilize uninit (tt-metal#47016) instead of programming its own."
    )
