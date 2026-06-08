# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Register-level regression for tt-llk#1161 (Wormhole _llk_unpack_tilize_uninit_).

The fix makes _llk_unpack_tilize_uninit_ restore the unpack tile-descriptor
z-dim to the operand's num_faces (the default state set by _llk_unpack_init_)
instead of a hardcoded 4. End-to-end PCC tests cannot catch this because plain
unpack drives its face iteration from the MOP loop count, not the descriptor
z-dim. So we assert the restored register value directly: run a tilize-only
kernel (which ends on the uninit) and read back THCON_SEC0_REG0_TileDescriptor
z-dim via get_tensix_state.
"""

from dataclasses import asdict

import pytest
from conftest import skip_for_coverage, skip_for_quasar
from fuser.fuser_config_parser import FuserConfigSchema
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.logger import logger
from ttexalens.tt_exalens_lib import get_tensix_state


@skip_for_quasar
@skip_for_coverage
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
