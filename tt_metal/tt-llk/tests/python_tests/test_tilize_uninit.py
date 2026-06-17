# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Register-level regression for tt-llk#1161 (Wormhole _llk_unpack_tilize_uninit_).

_llk_unpack_tilize_uninit_ must NOT clobber the unpack tile-descriptor z-dim.
The tilize datapath never modifies that register, so the value programmed by the
operand's _llk_unpack_hw_configure_ (z-dim == num_faces) must survive the uninit
untouched. A pre-fix uninit that wrote a hardcoded 4 left a <4-face operand in
the wrong state (tt-llk#1161); a later variant that wrote the consumer's
num_faces corrupted a following BFP matmul (tenstorrent/tt-metal#47016). The fix
simply stops writing z-dim in uninit, so each downstream op sees the value its
own hw_configure/reconfig programmed.

End-to-end PCC tests cannot catch this because plain unpack drives its face
iteration from the MOP loop count, not the descriptor z-dim. So we assert the
register value directly: run a tilize-only kernel (which ends on the uninit) and
read back THCON_SEC0_REG0_TileDescriptor z-dim via get_tensix_state. For a
num_faces=2 operand it must still read 2 (hw_configure's value, left intact).
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
        f"expected num_faces={num_faces} (the value hw_configure programmed, which "
        f"uninit must leave untouched). A value of 4 means uninit is clobbering "
        f"z-dim back to a hardcoded full-tile count (tt-llk#1161). NOTE: this test "
        f"only guards #1161 - it cannot detect tt-metal#47016, where uninit's "
        f"num_faces equals this operand's, so the register reads the expected value "
        f"either way; #47016 is covered behaviourally by test_repro_47016.py."
    )
