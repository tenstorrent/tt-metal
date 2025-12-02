# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from ..test_vae_wan2_1 import run_test_wan_decoder


@pytest.mark.parametrize(
    "mesh_device",
    [(4, 32)],
    indirect=True,
)
@pytest.mark.parametrize("num_links", [3], ids=["num_links_3"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("B, C, H, W"),
    [
        (1, 16, 60, 104),  # 480p, 10 frames
        (1, 16, 90, 160),  # 720p, 10 frames
    ],
    ids=[
        "480p",
        "720p",
    ],
)
@pytest.mark.parametrize("h_axis, w_axis", [[0, 1]], ids=["h0_w1"])
@pytest.mark.parametrize("T", [1, 10, 81], ids=["_1f", "10f", "81f"])
@pytest.mark.parametrize("mean, std", [(0, 1)])
@pytest.mark.parametrize("real_weights", [True], ids=["real_weights"])
@pytest.mark.parametrize("skip_check", [True], ids=["skip_check"])
def test_vae_wan2_1_exabox(
    mesh_device,
    num_links,
    device_params,
    B,
    C,
    T,
    H,
    W,
    mean,
    std,
    h_axis,
    w_axis,
    real_weights,
    skip_check,
    reset_seeds,
):
    run_test_wan_decoder(mesh_device, B, C, T, H, W, mean, std, h_axis, w_axis, num_links, real_weights, skip_check)
