# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
import gc
import ttnn


@pytest.fixture(autouse=True)
def ensure_gc():
    gc.collect()


@pytest.fixture(autouse=True)
def ensure_devices(ensure_devices_tg):
    pass


@pytest.fixture
def device_params(request, galaxy_type):
    # Get param dict passed in from test parametrize (or default to empty dict)
    params = getattr(request, "param", {}).copy()

    if "fabric_config" in params and params["fabric_config"] == True:
        params["fabric_config"] = (
            ttnn.FabricConfig.FABRIC_1D_RING if galaxy_type == "6U" else ttnn.FabricConfig.FABRIC_1D
        )

    return params
