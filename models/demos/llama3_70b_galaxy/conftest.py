# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import gc
import ttnn


@pytest.fixture(autouse=True)
def ensure_gc():
    gc.collect()


@pytest.fixture
def device_params(request):
    # Get param dict passed in from test parametrize (or default to empty dict)
    params = getattr(request, "param", {}).copy()

    if "fabric_config" in params and params["fabric_config"] == True:
        params["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING

    return params
