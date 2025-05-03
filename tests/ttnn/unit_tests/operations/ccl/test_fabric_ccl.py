# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.utility_functions import comp_pcc, skip_for_blackhole, run_for_wormhole_b0


@skip_for_blackhole()
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
def test_fabric_sanity(mesh_device):
    pass
