# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 8), id="8x8_grid")], indirect=True)
def test_visualize_mesh_device(mesh_device):
    ttnn.visualize_mesh_device(mesh_device)
