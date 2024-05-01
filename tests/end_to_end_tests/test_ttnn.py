# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import ttnn
import torch

import ttnn.operations.binary


@pytest.mark.eager_package_silicon
def test_ttnn_import(reset_seeds):
    with ttnn.manage_device(device_id=0) as device:
        pass


@pytest.mark.eager_package_silicon
def test_ttnn_add(reset_seeds):
    if os.environ["ARCH_NAME"] == "grayskull":
        pytest.skip("Hangs on grayskull, likely because of having copies of libraries")

    with ttnn.manage_device(device_id=0) as device:
        a_torch = torch.ones((5, 7))
        b_torch = torch.ones((1, 7))

        a = ttnn.from_torch(a_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        b = ttnn.from_torch(b_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        output = a + b
        output = ttnn.to_torch(output)
        print(output)
