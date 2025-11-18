# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

TILE_WIDTH = 32


def test_manual_seed(device):
    torch.manual_seed(0)

    ttnn.manual_seed(device, 42)
