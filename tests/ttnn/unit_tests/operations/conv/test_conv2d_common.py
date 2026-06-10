# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map, HS, WS, BS

SliceHeight = ttnn.Conv2dDRAMSliceHeight
SliceWidth = ttnn.Conv2dDRAMSliceWidth

__all__ = ["run_conv", "torch_tensor_map", "HS", "WS", "BS", "SliceHeight", "SliceWidth"]
