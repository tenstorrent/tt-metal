# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Inner test for the hang_plugin device validation. NOT meant to be run
directly under normal CI — it always hangs and costs a device reset.

The outer test at .claude/eval/tests/test_hang_plugin_device.py invokes
this as a subprocess to validate that hang_plugin.py:
  1. Detects the device timeout from the first parametrization.
  2. Skips the remaining parametrizations with "previous parametrization
     of test_intentional_hang hung".
"""

import os
import sys

import pytest
import torch
import ttnn

# Allow `from tests....` imports when invoked from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))

from tests.ttnn.unit_tests.operations.intentional_hang.intentional_hang_op import (  # noqa: E402
    intentional_hang,
)


@pytest.mark.parametrize("shape", [(32, 32), (32, 64), (32, 96)])
def test_intentional_hang(shape, device):
    torch_in = torch.ones(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out = intentional_hang(tt_in)
    # Force a synchronous read-back so the device-side deadlock manifests
    # as a Python exception during the call phase, not during teardown.
    ttnn.to_torch(tt_out)
