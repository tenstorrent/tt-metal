# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 2: block type sliding_moe (layer 1) with attention swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 1 (sliding_moe) with these steps on the device and the rest on the CPU reference:
    attn_norm
    attention
"""

from models.demos.common.bringup.testing.component import run_swap_test
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
BLOCK_TYPE = "sliding_moe"
SWAPPED = ["attn_norm", "attention"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
CHECKS = "steps"  # also gate every swapped step's own output (testing/component.py); None = block out only


@mesh_parametrize
def test_swap(mesh_device):
    assert run_swap_test(S, BLOCK_TYPE, SWAPPED, mesh_device, THRESHOLD, CHECKS)
