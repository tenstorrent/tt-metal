# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 8: block type dense (layer 0) with ffn_collapse swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 0 (dense) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_collapse
    attn_norm
    q_a
    attention
    attn_residual
    ffn_hc
    ffn_collapse
"""

from models.demos.common.bringup.testing.component import run_swap_test
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
BLOCK_TYPE = "dense"
SWAPPED = ["attn_hc", "attn_collapse", "attn_norm", "q_a", "attention", "attn_residual", "ffn_hc", "ffn_collapse"]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
CHECKS = "steps"  # also gate every swapped step's own output (testing/component.py); None = block out only


@mesh_parametrize
def test_swap(mesh_device):
    assert run_swap_test(S, BLOCK_TYPE, SWAPPED, mesh_device, THRESHOLD, CHECKS)
