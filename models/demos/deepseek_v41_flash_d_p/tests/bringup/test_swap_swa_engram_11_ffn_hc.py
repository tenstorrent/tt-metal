# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 11: block type swa_engram (layer 1) with ffn_hc swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 1 (swa_engram) with these steps on the device and the rest on the CPU reference:
    engram_rows
    engram_kv
    engram_add
    attn_hc
    attn_collapse
    attn_norm
    q_a
    kv
    attention
    attn_residual
    ffn_hc
"""

from models.demos.common.bringup.testing.component import run_swap_test
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
BLOCK_TYPE = "swa_engram"
SWAPPED = [
    "engram_rows",
    "engram_kv",
    "engram_add",
    "attn_hc",
    "attn_collapse",
    "attn_norm",
    "q_a",
    "kv",
    "attention",
    "attn_residual",
    "ffn_hc",
]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
CHECKS = "steps"  # also gate every swapped step's own output (testing/component.py); None = block out only


@mesh_parametrize
def test_swap(mesh_device):
    assert run_swap_test(S, BLOCK_TYPE, SWAPPED, mesh_device, THRESHOLD, CHECKS)
