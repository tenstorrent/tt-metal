# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Explicit-only North-Mini shape probe for experimental ``moe_compute``.

This file deliberately does not match pytest's default ``test_*.py`` collection
pattern.  Run it directly while no other process is using TT hardware:

    pytest -q \
      models/autoports/coherelabs_north_mini_code_1_0/tests/moe_compute_probe.py \
      -s

The probe reuses the operation's canonical single-card harness so the sparse
dispatch buffers, drain-core sharding, packed rank-6 weights, output tripwire,
and numerical goldens cannot drift from the operation's own contract.
"""

import os

import pytest
from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size
from ttnn.operations.ccl import MoEActivationFunction

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.test_moe_compute_single_card import (
    _run_moe_compute_single_card_test,
)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 500000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_north_mini_moe_compute_only_shape(mesh_device, mesh_shape):
    """Probe E=128, top-8, H=2048, I=768 with BF16 input and SwiGLU.

    ``moe_compute`` itself requires its specially packed weights in BFLOAT4_B;
    BF16 here is the activation/score dtype, matching the only supported input
    contract.  Consequently a passing probe is not by itself evidence that the
    op preserves North-Mini's current BF16 weight precision policy.
    """

    ring_size = effective_matmul_ring_size(mesh_device)
    # One token is the decoder batch-1 shape but currently terminates the host
    # process with SIGFPE inside the public op call.  Keep it reproducible via
    # NORTH_MOE_PROBE_TOKENS=1 while using the canonical 32-token tile control
    # by default.
    tokens = int(os.environ.get("NORTH_MOE_PROBE_TOKENS", "32"))
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=128,
        tokens_per_device=tokens,
        selected_experts_k=8,
        N=768,
        hidden_size=2048,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(2048, matmul_ring_size=ring_size),
        dtype=ttnn.bfloat16,
        activation_type=MoEActivationFunction.SWIGLU,
        has_bias=False,
    )
