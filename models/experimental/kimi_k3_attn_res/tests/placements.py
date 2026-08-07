# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The mesh placements and per-chip row count every AttnRes suite runs at.

Shared rather than repeated so the suites cannot drift onto different shapes and
quietly stop gating the same op.

**No single-device placement.** Production splits `d` across the TP axis, and at
`tp_factor == 1` `_reduce_stats` is the identity — the score chain then reduces a
rank-local `d` and divides by the global one without a collective to reconcile them.
That is a different computation, so a green `(1, 1)` arm says nothing about the op
that ships. Every gate in this module runs sharded.

**640 rows per chip.** Prefill chunks 5120 tokens across the sequence-parallel axis,
so that is what a chip sees on the Galaxy. It is not an arbitrary size: the collective
picks its algorithm from the payload, and below roughly 313 rows/chip the split form
stops paying for itself (`ROOFLINE.md` §5, §7). A suite at 64 tokens spends its arms
on a reduction production never issues.
"""

import pytest
import ttnn

PER_CHIP_TOKENS = 640

# `ttnn.all_reduce` needs an initialized fabric context — without it the op dies in the
# control plane (`control_plane.cpp:2186`) rather than returning wrong numbers.
# FABRIC_1D is what the analog's own 2x4 prefill config uses
# (`test_prefill_block.py:513-517`) and is the right pairing for `Topology.Linear` on a
# single cluster axis.
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}

# `(2, 4)` is the LoudBox. Of the three valid 8-chip meshes it is the one that
# exercises both axes, and its TP factor of 4 is Galaxy's, so it covers the reduction
# Galaxy runs. `(8, 4)` is the Galaxy itself: the same TP factor over a wider sequence
# axis, which the op is indifferent to by construction, so that arm is here to be run
# on the box rather than to add coverage. `mesh_device` skips a placement asking for
# more chips than the host has, so `(8, 4)` is inert below 32 chips.
on_placements = pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((2, 4), FABRIC, id="mesh-2x4"),
        pytest.param((8, 4), FABRIC, id="mesh-8x4"),
    ],
    indirect=["mesh_device", "device_params"],
)
