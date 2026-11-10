# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.moe_block.moe_tail on a Galaxy: determinism. The op's inputs (the combine's owner rows y_own,
the head's shared expert output) are the real MoE head -> sparse_dispatch_2d -> routed_experts_combine outputs on
seeded random weights and x (the routing steered through the router's bias), run once and kept; the tail then runs
alone (a latent model's W_up seeded random as well). Cases, routings, weights and the upstream ops: block_stages.py
(the experts + combine checks' block).

* test_moe_tail_determinism: CALLS tails of the same inputs; the output bit-identical across calls, compared on the
  device."""

import sys
from pathlib import Path

import pytest

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from block_stages import (  # noqa: E402
    CALLS,
    PARAMS,
    BlockStages,
    host_weights,  # noqa: F401 (the module's host weight cache fixture)
    skip_unless_galaxy,
)
from op_checks import DeterminismMarker  # noqa: E402


@PARAMS
@pytest.mark.timeout(1800)
def test_moe_tail_determinism(case, routing, mesh_device, device_params, host_weights):  # noqa: F811
    skip_unless_galaxy()
    st = BlockStages(mesh_device, case, host_weights)
    try:
        st.route(routing)
        y_own = st()  # (the combine's persistent owner rows: not written again below)
        det = DeterminismMarker(mesh_device)
        for _ in range(CALLS):
            y = st.tail(y_own)
            det.add({"y": y})  # (the first call's cloned as the reference)
            ttnn.deallocate(y)
        det.assert_identical(f"moe_tail {case.id} {routing}")
    finally:
        st.close()
