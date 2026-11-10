# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.moe_block.sparse_dispatch_2d on a Galaxy: determinism. The op's inputs (xbuf with every chip's own
rows, the route words) are the real MoE head's outputs on seeded random weights and x (the routing steered through
the router's bias), run once and kept; the dispatch then runs alone. Cases, routings, weights and the upstream ops:
block_stages.py (the experts + combine checks' block).

* test_sparse_dispatch_2d_determinism: CALLS dispatches of the same rows; the filled xbuf (in place: every chip's own
  rows and every row it received) bit-identical across calls, compared on the device."""

import sys
from pathlib import Path

import pytest

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
def test_sparse_dispatch_2d_determinism(case, routing, mesh_device, device_params, host_weights):  # noqa: F811
    skip_unless_galaxy()
    st = BlockStages(mesh_device, case, host_weights)
    try:
        st.route(routing)  # (the head, then a first dispatch)
        det = DeterminismMarker(mesh_device)
        det.add({"xbuf": st.xg})  # (the route's dispatch: the reference)
        for _ in range(CALLS):
            det.add({"xbuf": st.dispatch()})  # (the same buffer, filled again)
        det.assert_identical(f"sparse_dispatch_2d {case.id} {routing}")
    finally:
        st.close()
