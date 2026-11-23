# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug matrix for all_gather_rms_norm (DO NOT DELETE — documents the debugging process).

Isolates the R_U (W_local % 32 != 0) LayerNorm failure of the acceptance case 1x1x32x50 LN fp32 TILE gamma L2:
vary one factor at a time (dtype, gamma layout / presence, norm type, links, topology).
"""

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.all_gather_rms_norm.test_all_gather_rms_norm import DEVICE_PARAMS, _run_case

R = ttnn.Topology.Ring
L = ttnn.Topology.Linear
RM = ttnn.ROW_MAJOR_LAYOUT
TL = ttnn.TILE_LAYOUT

CASES = [
    # (shape, norm, gamma layout, beta, topo, links, dtype)
    ((1, 1, 32, 50), "layer_norm", TL, False, L, 2, ttnn.float32),  # the failing acceptance case
    ((1, 1, 32, 50), "layer_norm", None, False, L, 2, ttnn.float32),
    ((1, 1, 32, 50), "layer_norm", RM, False, L, 2, ttnn.float32),
    ((1, 1, 32, 50), "layer_norm", TL, False, L, 2, ttnn.bfloat16),
    ((1, 1, 32, 50), "rms_norm", TL, False, L, 2, ttnn.float32),
    ((1, 1, 32, 50), "layer_norm", TL, False, L, 1, ttnn.float32),
    ((1, 1, 32, 64), "layer_norm", TL, False, L, 2, ttnn.float32),
    ((1, 1, 32, 18), "layer_norm", None, False, L, 1, ttnn.float32),
]


def _id(c):
    shape, norm, gl, beta, topo, links, dtype = c
    g = "nog" if gl is None else ("gRM" if gl == RM else "gT")
    return f"{'x'.join(map(str, shape))}-{norm}-{g}-{topo.name}-L{links}-{'fp32' if dtype == ttnn.float32 else 'bf16'}"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4")], indirect=True)
@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
def test_debug_matrix(mesh_device, case):
    shape, norm_type, gamma_layout, with_beta, topology, num_links, dtype = case
    _run_case(
        mesh_device,
        shape,
        cluster_axis=1,
        dtype=dtype,
        norm_type=norm_type,
        gamma_layout=gamma_layout,
        with_beta=with_beta,
        topology=topology,
        num_links=num_links,
    )
