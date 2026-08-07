# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Variable-width `chunks` for all_gather_minimal_matmul_async on the Blackhole galaxy.

`chunk_sizes` lets the op's N-split produce chunks of differing widths (e.g. a fused QKV+gate).
Each case asserts PCC per chunk, since the per-chunk row stride is the failure mode a shared width
would reintroduce (chunk 0 stays correct while later chunks scatter).
"""

import pytest
import ttnn
from models.common.utility_functions import is_wormhole_b0

from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import (
    create_fabric_router_config,
    run_test_linear,
)

# Per-device chunk widths in ELEMENTS. All must be multiples of TILE_WIDTH (32) and sum to N.
_CHUNK_CASES = [
    # Uniform, as a control: the variable-width code path must reproduce the legacy behaviour.
    pytest.param(3072, [1024, 1024, 1024], id="uniform_3x1024"),
    # The motivating shape: fused QKV + per-head gate
    pytest.param(3104, [1024, 1024, 1024, 32], id="qkv_plus_gate"),
    # Asymmetric widths whose boundaries do NOT fall on tidy multiples
    pytest.param(3104, [512, 1536, 1024, 32], id="asymmetric"),
    # Two chunks, wildly unbalanced -- smallest legal chunk against a large one.
    pytest.param(2080, [2048, 32], id="lopsided_2chunk"),
]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("N, chunk_sizes", _CHUNK_CASES)
@pytest.mark.parametrize(
    "device_params, topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                # 4096-B payload: 8192 overflows the fabric mux L1 map at this client count.
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_all_gather_minimal_matmul_variable_chunks(mesh_device, topology, N, chunk_sizes):
    if is_wormhole_b0():
        pytest.skip("core grid (12, 9) exceeds wormhole_b0 compute grid (8x8), blackhole-only config")

    assert sum(chunk_sizes) == N, f"chunk_sizes {chunk_sizes} must sum to N={N}"

    check_result = run_test_linear(
        mesh_device,
        M=4864,
        K=4096,
        N=N,
        chunks=len(chunk_sizes),
        chunk_sizes=chunk_sizes,
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=2,
        subblock_w=1,
        topology=topology,
        core_grid=ttnn.CoreCoord(12, 9),
        num_workers_per_link=6,
        num_links=2,
        use_bias=True,
        use_non_fused=False,
        sp_axis=1,
        tp_axis=0,
        cluster_axis=0,
    )

    # Per chunk, per device -- a mis-strided write shows up as one bad chunk, not a bad total.
    for c in range(len(chunk_sizes)):
        for i, result in enumerate(check_result[0][c]):
            assert result["pcc"] > 0.999_500, f"chunk {c} (width {chunk_sizes[c]}), device {i}: pcc {result['pcc']}"
            assert (
                result["relative_rmse"] < 0.02
            ), f"chunk {c} (width {chunk_sizes[c]}), device {i}: relative_rmse {result['relative_rmse']}"
