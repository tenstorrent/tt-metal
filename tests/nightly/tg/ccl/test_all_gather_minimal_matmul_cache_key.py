# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The all-gather matmul's program-cache key must tell apart programs with the same shapes but a different compute
config, output dtype or fused epilogue; a collision silently serves the first compiled program to the others."""

import pytest
import ttnn
from models.common.utility_functions import is_wormhole_b0

from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import (
    create_fabric_router_config,
    run_test_linear,
)

SHAPE = dict(
    M=3072,
    K=5120,
    N=1280,
    M_block_size=8,
    K_block_size=8,
    N_block_size=8,
    subblock_h=2,
    subblock_w=1,
    core_grid=ttnn.CoreCoord(12, 9),
    num_workers_per_link=6,
    num_links=2,
    use_bias=True,
    sp_axis=1,
    tp_axis=0,
    cluster_axis=0,
)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize(
    "device_params, topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_all_gather_minimal_matmul_cache_key(mesh_device, topology):
    if is_wormhole_b0():
        pytest.skip("core grid (12, 9) exceeds wormhole_b0 compute grid (8x8), blackhole-only config")
    mesh_device.enable_program_cache()

    def new_entries(min_pcc=0.999, **kwargs):
        before = mesh_device.num_program_cache_entries()
        checks = run_test_linear(mesh_device, topology=topology, **SHAPE, **kwargs)
        for per_chunk in checks[0]:
            for check in per_chunk:
                assert check["pcc"] > min_pcc
        return mesh_device.num_program_cache_entries() - before

    assert new_entries(math_fidelity=ttnn.MathFidelity.HiFi2) > 0
    assert new_entries(math_fidelity=ttnn.MathFidelity.LoFi, min_pcc=0.99) > 0
    assert new_entries(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_acc=False) > 0
    assert new_entries(math_fidelity=ttnn.MathFidelity.HiFi2, output_dtype=ttnn.bfloat8_b) > 0
    assert new_entries(math_fidelity=ttnn.MathFidelity.HiFi2, fuse_addcmul=True, addcmul_scalar=1.0) > 0
    assert new_entries(math_fidelity=ttnn.MathFidelity.HiFi2) == 0
