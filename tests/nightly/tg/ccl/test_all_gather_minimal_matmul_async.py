# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from ttnn import ShardTensor2dMesh, ConcatMesh2dToTensor

from tests.nightly.t3000.ccl.test_all_gather_minimal_matmul_async import run_test_linear

from tracy.process_model_log import (
    get_latest_ops_log_filename,
    run_device_profiler,
)


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "M, K, N, core_grid_x, core_grid_y, num_workers_per_link, num_links, force_transpose",
    [
        (4096, 4096, 4096, 4, 4, 4, 1, True),
        (4096, 4096, 4096, 8, 8, 8, 1, True),
        (4096, 4096, 4096, 8, 8, 4, 2, True),
        (4096, 4096, 4096, 8, 8, 2, 4, True),
    ],
    ids=[
        "4K4K4Ksmallgrid",
        "4K4K4Kfullgrid",
        "2links",
        "4links",
    ],
)
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(8, 8, 8, 2, 2)],
)
@pytest.mark.parametrize(
    "use_non_fused",
    [
        True,
        False,
    ],
    ids=["separate", "fused"],
)
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_linear(
    mesh_device,
    M,
    K,
    N,
    M_block_size,
    K_block_size,
    N_block_size,
    subblock_h,
    subblock_w,
    topology,
    core_grid_x,
    core_grid_y,
    num_workers_per_link,
    num_links,
    use_non_fused,
    force_transpose,
):
    check_result = run_test_linear(
        mesh_device,
        M,
        K,
        N,
        M_block_size,
        K_block_size,
        N_block_size,
        subblock_h,
        subblock_w,
        topology,
        core_grid=ttnn.CoreCoord(core_grid_x, core_grid_y),
        num_workers_per_link=num_workers_per_link,
        num_links=num_links,
        use_non_fused=use_non_fused,
        force_transpose=force_transpose,
    )
    for i in range(mesh_device.get_num_devices()):
        assert check_result[i]["pcc"] > 0.999_500
        assert check_result[i]["relative_rmse"] < 0.02
