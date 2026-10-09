# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.tt_dit.tests.models.wan2_2.test_all_gather_minimal_matmul_async import run_test_linear
from tests.nightly.t3000.ccl.test_strided_all_gather_minimal_matmul_async import (
    run_strided_all_gather_minimal_matmul_impl,
)


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "residual_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["residual_dram", "residual_l1"]
)
@pytest.mark.parametrize(
    "gate_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["gate_dram", "gate_l1"]
)
@pytest.mark.parametrize("force_transpose", [True, False], ids=["transpose", "no_transpose"])
def test_all_gather_minimal_matmul_ternary_memory(
    mesh_device, residual_memory_config, gate_memory_config, force_transpose
):
    """Legacy accessor types differ for DRAM and L1 even without Metal 2.0 binding IDs."""
    torch.manual_seed(0)
    results = run_test_linear(
        mesh_device,
        M=512,
        K=512,
        N=512,
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=2,
        subblock_w=2,
        topology=ttnn.Topology.Linear,
        core_grid=ttnn.CoreCoord(4, 4),
        num_workers_per_link=4,
        num_links=1,
        num_buffers_per_channel=8,
        use_bias=False,
        fuse_addcmul=True,
        force_transpose=force_transpose,
        residual_memory_config=residual_memory_config,
        gate_memory_config=gate_memory_config,
    )
    for iteration in results:
        for chunk in iteration:
            for result in chunk:
                assert result["pcc"] > 0.9995
                assert result["relative_rmse"] < 0.02


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "residual_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["residual_dram", "residual_l1"]
)
@pytest.mark.parametrize(
    "gate_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["gate_dram", "gate_l1"]
)
@pytest.mark.parametrize("M, N", [(512, 512), (1024, 256)], ids=["no_transpose", "transpose"])
def test_strided_all_gather_minimal_matmul_ternary_memory(
    mesh_device, residual_memory_config, gate_memory_config, M, N
):
    """Exercise the fabric-bound ternary reader with independently placed operands."""
    run_strided_all_gather_minimal_matmul_impl(
        mesh_device,
        num_devices=mesh_device.get_num_devices(),
        M=M,
        K=512,
        N=N,
        dim=3,
        other_dim=2,
        num_links=1,
        ag_input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mem_config_input=ttnn.DRAM_MEMORY_CONFIG,
        mem_config_ag=ttnn.DRAM_MEMORY_CONFIG,
        mem_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        all_gather_topology=ttnn.Topology.Linear,
        mm_block_m=64,
        mm_block_k=64,
        mm_block_n=64,
        subblock_h=1,
        subblock_w=2,
        num_workers_per_link=2,
        num_buffers_per_channel=8,
        mm_core_grid=ttnn.CoreCoord(4, 4),
        ag_core_grid_offset=(0, 4),
        enable_trace=False,
        use_non_fused=False,
        use_ternary=True,
        residual_memory_config=residual_memory_config,
        gate_memory_config=gate_memory_config,
    )
