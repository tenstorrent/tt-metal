# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# AG+MM fused op tests for Llama 70B 8K ISL - Attn Out layer
# M=8192, K=1024, N=2048 (WO projection after SDPA)

import pytest
from dataclasses import dataclass
import ttnn
from models.common.utility_functions import skip_for_blackhole

from tests.nightly.t3000.ccl.test_strided_all_gather_minimal_matmul_async import (
    run_strided_all_gather_minimal_matmul_impl,
)


@dataclass
class TestConfig:
    M: int
    K: int
    N: int
    dim: int = 3
    other_dim: int = 2
    mm_block_m: int = 256
    mm_block_k: int = 256
    mm_block_n: int = 256
    subblock_h: int = 2
    subblock_w: int = 2
    num_workers_per_link: int = 2
    mm_core_grid: object = None


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param(
            TestConfig(M=8192, K=1024, N=2048, mm_core_grid=ttnn.CoreCoord(4, 8)),
            id="attn_out_4x8",
        ),
        pytest.param(
            TestConfig(M=8192, K=1024, N=2048, mm_core_grid=ttnn.CoreCoord(8, 4)),
            id="attn_out_8x4",
        ),
        pytest.param(
            TestConfig(M=8192, K=1024, N=2048, mm_core_grid=ttnn.CoreCoord(7, 8)),
            id="attn_out_7x8",
        ),
        pytest.param(
            TestConfig(M=8192, K=1024, N=2048, mm_core_grid=ttnn.CoreCoord(8, 8)),
            id="attn_out_8x8",
        ),
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag, mem_config_mm",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 266240},
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_ag_mm_llama_8k_attn_out(
    mesh_device,
    test_config,
    num_links,
    mem_config_input,
    mem_config_ag,
    mem_config_mm,
    device_params,
    all_gather_topology,
):
    TILE_SIZE = 32
    cfg = test_config
    assert (cfg.M // TILE_SIZE) % cfg.num_workers_per_link == 0
    Nt = cfg.N // TILE_SIZE
    Nt_per_core = Nt // cfg.mm_core_grid.x
    assert Nt_per_core > (cfg.mm_block_n // TILE_SIZE)

    run_strided_all_gather_minimal_matmul_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        cfg.M,
        cfg.K,
        cfg.N,
        cfg.dim,
        cfg.other_dim,
        num_links,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        mem_config_input,
        mem_config_ag,
        mem_config_mm,
        all_gather_topology=all_gather_topology,
        cluster_axis=1,
        enable_trace=False,
        num_iters=1,
        num_workers_per_link=cfg.num_workers_per_link,
        mm_block_m=cfg.mm_block_m,
        mm_block_k=cfg.mm_block_k,
        mm_block_n=cfg.mm_block_n,
        subblock_h=cfg.subblock_h,
        subblock_w=cfg.subblock_w,
        mm_core_grid=cfg.mm_core_grid,
        use_non_fused=False,
        shard_weights=False,
        read_local_slice_from_input=True,
    )
