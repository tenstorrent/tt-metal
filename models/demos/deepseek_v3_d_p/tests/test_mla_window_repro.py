# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone ring_mla replay that trips the dev10 GDDR EDC counters and corrupts bank 3.

Shapes, dtypes, memory configs, program configs and CCL setup are the Kimi-K2.6 8x4 fabric2d
values captured from the Inspector runtime entries of the tripped run; nothing is imported from
the model, so this file is unaffected by edits to mla.py.

Env: MLA_WINDOW_ITERS (loop count, default 100), EDC_PROBE=1 to arm the per-op EDC probe.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.edc_probe import make_probe
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size

MESH_SHAPE = (8, 4)
NUM_HEADS_LOCAL = 16
KV_LORA_RANK = 512
KVPE_DIM = 576
SEQ_LEN_LOCAL = 640
KV_ACTUAL_GLOBAL = 51200
CACHE_LEN_GLOBAL = 56320
CACHE_LEN_LOCAL = 7040
DRAM_BANKS = 8
BANK_SHARD_TOKENS = 32
CCL_NUM_LINKS = 2

# Grids as they resolve on Blackhole (compute_with_storage_grid_size() = 12x10).
COMPUTE_GRID = (11, 10)
RING_CCL_CORE_OFFSET = (11, 0)
CCL_SUB_DEVICE_CRS = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 9))})

# 192 ** -0.5 scaled by the YaRN mscale (factor=64.0, mscale=1.0).
SDPA_SCALE = 0.14467962580268923

SDPA_PROGRAM_CONFIG = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=COMPUTE_GRID,
    q_chunk_size=32,
    k_chunk_size=640,
    exp_approx_mode=False,
)

KV_CACHE_MEMORY_CONFIG = ttnn.MemoryConfig(
    buffer_type=ttnn.BufferType.DRAM,
    nd_shard_spec=ttnn.NdShardSpec(
        shard_shape=[1, 1, BANK_SHARD_TOKENS, KVPE_DIM],
        grid=ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0)) for bank in range(DRAM_BANKS)]
        ),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    ),
)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "l1_small_size": 1152,
        }
    ],
    ids=["fabric2d"],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], ids=["8x4"], indirect=True)
@pytest.mark.timeout(0)
def test_mla_window_repro(mesh_device):
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    ccl_semaphores = [ttnn.create_global_semaphore(mesh_device, CCL_SUB_DEVICE_CRS, 0) for _ in range(2)]

    # Left uninitialised: nothing checks ring_mla's output, so only the spec and the CCL
    # topology matter, and whatever is already in DRAM is the payload.
    kv_cache = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, CACHE_LEN_LOCAL, KVPE_DIM]),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        mesh_device,
        KV_CACHE_MEMORY_CONFIG,
    )
    kv_cache.update_tensor_topology(
        ttnn.TensorTopology(
            ttnn.MeshShape([MESH_SHAPE[0] * MESH_SHAPE[1]]),
            [ttnn.PlacementReplicate()],
            list(ttnn.MeshCoordinateRange(ttnn.MeshShape(*MESH_SHAPE))),
        )
    )
    gathered_kv = ttnn.from_torch(
        torch.randn(1, 1, CACHE_LEN_GLOBAL, KVPE_DIM, dtype=torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=[None, None]),
    )
    tt_q = ttnn.from_torch(
        torch.randn(1, NUM_HEADS_LOCAL, SEQ_LEN_LOCAL, KVPE_DIM, dtype=torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    mesh_device.enable_program_cache()
    probe = make_probe(mesh_device)
    iters = int(os.environ.get("MLA_WINDOW_ITERS", "100"))
    logger.info(f"MLA window repro: {iters} iters, kv_actual={KV_ACTUAL_GLOBAL}")
    for i in range(iters):
        logger.info(f"iter {i}")
        attn_out, _ = ttnn.transformer.ring_mla(
            tt_q,
            kv_cache,
            persistent_output_buffer_kv=gathered_kv,
            head_dim_v=KV_LORA_RANK,
            logical_n=CACHE_LEN_GLOBAL,
            program_config=SDPA_PROGRAM_CONFIG,
            scale=SDPA_SCALE,
            compute_kernel_config=compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=ccl_semaphores,
            num_links=CCL_NUM_LINKS,
            cluster_axis=0,
            mesh_device=mesh_device,
            topology=ttnn.Topology.Linear,
            ccl_core_grid_offset=RING_CCL_CORE_OFFSET,
            use_column_major_ccl=True,
            is_balanced=False,
            kv_cache_batch_idx=0,
            kv_actual_isl=KV_ACTUAL_GLOBAL,
        )
        probe.check()
        ttnn.deallocate(attn_out)

    ttnn.synchronize_device(mesh_device)
    logger.success(f"MLA window repro survived {iters} iters")
