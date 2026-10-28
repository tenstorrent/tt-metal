# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""test_mla_window_repro on a single UBB: a 4x2 FABRIC_2D mesh, e.g. TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15.

Per-device work matches the 8x4 test: the same 640 local Q rows over the same 56320-token gathered
KV. The ring is 4 long, so each device holds twice the local KV and the new-token window is
4 * 640 = 2560, carved from the top of the unchanged cache.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.edc_probe import make_probe
from models.demos.deepseek_v3_d_p.tests.test_mla_window_repro import (
    CACHE_LEN_GLOBAL,
    CCL_NUM_LINKS,
    CCL_SUB_DEVICE_CRS,
    KV_CACHE_MEMORY_CONFIG,
    KV_LORA_RANK,
    KVPE_DIM,
    NUM_HEADS_LOCAL,
    RING_CCL_CORE_OFFSET,
    SDPA_PROGRAM_CONFIG,
    SDPA_SCALE,
    SEQ_LEN_LOCAL,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size

MESH_SHAPE = (4, 2)
RING_SIZE = MESH_SHAPE[0]
CACHE_LEN_LOCAL = CACHE_LEN_GLOBAL // RING_SIZE
KV_ACTUAL_GLOBAL = CACHE_LEN_GLOBAL - RING_SIZE * SEQ_LEN_LOCAL


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
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], ids=["4x2"], indirect=True)
@pytest.mark.timeout(0)
def test_mla_window_repro_ubb(mesh_device):
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    ccl_semaphores = [ttnn.create_global_semaphore(mesh_device, CCL_SUB_DEVICE_CRS, 0) for _ in range(2)]

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
    logger.info(f"MLA window repro (UBB 4x2): {iters} iters, kv_actual={KV_ACTUAL_GLOBAL}")
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
    logger.success(f"MLA window repro (UBB 4x2) survived {iters} iters")
