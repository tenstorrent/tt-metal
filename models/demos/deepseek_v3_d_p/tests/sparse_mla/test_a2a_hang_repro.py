# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Looped reproducer for the GLM-5.2 sparse-MLA AllToAll hang (Blaze prefill CI, run 35689027384).

tt-triage on the hung prefill shows op window
    92 TopkLargeIndices -> 93 HighBwAllGather (KVPE prefix, on the overlap sub-device)
    -> [sub-device manager cleared] -> 94 AllToAllAsyncGeneric RUNNING on all 32 chips
    -> 95 Untilize / 96 SparseSDPA / 97 Tilize never started.
Op 94 is the thin-head -> sequence reshard of q in ``ttMLA._sparse_mla``: [1,16,640,576] bf16 TILE DRAM,
in_dim=1 -> out_dim=2 over cluster_axis=1 (TP=4), num_links=2.

Modes add the surrounding production ops one layer at a time:
  a2a      : head->seq all-to-all only
  a2a_rt   : head->seq, untilize, tilize, seq->head (the full _sparse_mla CCL round trip)
  overlap  : load the 80/40 sparse-MLA sub-device manager, run the SP KVPE high_bw_all_gather on the
             40-core gather sub-device with the caller-owned semaphores, clear the manager, then a2a_rt

Arm CI's hang detector so a hang raises instead of spinning:
  TT_METAL_OPERATION_TIMEOUT_SECONDS=120
Iterations: A2A_REPRO_ITERS (default 2000); a host sync every A2A_REPRO_SYNC_EVERY iterations (default 50).
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

SP_AXIS, TP_AXIS = 0, 1
NUM_LINKS = 2
GLM_L1_SMALL_SIZE = 1216  # test_prefill_transformer_chunked.GLM_L1_SMALL_SIZE
CHUNK = 5120
NUM_HEADS = 64
KVPE_DIM = 576
KV_LORA_RANK = 512
KV_PREFIX_TOKENS = 15 * 1024  # KV prefix depth gathered before attention (mid-sequence chunk)

ITERS = int(os.environ.get("A2A_REPRO_ITERS", "2000"))
SYNC_EVERY = int(os.environ.get("A2A_REPRO_SYNC_EVERY", "50"))


def _mesh_tensor(mesh_device, shape, placements, layout, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        torch.randn(shape).to(torch.bfloat16) if dtype == ttnn.bfloat16 else torch.zeros(shape, dtype=torch.int32),
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(list(placements), mesh_device.shape)),
    )


def _head_to_seq(q):
    return ttnn.experimental.all_to_all_async_generic(
        q, in_dim=1, out_dim=2, num_links=NUM_LINKS, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=TP_AXIS
    )


def _seq_to_head(x):
    return ttnn.experimental.all_to_all_async_generic(
        x, in_dim=2, out_dim=1, num_links=NUM_LINKS, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=TP_AXIS
    )


def _a2a_round_trip(q):
    """Mirror of _sparse_mla's CCL skeleton with sparse_sdpa replaced by a slice to v_dim."""
    q_seq = _head_to_seq(q)  # [1,64,160,576]
    q_rm = ttnn.to_layout(q_seq, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(q_seq)
    out = ttnn.slice(q_rm, [0, 0, 0, 0], [1, NUM_HEADS, q_rm.shape[2], KV_LORA_RANK])  # sparse_sdpa stand-in
    ttnn.deallocate(q_rm)
    ret = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
    ttnn.deallocate(out)
    head = _seq_to_head(ret)  # [1,16,640,512]
    ttnn.deallocate(ret)
    return head


@pytest.mark.parametrize("mode", ["a2a", "a2a_rt", "overlap"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=GLM51Config.FABRIC_PAYLOAD_SIZE, l1_small_size=GLM_L1_SMALL_SIZE
            ),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_glm_sparse_mla_a2a_hang_repro(mesh_device, device_params, mode):
    sp, tp = tuple(mesh_device.shape)
    seq_local = CHUNK // sp  # 640
    # q: SP shards the sequence, TP shards the 64 heads -> per-chip [1,16,640,576]
    q = _mesh_tensor(
        mesh_device,
        [1, NUM_HEADS, CHUNK, KVPE_DIM],
        (ttnn.PlacementShard(2), ttnn.PlacementShard(1)),
        ttnn.TILE_LAYOUT,
    )
    assert list(ttnn.get_device_tensors(q)[0].shape) == [1, NUM_HEADS // tp, seq_local, KVPE_DIM]

    overlap = None
    if mode == "overlap":
        overlap = get_tt_ccl(mesh_device).get_sparse_mla_overlap_resources("galaxy_80_40")
        kv_total = KV_PREFIX_TOKENS + CHUNK
        kv_local = _mesh_tensor(
            mesh_device,
            [1, 1, kv_total, KVPE_DIM],
            (ttnn.PlacementShard(2), ttnn.PlacementReplicate()),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        kv_gathered = _mesh_tensor(
            mesh_device,
            [1, 1, kv_total, KVPE_DIM],
            (ttnn.PlacementReplicate(), ttnn.PlacementReplicate()),
            ttnn.ROW_MAJOR_LAYOUT,
        )

    ttnn.synchronize_device(mesh_device)
    logger.info(f"A2A repro mode={mode} iters={ITERS} q_local={list(ttnn.get_device_tensors(q)[0].shape)}")
    t0 = time.time()
    for it in range(ITERS):
        if overlap is not None:
            mesh_device.load_sub_device_manager(overlap.manager_id)
            ttnn.experimental.high_bw_all_gather(
                kv_local,
                dim=2,
                output_tensor=kv_gathered,
                num_links=NUM_LINKS,
                cluster_axis=SP_AXIS,
                subdevice_id=overlap.gather_subdevice_id,
                sub_core_grids=overlap.gather_core_grid,
                ready_semaphore=overlap.ready_semaphore,
                data_valid_semaphore=overlap.data_valid_semaphore,
            )
            mesh_device.clear_loaded_sub_device_manager()
        if mode == "a2a":
            out = _head_to_seq(q)
        else:
            out = _a2a_round_trip(q)
        ttnn.deallocate(out)
        if (it + 1) % SYNC_EVERY == 0:
            ttnn.synchronize_device(mesh_device)
            logger.info(f"iter {it + 1}/{ITERS} ok ({time.time() - t0:.1f}s)")
    ttnn.synchronize_device(mesh_device)
    logger.info(f"A2A repro mode={mode}: {ITERS} iterations completed without a hang in {time.time() - t0:.1f}s")
