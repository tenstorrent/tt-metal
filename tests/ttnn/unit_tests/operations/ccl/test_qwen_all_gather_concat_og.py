# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone `ttnn.experimental.all_gather_concat` repro for Qwen Galaxy decode (no models/, no checkpoints).

Tensor layout follows `nlp_create_qkv_heads_decode` Q output: **`[1, batch, n_local_heads, head_dim]`** — not HF
**`[..., heads, batch, dim]`**. Primary input uses a minimal L1 HEIGHT-sharded layout matching `all_gather_concat`
validation (not `SCORES_BATCHED_MM_OUTPUT_MEMCFG`). Build via **L1 ROW_MAJOR → `to_layout` TILE → input memcfg**;
**`ReplicateTensorToMesh`** so each chip is **`[1, 8, 8, 128]`**.

Config mirrors `qwen_model_config` prefetcher ring path: all_gather_concat output uses
**`SHARDED_ATTN_WO_INPUT_RING_MEMCFG`** (`PREFETCHER_NOC1_GRID`), intermediate from
**`TT_CCL.get_all_gather_concat_inter_buffer`**, full-device gather semaphore when worker subdevice is unset,
**`FABRIC_1D_RING`** + **Ring** CCL topology.
"""


import math

import pytest
import torch

import ttnn


CLUSTER_SHAPE = (8, 4)
MAX_BATCH_USERS = 32
N_KV_HEADS = 8
N_HEADS = 64
HEAD_DIM_MM = 128  # fused concat kernel padded dim 3

RING_SIZE = 24
_TILE_BATCH_ROWS = 32  # padded batch rows for WO / intermediate host shape
ATTN_WO_INPUT_SHARD_W = 12288 // 8 // RING_SIZE  # SHARDED_ATTN_WO_INPUT_RING_MEMCFG column width split

# PREFETCHER_NOC1_GRID / ring_core_range_set from qwen_model_config
PREFETCHER_NOC1_GRID = (
    (6, 6), (6, 7), (6, 9), (6, 0), (6, 1), (6, 2), (6, 4), (6, 5),
    (5, 5), (5, 6), (5, 7), (5, 9), (5, 0), (5, 1), (5, 2), (5, 4),
    (1, 4), (1, 5), (1, 9), (1, 0), (2, 0), (2, 4), (2, 5), (2, 9),
)
# llama_ccl.get_all_gather_concat_inter_buffer singleton noc1 cores
INTERMEDIATE_RING_CORES = (
    (6, 6), (6, 7), (6, 9), (6, 0), (6, 1), (6, 2), (6, 4), (6, 5),
    (5, 5), (5, 6), (5, 7), (5, 9), (5, 0), (5, 1), (5, 2), (5, 4), (1, 5),
)


def _gather_semaphore_grid_no_prefetch() -> ttnn.CoreRangeSet:
    """`TT_CCL` full grid when `worker_sub_device_id` is None."""
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])


def _noc1_ring_core_rangeset() -> ttnn.CoreRangeSet:
    """`ring_core_range_set` / PREFETCHER_NOC1_GRID from qwen_model_config."""
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in PREFETCHER_NOC1_GRID]
    )


def _all_gather_concat_input_memcfg(n_local_heads: int) -> ttnn.MemoryConfig:
    """Minimal L1 HEIGHT sharded input; core ranges match all_gather_concat device validation."""
    pad_h = math.ceil(n_local_heads / 32) * 32
    core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
        ]
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, [pad_h, HEAD_DIM_MM], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _sharded_attn_wo_input_ring_memcfg() -> ttnn.MemoryConfig:
    """`qwen_model_config['SHARDED_ATTN_WO_INPUT_RING_MEMCFG']`."""
    return ttnn.create_sharded_memory_config(
        shape=(_TILE_BATCH_ROWS, ATTN_WO_INPUT_SHARD_W),
        core_grid=_noc1_ring_core_rangeset(),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _concat_intermediate_memcfg() -> ttnn.MemoryConfig:
    """`TT_CCL.get_all_gather_concat_inter_buffer` (WH Galaxy)."""
    crs = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 4))]
        + [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for x, y in INTERMEDIATE_RING_CORES
        ]
    )

    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, [_TILE_BATCH_ROWS, HEAD_DIM_MM], ttnn.ShardOrientation.ROW_MAJOR),
    )


@pytest.mark.parametrize("mesh_device", [pytest.param(CLUSTER_SHAPE, id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_qwen_decode_all_gather_concat_repro(mesh_device):
    cluster_shape = CLUSTER_SHAPE
    num_devices = cluster_shape[0] * cluster_shape[1]
    assert num_devices == mesh_device.get_num_devices()

    n_groups = num_devices // N_KV_HEADS
    batch_per_chip = max(MAX_BATCH_USERS // n_groups, 1)
    n_local_heads = N_HEADS // N_KV_HEADS

    input_mem = _all_gather_concat_input_memcfg(n_local_heads)
    out_mem = _sharded_attn_wo_input_ring_memcfg()
    topology = ttnn.Topology.Ring

    inter_mem = _concat_intermediate_memcfg()
    intermediate = ttnn.from_torch(
        torch.zeros(cluster_shape[0], _TILE_BATCH_ROWS * cluster_shape[1], _TILE_BATCH_ROWS, HEAD_DIM_MM, dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=inter_mem,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[0, 1], mesh_shape=list(cluster_shape)),
    )

    inp = torch.randn(1, batch_per_chip, n_local_heads, HEAD_DIM_MM)
    primary = ttnn.to_memory_config(
        ttnn.to_layout(
            ttnn.from_torch(
                inp,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            ttnn.TILE_LAYOUT,
        ),
        input_mem,
    )
    shards = ttnn.get_device_tensors(primary)
    assert shards
    for s in shards:
        ps = tuple(s.padded_shape)
        assert ps[1] == batch_per_chip and ps[3] == HEAD_DIM_MM, (
            f"expected per-chip [*,{batch_per_chip},*,{HEAD_DIM_MM}] padded got logical={tuple(s.shape)} padded={ps}"
        )

    semaphore = ttnn.create_global_semaphore(mesh_device, _gather_semaphore_grid_no_prefetch(), 0)
    sub_dev = ttnn.SubDeviceId(0)

    output = ttnn.experimental.all_gather_concat(
        primary,
        intermediate,
        1,
        cluster_axis=1,
        mesh_device=mesh_device,
        topology=topology,
        multi_device_global_semaphore=semaphore,
        num_links=1,
        num_heads=n_local_heads,
        memory_config=out_mem,
        subdevice_id=sub_dev,
        use_noc1_only=False,
    )
    print("before sync")
    ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_dev])
    print("after sync")

    assert tuple(output.shape) == (1, 1, MAX_BATCH_USERS, n_local_heads * HEAD_DIM_MM)
