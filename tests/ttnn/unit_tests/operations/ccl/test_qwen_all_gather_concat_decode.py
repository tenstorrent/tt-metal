# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone `ttnn.experimental.all_gather_concat` repro for Qwen Galaxy decode (no models/, no checkpoints).

Tensor layout follows `nlp_create_qkv_heads_decode` Q output: **`[1, batch, n_local_heads, head_dim]`** — not HF
**`[..., heads, batch, dim]`**. Build primary input via **DRAM ROW_MAJOR → `to_layout` TILE → SCORES L1 HEIGHT**; tilizing already
HEIGHT_SHARDED ROW_MAJOR corrupts BH `padded_shape`. Primary uses **`ReplicateTensorToMesh`** so each chip matches SDPA-like
**`[1, 8, 8, 128]`** (full local batch/heads); `ConcatMesh2dToTensor` in demos is host compose only.

Config mirrors `models/demos/llama3_70b_galaxy` no-prefetch path: WO ring **`PREFETCHER_NOC1_GRID`**, full-device gather semaphore when
worker subdevice is unset. BH 6U UBB: **`fabric_config=True` → `FABRIC_1D`** + **Linear** CCL (see `qwen_model_config.ccl_topology`).
WH 6U uses `FABRIC_1D_RING` + Ring.
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
WO_INPUT_SHARD_W = 12288 // 8 // RING_SIZE  # SHARDED_ATTN_WO_INPUT_RING column width split


def _gather_semaphore_grid_no_prefetch() -> ttnn.CoreRangeSet:
    """`TT_CCL` full grid when `worker_sub_device_id` is None."""
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])


def _noc1_ring_core_rangeset() -> ttnn.CoreRangeSet:
    """`PREFETCHER_NOC1_GRID` singleton cores from model_config."""
    pts = (
        (6, 6),
        (6, 7),
        (6, 9),
        (6, 0),
        (6, 1),
        (6, 2),
        (6, 4),
        (6, 5),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 0),
        (5, 1),
        (5, 2),
        (5, 4),
        (1, 4),
        (1, 5),
        (1, 9),
        (1, 0),
        (2, 0),
        (2, 4),
        (2, 5),
        (2, 9),
    )
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in pts])


def _galaxy_sub_core_grids(max_y: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, max_y)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, max_y)),
        ]
    )


def _sdpa_concat_input_memcfg(
    batch_per_device_group: int, n_local_heads: int, sub_grids: ttnn.CoreRangeSet
) -> ttnn.MemoryConfig:
    """SCORES `BATCHED_MM_OUTPUT`: HEIGHT_SHARDED (ceil(n_heads/32)*32) × HEAD_DIM_MM."""
    pad_h = math.ceil(n_local_heads / 32) * 32
    return ttnn.create_sharded_memory_config(
        shape=(pad_h, HEAD_DIM_MM),
        core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(
            ttnn.CoreCoord(1, 0), batch_per_device_group, sub_grids, row_wise=True
        ),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _wo_input_memcfg() -> ttnn.MemoryConfig:
    return ttnn.create_sharded_memory_config(
        shape=(_TILE_BATCH_ROWS, WO_INPUT_SHARD_W),
        core_grid=_noc1_ring_core_rangeset(),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _concat_intermediate_memcfg(is_blackhole: bool) -> ttnn.MemoryConfig:
    """`TT_CCL.get_all_gather_concat_inter_buffer` split by arch."""
    if is_blackhole:
        sub_crs = _gather_semaphore_grid_no_prefetch()
        ncores = min(32, sub_crs.num_cores())
        crs = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(1, 0), ncores, sub_crs, row_wise=True)
    else:
        crs = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 4)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 6), ttnn.CoreCoord(6, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 7), ttnn.CoreCoord(6, 7)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 9), ttnn.CoreCoord(6, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(6, 0)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 1), ttnn.CoreCoord(6, 1)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 2), ttnn.CoreCoord(6, 2)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 4), ttnn.CoreCoord(6, 4)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 5), ttnn.CoreCoord(6, 5)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 5), ttnn.CoreCoord(5, 5)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 6), ttnn.CoreCoord(5, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 7), ttnn.CoreCoord(5, 7)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 9), ttnn.CoreCoord(5, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, 0)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 1), ttnn.CoreCoord(5, 1)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 2), ttnn.CoreCoord(5, 2)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 4), ttnn.CoreCoord(5, 4)),
                ttnn.CoreRange(ttnn.CoreCoord(1, 5), ttnn.CoreCoord(1, 5)),
            ]
        )

    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, [32, HEAD_DIM_MM], ttnn.ShardOrientation.ROW_MAJOR),
    )


@pytest.mark.parametrize("mesh_device", [pytest.param(CLUSTER_SHAPE, id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
def test_qwen_decode_all_gather_concat_repro(mesh_device):
    cluster_shape = CLUSTER_SHAPE
    num_devices = cluster_shape[0] * cluster_shape[1]
    assert num_devices == mesh_device.get_num_devices()

    n_groups = num_devices // N_KV_HEADS
    batch_per_chip = max(MAX_BATCH_USERS // n_groups, 1)
    n_local_heads = N_HEADS // N_KV_HEADS

    sub_grids = _galaxy_sub_core_grids(7 if ttnn.get_arch_name().lower() == "blackhole" else 9)

    sdpa_mem = _sdpa_concat_input_memcfg(batch_per_chip, n_local_heads, sub_grids)
    out_mem = _wo_input_memcfg()
    topology = ttnn.Topology.Linear if ttnn.get_arch_name().lower() == "blackhole" else ttnn.Topology.Ring
    is_bh = ttnn.get_arch_name().lower() == "blackhole"

    inter_mem = _concat_intermediate_memcfg(is_bh)
    intermediate = ttnn.from_torch(
        torch.zeros(
            cluster_shape[0], _TILE_BATCH_ROWS * cluster_shape[1], _TILE_BATCH_ROWS, HEAD_DIM_MM, dtype=torch.bfloat16
        ),
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
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            ttnn.TILE_LAYOUT,
        ),
        sdpa_mem,
    )
    shards = ttnn.get_device_tensors(primary)
    assert shards
    for s in shards:
        ps = tuple(s.padded_shape)
        assert (
            ps[1] == batch_per_chip and ps[3] == HEAD_DIM_MM
        ), f"expected per-chip [*,{batch_per_chip},*,{HEAD_DIM_MM}] padded got logical={tuple(s.shape)} padded={ps}"

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
    ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_dev])

    assert tuple(output.shape) == (1, 1, MAX_BATCH_USERS, n_local_heads * HEAD_DIM_MM)
