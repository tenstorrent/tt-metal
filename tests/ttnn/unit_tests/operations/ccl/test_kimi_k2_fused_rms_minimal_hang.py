# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal reproducer for the kimi-k2 batch=128 decode hang in
``ttnn.fused_rms_minimal`` (a.k.a. ``rms_allgather``), reported as
https://github.com/tenstorrent/tt-xla/issues/4565.

Bisected upstream cause: tt-mlir #8099 ("Enable DistributedRMSNorm in
Llama") loosens the decomposition check on the ``(1,1,32,M)`` shape, which
sends kimi-k2's decode-time op through the ``fused_rms_minimal`` kernel.
The kernel then deadlocks before its first call returns. tt-xla bisect
confirmed the hang surfaces inside the kernel itself (host runtime hangs
on a post-op ``Synchronize``), independent of fabric mode (FABRIC_1D /
FABRIC_1D_RING) or topology (Linear / Ring).

What distinguishes kimi-k2's invocation from the Llama-70B path that
``fused_rms_minimal`` was tuned for: the input's width-shard core grid is
**non-rectangular**. kimi-k2's per-chip hidden width is 896 → 28 tiles →
28 cores, laid out as ``(0,0)-(7,2) ∪ (0,3)-(3,3)`` on an 8-wide grid with
a ``virt_to_physical_map``. Llama's distributed RMSNorm uses 32 (rectangular).

The reproducer below mirrors that configuration:

  - 8 chips on the cluster axis (matches kimi-k2's model parallel = 8)
  - per-chip ``(1, 1, 32, 896)`` bf16 input
  - 28-core non-rectangular width shard
  - ``program_config = LayerNormShardedMultiCore<grid=(8,4), block_w=1>``
  - Llama-style call params otherwise (Topology.Ring + FABRIC_1D_RING)

Expected outcome on a Galaxy 6U: the first ``fused_rms_minimal`` call
never returns; ``ttnn.synchronize_device`` after it triggers the standard
Metal ``Timeout detected (metal_context)`` → ``TT_THROW: TIMEOUT`` chain.
"""

import math

import pytest
import torch
from loguru import logger

import ttnn
from conftest import is_6u


def _kimi_k2_decode_input_shard_grid():
    """28-core non-rectangular shard grid as produced by tt-mlir for
    kimi-k2 decode (per ``DistributedRMSNormWidthShardInputRewritePattern``
    when ``numWidthTiles=28`` on an 8-wide physical grid: ``8*3 + 4 = 28``).
    """
    return ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 2)),  # 8 x 3 = 24
            ttnn.CoreRange(ttnn.CoreCoord(0, 3), ttnn.CoreCoord(3, 3)),  # 4 x 1 = 4
        ]
    )


@pytest.mark.skipif(not is_6u(), reason="Only reproduces on Galaxy 6U (8x4 mesh)")
@pytest.mark.parametrize(
    "device_params",
    [
        # FABRIC_1D matches the topology tt-xla initializes on this Galaxy 6U
        # (FABRIC_1D_RING failed mesh-mapping on CPLD < v1.16); the hang
        # reproduces under either fabric mode in any case.
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((8, 4), id="8x4_grid")],
    indirect=True,
)
def test_fused_rms_minimal_kimi_k2_decode_hang(mesh_device):
    # Cluster axis = 1 (the 8-chip model-parallel axis on a 4x8 mesh).
    num_devices = 8
    cluster_axis = 1
    # kimi-k2 hidden_size; 8-way model parallel; per chip width = 896.
    elements_per_batch = 7168
    input_shard_grid = _kimi_k2_decode_input_shard_grid()
    num_cores = input_shard_grid.num_cores()
    assert num_cores == 28, f"expected 28 non-rectangular cores; got {num_cores}"

    total_cores = num_cores * num_devices
    padded_dim_per_core = int(math.ceil(elements_per_batch / total_cores / 32) * 32)
    padded_dim = padded_dim_per_core * total_cores
    assert padded_dim == 7168, padded_dim
    size_per_device = padded_dim // num_devices  # 896

    input_shape = (1, 1, 32, padded_dim)

    # Set up a CCL subdevice that fully contains the 28-core non-rectangular
    # input shard grid (which spans x in [0..7] and y in [0..3]). Using a
    # wider 8x8 range is safe — fused_rms_minimal only uses cores listed in
    # the input shard grid.
    ccl_sub_device_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    torch.manual_seed(1234)

    # Width-sharded L1 input across our 28-core non-rectangular grid.
    input_memory_config = ttnn.create_sharded_memory_config(
        shape=(32, 32),
        core_grid=input_shard_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Stats scratch tensor as required by fused_rms_minimal: 1 tile per device.
    ag_shape = [1, 1, 32, num_devices]
    ag_memory_config = ttnn.create_sharded_memory_config(
        shape=(32, 32),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    stats_tensor = torch.zeros(ag_shape, dtype=torch.bfloat16)
    tt_stats = ttnn.from_torch(
        stats_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ag_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(None, 3), mesh_shape=list(ttnn.MeshShape(1, num_devices))
        ),
    )

    # Output uses the same shard layout as input (skip_write_back path).
    output_memory_config = input_memory_config

    # Build the input tensors. Hidden dim is sharded across `cluster_axis` (=1)
    # by mapping dim 3 onto the 1xN sub-mesh along that axis.
    input_tensor_torch = torch.randn(input_shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn((1, 1, 1, padded_dim), dtype=torch.bfloat16)

    input_tensor = ttnn.as_tensor(
        input_tensor_torch,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device=mesh_device, dims=(None, 3), mesh_shape=list(ttnn.MeshShape(1, num_devices))
        ),
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_memory_config,
    )
    gamma_tensor = ttnn.as_tensor(
        gamma_torch.reshape([1, 1, padded_dim // 32, 32]),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device=mesh_device, dims=(None, 2), mesh_shape=list(ttnn.MeshShape(1, num_devices))
        ),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Program config matches what tt-mlir emits for the kimi-k2 decode shape
    # (the offending IR's `<grid=<9,4>, block_w=1>` — here we keep block_w=1
    # for the 1-tile-per-core slice and use a (8,4) compute grid that covers
    # the 28 cores).
    layer_norm_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        subblock_w=1,
        block_h=1,
        block_w=1,
        inplace=False,
    )

    # GlobalSemaphore, fresh value 0 just like the runtime emits.
    ccl_semaphore = ttnn.create_global_semaphore(mesh_device, input_shard_grid, 0)

    logger.info(
        f"calling ttnn.fused_rms_minimal: input={tuple(input_shape)} per-chip=(1,1,32,{size_per_device}) "
        f"shard_grid={num_cores}cores non_rect cluster_axis={cluster_axis}"
    )
    tt_out = ttnn.fused_rms_minimal(
        input_tensor,
        layer_norm_config,
        cluster_axis,
        mesh_device,
        ccl_semaphore,
        # Matches the topology tt-mlir's runtime defaults to for the
        # ttnn.distributed_rms_norm op when the MLIR op has no explicit
        # topology attribute (which is what tt-xla emits for kimi-k2).
        topology=ttnn.Topology.Linear,
        memory_config=output_memory_config,
        epsilon=1e-5,
        weight=gamma_tensor,
        stats=tt_stats,
    )

    # If the op actually completed, this Synchronize will return promptly.
    # If the kernel deadlocked the fabric, this is where the Metal-side
    # "Timeout detected (metal_context)" surfaces.
    logger.info("synchronize_device after fused_rms_minimal — expect hang for issue #4565")
    ttnn.synchronize_device(mesh_device)
    logger.info("synchronize_device returned — op completed (unexpected on a buggy kernel)")

    # Light sanity check on shape.
    tt_out_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=(1, num_devices)),
    )
    assert tt_out_torch.shape[-1] == padded_dim
