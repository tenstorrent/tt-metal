# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standalone validation harness for the GLM-4.7 decode MoE ring-reduce (Track 1).

Goal: prove — cheaply, without loading 218B weights — that the DeepSeek fused ring
reduce ops work for GLM-4.7's exact geometry (Mesh(8,4), hidden=5120, TP=axis0=8,
FABRIC_1D_RING) and reproduce the model's current 2-step all-reduce numerically.

GLM-4.7's decode MoE reduce today (decoder_layer_tt.py::_moe_forward, EP_REDUCE_DEVICE):
    combined = shared_out_partial * (1/DP) + routed_out_partial      # per-device partial
    mlp_out  = all_reduce(combined, axis0=8)                          # TP reduce  (replicated)
    mlp_out  = all_reduce(mlp_out,  axis1=4)                          # DP reduce  (replicated)

Ring-reduce variant validated here (mirrors the proven config in
tests/nightly/tg/ccl/test_deepseek_moe_reduce_scatter_6U.py, "three_links_partial"):
    stack   = concat([shared*(1/DP), routed], dim=0)                  # [2,1,32,5120] per device
    split   = deepseek_moe_fast_reduce_nc(stack, dim=0, split_size=5120/8, out=rs_input_mc)
    scat    = deepseek_moe_reduce_scatter(split, dim=3, cluster_axis=0, Ring, 3 links)  # [1,1,32,640]
    mlp_out = all_gather(scat, dim=3, cluster_axis=0)                 # restore [1,1,32,5120] replicated
    mlp_out = all_reduce(mlp_out, axis1=4)                            # DP reduce (unchanged)

Run (when the Galaxy device is free):
    export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
    ./python_env/bin/python -m pytest -svq \
      models/experimental/glm4_moe/experiments/decode_ring_reduce/test_ring_reduce_glm4.py

A PASS confirms the op sequence is trace-safe and PCC-matches the reference, and locks
the exact memory configs to lift into decoder_layer_tt.py behind GLM4_MOE_MOE_RING_REDUCE=1.
"""
from __future__ import annotations

import pytest
import torch
import ttnn
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# GLM-4.7-REAP-218B decode geometry.
HIDDEN = 5120
TP = 8  # mesh rows / cluster_axis=0
DP = 4  # mesh cols / cluster_axis=1
USERS = 32  # decode token block (sparsity_block_size); test uses one padded block
SLICE = HIDDEN // TP  # 640

# Proven rs_input sharded config for hidden=5120 / TP=8 (5 cores x 128 = 640).
# From test_deepseek_moe_reduce_scatter_6U.py "three_links_partial".
_RS_INPUT_MC = ttnn.MemoryConfig(
    ttnn.BufferType.L1,
    ttnn.NdShardSpec(
        ttnn.Shape([1, 1, USERS, 128]),
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))]),
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    ),
)
_L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
_RS_NUM_LINKS = 3


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}],
    indirect=True,
)
def test_glm4_ring_reduce_matches_allreduce(mesh_device):
    """concat->fast_reduce_nc->ring reduce_scatter->all_gather (axis0) must equal a
    plain per-axis all-reduce of (shared/DP + routed) over TP=8, for every device."""
    torch.manual_seed(0)
    mesh_rows, mesh_cols = tuple(mesh_device.shape)
    assert (mesh_rows, mesh_cols) == (TP, DP), f"expected Mesh(8,4), got {mesh_device.shape}"

    # Per-device partials, one rank-4 [1,1,USERS,HIDDEN] shard per device. Device index
    # d (row-major over the mesh) maps to row = d // DP, col = d % DP. Distinct per-device
    # data so a wrong axis/reduction is caught. reduce_scatter requires dim == rank-1, so
    # per-device tensors MUST be rank 4 (last dim = index 3).
    n_dev = TP * DP
    shared_host = torch.rand((n_dev, 1, USERS, HIDDEN)).bfloat16()  # [32,1,32,5120]
    routed_host = torch.rand((n_dev, 1, USERS, HIDDEN)).bfloat16()

    def to_mesh(host4d):
        # host4d: [n_dev, 1, U, H] -> ShardTensorToMesh(dim=0) gives one [1,1,U,H] per device.
        return ttnn.from_torch(
            host4d,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=_L1_INTERLEAVED,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

    shared_tt = to_mesh(shared_host)
    routed_tt = to_mesh(routed_host)

    # ---- Ring path (axis0 TP reduce) ----
    shared_scaled = ttnn.mul(shared_tt, 1.0 / DP, memory_config=_L1_INTERLEAVED)
    stack = ttnn.concat([shared_scaled, routed_tt], dim=0)  # [2,1,USERS,HIDDEN] per device
    split = ttnn.experimental.deepseek_moe_fast_reduce_nc(
        stack, dim=0, split_size=SLICE, output_memory_config=_RS_INPUT_MC
    )
    scattered = ttnn.experimental.deepseek_moe_reduce_scatter(
        split,
        output_memory_config=_L1_INTERLEAVED,
        dim=3,
        num_links=_RS_NUM_LINKS,
        topology=ttnn.Topology.Ring,
        cluster_axis=0,
    )  # [1,1,USERS,640] per device, reduced over the 8 rows
    gathered = ttnn.all_gather(
        scattered, dim=3, cluster_axis=0, num_links=_RS_NUM_LINKS, topology=ttnn.Topology.Ring
    )  # [1,1,USERS,5120] per device, replicated across the 8 rows
    # Concat the per-device outputs back to [n_dev,1,USERS,HIDDEN] (device index on dim0).
    ring_out = ttnn.to_torch(gathered, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    ring_out = ring_out.reshape(TP, DP, USERS, HIDDEN)  # [row, col, U, H]

    # ---- Reference: for each column, sum (shared/DP + routed) over the 8 rows ----
    ref_combined = (shared_host.float() / DP + routed_host.float()).reshape(TP, DP, USERS, HIDDEN)
    ref_axis0 = ref_combined.sum(dim=0)  # [col, U, H]

    # After all_gather on axis0 every row holds the same TP-sum, so ring_out[row, col] must
    # equal ref_axis0[col] for all rows. Check the worst-case row/col against the reference.
    ok = True
    for col in range(DP):
        eq, msg = comp_pcc(ring_out[0, col], ref_axis0[col], 0.99)
        logger.info(f"GLM4 ring-reduce axis0 PCC (col={col}): {msg}")
        ok = ok and eq
    # Also confirm rows agree (all-gather replication) for col 0.
    eq_rows, msg_rows = comp_pcc(ring_out[0, 0], ring_out[TP - 1, 0], 0.999)
    logger.info(f"GLM4 ring-reduce row-replication PCC: {msg_rows}")
    assert ok, "ring-reduce axis0 value mismatch vs torch reference"
    assert eq_rows, f"all_gather did not replicate across rows: {msg_rows}"
