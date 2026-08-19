# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from loguru import logger

from tests.nightly.t3000.ccl.test_strided_all_gather_minimal_matmul_async import (
    run_strided_all_gather_minimal_matmul_impl,
)
from models.common.utility_functions import (
    skip_for_wormhole_b0,
    skip_for_n_or_less_dev,
)


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


# SP=8 / TP=4 on the blackhole galaxy
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("num_links", [2], ids=["2link"])
@pytest.mark.parametrize(
    "M, K, N, dim, other_dim, num_workers_per_link, layout, ag_input_dtype, mm_block_m, mm_block_k, mm_block_n, subblock_h, subblock_w, mm_core_grid, shard_weights, ag_offset",
    [
        (
            38912,  # 4864 per device (SP=8): 152 tiles; against a 16-tile mm_block_m this is 9.5 blocks -> ragged last M-block (hang repro)
            4096,
            1024,
            3,
            2,
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            512,
            256,
            128,
            2,
            2,
            ttnn.CoreCoord(12, 8),
            False,
            (0, 8),
        ),
    ],
    ids=["wan1"],
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
    "enable_trace,num_iters",
    [
        (True, 3),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "use_non_fused",
    [
        False,
    ],
    ids=["fused"],
)
# One axis for the mutually-exclusive fused-output variants
@pytest.mark.parametrize(
    "fused_op_variant",
    [None, "addcmul", "gelu_tanh", "chunks2"],
    ids=["plain", "addcmul", "gelu_tanh", "chunks2"],
)
# Bias is orthogonal to the variant (LTX uses bias on every AG-matmul call)
@pytest.mark.parametrize(
    "use_bias",
    [False, True],
    ids=["no_bias", "bias"],
)
@pytest.mark.parametrize(
    "read_local_slice_from_input",
    [
        True,
    ],
    ids=["read_local"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(8192),
                "trace_region_size": 1171456,
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_strided_all_gather_minimal_matmul_async(
    mesh_device,
    M,
    K,
    N,
    dim,
    other_dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    mem_config_mm,
    enable_trace,
    all_gather_topology,
    num_iters,
    num_workers_per_link,
    mm_block_m,
    mm_block_k,
    mm_block_n,
    subblock_h,
    subblock_w,
    mm_core_grid,
    use_non_fused,
    fused_op_variant,
    use_bias,
    shard_weights,
    ag_offset,
    read_local_slice_from_input,
):
    logger.info(f"fabric max payload = {ttnn._ttnn.fabric.get_tt_fabric_max_payload_size_bytes()} B")

    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x < mm_core_grid.x or grid.y < ag_offset[1] + 1:
        pytest.skip(f"Requires worker grid >= {mm_core_grid.x}x{ag_offset[1] + 1}, got {grid.x}x{grid.y}")

    use_ternary = fused_op_variant == "addcmul"
    activation = fused_op_variant if fused_op_variant == "gelu_tanh" else None
    chunks = 2 if fused_op_variant == "chunks2" else 1

    run_strided_all_gather_minimal_matmul_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        M,
        K,
        N,
        dim,
        other_dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        mem_config_mm,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=8,
        mm_block_m=mm_block_m,
        mm_block_k=mm_block_k,
        mm_block_n=mm_block_n,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        mm_core_grid=mm_core_grid,
        use_non_fused=use_non_fused,
        use_ternary=use_ternary,
        activation=activation,
        chunks=chunks,
        use_bias=use_bias,
        shard_weights=shard_weights,
        ag_core_grid_offset=ag_offset,
        read_local_slice_from_input=read_local_slice_from_input,
        # Blackhole has the cores for the aggregators; require them so a config change cannot silently
        # drop back to reader-signaled matmul and lose the overlap.
        mm_signal_aggregator_mode=ttnn.MMSignalAggregatorMode.On,
    )


# Fused SwiGLU on the strided AGMM. N is the OUTPUT width; the device weight is the 2N-wide packed
# [up|gate]. On a 12-wide core grid the factory splits gate/up PAIRS, so out_N_tiles = N/32 sets the
# per-core width: "aligned" divides 12 exactly, "ragged" does not and exercises the pair-aware padding.
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "M, K, N, mm_block_m, mm_block_k, mm_block_n, subblock_h, subblock_w",
    [
        (9728, 4096, 1536, 512, 256, 128, 2, 2),
        (9728, 4096, 1024, 512, 256, 128, 2, 2),
        # Same shape as "aligned" but the N block spans a core's whole weight range (1 block/core).
        # M block is halved to keep the block CBs inside L1; the N block itself is not a swiglu limit.
        (9728, 4096, 1536, 256, 256, 256, 2, 2),
    ],
    ids=["aligned", "ragged", "full_block"],
)
@pytest.mark.parametrize("use_bias", [False, True], ids=["no_bias", "bias"])
# num_iters > 1 also covers the cached-program override path, not just create().
@pytest.mark.parametrize("enable_trace,num_iters", [(False, 2)], ids=["check"])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(8192),
                "trace_region_size": 1171456,
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_strided_all_gather_minimal_matmul_swiglu(
    mesh_device,
    M,
    K,
    N,
    mm_block_m,
    mm_block_k,
    mm_block_n,
    subblock_h,
    subblock_w,
    use_bias,
    enable_trace,
    num_iters,
    all_gather_topology,
):
    mm_core_grid = ttnn.CoreCoord(12, 8)
    ag_offset = (0, 8)
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x < mm_core_grid.x or grid.y < ag_offset[1] + 1:
        pytest.skip(f"Requires worker grid >= {mm_core_grid.x}x{ag_offset[1] + 1}, got {grid.x}x{grid.y}")

    assert (mm_block_n // 32) % 2 == 0, "fuse_swiglu requires an even mm_block_n in tiles"

    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_strided_all_gather_minimal_matmul_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        M,
        K,
        N,
        3,  # dim (AG/K shard axis)
        2,  # other_dim (M/sequence shard axis)
        2,  # num_links
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        dram,
        dram,
        dram,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        num_workers_per_link=3,
        num_buffers_per_channel=8,
        mm_block_m=mm_block_m,
        mm_block_k=mm_block_k,
        mm_block_n=mm_block_n,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        mm_core_grid=mm_core_grid,
        use_non_fused=False,
        use_bias=use_bias,
        shard_weights=False,
        ag_core_grid_offset=ag_offset,
        read_local_slice_from_input=True,
        fuse_swiglu=True,
    )


# The AG-matmul configurations the LTX transformer block actually uses
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "ltx_layer, M, K, N, chunks, use_bias, activation, use_ternary, M_block, K_block, N_block, subblock_h, subblock_w",
    [
        # M is the full (SP=8) sequence length
        ("v_q_out", 38912, 4096, 1024, 1, True, None, False, 16, 8, 4, 2, 2),
        ("v_q_out_addcmul", 38912, 4096, 1024, 1, True, None, True, 16, 8, 4, 2, 2),
        ("v_q_out_s1", 9728, 4096, 1024, 1, True, None, False, 16, 8, 4, 2, 2),
        ("v_q_out_addcmul_s1", 9728, 4096, 1024, 1, True, None, True, 16, 8, 4, 2, 2),
        # ("v_out_addcmul", 38912, 4096, 1024, 1, True, None, True, 16, 8, 4, 2, 2)
        ("v_gate", 38912, 4096, 8, 1, True, None, False, 16, 8, 1, 2, 1),
        ("v_gate_s1", 9728, 4096, 8, 1, True, None, False, 16, 8, 1, 2, 1),
        # Audio block (K = audio_dim = 2048), same fusion structure with halved dims
        ("a_kv", 38912, 2048, 1024, 1, True, None, False, 16, 8, 4, 2, 2),
        ("a_kv_s1", 9728, 2048, 1024, 1, True, None, False, 16, 8, 4, 2, 2),
        # ("a_q_out", 38912, 2048, 512, 1, True, None, False, 16, 8, 4, 2, 2)
        ("a_gate", 38912, 2048, 8, 1, True, None, False, 16, 8, 1, 2, 1),
        ("a_gate_s1", 9728, 2048, 8, 1, True, None, False, 16, 8, 1, 2, 1),
        # N_block=2 to match the in-model registry (N_block=1 deadlocks via the split-write path).
        ("a2v_attn_s1", 9728, 4096, 512, 1, True, None, False, 16, 8, 2, 2, 2),
        ("a2v_attn_s2", 38912, 4096, 512, 1, True, None, False, 16, 8, 2, 2, 2),
    ],
    ids=[
        "v_q_out",
        "v_q_out_addcmul",
        "v_q_out_s1",
        "v_q_out_addcmul_s1",
        "v_gate",
        "v_gate_s1",
        "a_kv",
        "a_kv_s1",
        "a_gate",
        "a_gate_s1",
        "a2v_attn_s1",
        "a2v_attn_s2",
    ],
)
@pytest.mark.parametrize(
    "enable_trace,num_iters",
    [
        (True, 3),
        (False, 1),
        # Repeated eager invocations (no trace/warmup) to cover back-to-back sAGMM ops like the e2e.
        (False, 8),
    ],
    ids=["perf", "check", "check_multi"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(8192),
                "trace_region_size": 1171456,
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_strided_all_gather_minimal_matmul_ltx_configs(
    mesh_device,
    ltx_layer,
    M,
    K,
    N,
    chunks,
    use_bias,
    activation,
    use_ternary,
    M_block,
    K_block,
    N_block,
    subblock_h,
    subblock_w,
    enable_trace,
    num_iters,
    all_gather_topology,
):
    mm_core_grid = ttnn.CoreCoord(12, 8)
    ag_offset = (0, 8)
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x < mm_core_grid.x or grid.y < ag_offset[1] + 1:
        pytest.skip(f"Requires worker grid >= {mm_core_grid.x}x{ag_offset[1] + 1}, got {grid.x}x{grid.y}")

    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_strided_all_gather_minimal_matmul_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        M,  # M (sequence, sharded on SP=8)
        K,  # video_dim=4096 or audio_dim=2048, gathered across TP=4
        N,
        3,  # dim (AG/K shard axis)
        2,  # other_dim (M/sequence shard axis)
        2,  # num_links
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        dram,
        dram,
        dram,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        num_workers_per_link=3,
        num_buffers_per_channel=8,
        mm_block_m=M_block * 32,
        mm_block_k=K_block * 32,
        mm_block_n=N_block * 32,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        mm_core_grid=mm_core_grid,
        use_non_fused=False,
        use_ternary=use_ternary,
        activation=activation,
        chunks=chunks,
        use_bias=use_bias,
        shard_weights=False,
        ag_core_grid_offset=ag_offset,
        read_local_slice_from_input=True,
        mm_signal_aggregator_mode=ttnn.MMSignalAggregatorMode.On,
    )
