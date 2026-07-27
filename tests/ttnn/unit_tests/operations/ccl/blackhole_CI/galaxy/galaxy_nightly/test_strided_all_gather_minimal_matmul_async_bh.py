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


# SP=8 / TP=4 on the blackhole galaxy. The (8, 4) mesh puts the M/sequence shard (other_dim) on
# axis 0 (SP=8) and the K/contraction shard (dim) on axis 1 (TP=4), matching the impl's
# shard_dims = [other_dim, dim]. The all-gather reconstructs full K across the TP group, so it
# rides axis 1 -- the impl's default cluster_axis=1. K gathers over TP=4, so per-device K is
# 4096/4 = 32 tiles; mm_block_k is 256 (8 tiles), which divides it.
#
# Core-grid partition (blackhole 12x10 grid): matmul takes the lower mm_core_grid.y rows at width
# mm_core_grid.x; the strided all-gather workers take the rows above, starting at ag_offset, so
# ag_offset.y must equal mm_core_grid.y to keep the two regions disjoint.
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
# One axis for the mutually-exclusive fused-output variants. addcmul and fused_activation cannot be
# combined on the op (validate); chunks (output split) is exercised on the plain projection, matching how
# LTX uses it (qkv/kv projections have no activation/addcmul). "chunks2" needs N divisible by chunks.
@pytest.mark.parametrize(
    "fused_op_variant",
    [None, "addcmul", "gelu_tanh", "chunks2"],
    ids=["plain", "addcmul", "gelu_tanh", "chunks2"],
)
# Bias is orthogonal to the variant (LTX uses bias on every AG-matmul call). Note bias disables the
# two-NoC split write (the !use_bias gate), so bias+chunks2 uses the single-NoC chunk write.
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
    )


# The AG-matmul configurations the LTX transformer block actually uses (see models/tt_dit LTXAttention /
# ParallelFeedForward), for both the video (dim=4096) and audio (dim=2048) blocks. Every call has bias; only
# K (=dim, gathered across TP=4), the output width N, the chunk count, the fused epilogue, and the N blocking
# differ. M reuses the wan1 sequence for all (see caveat: the real M is the text sequence for *_kv and the
# audio sequence for a_*). N below is the FULL model width; the table uses the per-device value (full / TP=4).
# The *_gate case (N = num_heads/TP = 8, a sub-tile width) is commented out for now — TODO.
#   *_qkv         : attn1.to_qkv       bias, chunks=3  (Q|K|V,  N = 3*dim)
#   *_kv          : attn2.to_kv        bias, chunks=2  (K|V,    N = 2*dim, M = context/text seq)
#   *_q_out       : to_q / to_out      bias, chunks=1, N = dim
#   *_out_addcmul : attn.to_out fused  bias + addcmul, chunks=1
#   *_ff1         : ffn.ff1 / audio_ff bias + gelu_tanh, chunks=1, N = ffn_dim
#   *_gate        : to_gate_logits     bias, chunks=1, N = num_heads = 32 (disabled)
@skip_for_wormhole_b0()
@skip_for_n_or_less_dev(1)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "ltx_layer, M, K, N, chunks, use_bias, activation, use_ternary, M_block, K_block, N_block, subblock_h, subblock_w",
    [
        # M is the full (SP=8) sequence length; per-device M = M / 8 (e.g. 38912 -> 4864, 9728 -> 1216).
        # M_block/K_block/N_block/subblock_h/subblock_w are in TILES. Video block (K = video_dim = 4096);
        # q_out folds to_q / to_out(plain) / cross-attn q. N is the PER-DEVICE output width (full N / TP=4),
        # since shard_weights=False replicates a per-device-sized weight (K is full, gathered by the AG).
        # Multi-N-block configs (v_qkv/v_kv/v_ff1 and audio) are commented out: with N_blocks/core > 1 the split
        # write takes the deferred path, which is not split-aware and deadlocks. These are the wide-N (compute-
        # bound) matmuls, so the fabric-bound AG-overlap path doesn't matter for them and we don't care to run them.
        # ("v_qkv", 38912, 4096, 3072, 3, True, None, False, 16, 8, 4, 2, 2), good
        # ("v_kv", 38912, 4096, 2048, 2, True, None, False, 16, 8, 4, 2, 2), check this dont see it on device
        ("v_q_out", 38912, 4096, 1024, 1, True, None, False, 16, 8, 4, 2, 2),
        ("v_q_out_addcmul", 38912, 4096, 1024, 1, True, None, True, 16, 8, 4, 2, 2),
        ("v_q_out_s1", 9728, 4096, 1024, 1, True, None, False, 16, 8, 4, 2, 2),
        ("v_q_out_addcmul_s1", 9728, 4096, 1024, 1, True, None, True, 16, 8, 4, 2, 2),
        # ("v_out_addcmul", 38912, 4096, 1024, 1, True, None, True, 16, 8, 4, 2, 2),
        # ("v_ff1", 38912, 4096, 4096, 1, True, "gelu_tanh", False, 16, 8, 4, 2, 2),
        ("v_gate", 38912, 4096, 8, 1, True, None, False, 16, 8, 1, 2, 1),
        ("v_gate_s1", 9728, 4096, 8, 1, True, None, False, 16, 8, 1, 2, 1),
        # Audio block (K = audio_dim = 2048), same fusion structure with halved dims.
        # ("a_qkv", 38912, 2048, 1536, 3, True, None, False, 16, 8, 4, 2, 2),
        ("a_kv", 38912, 2048, 1024, 2, True, None, False, 16, 8, 4, 2, 2),
        ("a_kv_s1", 9728, 2048, 1024, 2, True, None, False, 16, 8, 4, 2, 2),
        # ("a_q_out", 38912, 2048, 512, 1, True, None, False, 16, 8, 4, 2, 2),
        # ("a_out_addcmul", 38912, 2048, 512, 1, True, None, True, 16, 8, 4, 2, 2),
        # ("a_ff1", 38912, 2048, 2048, 1, True, "gelu_tanh", False, 16, 8, 4, 2, 2),
        ("a_gate", 38912, 2048, 8, 1, True, None, False, 16, 8, 1, 2, 1),
        ("a_gate_s1", 9728, 2048, 8, 1, True, None, False, 16, 8, 1, 2, 1),
        ("a2v_attn_s1", 9728, 4096, 512, 1, True, None, False, 16, 8, 1, 2, 1),
        ("a2v_attn_s2", 38912, 4096, 512, 1, True, None, False, 16, 8, 1, 2, 1),
    ],
    ids=[
        # "v_qkv",
        # "v_kv",
        "v_q_out",
        "v_out_addcmul",
        # "v_ff1",
        # "v_gate",
        # "a_qkv",
        # "a_kv",
        "a_q_out",
        "a_out_addcmul",
        # "a_ff1",
        # "a_gate",
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
    )
