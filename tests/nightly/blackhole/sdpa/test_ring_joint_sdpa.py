# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Ring Joint Attention SDPA Tests for Video Generation and MLA Models on Blackhole

Tests Ring Joint Attention accuracy and determinism using:
- WAN 2.2 model shapes: standard attention with non-causal, non-balanced mode
- DeepSeek MLA (Multi-Latent Attention): causal attention with balanced zigzag work distribution

Runs on BH multi-chip setups (single ring 1xN or Galaxy 4x8 mesh).
Perf tests are included but skipped on CI.

Model Configurations:
- WAN: nhq == nhk == nhv, d_q == d_k == d_v == 128, bfloat16 for all tensors
- MLA: nhk == 1, d_q == d_k == 576, d_v == 128, bfloat16 for Q, bfloat8_b for K/V

BH adaptation: uses init_device_compute_kernel_config instead of WormholeComputeKernelConfig.
"""
import os
import math
from unittest import mock

import torch
from dataclasses import dataclass, field
from itertools import product
from typing import List, Dict
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
import ttnn
from ttnn.operations.ccl import Topology
from loguru import logger
import pytest

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
from tests.nightly.sdpa_perf_utils import MeshConfig

MESH_CONFIG = MeshConfig.detect()

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================
#
# Each model defines per-device sequence lengths for Galaxy and Quiet Box,
# tuned so per-core work matches between platforms (11 vs 10 cores/head).
# See test_instructions.md for full parallelization context.

BATCH_SIZE = 1


@dataclass
class ModelConfig:
    """Benchmark configuration for a video generation or MLA model."""

    # Fixed model parameters
    name: str
    nhq: int  # Can be hardware-dependent to keep per-core work balanced between Galaxy and Quiet Box
    nhk: int
    nhv: int  # Can be hardware-dependent to keep per-core work balanced between Galaxy and Quiet Box
    d_q: int
    d_k: int
    d_v: int
    is_causal: bool
    is_balanced: bool

    # Dtypes
    q_dtype: ttnn.DataType
    kv_dtype: ttnn.DataType

    # Sweep parameters (chunk sizes to test)
    q_chunk_sizes: List[int]
    k_chunk_sizes: List[int]

    # Can be hardware-dependent to keep per-core work balanced between Galaxy and Quiet Box
    seq_len: int


def generate_model_configs(mesh_config: MeshConfig) -> Dict[str, ModelConfig]:
    """Generate model configurations for WAN and MLA based on mesh configuration."""
    configs = []

    # WAN 2.2 — 1×Galaxy deployment (32 chips, ~75K total tokens at 720p)

    configs.append(
        ModelConfig(
            name="wan2_2_1xGLX",
            nhq=10,
            nhk=10,
            nhv=10,
            d_q=128,
            d_k=128,
            d_v=128,
            is_causal=False,
            is_balanced=False,
            q_dtype=ttnn.bfloat16,
            kv_dtype=ttnn.bfloat16,
            q_chunk_sizes=[224, 256, 288],
            k_chunk_sizes=[128, 256, 512],
            seq_len=9472 if mesh_config.is_galaxy else 8544,  # Tuned for each platform to match per-core work
        )
    )

    # WAN 2.2 — 4×Galaxy deployment (128 chips, 720p split across 4 Galaxies)
    configs.append(
        ModelConfig(
            name="wan2_2_4xGLX",
            nhq=10,
            nhk=10,
            nhv=10,
            d_q=128,
            d_k=128,
            d_v=128,
            is_causal=False,
            is_balanced=False,
            q_dtype=ttnn.bfloat16,
            kv_dtype=ttnn.bfloat16,
            q_chunk_sizes=[224, 256, 288],
            k_chunk_sizes=[128, 256, 512],
            seq_len=2368 if mesh_config.is_galaxy else 2240,  # Tuned for each platform to match per-core work
        )
    )

    # VideogenModel1 720p — 1×Galaxy deployment (115,200 total tokens)
    # Single benchmark config: Sq_chunk_t=7 (q=224), k=512
    # Galaxy: 14400/dev, q_per_core=6. QB: 13440/dev, q_per_core=6.
    configs.append(
        ModelConfig(
            name="videogen_model1_720p",
            nhq=10,
            nhk=10,
            nhv=10,
            d_q=128,
            d_k=128,
            d_v=128,
            is_causal=False,
            is_balanced=False,
            q_dtype=ttnn.bfloat16,
            kv_dtype=ttnn.bfloat16,
            q_chunk_sizes=[224],
            k_chunk_sizes=[512],
            seq_len=14400 if mesh_config.is_galaxy else 13440,  # Tuned for each platform to match per-core work
        )
    )

    # VideogenModel1 480p — 1×Galaxy deployment (49,920 total tokens)
    # q288 chosen for zero slot waste on Galaxy (22 chunks / 11 cores = 2 each).
    # Galaxy: 6240/dev, q_per_core=2. QB: 5760/dev, q_per_core=2.
    configs.append(
        ModelConfig(
            name="videogen_model1_480p",
            nhq=10,
            nhk=10,
            nhv=10,
            d_q=128,
            d_k=128,
            d_v=128,
            is_causal=False,
            is_balanced=False,
            q_dtype=ttnn.bfloat16,
            kv_dtype=ttnn.bfloat16,
            q_chunk_sizes=[288],
            k_chunk_sizes=[512],
            seq_len=6240 if mesh_config.is_galaxy else 5760,  # Tuned for each platform to match per-core work
        )
    )

    # VideogenModel1 768×512 — 1×Galaxy deployment (49,152 total tokens)
    # q288 chosen for zero slot waste on Galaxy (22 chunks / 11 cores = 2 each).
    # Zero K pad waste (6144 = 12×512 exactly).
    # Galaxy: 6144/dev, q_per_core=2. QB: 5760/dev, q_per_core=2.
    configs.append(
        ModelConfig(
            name="videogen_model1_768x512",
            nhq=10,
            nhk=10,
            nhv=10,
            d_q=128,
            d_k=128,
            d_v=128,
            is_causal=False,
            is_balanced=False,
            q_dtype=ttnn.bfloat16,
            kv_dtype=ttnn.bfloat16,
            q_chunk_sizes=[288],
            k_chunk_sizes=[512],
            seq_len=6144 if mesh_config.is_galaxy else 5760,  # Tuned for each platform to match per-core work
        )
    )

    # DeepSeek MLA configuration
    mla_nhq = 32 if mesh_config.is_galaxy else 29  # Tuned for each platform to match per-core work

    configs.append(
        ModelConfig(
            name="mla_100k",
            nhq=mla_nhq,
            nhk=1,
            nhv=mla_nhq,
            d_q=576,
            d_k=576,
            d_v=128,
            is_causal=True,
            is_balanced=True,
            q_dtype=ttnn.bfloat16,
            kv_dtype=ttnn.bfloat8_b,
            q_chunk_sizes=[160],
            # k=160 -> Sk_chunk_t=5; k=256 straddles (3200%256=128); k=320 -> Sk_chunk_t=10 with kt-sub=2.
            k_chunk_sizes=[160, 256, 320],
            seq_len=3200,
        )
    )

    configs.append(
        ModelConfig(
            name="mla_128k",
            nhq=mla_nhq,
            nhk=1,
            nhv=mla_nhq,
            d_q=576,
            d_k=576,
            d_v=128,
            is_causal=True,
            is_balanced=True,
            q_dtype=ttnn.bfloat16,
            kv_dtype=ttnn.bfloat8_b,
            q_chunk_sizes=[128],
            k_chunk_sizes=[128],
            seq_len=4096,
        )
    )

    return {config.name: config for config in configs}


MODEL_CONFIGS = generate_model_configs(MESH_CONFIG)


# Accuracy threshold constants
DEFAULT_PCC_THRESHOLD = 0.994
DEFAULT_RMSE_THRESHOLD = 0.05
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32

from tests.nightly.sdpa_perf_utils import (
    ARCH_CONSTANTS,
    post_process_ops_log,
    compute_sdpa_flops,
    compute_math_utilization as compute_ring_joint_utilization,
    create_balanced_chunk_order,
    reorder_tensor_chunks,
    reverse_reorder_tensor_chunks,
)


def fa_rand(*shape):
    """
    Generate random tensors with Flash Attention-style distribution.
    """
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def torch_sdpa_reference(q, k, v, is_causal=False):
    """
    Memory-efficient PyTorch reference for ring joint attention.

    Chunks over heads and the combined Q sequence so the [B, H, Sq, Sk]
    attention matrix never materializes at full size — CPU SDPA's math
    kernel otherwise allocates it in fp32 and OOMs on long sequences.
    """
    SEQ_CHUNK = 4096
    HEAD_CHUNK = 16

    B, H, total_seq, _ = q.shape
    Dv = v.shape[-1]

    def take_heads(t, h_start, h_end):
        # MLA broadcasts a single KV head across all Q heads via expand (no copy).
        if t.shape[1] != H:
            return t.expand(B, h_end - h_start, -1, -1)
        return t[:, h_start:h_end]

    if total_seq <= SEQ_CHUNK and H <= HEAD_CHUNK:
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q,
            take_heads(k, 0, H),
            take_heads(v, 0, H),
            is_causal=is_causal,
        )
    else:
        attn_out = torch.empty(B, H, total_seq, Dv, dtype=q.dtype)
        for h_start in range(0, H, HEAD_CHUNK):
            h_end = min(h_start + HEAD_CHUNK, H)
            q_heads = q[:, h_start:h_end]
            k_heads = take_heads(k, h_start, h_end)
            v_heads = take_heads(v, h_start, h_end)
            for seq_start in range(0, total_seq, SEQ_CHUNK):
                seq_end = min(seq_start + SEQ_CHUNK, total_seq)
                q_chunk = q_heads[:, :, seq_start:seq_end]
                if is_causal:
                    q_pos = torch.arange(seq_start, seq_end).unsqueeze(1)
                    k_pos = torch.arange(seq_end).unsqueeze(0)
                    mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q_chunk, k_heads[:, :, :seq_end], v_heads[:, :, :seq_end], attn_mask=mask
                    )
                else:
                    out = torch.nn.functional.scaled_dot_product_attention(q_chunk, k_heads, v_heads)
                attn_out[:, h_start:h_end, seq_start:seq_end] = out

    return attn_out


def nd_sharded_dram_memory_config(device, head_dim):
    num_dram_banks = device.dram_grid_size().x
    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(num_dram_banks)
    ]
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)


# ============================================================================
# TEST CASE GENERATION
# ============================================================================


def get_test_case_id(config: ModelConfig, q_chunk_size: int, k_chunk_size: int) -> str:
    """Generate a unique test case ID based on model config and chunk sizes."""
    return f"{config.name}-q{q_chunk_size}-k{k_chunk_size}"


def generate_test_configs(mesh_config: MeshConfig, model_configs: Dict[str, ModelConfig]):
    """
    Generate (b, sq, nhq, nhk, nhv, d_q, d_k, d_v, q_chunk_size, k_chunk_size, is_causal, is_balanced, q_dtype, kv_dtype) tuples for all model configs.

    Each model defines its own Q/K chunk sizes, so the cross-product is per-model.

    NOTE: Uses detect_devices_without_opening() to avoid holding device locks
    during pytest collection, which would block subprocess profiling.
    """
    if mesh_config.num_devices < 2:
        return [], []

    configs = []
    config_ids = []

    for _, model in model_configs.items():
        for q_chunk, k_chunk in product(model.q_chunk_sizes, model.k_chunk_sizes):
            configs.append(
                (
                    BATCH_SIZE,
                    model.seq_len * mesh_config.sp_size,  # Global sequence length across all devices in the ring
                    model.nhq * mesh_config.tp_size,  # Total query heads across all TP shards
                    model.nhk
                    * (
                        mesh_config.tp_size if model.nhk != 1 else 1
                    ),  # Total key heads across all TP shards (handle nhk=1 case for MLA)
                    model.nhv * mesh_config.tp_size,  # Total value heads across all TP shards
                    model.d_q,
                    model.d_k,
                    model.d_v,
                    q_chunk,
                    k_chunk,
                    model.is_causal,
                    model.is_balanced,
                    model.q_dtype,
                    model.kv_dtype,
                )
            )
            config_ids.append(get_test_case_id(model, q_chunk, k_chunk))

    return configs, config_ids


def create_global_semaphores(mesh_device, cores, initial_value):
    """Create global semaphore handles for CCL coordination."""
    return [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]


def to_balanced_growing_cache_layout(src_full, sp_size, chunk_size, last_uploaded_chunk):
    """Pack a growing chunked K/V cache into the per-device slab layout expected by ring joint SDPA.

    "Balanced" here names the chunked cache storage layout, not the causal zigzag work balancing
    controlled by the is_balanced flag.
    """
    n_populated = last_uploaded_chunk + 1
    slab_rows = chunk_size // sp_size
    K_local_curr = n_populated * slab_rows
    populated_len = n_populated * chunk_size
    b, nh, _, d = src_full.shape
    perm = torch.zeros(b, nh, populated_len, d, dtype=src_full.dtype, device=src_full.device)
    for dev in range(sp_size):
        for chunk in range(n_populated):
            local_start = dev * K_local_curr + chunk * slab_rows
            global_start = chunk * chunk_size + dev * slab_rows
            perm[:, :, local_start : local_start + slab_rows, :] = src_full[
                :, :, global_start : global_start + slab_rows, :
            ]
    return perm


def call_sdpa(
    tt_q,
    tt_k,
    tt_v,
    logical_n,
    is_causal,
    is_balanced,
    p_buf_k,
    p_buf_v,
    program_config,
    compute_kernel_config,
    ccl_semaphore_handles,
    num_links,
    sp_axis,
    mesh_device,
    topology,
    worker_sub_device_id,
    ccl_column,
    *,
    scale=None,
    cache_batch_idx=None,
    kv_actual_isl=None,
):
    tt_out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        None,
        None,
        None,
        persistent_output_buffer_k=p_buf_k,
        persistent_output_buffer_v=p_buf_v,
        joint_strategy="rear",
        logical_n=logical_n,
        is_causal=is_causal,
        is_balanced=is_balanced,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl_semaphore_handles,
        num_links=num_links,
        cluster_axis=sp_axis,
        mesh_device=mesh_device,
        topology=topology,
        subdevice_id=worker_sub_device_id,
        ccl_core_grid_offset=(ccl_column, 0),
        use_column_major_ccl=True,
        scale=scale,
        cache_batch_idx=cache_batch_idx,
        kv_actual_isl=kv_actual_isl,
    )
    return tt_out


def run_ring_joint_sdpa(
    mesh_config,
    b,
    nhq,
    nhk,
    sq,
    d_q,
    q_chunk_size,
    k_chunk_size,
    q_dtype,
    *,
    nhv=None,
    d_k=None,
    d_v=None,
    kv_dtype=None,
    is_causal=False,
    is_balanced=False,
    pcc_threshold=DEFAULT_PCC_THRESHOLD,
    rmse_threshold=None,
    do_check=True,
    num_iterations=1,
):
    """
    Run Ring Joint Attention SDPA using direct ttnn operations with auto-detected devices.

    Args:
        b: Batch size (typically 1)
        nhq: Number of query heads
        nhk: Number of key heads (can be 1 for MLA)
        sq: Base sequence length (will be distributed across ring)
        d_q: Query head dimension
        q_chunk_size: Query chunk size for tiling
        k_chunk_size: Key chunk size for tiling
        q_dtype: Data type for Q tensor (e.g., ttnn.bfloat16)
        nhv: Number of value heads (defaults to nhq)
        d_k: Key head dimension (defaults to d_q)
        d_v: Value head dimension (defaults to d_q)
        kv_dtype: Data type for K/V tensors (defaults to q_dtype, can be ttnn.bfloat8_b for MLA)
        is_causal: Whether to use causal attention mask
        is_balanced: Whether to use balanced zigzag work distribution (for causal attention)
        pcc_threshold: Pearson correlation threshold for accuracy
        rmse_threshold: Root mean square error threshold
        do_check: Whether to verify accuracy against PyTorch reference
        num_iterations: Number of times to run the op (>1 for determinism testing)
    """
    # Apply defaults for optional parameters
    if nhv is None:
        nhv = nhq
    if d_k is None:
        d_k = d_q
    if d_v is None:
        d_v = d_q
    if kv_dtype is None:
        kv_dtype = q_dtype

    logger.debug(
        f"run_ring_joint_sdpa params: b={b}, nhq={nhq}, nhk={nhk}, nhv={nhv}, "
        f"sq={sq}, d_q={d_q}, d_k={d_k}, d_v={d_v}, "
        f"q_chunk={q_chunk_size}, k_chunk={k_chunk_size}, "
        f"q_dtype={q_dtype}, kv_dtype={kv_dtype}, "
        f"is_causal={is_causal}, is_balanced={is_balanced}, "
        f"pcc_threshold={pcc_threshold}, rmse_threshold={rmse_threshold}, "
        f"do_check={do_check}, num_iterations={num_iterations}"
    )

    # Ensure reproducible results
    torch.manual_seed(1234)

    # Validate head count constraints
    # For WAN: nhq == nhk == nhv (standard attention)
    # For MLA: nhk == 1, nhq == nhv (multi-latent attention with single K head)
    if nhk != 1 and nhq != nhk:
        pytest.skip(f"Ring joint attention requires nhq == nhk or nhk == 1, got nhq={nhq}, nhk={nhk}")

    # Auto-detect mesh configuration based on available devices

    # Ring topology requires >2 devices; fall back to linear for <=2
    use_ring = mesh_config.sp_size > 2
    fabric_config = ttnn.FabricConfig.FABRIC_1D_RING if use_ring else ttnn.FabricConfig.FABRIC_1D
    topology = Topology.Ring if use_ring else Topology.Linear

    # Configure fabric for ring joint attention
    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )

    # Mesh axis configuration
    sp_axis = 1  # Column axis for sequence parallel (ring axis)
    tp_axis = 0  # Row axis for tensor parallel (head axis)

    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    # Open mesh device based on calculated configuration
    mesh_shape = ttnn.MeshShape(mesh_config.tp_size, mesh_config.sp_size)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    num_links = 2

    try:
        if mesh_config.tp_size > 1 and nhq % mesh_config.tp_size != 0:
            pytest.skip(
                f"num_heads ({nhq}) must be divisible by TP size ({mesh_config.tp_size}) for multi-ring architecture"
            )

        # Configure compute grid and CCL coordination
        sdpa_compute_grid = (mesh_config.sdpa_cols, mesh_config.grid_rows)
        ccl_column = mesh_config.ccl_column

        # Get actual device grid for sub-device creation
        full_compute_grid = mesh_device.compute_with_storage_grid_size()

        # Create sub-device for CCL operations - Must include ALL cores that operations will use
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
        )
        worker_sub_device = ttnn.SubDevice(
            [
                ccl_sub_device_crs,
            ]
        )
        worker_sub_device_id = ttnn.SubDeviceId(0)

        # Set up sub-device manager with stall group
        sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
        sub_device_stall_group = [worker_sub_device_id]
        mesh_device.set_sub_device_stall_group(sub_device_stall_group)

        ccl_semaphore_handles = create_global_semaphores(mesh_device, ccl_sub_device_crs, 0)

        # Create tensors with appropriate shapes
        # Q: [b, nhq, sq, d_q]
        # K: [b, nhk, sq, d_k] - nhk can be 1 for MLA (broadcast to all Q heads)
        # V: [b, nhv, sq, d_v] - nhv typically equals nhq
        Q = fa_rand(b, nhq, sq, d_q)
        K = fa_rand(b, nhk, sq, d_k)
        V = fa_rand(b, nhv, sq, d_v)

        # Keep original tensors for reference comparison (before any reordering)
        Q_original, K_original, V_original = Q, K, V

        # Apply balanced reordering if enabled (for causal attention workload balancing)
        chunk_order = None
        if is_balanced:
            chunk_order = create_balanced_chunk_order(mesh_config.sp_size)
            Q = reorder_tensor_chunks(Q, chunk_order)
            K = reorder_tensor_chunks(K, chunk_order)
            V = reorder_tensor_chunks(V, chunk_order)

        # Create persistent output buffers
        kv_shard_dims = [None, None]
        kv_shard_dims[sp_axis] = None
        if mesh_config.tp_size > 1:
            kv_shard_dims[tp_axis] = 1

        # Persistent K output buffer uses nhk and d_k dimensions
        # Persistent V output buffer uses nhv and d_v dimensions
        expected_output_seq_len = sq

        # For K buffer: handle nhk=1 case (MLA) - may need different sharding
        persistent_k_shard_dims = [None, None]
        persistent_k_shard_dims[sp_axis] = None
        if mesh_config.tp_size > 1 and nhk != 1:
            persistent_k_shard_dims[tp_axis] = 1

        persistent_output_buffer_k = ttnn.from_torch(
            torch.zeros(b, nhk, expected_output_seq_len, d_k),
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_k_shard_dims
            ),
        )
        persistent_output_buffer_v = ttnn.from_torch(
            torch.zeros(b, nhv, expected_output_seq_len, d_v),
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kv_shard_dims),
        )

        # Create program config
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_compute_grid,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        # BH adaptation: use init_device_compute_kernel_config instead of WormholeComputeKernelConfig
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Convert to TT tensors with appropriate mesh sharding
        sdpa_input_shard_dims = [None, None]
        sdpa_input_shard_dims[sp_axis] = 2
        if mesh_config.tp_size > 1:
            sdpa_input_shard_dims[tp_axis] = 1

        # K tensor may have nhk=1 (MLA), different sharding
        sdpa_k_shard_dims = [None, None]
        sdpa_k_shard_dims[sp_axis] = 2
        if mesh_config.tp_size > 1 and nhk != 1:
            sdpa_k_shard_dims[tp_axis] = 1

        # Q tensor uses q_dtype
        tt_Q = ttnn.from_torch(
            Q,
            dtype=q_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
            ),
        )
        # K, V tensors use kv_dtype (can be bfloat8_b for MLA)
        tt_K = ttnn.from_torch(
            K,
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_k_shard_dims
            ),
        )
        tt_V = ttnn.from_torch(
            V,
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
            ),
        )

        # Set logical_n to the original full sequence length
        corrected_logical_n = sq

        # Precompute mesh composer dims
        main_row_dim = sdpa_input_shard_dims[0] if sdpa_input_shard_dims[0] is not None else -1
        main_col_dim = sdpa_input_shard_dims[1] if sdpa_input_shard_dims[1] is not None else -1

        # Run ring joint attention
        reference_output = None
        for i in range(num_iterations):
            tt_out = call_sdpa(
                tt_Q,
                tt_K,
                tt_V,
                corrected_logical_n,
                is_causal=is_causal,
                is_balanced=is_balanced,
                p_buf_k=persistent_output_buffer_k,
                p_buf_v=persistent_output_buffer_v,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                ccl_semaphore_handles=ccl_semaphore_handles,
                num_links=num_links,
                sp_axis=sp_axis,
                mesh_device=mesh_device,
                topology=topology,
                worker_sub_device_id=worker_sub_device_id,
                ccl_column=ccl_column,
            )

            # Convert main output to torch and slice out tile-padding
            tt_out_torch = ttnn.to_torch(
                tt_out,
                mesh_composer=ttnn.create_mesh_composer(
                    mesh_device, ttnn.MeshComposerConfig(main_row_dim, main_col_dim)
                ),
            )
            tt_out_torch = tt_out_torch[:, :, :sq, :]

            # Reverse balanced reordering if enabled (restore original sequence order)
            if is_balanced and chunk_order is not None:
                tt_out_torch = reverse_reorder_tensor_chunks(tt_out_torch, chunk_order)

            # Determinism mode: compare each output to the first
            if num_iterations > 1:
                if reference_output is None:
                    reference_output = tt_out_torch
                elif not torch.equal(reference_output, tt_out_torch):
                    diff_mask = reference_output != tt_out_torch
                    num_diffs = diff_mask.sum().item()
                    max_diff = (reference_output - tt_out_torch).abs().max().item()
                    pytest.fail(
                        f"Ring joint SDPA output at iteration {i} differs from iteration 0: "
                        f"{num_diffs} differing elements, max diff = {max_diff}"
                    )

        if num_iterations > 1:
            logger.info(f"Ring joint SDPA determinism verified: all {num_iterations} outputs are exactly equal")
            return

        if not do_check:
            return

        # Compute PyTorch reference on ORIGINAL data (before balanced reordering)
        gt_main = torch_sdpa_reference(Q_original, K_original, V_original, is_causal=is_causal)

        # Verify accuracy for main output
        out_pass_main, out_pcc_main = comp_pcc(gt_main, tt_out_torch, pcc_threshold)
        rmse_main = torch.sqrt(((gt_main - tt_out_torch) ** 2).mean()).item()
        logger.info(f"Main output - PCC: {out_pcc_main}, RMSE: {rmse_main:.6f}")

        if rmse_threshold is not None:
            assert rmse_main < rmse_threshold, f"Main RMSE {rmse_main:.6f} exceeds threshold {rmse_threshold}"
        assert out_pass_main, f"Main PCC {out_pcc_main} below threshold {pcc_threshold}"

    finally:
        # Clean up mesh device
        ttnn.close_mesh_device(mesh_device)

        # Restore fabric to disabled state
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ============================================================================
# CHUNKED-PREFILL VALIDATION
# ============================================================================
CHUNKED_PREFILL_PER_DEVICE_CHUNK = 640
CHUNKED_PREFILL_N_CHUNKS = 11
CHUNKED_PREFILL_CHUNK_SIZE = CHUNKED_PREFILL_PER_DEVICE_CHUNK * MESH_CONFIG.sp_size
CHUNKED_PREFILL_TOTAL_SEQ = CHUNKED_PREFILL_CHUNK_SIZE * CHUNKED_PREFILL_N_CHUNKS
CHUNKED_PREFILL_PCC_THRESHOLD = 0.99
# Q/V heads are sharded across tp_axis, so every device in a ring holds the same
# head shard => heads-per-ring == heads-per-device. nhq/nhv below are PER RING; the
# run multiplies by tp_size for the total head count (e.g. 16 per ring => 64 total on
# galaxy 4x8), matching the per-ring convention used by the sweep configs.
CHUNKED_PREFILL_HEADS_PER_RING = 16
CHUNKED_PREFILL_SEED = 1234


def run_ring_joint_sdpa_chunked(
    mesh_config,
    model: ModelConfig,
    chunk_size: int = CHUNKED_PREFILL_CHUNK_SIZE,
    total_seq: int = CHUNKED_PREFILL_TOTAL_SEQ,
    pcc_threshold: float = CHUNKED_PREFILL_PCC_THRESHOLD,
    rmse_threshold: float = None,
    q_chunk_size: int = None,
    k_chunk_size: int = None,
    num_iterations: int = 1,
    indexed_nd_sharded_kv_cache: bool = False,
    cache_batch_idx: int = 1,
    cache_batch: int = 2,
    persistent_buffer_mode: str = "exact_per_chunk",
):
    """
    Validate ring joint SDPA chunked-prefill against a full-sequence torch oracle.

    SUT: n_chunks calls; each call passes a short Q chunk at absolute positions
    [i*c, (i+1)*c) against a K/V cache holding the first (i+1)*c rows.
    If persistent_buffer_mode="reuse_max", one max-length persistent K/V buffer
    pair is allocated once and reused across all chunks and iterations.

    num_iterations > 1 switches to determinism mode: replay the full n_chunks
    sequence num_iterations times and require bit-exact equality of per-chunk
    outputs to iteration 0. PCC check is skipped in determinism mode.
    """
    torch.manual_seed(CHUNKED_PREFILL_SEED)

    sp_size = mesh_config.sp_size
    if sp_size < 2:
        pytest.skip(f"Ring joint chunked prefill requires at least 2 devices in ring, got SP={sp_size}")

    assert total_seq % sp_size == 0, f"total_seq {total_seq} must divide sp_size {sp_size}"
    assert chunk_size % sp_size == 0, f"chunk_size {chunk_size} must divide sp_size {sp_size}"
    assert total_seq % chunk_size == 0, f"total_seq {total_seq} must be a multiple of chunk_size {chunk_size}"
    assert persistent_buffer_mode in (
        "exact_per_chunk",
        "reuse_max",
    ), f"Unsupported persistent_buffer_mode={persistent_buffer_mode}"
    if indexed_nd_sharded_kv_cache:
        assert BATCH_SIZE == 1, "Indexed K/V cache test path assumes query batch is 1"
        assert 0 <= cache_batch_idx < cache_batch, f"cache_batch_idx {cache_batch_idx} must be in [0, {cache_batch})"
        assert (
            persistent_buffer_mode == "exact_per_chunk"
        ), "Reusable max buffers are not combined with indexed K/V cache"

    n_chunks = total_seq // chunk_size

    if q_chunk_size is None:
        q_chunk_size = model.q_chunk_sizes[0]
    if k_chunk_size is None:
        k_chunk_size = model.k_chunk_sizes[0]
    # model.nhq/nhk/nhv are PER RING; scale to total head counts across all TP shards
    # (mirrors the sweep convention; nhk=1 stays 1 for MLA's single shared K head).
    nhq = model.nhq * mesh_config.tp_size
    nhk = model.nhk * (mesh_config.tp_size if model.nhk != 1 else 1)
    nhv = model.nhv * mesh_config.tp_size
    d_q, d_k, d_v = model.d_q, model.d_k, model.d_v
    q_dtype, kv_dtype = model.q_dtype, model.kv_dtype
    is_balanced = False

    b = BATCH_SIZE

    use_ring = sp_size > 2
    fabric_config = ttnn.FabricConfig.FABRIC_1D_RING if use_ring else ttnn.FabricConfig.FABRIC_1D
    topology = Topology.Ring if use_ring else Topology.Linear

    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )

    sp_axis = 1
    tp_axis = 0
    num_links = 2

    logger.info(f"[chunked-test] opening mesh device shape=({mesh_config.tp_size}, {sp_size})")
    mesh_shape = ttnn.MeshShape(mesh_config.tp_size, sp_size)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    logger.info("[chunked-test] mesh device opened")

    try:
        sdpa_compute_grid = (mesh_config.sdpa_cols, mesh_config.grid_rows)
        ccl_column = mesh_config.ccl_column
        full_compute_grid = mesh_device.compute_with_storage_grid_size()

        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
        )
        worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        logger.info("[chunked-test] creating sub-device manager")
        sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
        mesh_device.set_sub_device_stall_group([worker_sub_device_id])
        logger.info("[chunked-test] sub-device manager loaded")

        logger.info("[chunked-test] creating global semaphores")
        ccl_semaphore_handles = create_global_semaphores(mesh_device, ccl_sub_device_crs, 0)
        logger.info("[chunked-test] global semaphores created")

        torch.manual_seed(CHUNKED_PREFILL_SEED)
        logger.info(f"[chunked-test] building torch tensors (total_seq={total_seq}, nhq={nhq}, nhk={nhk}, nhv={nhv})")
        Q_full = fa_rand(b, nhq, total_seq, d_q)
        K_full = fa_rand(b, nhk, total_seq, d_k)
        V_full = fa_rand(b, nhv, total_seq, d_v)
        logger.info("[chunked-test] computing torch reference (full causal SDPA)")
        ref_full = torch_sdpa_reference(Q_full, K_full, V_full, is_causal=True)
        logger.info("[chunked-test] torch reference computed")

        sdpa_input_shard_dims = [None, None]
        sdpa_input_shard_dims[sp_axis] = 2
        if mesh_config.tp_size > 1:
            sdpa_input_shard_dims[tp_axis] = 1

        sdpa_k_shard_dims = [None, None]
        sdpa_k_shard_dims[sp_axis] = 2
        if mesh_config.tp_size > 1 and nhk != 1:
            sdpa_k_shard_dims[tp_axis] = 1

        persistent_k_shard_dims = [None, None]
        if mesh_config.tp_size > 1 and nhk != 1:
            persistent_k_shard_dims[tp_axis] = 1

        # K can be nhk=1 and replicated across TP; V follows nhv/nhq and stays TP-sharded.
        kv_persistent_shard_dims = [None, None]
        if mesh_config.tp_size > 1:
            kv_persistent_shard_dims[tp_axis] = 1

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_compute_grid,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        main_row_dim = sdpa_input_shard_dims[0] if sdpa_input_shard_dims[0] is not None else -1
        main_col_dim = sdpa_input_shard_dims[1] if sdpa_input_shard_dims[1] is not None else -1

        def upload_q(q_host):
            return ttnn.from_torch(
                q_host,
                dtype=q_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
                ),
            )

        def upload_k(k_host, memory_config=None):
            kwargs = {}
            if memory_config is not None:
                kwargs["memory_config"] = memory_config
            return ttnn.from_torch(
                k_host,
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_k_shard_dims
                ),
                **kwargs,
            )

        def upload_v(v_host, memory_config=None):
            kwargs = {}
            if memory_config is not None:
                kwargs["memory_config"] = memory_config
            return ttnn.from_torch(
                v_host,
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
                ),
                **kwargs,
            )

        def to_host(tt_out, expected_q_len):
            out = ttnn.to_torch(
                tt_out,
                mesh_composer=ttnn.create_mesh_composer(
                    mesh_device, ttnn.MeshComposerConfig(main_row_dim, main_col_dim)
                ),
            )
            return out[:, :, :expected_q_len, :]

        def create_persistent_buffers(seq_len, kv_buffer_batch):
            persistent_output_buffer_k = ttnn.from_torch(
                torch.zeros(kv_buffer_batch, nhk, seq_len, d_k),
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_k_shard_dims
                ),
            )
            persistent_output_buffer_v = ttnn.from_torch(
                torch.zeros(kv_buffer_batch, nhv, seq_len, d_v),
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kv_persistent_shard_dims
                ),
            )
            return persistent_output_buffer_k, persistent_output_buffer_v

        logger.info(
            f"Chunked prefill: model={model.name}, total_seq={total_seq}, "
            f"sp_size={sp_size}, per-device Q seq_len={total_seq // sp_size}, "
            f"q_chunk={q_chunk_size}, k_chunk={k_chunk_size}, "
            f"indexed_nd_sharded_kv_cache={indexed_nd_sharded_kv_cache}, "
            f"persistent_buffer_mode={persistent_buffer_mode}"
        )

        # Chunked K/V cache layout: device d's local slab c holds global rows
        # [c*chunk_size + d*slab_rows, c*chunk_size + (d+1)*slab_rows). This packing is
        # independent of the is_balanced causal zigzag mode; the cache grows one chunk per call.
        slab_rows = chunk_size // sp_size
        assert chunk_size % sp_size == 0, f"chunk_size {chunk_size} not divisible by sp_size {sp_size}"
        assert slab_rows % 32 == 0, f"slab_rows {slab_rows} not tile-aligned (TILE_HEIGHT=32)"

        k_memory_config = nd_sharded_dram_memory_config(mesh_device, d_k) if indexed_nd_sharded_kv_cache else None
        v_memory_config = nd_sharded_dram_memory_config(mesh_device, d_v) if indexed_nd_sharded_kv_cache else None
        shared_persistent_buffers = (
            create_persistent_buffers(total_seq, b) if persistent_buffer_mode == "reuse_max" else None
        )

        # SUT: per-chunk calls with growing K/V cache + growing logical_n.
        # In determinism mode (num_iterations > 1), the entire n_chunks sequence is
        # replayed and per-chunk outputs from iteration 0 are compared bit-exact.
        reference_outputs = None
        per_chunk_results = []
        logger.info(f"[chunked-test] starting chunk loop: n_chunks={n_chunks}, num_iterations={num_iterations}")
        for it in range(num_iterations):
            iter_outputs = [] if num_iterations > 1 else None
            for i in range(n_chunks):
                s, e = i * chunk_size, (i + 1) * chunk_size
                logger.info(f"[chunked-test] iter={it} chunk={i} range=[{s}, {e}) starting")

                K_balanced = to_balanced_growing_cache_layout(K_full, sp_size, chunk_size, i)
                V_balanced = to_balanced_growing_cache_layout(V_full, sp_size, chunk_size, i)
                Q_chunk = Q_full[:, :, s:e, :].contiguous()
                kv_buffer_batch = b
                cache_batch_idx_arg = None

                if indexed_nd_sharded_kv_cache:
                    kv_buffer_batch = cache_batch
                    cache_batch_idx_arg = cache_batch_idx
                    K_input = torch.randn(cache_batch, nhk, e, d_k, dtype=K_balanced.dtype) * 100
                    V_input = torch.randn(cache_batch, nhv, e, d_v, dtype=V_balanced.dtype) * 100
                    K_input[cache_batch_idx : cache_batch_idx + 1] = K_balanced
                    V_input[cache_batch_idx : cache_batch_idx + 1] = V_balanced
                else:
                    K_input = K_balanced
                    V_input = V_balanced

                logger.info(f"[chunked-test] iter={it} chunk={i} uploading Q/K/V to device")
                tt_Q = upload_q(Q_chunk)
                tt_K = upload_k(K_input, memory_config=k_memory_config)
                tt_V = upload_v(V_input, memory_config=v_memory_config)
                logger.info(f"[chunked-test] iter={it} chunk={i} Q/K/V uploaded")

                if persistent_buffer_mode == "reuse_max":
                    persistent_output_buffer_k, persistent_output_buffer_v = shared_persistent_buffers
                else:
                    # AllGather output buffer sized to post-gather K/V: N_global == (i+1) * chunk_size.
                    persistent_output_buffer_k, persistent_output_buffer_v = create_persistent_buffers(
                        e, kv_buffer_batch
                    )

                logger.info(f"[chunked-test] iter={it} chunk={i} calling SDPA op (logical_n={e})")
                try:
                    tt_out = call_sdpa(
                        tt_Q,
                        tt_K,
                        tt_V,
                        e,
                        is_causal=True,
                        is_balanced=is_balanced,
                        p_buf_k=persistent_output_buffer_k,
                        p_buf_v=persistent_output_buffer_v,
                        program_config=program_config,
                        compute_kernel_config=compute_kernel_config,
                        ccl_semaphore_handles=ccl_semaphore_handles,
                        num_links=num_links,
                        sp_axis=sp_axis,
                        mesh_device=mesh_device,
                        topology=topology,
                        worker_sub_device_id=worker_sub_device_id,
                        ccl_column=ccl_column,
                        cache_batch_idx=cache_batch_idx_arg,
                    )
                except Exception as exc:
                    pytest.fail(
                        f"Chunked prefill SDPA call raised on iter {it}, chunk {i} "
                        f"(Q rows [{s}, {e}), logical_n={e}): {type(exc).__name__}: {exc}"
                    )
                logger.info(f"[chunked-test] iter={it} chunk={i} SDPA op returned, reading back to host")

                out_i = to_host(tt_out, chunk_size)
                logger.info(f"[chunked-test] iter={it} chunk={i} readback complete")

                if num_iterations > 1:
                    iter_outputs.append(out_i)
                    continue

                expected_i = ref_full[:, :, s:e, :]
                passed, pcc = comp_pcc(expected_i, out_i, pcc_threshold)
                rmse = torch.sqrt(((expected_i - out_i) ** 2).mean()).item()
                rmse_passed = rmse_threshold is None or rmse < rmse_threshold
                logger.info(
                    f"Chunk {i:2d} [{s:6d}, {e:6d}) logical_n={e}: PCC={pcc} RMSE={rmse:.6f} "
                    f"-> {'PASS' if passed and rmse_passed else 'FAIL'}"
                )
                per_chunk_results.append((i, e, passed, pcc, rmse_passed, rmse))

            if num_iterations > 1:
                if reference_outputs is None:
                    reference_outputs = iter_outputs
                else:
                    diffs = []
                    for i, (ref, cur) in enumerate(zip(reference_outputs, iter_outputs)):
                        if not torch.equal(ref, cur):
                            num_diffs = (ref != cur).sum().item()
                            max_diff = (ref - cur).abs().max().item()
                            diffs.append((i, num_diffs, max_diff))
                            logger.warning(f"  iter {it} chunk {i}: {num_diffs} diffs, max={max_diff}")
                    if diffs:
                        details = "; ".join(f"chunk {i}: {n} diffs, max={m}" for i, n, m in diffs)
                        pytest.fail(f"Chunked prefill determinism failed at iter {it}: {details}")

        if num_iterations > 1:
            logger.info(
                f"Chunked prefill determinism verified: all {num_iterations} runs of {n_chunks} chunks are exactly equal"
            )
            return

        failures = [
            (i, e, pcc, rmse)
            for i, e, passed, pcc, rmse_passed, rmse in per_chunk_results
            if not passed or not rmse_passed
        ]
        if failures:
            details = "; ".join(
                f"chunk {i} (logical_n={e}): PCC={pcc}, RMSE={rmse:.6f}" for i, e, pcc, rmse in failures
            )
            pytest.fail(
                f"Chunked prefill PCC/RMSE failures "
                f"(pcc_threshold={pcc_threshold}, rmse_threshold={rmse_threshold}): {details}"
            )

    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def torch_chunked_causal_sdpa_reference(q, k, v, q_abs_offset, scale=None):
    """Chunked causal SDPA reference where Q rows start at absolute position q_abs_offset."""
    nhq = q.shape[1]
    k = k.expand(-1, nhq, -1, -1) if k.shape[1] == 1 else k
    v = v.expand(-1, nhq, -1, -1) if v.shape[1] == 1 else v
    if scale is None:
        scale = q.shape[-1] ** -0.5

    sq = q.shape[2]
    sk = k.shape[2]
    scores = (q.float() @ k.transpose(-2, -1).float()) * scale
    q_pos = torch.arange(sq) + q_abs_offset
    k_pos = torch.arange(sk)
    scores = scores.masked_fill(k_pos.unsqueeze(0) > q_pos.unsqueeze(1), float("-inf"))
    return (torch.softmax(scores, dim=-1) @ v.float()).to(q.dtype)


def kv_pad_rotation_destinations(kv_actual_isl, new_actual_isl, sp_size, chunk_size_local):
    """Map current valid tokens into the slab-major cache layout used by kv_actual_isl."""
    chunk_size_global = sp_size * chunk_size_local
    q_fill_cursor = [0] * sp_size
    destinations = []

    for token_idx in range(new_actual_isl):
        global_pos = kv_actual_isl + token_idx
        group = global_pos // chunk_size_global
        within_group = global_pos % chunk_size_global
        dev = within_group // chunk_size_local
        cell = within_group % chunk_size_local
        cache_row = group * chunk_size_local + cell
        q_row = q_fill_cursor[dev]
        assert q_row < chunk_size_local, "current chunk overfilled one device's Q slab"
        q_fill_cursor[dev] += 1
        destinations.append((token_idx, dev, cache_row, q_row, global_pos))

    return destinations


def build_kv_pad_rotation_inputs(
    old_cache_k,
    old_cache_v,
    new_tokens_q,
    new_tokens_k,
    new_tokens_v,
    kv_actual_isl,
    sp_size,
    chunk_size_local,
):
    """Build padded Q/K/V host tensors for one KV-pad-aware Ring SDPA call."""
    assert old_cache_k.shape[0] == 1
    assert old_cache_v.shape[0] == 1
    assert new_tokens_q.shape[0] == 1

    nhq = new_tokens_q.shape[1]
    nhk = new_tokens_k.shape[1]
    nhv = new_tokens_v.shape[1]
    d_q = new_tokens_q.shape[3]
    d_k = new_tokens_k.shape[3]
    d_v = new_tokens_v.shape[3]
    new_actual_isl = new_tokens_q.shape[2]
    chunk_size_global = sp_size * chunk_size_local
    logical_n = kv_actual_isl + new_actual_isl
    num_cache_slabs = max(2, math.ceil(logical_n / chunk_size_global))
    cache_seq_per_dev = num_cache_slabs * chunk_size_local

    k_per_dev = torch.zeros(sp_size, nhk, cache_seq_per_dev, d_k, dtype=torch.bfloat16)
    v_per_dev = torch.zeros(sp_size, nhv, cache_seq_per_dev, d_v, dtype=torch.bfloat16)
    for global_pos in range(kv_actual_isl):
        group = global_pos // chunk_size_global
        within_group = global_pos % chunk_size_global
        dev = within_group // chunk_size_local
        cell = within_group % chunk_size_local
        cache_row = group * chunk_size_local + cell
        k_per_dev[dev, :, cache_row, :] = old_cache_k[0, :, global_pos, :]
        v_per_dev[dev, :, cache_row, :] = old_cache_v[0, :, global_pos, :]

    q_per_dev = torch.zeros(sp_size, nhq, chunk_size_local, d_q, dtype=torch.bfloat16)
    q_global_pos_per_dev = [[None] * chunk_size_local for _ in range(sp_size)]
    destinations = kv_pad_rotation_destinations(kv_actual_isl, new_actual_isl, sp_size, chunk_size_local)
    assert len(destinations) == new_actual_isl
    for token_idx, dev, cache_row, q_row, global_pos in destinations:
        k_per_dev[dev, :, cache_row, :] = new_tokens_k[0, :, token_idx, :]
        v_per_dev[dev, :, cache_row, :] = new_tokens_v[0, :, token_idx, :]
        q_per_dev[dev, :, q_row, :] = new_tokens_q[0, :, token_idx, :]
        q_global_pos_per_dev[dev][q_row] = global_pos

    q_host = q_per_dev.permute(1, 0, 2, 3).reshape(1, nhq, chunk_size_global, d_q)
    k_host = k_per_dev.permute(1, 0, 2, 3).reshape(1, nhk, sp_size * cache_seq_per_dev, d_k)
    v_host = v_per_dev.permute(1, 0, 2, 3).reshape(1, nhv, sp_size * cache_seq_per_dev, d_v)

    combined_q_global_pos = []
    for dev in range(sp_size):
        combined_q_global_pos.extend(q_global_pos_per_dev[dev])

    valid_rows = [None] * new_actual_isl
    for row, q_pos in enumerate(combined_q_global_pos):
        if q_pos is not None:
            valid_rows[q_pos - kv_actual_isl] = row
    assert all(row is not None for row in valid_rows)

    return q_host, k_host, v_host, valid_rows, num_cache_slabs


def run_ring_joint_sdpa_kv_pad_rotation_case(
    mesh_config,
    kv_actual_isl,
    new_actual_isl,
    seed,
    chunk_size_local=64,
    pcc_threshold=CHUNKED_PREFILL_PCC_THRESHOLD,
    rmse_threshold=DEFAULT_RMSE_THRESHOLD,
):
    sp_size = mesh_config.sp_size
    if sp_size < 2:
        pytest.skip(f"Ring joint KV-pad rotation requires at least 2 devices in ring, got SP={sp_size}")

    tile_height = 32
    assert chunk_size_local % tile_height == 0
    assert kv_actual_isl % tile_height == 0
    assert new_actual_isl % tile_height == 0

    b = BATCH_SIZE
    local_heads = 4
    nhq = local_heads * mesh_config.tp_size
    nhk = 1
    nhv = nhq
    d_q = 64
    d_k = 64
    d_v = 32
    q_dtype = ttnn.bfloat16
    kv_dtype = ttnn.bfloat16
    logical_n = kv_actual_isl + new_actual_isl
    chunk_size_global = chunk_size_local * sp_size

    torch.manual_seed(seed)
    old_cache_k = fa_rand(b, nhk, kv_actual_isl, d_k)
    old_cache_v = fa_rand(b, nhv, kv_actual_isl, d_v)
    new_tokens_q = fa_rand(b, nhq, new_actual_isl, d_q)
    new_tokens_k = fa_rand(b, nhk, new_actual_isl, d_k)
    new_tokens_v = fa_rand(b, nhv, new_actual_isl, d_v)

    q_host, k_host, v_host, valid_rows, num_cache_slabs = build_kv_pad_rotation_inputs(
        old_cache_k,
        old_cache_v,
        new_tokens_q,
        new_tokens_k,
        new_tokens_v,
        kv_actual_isl,
        sp_size,
        chunk_size_local,
    )
    cache_seq_per_dev = num_cache_slabs * chunk_size_local

    use_ring = sp_size > 2
    fabric_config = ttnn.FabricConfig.FABRIC_1D_RING if use_ring else ttnn.FabricConfig.FABRIC_1D
    topology = Topology.Ring if use_ring else Topology.Linear
    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )

    sp_axis = 1
    tp_axis = 0
    num_links = 2
    mesh_shape = ttnn.MeshShape(mesh_config.tp_size, sp_size)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    try:
        sdpa_compute_grid = (mesh_config.sdpa_cols, mesh_config.grid_rows)
        ccl_column = mesh_config.ccl_column
        full_compute_grid = mesh_device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
        )
        worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
        mesh_device.set_sub_device_stall_group([worker_sub_device_id])
        ccl_semaphore_handles = create_global_semaphores(mesh_device, ccl_sub_device_crs, 0)

        sdpa_input_shard_dims = [None, None]
        sdpa_input_shard_dims[sp_axis] = 2
        if mesh_config.tp_size > 1:
            sdpa_input_shard_dims[tp_axis] = 1

        sdpa_k_shard_dims = [None, None]
        sdpa_k_shard_dims[sp_axis] = 2

        # KV-pad rotation uses nhk=1, so K stays TP-replicated while V is TP-sharded with Q/nhv.
        persistent_k_shard_dims = [None, None]
        kv_persistent_shard_dims = [None, None]
        if mesh_config.tp_size > 1:
            kv_persistent_shard_dims[tp_axis] = 1

        def upload(host_tensor, dtype, shard_dims):
            return ttnn.from_torch(
                host_tensor,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
            )

        tt_q = upload(q_host, q_dtype, sdpa_input_shard_dims)
        tt_k = upload(k_host, kv_dtype, sdpa_k_shard_dims)
        tt_v = upload(v_host, kv_dtype, sdpa_input_shard_dims)

        persistent_output_buffer_k = ttnn.from_torch(
            torch.zeros(b, nhk, sp_size * cache_seq_per_dev, d_k),
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_k_shard_dims
            ),
        )
        persistent_output_buffer_v = ttnn.from_torch(
            torch.zeros(b, nhv, sp_size * cache_seq_per_dev, d_v),
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kv_persistent_shard_dims
            ),
        )

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_compute_grid,
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
        )
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        logger.info(
            f"KV-pad rotation: sp_size={sp_size}, chunk_size_global={chunk_size_global}, "
            f"kv_actual_isl={kv_actual_isl}, new_actual_isl={new_actual_isl}, cache_slabs={num_cache_slabs}"
        )

        tt_out = call_sdpa(
            tt_q,
            tt_k,
            tt_v,
            logical_n,
            is_causal=True,
            is_balanced=False,
            p_buf_k=persistent_output_buffer_k,
            p_buf_v=persistent_output_buffer_v,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            ccl_semaphore_handles=ccl_semaphore_handles,
            num_links=num_links,
            sp_axis=sp_axis,
            mesh_device=mesh_device,
            topology=topology,
            worker_sub_device_id=worker_sub_device_id,
            ccl_column=ccl_column,
            kv_actual_isl=kv_actual_isl,
        )

        main_row_dim = sdpa_input_shard_dims[0] if sdpa_input_shard_dims[0] is not None else -1
        main_col_dim = sdpa_input_shard_dims[1] if sdpa_input_shard_dims[1] is not None else -1
        out_host = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(main_row_dim, main_col_dim)),
        )
        out_valid_natural = out_host[:, :, valid_rows, :]

        natural_k = torch.cat([old_cache_k, new_tokens_k], dim=2)
        natural_v = torch.cat([old_cache_v, new_tokens_v], dim=2)
        ref_out = torch_chunked_causal_sdpa_reference(new_tokens_q, natural_k, natural_v, kv_actual_isl)
        passed, pcc = comp_pcc(ref_out, out_valid_natural, pcc_threshold)
        rmse = torch.sqrt(((ref_out - out_valid_natural) ** 2).mean()).item()
        rmse_passed = rmse_threshold is None or rmse < rmse_threshold
        logger.info(f"KV-pad rotation PCC={pcc}, RMSE={rmse:.6f}")
        assert passed and rmse_passed, (
            f"KV-pad rotation accuracy failed: PCC={pcc} (threshold={pcc_threshold}), "
            f"RMSE={rmse:.6f} (threshold={rmse_threshold})"
        )

    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.timeout(600)
def test_ring_joint_attention_chunked_nd_sharded_indexed_kv_cache_accuracy():
    """Validate chunked direct ND-sharded K/V cache inputs selected by cache_batch_idx."""
    mesh_config = MESH_CONFIG
    local_heads = 4
    chunk_seq_len = 64
    total_seq_len = 128
    model = ModelConfig(
        name="mla_indexed_nd_sharded_kv",
        nhq=local_heads,
        nhk=1,
        nhv=local_heads,
        d_q=64,
        d_k=64,
        d_v=32,
        is_causal=True,
        is_balanced=False,
        q_dtype=ttnn.bfloat16,
        kv_dtype=ttnn.bfloat16,
        q_chunk_sizes=[32],
        k_chunk_sizes=[32],
        seq_len=total_seq_len,
    )

    run_ring_joint_sdpa_chunked(
        mesh_config,
        model,
        chunk_size=chunk_seq_len * mesh_config.sp_size,
        total_seq=total_seq_len * mesh_config.sp_size,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        q_chunk_size=model.q_chunk_sizes[0],
        k_chunk_size=model.k_chunk_sizes[0],
        indexed_nd_sharded_kv_cache=True,
    )


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "case_name",
    [
        "cold_start_one_slot",
        "partial_old_pad",
        "no_old_pad",
        "multi_slab_wrap",
    ],
)
def test_ring_joint_attention_kv_pad_aware_rotation_accuracy(case_name):
    """Validate kv_actual_isl pad-aware rotation against a natural-order causal reference."""
    mesh_config = MESH_CONFIG
    chunk_size_local = 64
    chunk_size_global = chunk_size_local * mesh_config.sp_size
    cases = {
        "cold_start_one_slot": (0, 64),
        "partial_old_pad": (32, 64),
        "no_old_pad": (chunk_size_global, 64),
        "multi_slab_wrap": (2 * chunk_size_global - 32, 64),
    }
    kv_actual_isl, new_actual_isl = cases[case_name]
    run_ring_joint_sdpa_kv_pad_rotation_case(
        mesh_config,
        kv_actual_isl=kv_actual_isl,
        new_actual_isl=new_actual_isl,
        seed=4321 + list(cases).index(case_name),
        chunk_size_local=chunk_size_local,
    )


# Generate test parameters dynamically based on detected hardware for different models (WAN, MLA, VideGen...)
TEST_CONFIGS, TEST_CONFIG_IDS = generate_test_configs(MESH_CONFIG, MODEL_CONFIGS)
TEST_CONFIG_MODELS = list(MODEL_CONFIGS.keys())


# === TEST 1: PERFORMANCE SWEEP (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.parametrize(
    "b,sq,nhq,nhk,nhv,d_q,d_k,d_v,q_chunk_size,k_chunk_size,is_causal,is_balanced,q_dtype,kv_dtype",
    TEST_CONFIGS,
    ids=TEST_CONFIG_IDS,
)
def test_ring_joint_attention_sdpa_sweep_perf_impl(
    b,
    sq,
    nhq,
    nhk,
    nhv,
    d_q,
    d_k,
    d_v,
    q_chunk_size,
    k_chunk_size,
    is_causal,
    is_balanced,
    q_dtype,
    kv_dtype,
):
    """
    Performance sweep test for ring joint attention SDPA.
    Skipped on CI - run locally for performance measurement.
    Supports both WAN and MLA configurations.
    """
    mesh_config = MESH_CONFIG

    run_ring_joint_sdpa(
        mesh_config,
        b,
        nhq,
        nhk,
        sq,
        d_q,
        q_chunk_size,
        k_chunk_size,
        q_dtype,
        nhv=nhv,
        d_k=d_k,
        d_v=d_v,
        kv_dtype=kv_dtype,
        is_causal=is_causal,
        is_balanced=is_balanced,
        do_check=False,
    )


# === TEST 2: ACCURACY VERIFICATION ===
@pytest.mark.parametrize(
    "b,sq,nhq,nhk,nhv,d_q,d_k,d_v,q_chunk_size,k_chunk_size,is_causal,is_balanced,q_dtype,kv_dtype",
    TEST_CONFIGS,
    ids=TEST_CONFIG_IDS,
)
def test_ring_joint_attention_sdpa_accuracy(
    b,
    sq,
    nhq,
    nhk,
    nhv,
    d_q,
    d_k,
    d_v,
    q_chunk_size,
    k_chunk_size,
    is_causal,
    is_balanced,
    q_dtype,
    kv_dtype,
):
    """
    Accuracy verification test for ring joint attention SDPA.
    Supports both WAN and MLA configurations.

    ACCURACY METRICS:
    - PCC (Pearson Correlation Coefficient): Measures linear correlation
    - RMSE (Root Mean Square Error): Measures absolute error magnitude

    THRESHOLD RATIONALE:
    - PCC = 0.994: Relaxed for joint attention complexity
    """
    mesh_config = MESH_CONFIG

    pcc_threshold = DEFAULT_PCC_THRESHOLD
    rmse_threshold = DEFAULT_RMSE_THRESHOLD
    run_ring_joint_sdpa(
        mesh_config,
        b,
        nhq,
        nhk,
        sq,
        d_q,
        q_chunk_size,
        k_chunk_size,
        q_dtype,
        nhv=nhv,
        d_k=d_k,
        d_v=d_v,
        kv_dtype=kv_dtype,
        is_causal=is_causal,
        is_balanced=is_balanced,
        pcc_threshold=pcc_threshold,
        rmse_threshold=rmse_threshold,
    )


# === TEST 3: DETERMINISM VERIFICATION ===
@pytest.mark.parametrize(
    "b,sq,nhq,nhk,nhv,d_q,d_k,d_v,q_chunk_size,k_chunk_size,is_causal,is_balanced,q_dtype,kv_dtype",
    TEST_CONFIGS,
    ids=TEST_CONFIG_IDS,
)
def test_ring_joint_attention_sdpa_determinism(
    b,
    sq,
    nhq,
    nhk,
    nhv,
    d_q,
    d_k,
    d_v,
    q_chunk_size,
    k_chunk_size,
    is_causal,
    is_balanced,
    q_dtype,
    kv_dtype,
):
    """
    Test ring joint attention SDPA determinism: run 10 times with same inputs and verify outputs match exactly.
    """
    mesh_config = MESH_CONFIG

    num_iterations = 10
    run_ring_joint_sdpa(
        mesh_config,
        b,
        nhq,
        nhk,
        sq,
        d_q,
        q_chunk_size,
        k_chunk_size,
        q_dtype,
        nhv=nhv,
        d_k=d_k,
        d_v=d_v,
        kv_dtype=kv_dtype,
        is_causal=is_causal,
        is_balanced=is_balanced,
        num_iterations=num_iterations,
    )


# === TEST 4: PERFORMANCE TABLE GENERATOR (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.timeout(1000)
@pytest.mark.parametrize("model_name", TEST_CONFIG_MODELS)
def test_ring_joint_attention_create_perf_table(model_name):
    """
    Sweep chunk sizes for ring joint attention SDPA and print a performance table.
    Skipped on CI - run locally with tracy profiler.
    """
    from tracy.process_model_log import run_device_profiler

    mesh_config = MESH_CONFIG
    model_configs = MODEL_CONFIGS

    ring_size = mesh_config.sp_size

    if ring_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices, got {ring_size}")

    # Generate test configs for this model
    test_configs, test_config_ids = generate_test_configs(mesh_config, model_configs)
    sweep_configs = [
        (config, config_id)
        for config, config_id in zip(test_configs, test_config_ids)
        if config_id.startswith(model_name)
    ]

    # Look up model configuration
    model = model_configs[model_name]

    # Use hardware config values (cannot query device due to TLB conflicts with subprocess tests)
    full_grid_rows = mesh_config.grid_rows
    total_compute_cores = mesh_config.sdpa_cores
    total_cores = mesh_config.total_cores

    ccl_cores = full_grid_rows  # Full column height for CCL

    subdir = "ttnn_ring_joint_sdpa_performance"
    perf_results = []

    for config, config_id in sweep_configs:
        float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]", "PM FPU UTIL (%)"]
        cols = ["ATTRIBUTES"]

        command = (
            f"pytest tests/nightly/blackhole/sdpa/"
            f"test_ring_joint_sdpa.py::"
            f"test_ring_joint_attention_sdpa_sweep_perf_impl"
            f"[{config_id}]"
        )

        (
            b,
            sq,
            nhq,
            nhk,
            nhv,
            d_q,
            d_k,
            d_v,
            q_chunk_size,
            k_chunk_size,
            is_causal,
            is_balanced,
            q_dtype,
            kv_dtype,
        ) = config

        # Config now contains global (TP/SP-scaled) values
        # Convert to local (per-device) values for performance calculations
        s = sq  # sq is already global sequence length
        local_seq_len = sq // ring_size
        local_nhq = nhq // mesh_config.tp_size

        try:
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
            r = post_process_ops_log(
                subdir,
                float_columns=float_cols,
                columns=cols,
                op_name="RingJointSDPADeviceOperation",
                sum_vals=False,
                has_signposts=False,
            )

            measured_core_count = int(r["CORE COUNT"][0]) if len(r["CORE COUNT"]) > 0 else 0
            duration_ns = (
                int(r["DEVICE KERNEL DURATION [ns]"].max()) if len(r["DEVICE KERNEL DURATION [ns]"]) > 0 else 0
            )
            fpu_util_col = r.get("PM FPU UTIL (%)", [])
            fpu_util_min = float(fpu_util_col.min()) if len(fpu_util_col) > 0 else 0.0
            fpu_util_max = float(fpu_util_col.max()) if len(fpu_util_col) > 0 else 0.0

            B = b
            local_q_num_chunks = math.ceil(local_seq_len / q_chunk_size)
            local_k_num_chunks = math.ceil(local_seq_len / k_chunk_size)
            # Each ring step iterates over the device's local K shard (padded up to k_chunk_size),
            # so total K chunks traversed is ring_size * local_k_num_chunks — not ceil(s / k_chunk_size),
            # which would amortize per-device padding globally.
            k_num_chunks = ring_size * local_k_num_chunks

            total_work_items = B * local_nhq * local_q_num_chunks
            q_per_core = (
                math.ceil(total_work_items / total_compute_cores) if total_compute_cores > 0 else total_work_items
            )
            iters_per_core = q_per_core * k_num_chunks

            # Padding waste
            local_q_padded = local_q_num_chunks * q_chunk_size
            global_q_padded = local_q_padded * ring_size
            local_k_padded = local_k_num_chunks * k_chunk_size
            global_k_padded = local_k_padded * ring_size
            actual_work = s * s
            padded_work = global_q_padded * global_k_padded
            total_waste_pct = ((padded_work - actual_work) / padded_work) * 100 if padded_work > 0 else 0

            # Slot waste: leftover capacity after distributing (b, h, q) work items flatly across all cores.
            total_q_slots = q_per_core * total_compute_cores
            wasted_q_slots = max(0, total_q_slots - total_work_items)
            slot_waste_pct = (wasted_q_slots / total_q_slots) * 100 if total_q_slots > 0 else 0

            # Math utilization — tracy reports SDPA + CCL cores together; strip the CCL contribution
            # by rounding down to the nearest multiple of grid_rows (CCL adds < grid_rows cores).
            effective_cores = (measured_core_count // mesh_config.grid_rows) * mesh_config.grid_rows
            heads_per_device = local_nhq
            utilization = compute_ring_joint_utilization(
                local_seq_len, s, d_q, d_v, heads_per_device, duration_ns, effective_cores, is_causal
            )

            perf_results.append(
                {
                    "q_chunk_size": q_chunk_size,
                    "k_chunk_size": k_chunk_size,
                    "cores_used": effective_cores,
                    "iters_per_core": iters_per_core,
                    "total_waste_pct": total_waste_pct,
                    "slot_waste_pct": slot_waste_pct,
                    "duration_ns": duration_ns,
                    "duration_ms": duration_ns / 1e6,
                    "utilization": utilization,
                    "fpu_util_min": fpu_util_min,
                    "fpu_util_max": fpu_util_max,
                }
            )
            logger.info(
                f"q={q_chunk_size}, k={k_chunk_size}: {duration_ns/1e6:.3f} ms, "
                f"util={utilization:.1f}%, cores={effective_cores}/{total_compute_cores}, "
                f"iters/core={iters_per_core}"
            )

        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            logger.error(
                f"Error running ring joint SDPA with q_chunk_size={q_chunk_size}, k_chunk_size={k_chunk_size}: {e}"
            )
            perf_results.append(
                {
                    "q_chunk_size": q_chunk_size,
                    "k_chunk_size": k_chunk_size,
                    "duration_ns": None,
                }
            )

    # Sort by duration (best first)
    valid_results = [r for r in perf_results if r["duration_ns"] is not None]
    valid_results.sort(key=lambda x: x["duration_ns"])

    mm_flops = compute_sdpa_flops(s, s, d_q, d_v, nhq, is_causal)

    # Print summary table
    print(f"\n{'='*150}")
    print(
        f"Ring Joint Attention Performance Sweep ({model_name.upper()}): b={b}, nh={nhq} (global), s={s}, d_q={d_q}, d_v={d_v}, causal={is_causal}"
    )
    print(f"Architecture: {mesh_config.arch_type}, Ring size: {ring_size} devices, TP size: {mesh_config.tp_size}")
    print(f"Total MM FLOPs (all devices): {mm_flops:,} ({mm_flops/1e9:.2f} GFLOPs)")
    print(f"Per-device workload: Q={s // ring_size} tokens, K/V={s} tokens (via ring), {local_nhq} heads")
    print(f"Core Allocation: {total_compute_cores} compute + {ccl_cores} CCL = {total_cores} total cores")
    print(f"{'='*150}")
    header = "| Rank | q_chunk | k_chunk | Duration (ms) | Cores Used | Iters/Core | Pad Waste | Slot Waste | FPU Util (%)  | Math Util |"
    sep = "|------|---------|---------|---------------|------------|------------|-----------|------------|---------------|-----------|"
    print(header)
    print(sep)

    for rank, result in enumerate(valid_results, 1):
        fpu_range = f"{result['fpu_util_min']:.1f}-{result['fpu_util_max']:.1f}"
        print(
            f"| {rank:4d} | {result['q_chunk_size']:7d} | {result['k_chunk_size']:7d} | {result['duration_ms']:13.3f} | "
            f"{result['cores_used']:10d} | {result['iters_per_core']:10d} | "
            f"{result['total_waste_pct']:8.1f}% | {result['slot_waste_pct']:9.1f}% | {fpu_range:>13} | {result['utilization']:8.1f}% |"
        )

    failed_results = [r for r in perf_results if r["duration_ns"] is None]
    if failed_results:
        print(f"\nFailed configurations:")
        for result in failed_results:
            print(f"  q_chunk_size={result['q_chunk_size']}, k_chunk_size={result['k_chunk_size']}")

    if valid_results:
        best = valid_results[0]
        print(
            f"\nBest configuration: q_chunk_size={best['q_chunk_size']}, "
            f"k_chunk_size={best['k_chunk_size']} "
            f"({best['duration_ms']:.3f} ms, {best['utilization']:.1f}% math util, "
            f"{best['cores_used']}/{total_compute_cores} compute cores, {best['iters_per_core']} iters/core, "
            f"{best['total_waste_pct']:.1f}% pad waste, {best['slot_waste_pct']:.1f}% slot waste)"
        )

    print(f"{'='*150}\n")


# === TEST 5: PERFORMANCE CHECK (CI-gated by SDPA_PERF_CHECKS=1) ===
# Symmetric +/- band — catches both regressions and unexpected speedups.
RING_JOINT_PERF_MARGIN = 0.005

RING_JOINT_PERF_CHECK_CONFIGS = [
    # (model_name, q_chunk_size, k_chunk_size, ring_size, expected_util)
    # 4-device ring (QuietBox)
    ("wan2_2_1xGLX", 288, 512, 4, 68.9),
    ("mla_100k", 160, 320, 4, 63.2),
]


@pytest.mark.skipif(
    os.environ.get("SDPA_PERF_CHECKS") != "1",
    reason="Set SDPA_PERF_CHECKS=1 to run (CI: sdpa perf tests job)",
)
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "model_name, q_chunk_size, k_chunk_size, ring_size_expected, expected_util",
    RING_JOINT_PERF_CHECK_CONFIGS,
    ids=[f"{cfg[0]}-q{cfg[1]}-k{cfg[2]}-ring{cfg[3]}" for cfg in RING_JOINT_PERF_CHECK_CONFIGS],
)
def test_ring_joint_attention_perf_check(model_name, q_chunk_size, k_chunk_size, ring_size_expected, expected_util):
    """Measure ring joint SDPA math utilization via tracy and assert within +/- RING_JOINT_PERF_MARGIN."""
    from tracy.process_model_log import run_device_profiler

    if MESH_CONFIG.sp_size != ring_size_expected:
        pytest.skip(f"Expected ring size {ring_size_expected}, current topology has ring size {MESH_CONFIG.sp_size}")

    if model_name not in MODEL_CONFIGS:
        pytest.skip(f"Model {model_name} not available for current mesh config")

    model = MODEL_CONFIGS[model_name]
    config_id = get_test_case_id(model, q_chunk_size, k_chunk_size)

    sq = model.seq_len * MESH_CONFIG.sp_size
    local_seq_len = model.seq_len
    local_nhq = model.nhq

    subdir = "ttnn_ring_joint_sdpa_perf_check"
    command = (
        f"pytest tests/nightly/blackhole/sdpa/"
        f"test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_sweep_perf_impl"
        f"[{config_id}]"
    )

    float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
    cols = ["ATTRIBUTES"]

    with mock.patch.dict(os.environ, {"CI": "false"}):
        run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log(
        subdir, float_columns=float_cols, columns=cols, op_name="", sum_vals=False, has_signposts=False
    )

    assert (
        len(r["CORE COUNT"]) > 0 and len(r["DEVICE KERNEL DURATION [ns]"]) > 0
    ), "profiler returned no SDPA ops — inner test was skipped or did not produce a kernel run"

    measured_core_count = int(r["CORE COUNT"][0])
    duration_ns = int(r["DEVICE KERNEL DURATION [ns]"].max())

    # Match perf-table effective_cores rounding (ignore non-multiple-of-10 strays)
    effective_cores = measured_core_count - measured_core_count % 10
    assert (
        effective_cores > 0
    ), f"effective_cores=0 (measured_core_count={measured_core_count}) — profiler output incomplete"

    utilization = compute_ring_joint_utilization(
        local_seq_len, sq, model.d_q, model.d_v, local_nhq, duration_ns, effective_cores, model.is_causal
    )

    lower = expected_util * (1 - RING_JOINT_PERF_MARGIN)
    upper = expected_util * (1 + RING_JOINT_PERF_MARGIN)

    logger.info(
        f"Ring joint SDPA perf check {config_id}: "
        f"duration={duration_ns/1e6:.3f} ms, math_util={utilization:.2f}% "
        f"(expected {expected_util:.2f}%, band [{lower:.2f}, {upper:.2f}])"
    )

    assert lower <= utilization <= upper, (
        f"Math utilization {utilization:.2f}% outside band [{lower:.2f}, {upper:.2f}] "
        f"(expected {expected_util:.2f}%, margin +/- {RING_JOINT_PERF_MARGIN*100:.1f}%)"
    )


# === TEST 6: CHUNKED-PREFILL ACCURACY ===
# Chunked-prefill tests use a kimi-K2.6-style MLA config (16 Q heads, single KV head,
# DeepSeek V3 dims) kept separate from the global MODEL_CONFIGS so the kimi parameters
# don't leak into the non-chunked sweep/perf/determinism tests.
CHUNKED_PREFILL_MODEL_CONFIGS = {
    "kimi50k": ModelConfig(
        name="kimi50k",
        nhq=CHUNKED_PREFILL_HEADS_PER_RING,
        nhk=1,
        nhv=CHUNKED_PREFILL_HEADS_PER_RING,
        d_q=576,
        d_k=576,
        d_v=128,
        is_causal=True,
        is_balanced=True,
        q_dtype=ttnn.bfloat16,
        kv_dtype=ttnn.bfloat8_b,
        q_chunk_sizes=[64],
        k_chunk_sizes=[128, 256, 512],
        seq_len=CHUNKED_PREFILL_CHUNK_SIZE,  # unused by chunked path
    ),
}
CHUNKED_PREFILL_MODELS = list(CHUNKED_PREFILL_MODEL_CONFIGS.keys())


def _generate_chunked_configs():
    configs = []
    ids = []
    for model_name in CHUNKED_PREFILL_MODELS:
        model = CHUNKED_PREFILL_MODEL_CONFIGS[model_name]
        for q, k in product(model.q_chunk_sizes, model.k_chunk_sizes):
            configs.append((model_name, q, k))
            ids.append(f"{model_name}-q{q}-k{k}")
    return configs, ids


CHUNKED_CONFIGS, CHUNKED_CONFIG_IDS = _generate_chunked_configs()


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("chunk_size", [CHUNKED_PREFILL_CHUNK_SIZE], ids=[f"chunk{CHUNKED_PREFILL_CHUNK_SIZE}"])
@pytest.mark.parametrize(
    "model_name,q_chunk_size,k_chunk_size",
    CHUNKED_CONFIGS,
    ids=CHUNKED_CONFIG_IDS,
)
def test_ring_joint_attention_sdpa_chunked_accuracy(model_name, q_chunk_size, k_chunk_size, chunk_size):
    """Validate ring joint SDPA chunked prefill with reusable max-sized K/V buffers."""
    mesh_config = MESH_CONFIG

    run_ring_joint_sdpa_chunked(
        mesh_config,
        CHUNKED_PREFILL_MODEL_CONFIGS[model_name],
        chunk_size=chunk_size,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        persistent_buffer_mode="reuse_max",
    )


# === TEST 7: CHUNKED-PREFILL DETERMINISM ===
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("chunk_size", [CHUNKED_PREFILL_CHUNK_SIZE], ids=[f"chunk{CHUNKED_PREFILL_CHUNK_SIZE}"])
@pytest.mark.parametrize(
    "model_name,q_chunk_size,k_chunk_size",
    CHUNKED_CONFIGS,
    ids=CHUNKED_CONFIG_IDS,
)
def test_ring_joint_attention_sdpa_chunked_determinism(model_name, q_chunk_size, k_chunk_size, chunk_size):
    """Replay ring joint SDPA chunked prefill 3 times and require bit-exact per-chunk outputs."""
    mesh_config = MESH_CONFIG

    run_ring_joint_sdpa_chunked(
        mesh_config,
        CHUNKED_PREFILL_MODEL_CONFIGS[model_name],
        chunk_size=chunk_size,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        num_iterations=3,
    )


# === TEST 8: CHUNKED-PREFILL PERF TABLE (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("chunk_size", [CHUNKED_PREFILL_CHUNK_SIZE], ids=[f"chunk{CHUNKED_PREFILL_CHUNK_SIZE}"])
@pytest.mark.parametrize(
    "model_name,q_chunk_size,k_chunk_size",
    CHUNKED_CONFIGS,
    ids=CHUNKED_CONFIG_IDS,
)
def test_ring_joint_attention_create_chunked_perf_table(model_name, q_chunk_size, k_chunk_size, chunk_size):
    """Run chunked prefill once with tracy and print a per-chunk math-util table.

    Per-chunk work is rectangle (Q_chunk vs prefix K/V, non-causal) + triangle (Q_chunk vs
    current K/V, causal half), so later chunks have a larger prefix and should reach higher
    math utilization than chunk 0 (which is only the triangle).
    """
    from tracy.process_model_log import run_device_profiler

    mesh_config = MESH_CONFIG
    ring_size = mesh_config.sp_size

    if ring_size < 2:
        pytest.skip(f"Ring joint chunked prefill requires at least 2 devices, got {ring_size}")

    model = CHUNKED_PREFILL_MODEL_CONFIGS[model_name]
    total_seq = CHUNKED_PREFILL_TOTAL_SEQ
    n_chunks = total_seq // chunk_size

    config_id = f"{model_name}-q{q_chunk_size}-k{k_chunk_size}-chunk{chunk_size}"
    subdir = "ttnn_ring_joint_sdpa_chunked_performance"
    command = (
        f"pytest tests/nightly/blackhole/sdpa/"
        f"test_ring_joint_sdpa.py::"
        f"test_ring_joint_attention_sdpa_chunked_accuracy"
        f"[{config_id}]"
    )

    float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]", "PM FPU UTIL (%)"]
    cols = ["ATTRIBUTES"]

    run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log(
        subdir,
        float_columns=float_cols,
        columns=cols,
        op_name="RingJointSDPADeviceOperation",
        sum_vals=False,
        has_signposts=False,
    )

    durations = r["DEVICE KERNEL DURATION [ns]"].tolist()
    core_counts = r["CORE COUNT"].tolist()
    fpu_util_col = r.get("PM FPU UTIL (%)", [])

    # Tracy emits one entry per (chunk, device). Group every devs_per_chunk consecutive
    # entries into a chunk and take the max duration (critical path: chunk completes when
    # the slowest device finishes).
    assert (
        len(durations) % n_chunks == 0
    ), f"RingJointSDPADeviceOperation entry count ({len(durations)}) is not a multiple of n_chunks ({n_chunks})"
    devs_per_chunk = len(durations) // n_chunks
    expected_devs = mesh_config.tp_size * mesh_config.sp_size
    assert (
        devs_per_chunk == expected_devs
    ), f"Expected {expected_devs} entries per chunk (tp_size * sp_size), got {devs_per_chunk}"

    chunk_durations = [max(durations[i * devs_per_chunk : (i + 1) * devs_per_chunk]) for i in range(n_chunks)]
    chunk_core_counts = [
        max(int(c) for c in core_counts[i * devs_per_chunk : (i + 1) * devs_per_chunk]) for i in range(n_chunks)
    ]

    q_per_dev = chunk_size // ring_size
    # model.nhq is PER RING, and heads-per-ring == heads-per-device (heads shard across tp_axis).
    nh_per_dev = model.nhq
    d_q, d_v = model.d_q, model.d_v
    constants = ARCH_CONSTANTS["blackhole"]
    clock_ghz = constants["clock_ghz"]
    flops_per_cycle_per_core = constants["mm_flops_per_cycle_per_core"]

    per_chunk_rows = []
    for i, (dur_ns, ccount) in enumerate(zip(chunk_durations, chunk_core_counts)):
        prefix_k = i * chunk_size
        # Rectangle: Q_chunk (q_per_dev rows on this device) vs prefix K/V (i * chunk_size rows), non-causal.
        rect_flops = 2 * q_per_dev * prefix_k * (d_q + d_v) * nh_per_dev
        # Triangle: Q_chunk vs current chunk K/V, causal => c*c/2 valid (q,k) pairs.
        tri_flops = q_per_dev * chunk_size * (d_q + d_v) * nh_per_dev
        chunk_flops = rect_flops + tri_flops

        # Strip CCL contribution: round measured core count down to multiple of grid_rows.
        effective_cores = (ccount // mesh_config.grid_rows) * mesh_config.grid_rows
        cycles = dur_ns * clock_ghz
        theoretical_flops = effective_cores * cycles * flops_per_cycle_per_core
        util = (chunk_flops / theoretical_flops) * 100 if theoretical_flops > 0 else 0.0

        per_chunk_rows.append(
            {
                "chunk": i,
                "logical_n": (i + 1) * chunk_size,
                "prefix_k": prefix_k,
                "duration_ns": int(dur_ns),
                "cores": effective_cores,
                "chunk_flops": chunk_flops,
                "util": util,
            }
        )

    print(f"\n{'='*130}")
    print(
        f"Ring Joint Chunked-Prefill Per-Chunk Math Util: model={model_name}, "
        f"q_chunk={q_chunk_size}, k_chunk={k_chunk_size}, chunk_size={chunk_size}, total_seq={total_seq}"
    )
    print(f"Architecture: {mesh_config.arch_type}, Ring size: {ring_size}, TP size: {mesh_config.tp_size}")
    print(f"Per-device per-chunk Q rows: {q_per_dev}, heads/device: {nh_per_dev}, d_q={d_q}, d_v={d_v}")
    print(f"{'='*130}")
    header = "| Chunk | logical_n | prefix_K | Duration (ms) | Cores | Chunk FLOPs (G) | Math Util |"
    sep = "|-------|-----------|----------|---------------|-------|-----------------|-----------|"
    print(header)
    print(sep)
    for row in per_chunk_rows:
        print(
            f"| {row['chunk']:5d} | {row['logical_n']:9d} | {row['prefix_k']:8d} | "
            f"{row['duration_ns']/1e6:13.3f} | {row['cores']:5d} | {row['chunk_flops']/1e9:15.2f} | "
            f"{row['util']:8.1f}% |"
        )

    if len(fpu_util_col) > 0:
        print(
            f"\nTracy PM FPU UTIL range across all SDPA cores/chunks: "
            f"{float(fpu_util_col.min()):.1f}% - {float(fpu_util_col.max()):.1f}%"
        )

    utils = [row["util"] for row in per_chunk_rows]
    assert all(0.0 <= u <= 100.0 for u in utils), f"Math util out of [0, 100]: {[f'{u:.1f}' for u in utils]}"
    assert utils[-1] > utils[0], (
        f"Expected last chunk util ({utils[-1]:.1f}%) > first chunk util ({utils[0]:.1f}%) "
        f"— prefix grows with chunk index, so util should increase."
    )
