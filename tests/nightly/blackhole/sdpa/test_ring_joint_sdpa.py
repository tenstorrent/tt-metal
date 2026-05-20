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

from tests.nightly.sdpa_perf_utils import (
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


def make_paged_kv_cache(
    cache,
    page_block_size,
    shuffle_pages=True,
    permutation=None,
    shuffle_mode="global",
    sp_size=1,
    window_start_idx=0,
    window_seq_len=None,
):
    """Convert [B, H, S, D] K/V cache to paged [blocks, H, block, D] plus page table."""
    b, nheads, seq_len, head_dim = cache.shape
    assert seq_len % page_block_size == 0
    max_num_blocks_per_seq = seq_len // page_block_size
    max_num_blocks = b * max_num_blocks_per_seq
    window_start_page = window_start_idx // page_block_size
    if window_seq_len is None:
        window_end_page = max_num_blocks_per_seq
    else:
        window_end_page = math.ceil((window_start_idx + window_seq_len) / page_block_size)
    window_pages_per_seq = window_end_page - window_start_page
    assert window_end_page <= max_num_blocks_per_seq

    if permutation is None:
        if not shuffle_pages:
            permutation = torch.arange(max_num_blocks)
        elif shuffle_mode == "global":
            permutation = torch.randperm(max_num_blocks)
        elif shuffle_mode == "rank_local":
            assert window_start_idx % page_block_size == 0
            if window_seq_len is not None:
                assert window_seq_len % page_block_size == 0
            assert max_num_blocks_per_seq % sp_size == 0
            pages_per_rank = max_num_blocks_per_seq // sp_size
            assert window_pages_per_seq % sp_size == 0
            window_pages_per_rank = window_pages_per_seq // sp_size
            assert window_pages_per_rank <= pages_per_rank
            rank_local_permutation = []
            for batch in range(b):
                batch_page_offset = batch * max_num_blocks_per_seq
                batch_permutation = torch.full((max_num_blocks_per_seq,), -1, dtype=torch.long)
                for rank in range(sp_size):
                    rank_page_offset = batch_page_offset + rank * pages_per_rank
                    rank_pages = torch.arange(rank_page_offset, rank_page_offset + pages_per_rank)
                    if window_start_idx == 0 and window_pages_per_seq == max_num_blocks_per_seq:
                        rank_local_permutation.append(rank_pages[torch.randperm(pages_per_rank)])
                    else:
                        logical_window_start = batch_page_offset + window_start_page + rank * window_pages_per_rank
                        logical_pages = torch.arange(logical_window_start, logical_window_start + window_pages_per_rank)
                        physical_pages = rank_pages[torch.randperm(pages_per_rank)[:window_pages_per_rank]]
                        batch_permutation[physical_pages - batch_page_offset] = logical_pages
                if window_start_idx != 0 or window_pages_per_seq != max_num_blocks_per_seq:
                    remaining_physical = (batch_permutation == -1).nonzero(as_tuple=False).flatten()
                    assigned_logical = set(batch_permutation[batch_permutation != -1].tolist())
                    remaining_logical = torch.tensor(
                        [
                            batch_page_offset + page
                            for page in range(max_num_blocks_per_seq)
                            if batch_page_offset + page not in assigned_logical
                        ],
                        dtype=torch.long,
                    )
                    batch_permutation[remaining_physical] = remaining_logical[torch.randperm(len(remaining_logical))]
                    rank_local_permutation.append(batch_permutation)
            permutation = torch.cat(rank_local_permutation)
        else:
            raise ValueError(f"Unsupported shuffle_mode: {shuffle_mode}")
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(b, max_num_blocks_per_seq).to(torch.int32)

    paged_cache = (
        cache.reshape(b, nheads, max_num_blocks_per_seq, page_block_size, head_dim)
        .transpose(1, 2)
        .reshape(max_num_blocks, nheads, page_block_size, head_dim)
    )
    return paged_cache[permutation], page_table, reverse_permutation, permutation


def make_rank_local_permutation(b, max_num_blocks_per_seq, sp_size):
    """Create a deterministic rank-local non-identity permutation for paged K/V program-cache tests."""
    assert max_num_blocks_per_seq % sp_size == 0
    pages_per_rank = max_num_blocks_per_seq // sp_size
    permutation = []
    for batch in range(b):
        batch_page_offset = batch * max_num_blocks_per_seq
        for rank in range(sp_size):
            rank_page_offset = batch_page_offset + rank * pages_per_rank
            rank_pages = torch.arange(rank_page_offset, rank_page_offset + pages_per_rank)
            permutation.append(torch.flip(rank_pages, dims=[0]))
    return torch.cat(permutation)


def unpage_kv_cache(paged_cache, reverse_permutation, b, nheads, seq_len, head_dim, page_block_size):
    max_num_blocks_per_seq = seq_len // page_block_size
    unshuffled = paged_cache[reverse_permutation]
    return (
        unshuffled.reshape(b, max_num_blocks_per_seq, nheads, page_block_size, head_dim)
        .transpose(1, 2)
        .reshape(b, nheads, seq_len, head_dim)
    )


def torch_joint_sdpa_reference(q, k, v, joint_q, joint_k, joint_v, is_causal=False):
    """
    Memory-efficient PyTorch reference for ring joint attention.

    Chunks over heads and the combined Q sequence so the [B, H, Sq, Sk]
    attention matrix never materializes at full size — CPU SDPA's math
    kernel otherwise allocates it in fp32 and OOMs on long sequences.
    """
    SEQ_CHUNK = 4096
    HEAD_CHUNK = 16

    main_seq_len = q.size(2)

    combined_q = torch.cat([q, joint_q], dim=2)
    combined_k = torch.cat([k, joint_k], dim=2)
    combined_v = torch.cat([v, joint_v], dim=2)

    B, H, total_seq, _ = combined_q.shape
    Dv = combined_v.shape[-1]

    def take_heads(t, h_start, h_end):
        # MLA broadcasts a single KV head across all Q heads via expand (no copy).
        if t.shape[1] != H:
            return t.expand(B, h_end - h_start, -1, -1)
        return t[:, h_start:h_end]

    if total_seq <= SEQ_CHUNK and H <= HEAD_CHUNK:
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            combined_q,
            take_heads(combined_k, 0, H),
            take_heads(combined_v, 0, H),
            is_causal=is_causal,
        )
    else:
        attn_out = torch.empty(B, H, total_seq, Dv, dtype=combined_q.dtype)
        for h_start in range(0, H, HEAD_CHUNK):
            h_end = min(h_start + HEAD_CHUNK, H)
            q_heads = combined_q[:, h_start:h_end]
            k_heads = take_heads(combined_k, h_start, h_end)
            v_heads = take_heads(combined_v, h_start, h_end)
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

    return attn_out[:, :, :main_seq_len], attn_out[:, :, main_seq_len:]


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


def choose_paged_kv_page_block_size(local_seq_len: int, k_chunk_size: int):
    """Pick a tile-aligned page size that evenly shards across SP ranks."""
    for page_block_size in (64, 128, 96, 160, 224, 256, 288, 320, 512, 32):
        if page_block_size <= k_chunk_size and local_seq_len % page_block_size == 0:
            return page_block_size
    return None


PAGED_KV_PERF_PAGE_SIZE_CANDIDATES = (
    32,
    64,
    96,
    128,
    160,
    224,
    256,
    288,
    320,
    448,
    512,
    640,
    800,
    896,
    1024,
    1120,
    1280,
    1600,
    1792,
    2240,
    2560,
    3200,
    4096,
)

PAGED_KV_PERF_PAGE_TABLE_MODES = ("identity", "rank_local", "global")

# Keep the page-size sensitivity sweep focused enough for iterative Tracy runs,
# while covering both non-causal WAN and causal balanced MLA families.
PAGED_KV_PAGE_SIZE_PERF_TARGETS = (
    ("wan2_2_4xGLX", 224, 128),
    ("wan2_2_4xGLX", 224, 512),
    ("wan2_2_4xGLX", 288, 128),
    ("wan2_2_4xGLX", 288, 512),
    ("mla_100k", 160, 160),
    ("mla_100k", 160, 256),
    ("mla_100k", 160, 320),
    ("mla_128k", 128, 128),
)


def get_valid_paged_kv_page_block_sizes(seq_len: int, sp_size: int):
    """Return tile-aligned page sizes that divide the global K/V cache evenly across SP ranks."""
    return [
        page_block_size
        for page_block_size in PAGED_KV_PERF_PAGE_SIZE_CANDIDATES
        if page_block_size <= seq_len and seq_len % page_block_size == 0 and (seq_len // page_block_size) % sp_size == 0
    ]


def generate_paged_test_configs(mesh_config: MeshConfig, test_configs, test_config_ids):
    configs = []
    config_ids = []

    for config, config_id in zip(test_configs, test_config_ids):
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
        local_seq_len = sq // mesh_config.sp_size
        page_block_size = choose_paged_kv_page_block_size(local_seq_len, k_chunk_size)
        if page_block_size is None:
            continue
        configs.append(
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
                page_block_size,
            )
        )
        config_ids.append(f"{config_id}-page{page_block_size}")

    return configs, config_ids


def generate_paged_perf_variant_configs(mesh_config: MeshConfig, test_configs, test_config_ids):
    configs = []
    config_ids = []
    selected_targets = set(PAGED_KV_PAGE_SIZE_PERF_TARGETS)

    if mesh_config.num_devices < 2:
        return configs, config_ids

    for config, config_id in zip(test_configs, test_config_ids):
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
        model_name = config_id.rsplit("-q", 1)[0]
        if (model_name, q_chunk_size, k_chunk_size) not in selected_targets:
            continue

        for page_block_size in get_valid_paged_kv_page_block_sizes(sq, mesh_config.sp_size):
            for page_table_mode in PAGED_KV_PERF_PAGE_TABLE_MODES:
                configs.append(
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
                        page_block_size,
                        page_table_mode,
                    )
                )
                config_ids.append(f"{config_id}-page{page_block_size}-{page_table_mode}")

    return configs, config_ids


def create_global_semaphores(mesh_device, cores, initial_value):
    """Create global semaphore handles for CCL coordination."""
    return [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]


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
    use_paged_kv=False,
    page_block_size=64,
    v_page_block_size=None,
    paged_num_blocks_override=None,
    shuffle_pages=True,
    page_shuffle_mode="global",
    chunk_start_idx=0,
    kv_cache_seq_len_override=None,
    vary_page_table_per_iteration=False,
    paged_kv_page_table_is_rank_local=None,
    page_table_dtype=ttnn.int32,
    page_table_layout=ttnn.ROW_MAJOR_LAYOUT,
    page_table_transform=None,
    paged_extra_blocks=0,
    joint_seq_len=0,
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
        v_page_block_size: Optional V page block size override, for validation tests
        paged_num_blocks_override: Optional total paged K/V block count override, for validation tests
        page_table_dtype: Device dtype used for the page table, for validation tests
        page_table_layout: Device layout used for the page table, for validation tests
        page_table_transform: Optional test hook that mutates the generated torch page table before upload
        paged_kv_page_table_is_rank_local: Whether paged logical rank shards map only to physical pages owned by the
            same rank. Defaults from the test page shuffle mode.
        paged_extra_blocks: Extra unused physical K/V cache pages appended after the logical page-table range
        kv_cache_seq_len_override: Optional physical K/V cache sequence length for non-page-aligned window tests
        joint_seq_len: Joint sequence length
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
    if v_page_block_size is None:
        v_page_block_size = page_block_size
    if paged_kv_page_table_is_rank_local is None:
        paged_kv_page_table_is_rank_local = use_paged_kv and (not shuffle_pages or page_shuffle_mode == "rank_local")

    logger.debug(
        f"run_ring_joint_sdpa params: b={b}, nhq={nhq}, nhk={nhk}, nhv={nhv}, "
        f"sq={sq}, d_q={d_q}, d_k={d_k}, d_v={d_v}, "
        f"q_chunk={q_chunk_size}, k_chunk={k_chunk_size}, "
        f"q_dtype={q_dtype}, kv_dtype={kv_dtype}, "
        f"is_causal={is_causal}, is_balanced={is_balanced}, "
        f"pcc_threshold={pcc_threshold}, rmse_threshold={rmse_threshold}, "
        f"do_check={do_check}, num_iterations={num_iterations}, "
        f"use_paged_kv={use_paged_kv}, page_block_size={page_block_size}, v_page_block_size={v_page_block_size}, "
        f"paged_num_blocks_override={paged_num_blocks_override}, shuffle_pages={shuffle_pages}, "
        f"paged_extra_blocks={paged_extra_blocks}, "
        f"page_shuffle_mode={page_shuffle_mode}, chunk_start_idx={chunk_start_idx}, "
        f"kv_cache_seq_len_override={kv_cache_seq_len_override}, "
        f"vary_page_table_per_iteration={vary_page_table_per_iteration}, "
        f"paged_kv_page_table_is_rank_local={paged_kv_page_table_is_rank_local}"
    )

    # Ensure reproducible results
    torch.manual_seed(1234)

    # Validate head count constraints
    # For WAN: nhq == nhk == nhv (standard attention)
    # For MLA: nhk == 1, nhq == nhv (multi-latent attention with single K head)
    if nhk != 1 and nhq != nhk:
        pytest.skip(f"Ring joint attention requires nhq == nhk or nhk == 1, got nhq={nhq}, nhk={nhk}")
    if vary_page_table_per_iteration:
        assert use_paged_kv, "vary_page_table_per_iteration is only valid for paged K/V tests"
        assert num_iterations > 1, "vary_page_table_per_iteration requires num_iterations > 1"

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
        kv_cache_seq_len = sq + chunk_start_idx if use_paged_kv else sq
        if use_paged_kv and kv_cache_seq_len_override is not None:
            assert kv_cache_seq_len_override >= sq + chunk_start_idx
            kv_cache_seq_len = kv_cache_seq_len_override
        K_cache = fa_rand(b, nhk, kv_cache_seq_len, d_k)
        V_cache = fa_rand(b, nhv, kv_cache_seq_len, d_v)
        K = K_cache[:, :, chunk_start_idx : chunk_start_idx + sq, :]
        V = V_cache[:, :, chunk_start_idx : chunk_start_idx + sq, :]

        # Joint tensors - Use dummy tensors like WAN 2.2 (empty sequence, zero-filled)
        joint_Q = torch.zeros((b, nhq, joint_seq_len, d_q), dtype=torch.bfloat16)
        joint_K = torch.zeros((b, nhk, joint_seq_len, d_k), dtype=torch.bfloat16)
        joint_V = torch.zeros((b, nhv, joint_seq_len, d_v), dtype=torch.bfloat16)

        # Keep original tensors for reference comparison (before any reordering)
        Q_original, K_original, V_original = Q, K, V

        # Apply balanced reordering if enabled (for causal attention workload balancing)
        chunk_order = None
        if is_balanced:
            chunk_order = create_balanced_chunk_order(mesh_config.sp_size)
            Q = reorder_tensor_chunks(Q, chunk_order)
            K = reorder_tensor_chunks(K, chunk_order)
            V = reorder_tensor_chunks(V, chunk_order)

        if use_paged_kv and kv_cache_seq_len % page_block_size != 0:
            pytest.skip(
                f"Paged Ring Joint SDPA requires cache length {kv_cache_seq_len} divisible by "
                f"page_block_size {page_block_size}"
            )
        if use_paged_kv and kv_cache_seq_len % v_page_block_size != 0:
            pytest.skip(
                f"Paged Ring Joint SDPA requires cache length {kv_cache_seq_len} divisible by "
                f"v_page_block_size {v_page_block_size}"
            )
        if use_paged_kv and paged_extra_blocks:
            assert paged_num_blocks_override is None, "paged_extra_blocks and paged_num_blocks_override are exclusive"
            assert v_page_block_size == page_block_size, "paged_extra_blocks requires matching K/V page sizes"

        base_num_blocks = kv_cache_seq_len // page_block_size if use_paged_kv else 0
        base_num_v_blocks = kv_cache_seq_len // v_page_block_size if use_paged_kv else 0
        physical_num_blocks = (
            paged_num_blocks_override if paged_num_blocks_override is not None else base_num_blocks + paged_extra_blocks
        )
        physical_num_v_blocks = (
            paged_num_blocks_override
            if paged_num_blocks_override is not None
            else base_num_v_blocks + paged_extra_blocks
        )
        if use_paged_kv and physical_num_blocks % mesh_config.sp_size != 0:
            pytest.skip(
                f"Paged Ring Joint SDPA requires physical K pages {physical_num_blocks} to shard evenly "
                f"across SP={mesh_config.sp_size}"
            )
        if use_paged_kv and physical_num_v_blocks % mesh_config.sp_size != 0:
            pytest.skip(
                f"Paged Ring Joint SDPA requires physical V pages {physical_num_v_blocks} to shard evenly "
                f"across SP={mesh_config.sp_size}"
            )
        if use_paged_kv and paged_kv_page_table_is_rank_local and base_num_blocks % mesh_config.sp_size != 0:
            pytest.skip(
                f"Rank-local paged K/V requires logical pages {base_num_blocks} to shard evenly "
                f"across SP={mesh_config.sp_size}"
            )

        # Create persistent output buffers
        kv_shard_dims = [None, None]
        kv_shard_dims[sp_axis] = None
        if mesh_config.tp_size > 1:
            kv_shard_dims[tp_axis] = 1

        if use_paged_kv:
            persistent_k_shape = (physical_num_blocks, nhk, page_block_size, d_k)
            persistent_v_shape = (physical_num_v_blocks, nhv, v_page_block_size, d_v)
        else:
            persistent_k_shape = (b, nhk, sq, d_k)
            persistent_v_shape = (b, nhv, sq, d_v)

        # For K buffer: handle nhk=1 case (MLA) - may need different sharding
        persistent_k_shard_dims = [None, None]
        persistent_k_shard_dims[sp_axis] = None
        if mesh_config.tp_size > 1 and nhk != 1:
            persistent_k_shard_dims[tp_axis] = 1

        persistent_output_buffer_k = ttnn.from_torch(
            torch.zeros(persistent_k_shape),
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_k_shard_dims
            ),
        )
        persistent_output_buffer_v = ttnn.from_torch(
            torch.zeros(persistent_v_shape),
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
        sdpa_k_shard_dims[sp_axis] = 0 if use_paged_kv else 2
        if mesh_config.tp_size > 1 and nhk != 1:
            sdpa_k_shard_dims[tp_axis] = 1

        sdpa_v_shard_dims = [None, None]
        sdpa_v_shard_dims[sp_axis] = 0 if use_paged_kv else 2
        if mesh_config.tp_size > 1:
            sdpa_v_shard_dims[tp_axis] = 1

        sdpa_joint_shard_dims = [None, None]
        if mesh_config.tp_size > 1:
            sdpa_joint_shard_dims[tp_axis] = 1

        sdpa_joint_k_shard_dims = [None, None]
        if mesh_config.tp_size > 1 and nhk != 1:
            sdpa_joint_k_shard_dims[tp_axis] = 1

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
        page_tables_seen = []

        def create_tt_kv_inputs(iteration=0):
            if not use_paged_kv:
                tt_k = ttnn.from_torch(
                    K,
                    dtype=kv_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_k_shard_dims
                    ),
                )
                tt_v = ttnn.from_torch(
                    V,
                    dtype=kv_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_v_shard_dims
                    ),
                )
                return tt_k, tt_v, None

            forced_permutation = None
            if vary_page_table_per_iteration:
                if page_shuffle_mode != "rank_local" or chunk_start_idx != 0:
                    pytest.skip(
                        "Program-cache page-table permutation coverage currently uses zero-offset rank-local pages"
                    )
                max_num_blocks_per_seq = kv_cache_seq_len // page_block_size
                if iteration % 2 == 0:
                    forced_permutation = torch.arange(b * max_num_blocks_per_seq)
                else:
                    forced_permutation = make_rank_local_permutation(b, max_num_blocks_per_seq, mesh_config.sp_size)

            K_for_paging = K
            V_for_paging = V
            if chunk_start_idx != 0:
                K_for_paging = K_cache.clone()
                V_for_paging = V_cache.clone()
                if is_balanced:
                    K_for_paging[:, :, chunk_start_idx : chunk_start_idx + sq, :] = K
                    V_for_paging[:, :, chunk_start_idx : chunk_start_idx + sq, :] = V

            paged_K, page_table, _, permutation = make_paged_kv_cache(
                K_for_paging,
                page_block_size,
                shuffle_pages=shuffle_pages,
                permutation=forced_permutation,
                shuffle_mode=page_shuffle_mode,
                sp_size=mesh_config.sp_size,
                window_start_idx=chunk_start_idx,
                window_seq_len=sq,
            )
            paged_V, page_table_v, _, _ = make_paged_kv_cache(
                V_for_paging,
                v_page_block_size,
                shuffle_pages=shuffle_pages,
                permutation=permutation if v_page_block_size == page_block_size else None,
                shuffle_mode=page_shuffle_mode,
                sp_size=mesh_config.sp_size,
                window_start_idx=chunk_start_idx,
                window_seq_len=sq,
            )
            if v_page_block_size == page_block_size:
                assert torch.equal(page_table, page_table_v)
            if paged_extra_blocks:
                paged_K = torch.cat(
                    [
                        paged_K,
                        torch.zeros((paged_extra_blocks, nhk, page_block_size, d_k), dtype=paged_K.dtype),
                    ],
                    dim=0,
                ).contiguous()
                paged_V = torch.cat(
                    [
                        paged_V,
                        torch.zeros((paged_extra_blocks, nhv, v_page_block_size, d_v), dtype=paged_V.dtype),
                    ],
                    dim=0,
                ).contiguous()
            if paged_num_blocks_override is not None:
                assert paged_num_blocks_override % b == 0
                page_table_pages = paged_num_blocks_override // b
                paged_K = paged_K[:paged_num_blocks_override].contiguous()
                paged_V = paged_V[:paged_num_blocks_override].contiguous()
                page_table = page_table[:, :page_table_pages].contiguous()
            if page_table_transform is not None:
                page_table = page_table_transform(page_table)
            if vary_page_table_per_iteration:
                page_tables_seen.append(page_table)

            tt_k = ttnn.from_torch(
                paged_K,
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_k_shard_dims
                ),
            )
            tt_v = ttnn.from_torch(
                paged_V,
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_v_shard_dims
                ),
            )
            page_table_tt = ttnn.from_torch(
                page_table,
                dtype=page_table_dtype,
                layout=page_table_layout,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            return tt_k, tt_v, page_table_tt

        tt_K, tt_V, page_table_tt = (None, None, None)
        if not vary_page_table_per_iteration:
            tt_K, tt_V, page_table_tt = create_tt_kv_inputs()
        tt_joint_Q = ttnn.from_torch(
            joint_Q,
            dtype=q_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_shard_dims
            ),
        )
        tt_joint_K = ttnn.from_torch(
            joint_K,
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_k_shard_dims
            ),
        )
        tt_joint_V = ttnn.from_torch(
            joint_V,
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_shard_dims
            ),
        )

        # Set logical_n to the original full sequence length
        corrected_logical_n = sq

        # Precompute mesh composer dims
        main_row_dim = sdpa_input_shard_dims[0] if sdpa_input_shard_dims[0] is not None else -1
        main_col_dim = sdpa_input_shard_dims[1] if sdpa_input_shard_dims[1] is not None else -1

        # Run ring joint attention
        reference_output = None
        iteration_outputs = []
        for i in range(num_iterations):
            if vary_page_table_per_iteration:
                tt_K, tt_V, page_table_tt = create_tt_kv_inputs(i)

            tt_out, tt_joint_out, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                tt_Q,
                tt_K,
                tt_V,
                tt_joint_Q,
                tt_joint_K,
                tt_joint_V,
                persistent_output_buffer_k=persistent_output_buffer_k,
                persistent_output_buffer_v=persistent_output_buffer_v,
                joint_strategy="rear",
                logical_n=corrected_logical_n,
                is_causal=is_causal,
                is_balanced=is_balanced,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                dim=0 if use_paged_kv else 2,
                multi_device_global_semaphore=ccl_semaphore_handles,
                num_links=num_links,
                cluster_axis=sp_axis,
                mesh_device=mesh_device,
                topology=topology,
                subdevice_id=worker_sub_device_id,
                ccl_core_grid_offset=(ccl_column, 0),  # Point to CCL column
                use_column_major_ccl=True,
                page_table=page_table_tt,
                chunk_start_idx=chunk_start_idx if use_paged_kv else None,
                paged_kv_page_table_is_rank_local=paged_kv_page_table_is_rank_local,
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
            if vary_page_table_per_iteration:
                iteration_outputs.append(tt_out_torch)
            elif num_iterations > 1:
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

        if vary_page_table_per_iteration:
            if not any(not torch.equal(page_tables_seen[0], page_table) for page_table in page_tables_seen[1:]):
                pytest.fail("Program-cache permutation test did not create distinct page tables")
        elif num_iterations > 1:
            logger.info(f"Ring joint SDPA determinism verified: all {num_iterations} outputs are exactly equal")
            return

        if not do_check:
            return

        # Convert and verify joint output (only if joint_seq_len > 0)
        if joint_seq_len > 0:
            if mesh_config.arch_type.startswith("galaxy"):
                joint_row_dim = sdpa_joint_shard_dims[0] if sdpa_joint_shard_dims[0] is not None else -1
                joint_col_dim = sdpa_joint_shard_dims[1] if sdpa_joint_shard_dims[1] is not None else -1
                tt_joint_out_torch = ttnn.to_torch(
                    tt_joint_out,
                    mesh_composer=ttnn.create_mesh_composer(
                        mesh_device, ttnn.MeshComposerConfig(joint_row_dim, joint_col_dim)
                    ),
                )
            else:
                tt_joint_out_torch = ttnn.to_torch(
                    tt_joint_out,
                    mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(1, -1)),
                )

            if tt_joint_out_torch.shape[3] != d_v:
                tt_joint_out_torch = tt_joint_out_torch[:, :, :, :d_v]
            if tt_joint_out_torch.shape[0] > 1:
                tt_joint_out_torch = tt_joint_out_torch[0:1, :, :, :]
            tt_joint_out_torch = tt_joint_out_torch[:, :, :joint_seq_len, :]
        else:
            logger.info("Joint output - Dummy tensors (seq_len=0), skipping accuracy check (wan2.2 compatible)")

        # Compute PyTorch reference on ORIGINAL data (before balanced reordering)
        gt_main, gt_joint = torch_joint_sdpa_reference(
            Q_original, K_original, V_original, joint_Q, joint_K, joint_V, is_causal=is_causal
        )

        if vary_page_table_per_iteration:
            for i, iteration_output in enumerate(iteration_outputs):
                out_pass_main, out_pcc_main = comp_pcc(gt_main, iteration_output, pcc_threshold)
                rmse_main = torch.sqrt(((gt_main - iteration_output) ** 2).mean()).item()
                logger.info(f"Main output iteration {i} - PCC: {out_pcc_main}, RMSE: {rmse_main:.6f}")
                if rmse_threshold is not None:
                    assert rmse_main < rmse_threshold, f"Main RMSE {rmse_main:.6f} exceeds threshold {rmse_threshold}"
                assert out_pass_main, f"Main PCC {out_pcc_main} below threshold {pcc_threshold}"
            return

        # Verify accuracy for main output
        out_pass_main, out_pcc_main = comp_pcc(gt_main, tt_out_torch, pcc_threshold)
        rmse_main = torch.sqrt(((gt_main - tt_out_torch) ** 2).mean()).item()
        logger.info(f"Main output - PCC: {out_pcc_main}, RMSE: {rmse_main:.6f}")

        if rmse_threshold is not None:
            assert rmse_main < rmse_threshold, f"Main RMSE {rmse_main:.6f} exceeds threshold {rmse_threshold}"
        assert out_pass_main, f"Main PCC {out_pcc_main} below threshold {pcc_threshold}"

        # Verify accuracy for joint output
        if joint_seq_len > 0:
            out_pass_joint, out_pcc_joint = comp_pcc(gt_joint, tt_joint_out_torch, pcc_threshold)
            rmse_joint = torch.sqrt(((gt_joint - tt_joint_out_torch) ** 2).mean()).item()
            logger.info(f"Joint output - PCC: {out_pcc_joint}, RMSE: {rmse_joint:.6f}")
            if rmse_threshold is not None:
                assert rmse_joint < rmse_threshold, f"Joint RMSE {rmse_joint:.6f} exceeds threshold {rmse_threshold}"
            assert out_pass_joint, f"Joint PCC {out_pcc_joint} below threshold {pcc_threshold}"

    finally:
        # Clean up mesh device
        ttnn.close_mesh_device(mesh_device)

        # Restore fabric to disabled state
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# Generate test parameters dynamically based on detected hardware for different models (WAN, MLA, VideGen...)
TEST_CONFIGS, TEST_CONFIG_IDS = generate_test_configs(MESH_CONFIG, MODEL_CONFIGS)
PAGED_TEST_CONFIGS, PAGED_TEST_CONFIG_IDS = generate_paged_test_configs(MESH_CONFIG, TEST_CONFIGS, TEST_CONFIG_IDS)
PAGED_TEST_CONFIG_BASE_IDS = [config_id.rsplit("-page", 1)[0] for config_id in PAGED_TEST_CONFIG_IDS]
PAGED_TEST_CONFIG_BY_BASE_ID = dict(zip(PAGED_TEST_CONFIG_BASE_IDS, PAGED_TEST_CONFIGS))
PAGED_PERF_VARIANT_CONFIGS, PAGED_PERF_VARIANT_CONFIG_IDS = generate_paged_perf_variant_configs(
    MESH_CONFIG, TEST_CONFIGS, TEST_CONFIG_IDS
)
PAGED_PERF_VARIANT_CONFIG_BY_ID = dict(zip(PAGED_PERF_VARIANT_CONFIG_IDS, PAGED_PERF_VARIANT_CONFIGS))
TEST_CONFIG_MODELS = list(MODEL_CONFIGS.keys())


def test_paged_kv_cache_roundtrip_helper():
    torch.manual_seed(0)
    b, nheads, seq_len, head_dim, page_block_size = 1, 3, 256, 32, 64
    cache = fa_rand(b, nheads, seq_len, head_dim)
    paged_cache, page_table, reverse_permutation, _ = make_paged_kv_cache(cache, page_block_size, shuffle_pages=True)
    restored = unpage_kv_cache(paged_cache, reverse_permutation, b, nheads, seq_len, head_dim, page_block_size)

    assert page_table.shape == (b, seq_len // page_block_size)
    assert page_table.dtype == torch.int32
    assert torch.allclose(restored, cache)

    rank_local_seq_len = 512
    rank_local_cache = fa_rand(b, nheads, rank_local_seq_len, head_dim)
    rank_local_paged, _, rank_local_reverse_permutation, _ = make_paged_kv_cache(
        rank_local_cache,
        page_block_size,
        shuffle_pages=True,
        shuffle_mode="rank_local",
        sp_size=4,
    )
    rank_local_restored = unpage_kv_cache(
        rank_local_paged,
        rank_local_reverse_permutation,
        b,
        nheads,
        rank_local_seq_len,
        head_dim,
        page_block_size,
    )
    assert torch.allclose(rank_local_restored, rank_local_cache)

    windowed_seq_len = 768
    window_start_idx = 256
    window_seq_len = 512
    sp_size = 4
    windowed_cache = fa_rand(b, nheads, windowed_seq_len, head_dim)
    windowed_paged, windowed_page_table, windowed_reverse_permutation, _ = make_paged_kv_cache(
        windowed_cache,
        page_block_size,
        shuffle_pages=True,
        shuffle_mode="rank_local",
        sp_size=sp_size,
        window_start_idx=window_start_idx,
        window_seq_len=window_seq_len,
    )
    windowed_restored = unpage_kv_cache(
        windowed_paged,
        windowed_reverse_permutation,
        b,
        nheads,
        windowed_seq_len,
        head_dim,
        page_block_size,
    )
    assert torch.allclose(windowed_restored, windowed_cache)

    pages_per_rank = (windowed_seq_len // page_block_size) // sp_size
    window_start_page = window_start_idx // page_block_size
    window_pages_per_rank = (window_seq_len // page_block_size) // sp_size
    for logical_page in range(window_start_page, window_start_page + window_seq_len // page_block_size):
        expected_rank = (logical_page - window_start_page) // window_pages_per_rank
        physical_page = windowed_page_table[0, logical_page].item()
        assert physical_page // pages_per_rank == expected_rank


def test_ring_joint_attention_sdpa_paged_kv_smoke():
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    page_block_size = 64
    sq = mesh_config.sp_size * page_block_size
    nh = max(mesh_config.tp_size, 1)

    run_ring_joint_sdpa(
        mesh_config,
        b=1,
        nhq=nh,
        nhk=nh,
        sq=sq,
        d_q=128,
        q_chunk_size=64,
        k_chunk_size=64,
        q_dtype=ttnn.bfloat16,
        nhv=nh,
        d_k=128,
        d_v=128,
        kv_dtype=ttnn.bfloat16,
        is_causal=False,
        is_balanced=False,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        use_paged_kv=True,
        page_block_size=page_block_size,
        shuffle_pages=False,
    )


def test_ring_joint_attention_sdpa_paged_kv_rank_local_shuffle_smoke():
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    page_block_size = 64
    sq = mesh_config.sp_size * 2 * page_block_size
    nh = max(mesh_config.tp_size, 1)

    run_ring_joint_sdpa(
        mesh_config,
        b=1,
        nhq=nh,
        nhk=nh,
        sq=sq,
        d_q=128,
        q_chunk_size=64,
        k_chunk_size=64,
        q_dtype=ttnn.bfloat16,
        nhv=nh,
        d_k=128,
        d_v=128,
        kv_dtype=ttnn.bfloat16,
        is_causal=False,
        is_balanced=False,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        use_paged_kv=True,
        page_block_size=page_block_size,
        shuffle_pages=True,
        page_shuffle_mode="rank_local",
    )


def test_ring_joint_attention_sdpa_paged_kv_global_shuffle_smoke():
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    run_ring_joint_sdpa(
        mesh_config,
        b=1,
        nhq=1,
        nhk=1,
        sq=512,
        d_q=128,
        q_chunk_size=64,
        k_chunk_size=64,
        q_dtype=ttnn.bfloat16,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        use_paged_kv=True,
        page_block_size=64,
        shuffle_pages=True,
        page_shuffle_mode="global",
    )


def test_ring_joint_attention_sdpa_paged_kv_global_shuffle_causal_balanced_smoke():
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    run_ring_joint_sdpa(
        mesh_config,
        b=1,
        nhq=1,
        nhk=1,
        sq=mesh_config.sp_size * 128,
        d_q=128,
        q_chunk_size=64,
        k_chunk_size=64,
        q_dtype=ttnn.bfloat16,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        is_causal=True,
        is_balanced=True,
        use_paged_kv=True,
        page_block_size=64,
        shuffle_pages=True,
        page_shuffle_mode="global",
    )


def test_ring_joint_attention_sdpa_paged_kv_global_shuffle_causal_balanced_nonzero_chunk_start_smoke():
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    page_block_size = 64

    run_ring_joint_sdpa(
        mesh_config,
        b=1,
        nhq=1,
        nhk=1,
        sq=mesh_config.sp_size * 128,
        d_q=128,
        q_chunk_size=64,
        k_chunk_size=64,
        q_dtype=ttnn.bfloat16,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        is_causal=True,
        is_balanced=True,
        use_paged_kv=True,
        page_block_size=page_block_size,
        shuffle_pages=True,
        page_shuffle_mode="global",
        chunk_start_idx=mesh_config.sp_size * page_block_size,
    )


def _run_small_paged_validation_case(mesh_config, **kwargs):
    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    params = {
        "b": 1,
        "nhq": 1,
        "nhk": 1,
        "sq": mesh_config.sp_size * 128,
        "d_q": 128,
        "q_chunk_size": 64,
        "k_chunk_size": 64,
        "q_dtype": ttnn.bfloat16,
        "do_check": False,
        "use_paged_kv": True,
        "page_block_size": 64,
        "shuffle_pages": False,
    }
    params.update(kwargs)
    run_ring_joint_sdpa(mesh_config, **params)


def test_ring_joint_attention_sdpa_paged_kv_validation_rejects_bad_page_table_dtype():
    with pytest.raises(RuntimeError, match="Page table must be int32"):
        _run_small_paged_validation_case(MESH_CONFIG, page_table_dtype=ttnn.bfloat16)


def test_ring_joint_attention_sdpa_paged_kv_validation_rejects_page_table_tile_layout():
    with pytest.raises(RuntimeError, match="Page table must be row major"):
        _run_small_paged_validation_case(MESH_CONFIG, page_table_layout=ttnn.TILE_LAYOUT)


def test_ring_joint_attention_sdpa_paged_kv_validation_rejects_page_table_rank():
    def flatten_page_table(page_table):
        return page_table.flatten().contiguous()

    with pytest.raises(RuntimeError, match="Page table must be 2D"):
        _run_small_paged_validation_case(MESH_CONFIG, page_table_transform=flatten_page_table)


def test_ring_joint_attention_sdpa_paged_kv_validation_rejects_page_table_batch():
    def duplicate_batch(page_table):
        return page_table.repeat(2, 1).contiguous()

    with pytest.raises(RuntimeError, match="Page table batch size must match input batch size"):
        _run_small_paged_validation_case(MESH_CONFIG, page_table_transform=duplicate_batch)


def test_ring_joint_attention_sdpa_paged_kv_global_unsharded_page_table_smoke():
    mesh_config = MESH_CONFIG
    page_block_size = 64
    sq = mesh_config.sp_size * 160
    logical_pages = sq // page_block_size
    extra_blocks = (mesh_config.sp_size - (logical_pages % mesh_config.sp_size)) % mesh_config.sp_size
    if extra_blocks == 0:
        pytest.skip("Test requires logical page count that is not evenly sharded across the ring")

    _run_small_paged_validation_case(
        mesh_config,
        sq=sq,
        q_chunk_size=64,
        k_chunk_size=64,
        page_block_size=page_block_size,
        shuffle_pages=True,
        page_shuffle_mode="global",
        paged_extra_blocks=extra_blocks,
        do_check=True,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
    )


def test_ring_joint_attention_sdpa_paged_kv_validation_rejects_kv_page_block_mismatch():
    with pytest.raises(RuntimeError, match="Paged K and V page_block_size must match"):
        _run_small_paged_validation_case(MESH_CONFIG, v_page_block_size=128)


def test_ring_joint_attention_sdpa_paged_kv_validation_rejects_page_table_capacity():
    with pytest.raises(RuntimeError, match="page table capacity must cover"):
        _run_small_paged_validation_case(MESH_CONFIG, paged_num_blocks_override=MESH_CONFIG.sp_size)


def test_ring_joint_attention_sdpa_paged_kv_non_q_chunk_aligned_chunk_start_smoke():
    mesh_config = MESH_CONFIG
    sq = mesh_config.sp_size * 128
    chunk_start_idx = 64
    page_block_size = 64
    logical_pages = (sq + chunk_start_idx) // page_block_size
    extra_blocks = (mesh_config.sp_size - (logical_pages % mesh_config.sp_size)) % mesh_config.sp_size

    _run_small_paged_validation_case(
        mesh_config,
        sq=sq,
        q_chunk_size=128,
        k_chunk_size=64,
        page_block_size=page_block_size,
        chunk_start_idx=chunk_start_idx,
        shuffle_pages=True,
        page_shuffle_mode="global",
        paged_extra_blocks=extra_blocks,
        do_check=True,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
    )


def test_ring_joint_attention_sdpa_paged_kv_non_page_aligned_chunk_start_smoke():
    mesh_config = MESH_CONFIG
    sq = mesh_config.sp_size * 128
    chunk_start_idx = 32
    page_block_size = 64
    kv_cache_seq_len = math.ceil((sq + chunk_start_idx) / page_block_size) * page_block_size
    logical_pages = kv_cache_seq_len // page_block_size
    extra_blocks = (mesh_config.sp_size - (logical_pages % mesh_config.sp_size)) % mesh_config.sp_size

    _run_small_paged_validation_case(
        mesh_config,
        sq=sq,
        q_chunk_size=64,
        k_chunk_size=64,
        page_block_size=page_block_size,
        chunk_start_idx=chunk_start_idx,
        kv_cache_seq_len_override=kv_cache_seq_len,
        shuffle_pages=True,
        page_shuffle_mode="global",
        paged_extra_blocks=extra_blocks,
        do_check=True,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
    )


def test_ring_joint_attention_sdpa_paged_kv_validation_rejects_rank_local_non_page_aligned_chunk_start():
    mesh_config = MESH_CONFIG
    page_block_size = 64
    sq = mesh_config.sp_size * 128
    kv_cache_seq_len = mesh_config.sp_size * 3 * page_block_size

    with pytest.raises(RuntimeError, match="Rank-local paged K/V requires chunk_start_idx"):
        _run_small_paged_validation_case(
            mesh_config,
            sq=sq,
            q_chunk_size=64,
            k_chunk_size=64,
            page_block_size=page_block_size,
            chunk_start_idx=32,
            kv_cache_seq_len_override=kv_cache_seq_len,
            shuffle_pages=False,
            paged_kv_page_table_is_rank_local=True,
        )


@pytest.mark.parametrize(
    "shuffle_pages,page_shuffle_mode",
    [
        (False, "global"),
        (True, "rank_local"),
        (True, "global"),
    ],
    ids=["identity", "rank_local", "global"],
)
def test_ring_joint_attention_sdpa_paged_kv_joint_sequence_smoke(shuffle_pages, page_shuffle_mode):
    _run_small_paged_validation_case(
        MESH_CONFIG,
        do_check=True,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        joint_seq_len=64,
        shuffle_pages=shuffle_pages,
        page_shuffle_mode=page_shuffle_mode,
    )


def test_ring_joint_attention_sdpa_paged_kv_non_causal_nhk1_k_mcast_smoke():
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    q_chunk_size = 64
    local_seq_len = mesh_config.grid_rows * q_chunk_size
    nh = mesh_config.sdpa_cols * mesh_config.tp_size

    run_ring_joint_sdpa(
        mesh_config,
        b=1,
        nhq=nh,
        nhk=1,
        sq=mesh_config.sp_size * local_seq_len,
        d_q=128,
        q_chunk_size=q_chunk_size,
        k_chunk_size=64,
        q_dtype=ttnn.bfloat16,
        nhv=nh,
        d_k=128,
        d_v=128,
        kv_dtype=ttnn.bfloat16,
        is_causal=False,
        is_balanced=False,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        use_paged_kv=True,
        page_block_size=64,
        shuffle_pages=False,
    )


def test_ring_joint_attention_sdpa_paged_kv_multi_head_rank_local_shuffle_smoke():
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    page_block_size = 64
    sq = mesh_config.sp_size * 2 * page_block_size
    nh = max(2 * mesh_config.tp_size, 2)

    run_ring_joint_sdpa(
        mesh_config,
        b=1,
        nhq=nh,
        nhk=nh,
        sq=sq,
        d_q=128,
        q_chunk_size=64,
        k_chunk_size=64,
        q_dtype=ttnn.bfloat16,
        nhv=nh,
        d_k=128,
        d_v=128,
        kv_dtype=ttnn.bfloat16,
        is_causal=False,
        is_balanced=False,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        use_paged_kv=True,
        page_block_size=page_block_size,
        shuffle_pages=True,
        page_shuffle_mode="rank_local",
    )


def test_ring_joint_attention_sdpa_paged_kv_multi_head_global_shuffle_smoke():
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    page_block_size = 64
    sq = mesh_config.sp_size * 2 * page_block_size
    nh = max(2 * mesh_config.tp_size, 2)

    run_ring_joint_sdpa(
        mesh_config,
        b=1,
        nhq=nh,
        nhk=nh,
        sq=sq,
        d_q=128,
        q_chunk_size=64,
        k_chunk_size=64,
        q_dtype=ttnn.bfloat16,
        nhv=nh,
        d_k=128,
        d_v=128,
        kv_dtype=ttnn.bfloat16,
        is_causal=False,
        is_balanced=False,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        use_paged_kv=True,
        page_block_size=page_block_size,
        shuffle_pages=True,
        page_shuffle_mode="global",
    )


def test_ring_joint_attention_sdpa_paged_kv_nonzero_chunk_start_smoke():
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    run_ring_joint_sdpa(
        mesh_config,
        b=1,
        nhq=1,
        nhk=1,
        sq=512,
        d_q=128,
        q_chunk_size=64,
        k_chunk_size=64,
        q_dtype=ttnn.bfloat16,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        use_paged_kv=True,
        page_block_size=64,
        shuffle_pages=True,
        page_shuffle_mode="rank_local",
        chunk_start_idx=256,
    )


def test_ring_joint_attention_sdpa_paged_kv_determinism_smoke():
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    page_block_size = 64
    sq = mesh_config.sp_size * 2 * page_block_size
    nh = max(mesh_config.tp_size, 1)

    run_ring_joint_sdpa(
        mesh_config,
        b=1,
        nhq=nh,
        nhk=nh,
        sq=sq,
        d_q=128,
        q_chunk_size=64,
        k_chunk_size=64,
        q_dtype=ttnn.bfloat16,
        nhv=nh,
        d_k=128,
        d_v=128,
        kv_dtype=ttnn.bfloat16,
        is_causal=False,
        is_balanced=False,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        num_iterations=3,
        use_paged_kv=True,
        page_block_size=page_block_size,
        shuffle_pages=True,
        page_shuffle_mode="rank_local",
    )


def test_ring_joint_attention_sdpa_paged_kv_program_cache_permutation_smoke():
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    run_ring_joint_sdpa(
        mesh_config,
        b=1,
        nhq=1,
        nhk=1,
        sq=512,
        d_q=128,
        q_chunk_size=64,
        k_chunk_size=64,
        q_dtype=ttnn.bfloat16,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        num_iterations=2,
        use_paged_kv=True,
        page_block_size=64,
        shuffle_pages=True,
        page_shuffle_mode="rank_local",
        vary_page_table_per_iteration=True,
    )


@pytest.mark.parametrize("page_block_size", [64, 128], ids=["page64", "page128"])
def test_ring_joint_attention_sdpa_paged_kv_mla_balanced_smoke(page_block_size):
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    model = MODEL_CONFIGS["mla_128k"]
    q_chunk_size = model.q_chunk_sizes[0]
    k_chunk_size = model.k_chunk_sizes[0]

    run_ring_joint_sdpa(
        mesh_config,
        b=BATCH_SIZE,
        nhq=model.nhq * mesh_config.tp_size,
        nhk=model.nhk * (mesh_config.tp_size if model.nhk != 1 else 1),
        sq=model.seq_len * mesh_config.sp_size,
        d_q=model.d_q,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        q_dtype=model.q_dtype,
        nhv=model.nhv * mesh_config.tp_size,
        d_k=model.d_k,
        d_v=model.d_v,
        kv_dtype=model.kv_dtype,
        is_causal=model.is_causal,
        is_balanced=model.is_balanced,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        use_paged_kv=True,
        page_block_size=page_block_size,
        shuffle_pages=True,
        page_shuffle_mode="rank_local",
    )


def test_ring_joint_attention_sdpa_paged_kv_mla_balanced_global_shuffle_smoke():
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    model = MODEL_CONFIGS["mla_128k"]
    q_chunk_size = model.q_chunk_sizes[0]
    k_chunk_size = model.k_chunk_sizes[0]

    run_ring_joint_sdpa(
        mesh_config,
        b=BATCH_SIZE,
        nhq=model.nhq * mesh_config.tp_size,
        nhk=model.nhk * (mesh_config.tp_size if model.nhk != 1 else 1),
        sq=model.seq_len * mesh_config.sp_size,
        d_q=model.d_q,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        q_dtype=model.q_dtype,
        nhv=model.nhv * mesh_config.tp_size,
        d_k=model.d_k,
        d_v=model.d_v,
        kv_dtype=model.kv_dtype,
        is_causal=model.is_causal,
        is_balanced=model.is_balanced,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        use_paged_kv=True,
        page_block_size=64,
        shuffle_pages=True,
        page_shuffle_mode="global",
    )


@pytest.mark.parametrize(
    "b,sq,nhq,nhk,nhv,d_q,d_k,d_v,q_chunk_size,k_chunk_size,is_causal,is_balanced,q_dtype,kv_dtype,page_block_size",
    PAGED_TEST_CONFIGS,
    ids=PAGED_TEST_CONFIG_IDS,
)
def test_ring_joint_attention_sdpa_paged_kv_accuracy(
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
    page_block_size,
):
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
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        use_paged_kv=True,
        page_block_size=page_block_size,
        shuffle_pages=True,
        page_shuffle_mode="rank_local",
    )


# === PAGED TEST: DETERMINISM VERIFICATION ===
@pytest.mark.parametrize(
    "b,sq,nhq,nhk,nhv,d_q,d_k,d_v,q_chunk_size,k_chunk_size,is_causal,is_balanced,q_dtype,kv_dtype,page_block_size",
    PAGED_TEST_CONFIGS,
    ids=PAGED_TEST_CONFIG_IDS,
)
def test_ring_joint_attention_sdpa_paged_kv_determinism(
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
    page_block_size,
):
    """
    Test paged ring joint attention SDPA determinism across the same generated
    paged shapes used for accuracy.
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
        use_paged_kv=True,
        page_block_size=page_block_size,
        shuffle_pages=True,
        page_shuffle_mode="rank_local",
    )


# === PAGED TEST: PERFORMANCE SWEEP (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.parametrize(
    "b,sq,nhq,nhk,nhv,d_q,d_k,d_v,q_chunk_size,k_chunk_size,is_causal,is_balanced,q_dtype,kv_dtype,page_block_size",
    PAGED_TEST_CONFIGS,
    ids=PAGED_TEST_CONFIG_BASE_IDS,
)
def test_ring_joint_attention_sdpa_paged_kv_sweep_perf_impl(
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
    page_block_size,
):
    """
    Performance sweep test for paged ring joint attention SDPA.
    Skipped on CI - run locally for performance measurement.
    Uses the same paged WAN, MLA, and VideoGen configurations as accuracy.
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
        use_paged_kv=True,
        page_block_size=page_block_size,
        shuffle_pages=True,
        page_shuffle_mode="rank_local",
    )


# === PAGED TEST: PAGE-SIZE/MODE PERFORMANCE VARIANTS (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.parametrize(
    "b,sq,nhq,nhk,nhv,d_q,d_k,d_v,q_chunk_size,k_chunk_size,is_causal,is_balanced,q_dtype,kv_dtype,page_block_size,page_table_mode",
    PAGED_PERF_VARIANT_CONFIGS,
    ids=PAGED_PERF_VARIANT_CONFIG_IDS,
)
def test_ring_joint_attention_sdpa_paged_kv_perf_variant_impl(
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
    page_block_size,
    page_table_mode,
):
    """
    Performance variant for page-size and page-table-mode sensitivity.
    Skipped on CI - run locally for performance measurement.
    """
    mesh_config = MESH_CONFIG
    assert page_table_mode in PAGED_KV_PERF_PAGE_TABLE_MODES

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
        use_paged_kv=True,
        page_block_size=page_block_size,
        shuffle_pages=page_table_mode != "identity",
        page_shuffle_mode="global" if page_table_mode == "global" else "rank_local",
    )


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


def profile_ring_joint_sdpa_perf_command(command, subdir):
    """Run a profiler command and extract Ring Joint SDPA timing fields."""
    from tracy.process_model_log import run_device_profiler

    float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]", "PM FPU UTIL (%)"]
    cols = ["ATTRIBUTES"]

    # Each command profiles one RingJointSDPADeviceOperation. Keeping the profiler
    # op-support count tight avoids profiler metadata pushing size-sensitive
    # paged kernels over the Blackhole kernel config buffer limit.
    run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"], op_support_count=1)
    r = post_process_ops_log(
        subdir,
        float_columns=float_cols,
        columns=cols,
        op_name="RingJointSDPADeviceOperation",
        sum_vals=False,
        has_signposts=False,
    )

    if len(r["CORE COUNT"]) == 0 or len(r["DEVICE KERNEL DURATION [ns]"]) == 0:
        raise RuntimeError("Profiler returned no RingJointSDPADeviceOperation rows")

    fpu_util_col = r.get("PM FPU UTIL (%)", [])
    return {
        "measured_core_count": int(r["CORE COUNT"][0]),
        "duration_ns": int(r["DEVICE KERNEL DURATION [ns]"].max()),
        "fpu_util_min": float(fpu_util_col.min()) if len(fpu_util_col) > 0 else 0.0,
        "fpu_util_max": float(fpu_util_col.max()) if len(fpu_util_col) > 0 else 0.0,
    }


def compute_effective_ring_joint_compute_cores(measured_core_count, mesh_config):
    # Tracy reports SDPA + CCL cores together; strip the CCL contribution by
    # rounding down to the nearest full compute-grid row.
    return (measured_core_count // mesh_config.grid_rows) * mesh_config.grid_rows


# === TEST 4: PERFORMANCE TABLE GENERATOR (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.timeout(1000)
@pytest.mark.parametrize("model_name", TEST_CONFIG_MODELS)
def test_ring_joint_attention_create_perf_table(model_name):
    """
    Sweep chunk sizes for ring joint attention SDPA and print a performance table.
    Skipped on CI - run locally with tracy profiler.
    """
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
            profile_result = profile_ring_joint_sdpa_perf_command(command, subdir)
            measured_core_count = profile_result["measured_core_count"]
            duration_ns = profile_result["duration_ns"]
            fpu_util_min = profile_result["fpu_util_min"]
            fpu_util_max = profile_result["fpu_util_max"]

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

            effective_cores = compute_effective_ring_joint_compute_cores(measured_core_count, mesh_config)
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


# === PAGED TEST: HEAD-TO-HEAD PERFORMANCE TABLE GENERATOR (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.timeout(2000)
@pytest.mark.parametrize("model_name", TEST_CONFIG_MODELS)
def test_ring_joint_attention_create_paged_kv_overhead_perf_table(model_name):
    """
    Profile matched non-paged and paged Ring Joint SDPA configs and print a
    head-to-head overhead table.
    """
    mesh_config = MESH_CONFIG
    model_configs = MODEL_CONFIGS

    ring_size = mesh_config.sp_size
    if ring_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices, got {ring_size}")

    test_configs, test_config_ids = generate_test_configs(mesh_config, model_configs)
    paired_sweep_configs = []
    missing_paged_config_ids = []
    for config, config_id in zip(test_configs, test_config_ids):
        if not config_id.startswith(model_name):
            continue
        paged_config = PAGED_TEST_CONFIG_BY_BASE_ID.get(config_id)
        if paged_config is None:
            missing_paged_config_ids.append(config_id)
            continue
        paired_sweep_configs.append((config, paged_config, config_id))

    if not paired_sweep_configs:
        pytest.skip(f"No paged K/V perf configs match model {model_name}")

    for config_id in missing_paged_config_ids:
        logger.warning(f"No paged K/V perf config for {config_id}; skipping overhead comparison")

    subdir = "ttnn_ring_joint_sdpa_paged_kv_overhead"
    perf_results = []

    for config, paged_config, config_id in paired_sweep_configs:
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
        page_block_size = paged_config[-1]

        nonpaged_command = (
            f"pytest tests/nightly/blackhole/sdpa/"
            f"test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_sweep_perf_impl"
            f"[{config_id}]"
        )
        paged_command = (
            f"pytest tests/nightly/blackhole/sdpa/"
            f"test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_paged_kv_sweep_perf_impl"
            f"[{config_id}]"
        )

        local_seq_len = sq // ring_size
        local_nhq = nhq // mesh_config.tp_size

        try:
            nonpaged = profile_ring_joint_sdpa_perf_command(nonpaged_command, subdir)
            paged = profile_ring_joint_sdpa_perf_command(paged_command, subdir)

            nonpaged_effective_cores = compute_effective_ring_joint_compute_cores(
                nonpaged["measured_core_count"], mesh_config
            )
            paged_effective_cores = compute_effective_ring_joint_compute_cores(
                paged["measured_core_count"], mesh_config
            )
            nonpaged_duration_ns = nonpaged["duration_ns"]
            paged_duration_ns = paged["duration_ns"]
            nonpaged_utilization = compute_ring_joint_utilization(
                local_seq_len,
                sq,
                d_q,
                d_v,
                local_nhq,
                nonpaged_duration_ns,
                nonpaged_effective_cores,
                is_causal,
            )
            paged_utilization = compute_ring_joint_utilization(
                local_seq_len,
                sq,
                d_q,
                d_v,
                local_nhq,
                paged_duration_ns,
                paged_effective_cores,
                is_causal,
            )
            overhead_pct = (
                ((paged_duration_ns - nonpaged_duration_ns) / nonpaged_duration_ns) * 100
                if nonpaged_duration_ns > 0
                else 0
            )

            perf_results.append(
                {
                    "config_id": config_id,
                    "q_chunk_size": q_chunk_size,
                    "k_chunk_size": k_chunk_size,
                    "page_block_size": page_block_size,
                    "nonpaged_duration_ms": nonpaged_duration_ns / 1e6,
                    "paged_duration_ms": paged_duration_ns / 1e6,
                    "overhead_pct": overhead_pct,
                    "nonpaged_cores": nonpaged_effective_cores,
                    "paged_cores": paged_effective_cores,
                    "nonpaged_utilization": nonpaged_utilization,
                    "paged_utilization": paged_utilization,
                    "nonpaged_fpu_util_min": nonpaged["fpu_util_min"],
                    "nonpaged_fpu_util_max": nonpaged["fpu_util_max"],
                    "paged_fpu_util_min": paged["fpu_util_min"],
                    "paged_fpu_util_max": paged["fpu_util_max"],
                }
            )

            logger.info(
                f"{config_id}: non-paged={nonpaged_duration_ns/1e6:.3f} ms, "
                f"paged={paged_duration_ns/1e6:.3f} ms, overhead={overhead_pct:.1f}%"
            )

        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            logger.error(f"Error profiling paged K/V overhead for {config_id}: {e}")
            perf_results.append(
                {
                    "config_id": config_id,
                    "q_chunk_size": q_chunk_size,
                    "k_chunk_size": k_chunk_size,
                    "page_block_size": page_block_size,
                    "nonpaged_duration_ms": None,
                    "paged_duration_ms": None,
                }
            )

    valid_results = [r for r in perf_results if r["nonpaged_duration_ms"] is not None]

    print(f"\n{'='*170}")
    print(f"Ring Joint Attention Paged K/V Overhead Sweep ({model_name.upper()})")
    print(f"Architecture: {mesh_config.arch_type}, Ring size: {ring_size} devices, TP size: {mesh_config.tp_size}")
    print(f"{'='*170}")
    header = (
        "| Config | q_chunk | k_chunk | page | Non-Paged (ms) | Paged (ms) | Overhead | "
        "Cores NP/P | Math Util NP/P | FPU Util NP/P |"
    )
    sep = "|--------|---------|---------|------|----------------|------------|----------|------------|----------------|---------------|"
    print(header)
    print(sep)

    for result in valid_results:
        nonpaged_fpu = f"{result['nonpaged_fpu_util_min']:.1f}-{result['nonpaged_fpu_util_max']:.1f}"
        paged_fpu = f"{result['paged_fpu_util_min']:.1f}-{result['paged_fpu_util_max']:.1f}"
        print(
            f"| {result['config_id']} | {result['q_chunk_size']:7d} | {result['k_chunk_size']:7d} | "
            f"{result['page_block_size']:4d} | {result['nonpaged_duration_ms']:14.3f} | "
            f"{result['paged_duration_ms']:10.3f} | {result['overhead_pct']:7.1f}% | "
            f"{result['nonpaged_cores']:4d}/{result['paged_cores']:<4d} | "
            f"{result['nonpaged_utilization']:6.1f}%/{result['paged_utilization']:<6.1f}% | "
            f"{nonpaged_fpu}/{paged_fpu} |"
        )

    failed_results = [r for r in perf_results if r["nonpaged_duration_ms"] is None]
    if failed_results:
        print("\nFailed configurations:")
        for result in failed_results:
            print(f"  {result['config_id']}")

    if valid_results:
        avg_overhead = sum(result["overhead_pct"] for result in valid_results) / len(valid_results)
        worst = max(valid_results, key=lambda result: result["overhead_pct"])
        best = min(valid_results, key=lambda result: result["overhead_pct"])
        print(
            f"\nAverage overhead: {avg_overhead:.1f}% | "
            f"Best: {best['config_id']} ({best['overhead_pct']:.1f}%) | "
            f"Worst: {worst['config_id']} ({worst['overhead_pct']:.1f}%)"
        )

    print(f"{'='*170}\n")


# === PAGED TEST: PAGE-SIZE SENSITIVITY TABLE GENERATOR (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.timeout(3000)
@pytest.mark.parametrize(
    "model_name, q_chunk_size, k_chunk_size",
    PAGED_KV_PAGE_SIZE_PERF_TARGETS,
    ids=[f"{target[0]}-q{target[1]}-k{target[2]}" for target in PAGED_KV_PAGE_SIZE_PERF_TARGETS],
)
def test_ring_joint_attention_create_paged_kv_page_size_perf_table(model_name, q_chunk_size, k_chunk_size):
    """
    Profile matched non-paged and paged Ring Joint SDPA variants across page
    sizes and page-table modes.
    """
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices, got {mesh_config.sp_size}")

    if model_name not in MODEL_CONFIGS:
        pytest.skip(f"Model {model_name} not available for current mesh config")

    model = MODEL_CONFIGS[model_name]
    config_id = get_test_case_id(model, q_chunk_size, k_chunk_size)
    if config_id not in TEST_CONFIG_IDS:
        pytest.skip(f"Config {config_id} is not available for current mesh config")

    config = TEST_CONFIGS[TEST_CONFIG_IDS.index(config_id)]
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

    nonpaged_command = (
        f"pytest tests/nightly/blackhole/sdpa/"
        f"test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_sweep_perf_impl"
        f"[{config_id}]"
    )
    subdir = "ttnn_ring_joint_sdpa_paged_kv_page_size"
    negligible_overhead_pct = 5.0
    local_seq_len = sq // mesh_config.sp_size
    local_nhq = nhq // mesh_config.tp_size

    nonpaged = profile_ring_joint_sdpa_perf_command(nonpaged_command, subdir)
    nonpaged_effective_cores = compute_effective_ring_joint_compute_cores(nonpaged["measured_core_count"], mesh_config)
    nonpaged_duration_ns = nonpaged["duration_ns"]
    nonpaged_utilization = compute_ring_joint_utilization(
        local_seq_len,
        sq,
        d_q,
        d_v,
        local_nhq,
        nonpaged_duration_ns,
        nonpaged_effective_cores,
        is_causal,
    )

    valid_page_sizes = get_valid_paged_kv_page_block_sizes(sq, mesh_config.sp_size)
    perf_results = []
    for page_table_mode in PAGED_KV_PERF_PAGE_TABLE_MODES:
        for page_block_size in valid_page_sizes:
            variant_id = f"{config_id}-page{page_block_size}-{page_table_mode}"
            if variant_id not in PAGED_PERF_VARIANT_CONFIG_BY_ID:
                continue

            paged_command = (
                f"pytest tests/nightly/blackhole/sdpa/"
                f"test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_paged_kv_perf_variant_impl"
                f"[{variant_id}]"
            )

            try:
                paged = profile_ring_joint_sdpa_perf_command(paged_command, subdir)
                paged_effective_cores = compute_effective_ring_joint_compute_cores(
                    paged["measured_core_count"], mesh_config
                )
                paged_duration_ns = paged["duration_ns"]
                paged_utilization = compute_ring_joint_utilization(
                    local_seq_len,
                    sq,
                    d_q,
                    d_v,
                    local_nhq,
                    paged_duration_ns,
                    paged_effective_cores,
                    is_causal,
                )
                overhead_pct = ((paged_duration_ns - nonpaged_duration_ns) / nonpaged_duration_ns) * 100
                perf_results.append(
                    {
                        "page_table_mode": page_table_mode,
                        "page_block_size": page_block_size,
                        "pages_per_sequence": sq // page_block_size,
                        "paged_duration_ms": paged_duration_ns / 1e6,
                        "overhead_pct": overhead_pct,
                        "paged_cores": paged_effective_cores,
                        "paged_utilization": paged_utilization,
                        "paged_fpu_util_min": paged["fpu_util_min"],
                        "paged_fpu_util_max": paged["fpu_util_max"],
                    }
                )
                logger.info(
                    f"{variant_id}: page={page_block_size}, mode={page_table_mode}, "
                    f"paged={paged_duration_ns/1e6:.3f} ms, overhead={overhead_pct:.1f}%"
                )
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
                logger.error(f"Error profiling page-size variant {variant_id}: {e}")

    print(f"\n{'='*170}")
    print(f"Ring Joint Attention Paged K/V Page-Size Sensitivity ({config_id})")
    print(f"Architecture: {mesh_config.arch_type}, Ring size: {mesh_config.sp_size}, TP size: {mesh_config.tp_size}")
    print(
        f"Non-paged baseline: {nonpaged_duration_ns/1e6:.3f} ms, "
        f"math_util={nonpaged_utilization:.1f}%, cores={nonpaged_effective_cores}"
    )
    print(f"{'='*170}")
    header = "| Mode | Page | Pages/Seq | Paged (ms) | Overhead | Cores | Math Util | " "FPU Util (%) | Negligible |"
    sep = "|------|------|-----------|------------|----------|-------|-----------|--------------|------------|"
    print(header)
    print(sep)

    for result in perf_results:
        fpu_range = f"{result['paged_fpu_util_min']:.1f}-{result['paged_fpu_util_max']:.1f}"
        negligible = "yes" if abs(result["overhead_pct"]) <= negligible_overhead_pct else "no"
        print(
            f"| {result['page_table_mode']} | {result['page_block_size']:4d} | "
            f"{result['pages_per_sequence']:9d} | {result['paged_duration_ms']:10.3f} | "
            f"{result['overhead_pct']:7.1f}% | {result['paged_cores']:5d} | "
            f"{result['paged_utilization']:8.1f}% | {fpu_range:>12} | {negligible:>10} |"
        )

    for page_table_mode in PAGED_KV_PERF_PAGE_TABLE_MODES:
        mode_results = [result for result in perf_results if result["page_table_mode"] == page_table_mode]
        negligible_results = [
            result for result in mode_results if abs(result["overhead_pct"]) <= negligible_overhead_pct
        ]
        if negligible_results:
            smallest = min(negligible_results, key=lambda result: result["page_block_size"])
            print(
                f"\nSmallest negligible-overhead page for {page_table_mode}: "
                f"{smallest['page_block_size']} ({smallest['overhead_pct']:.1f}%)"
            )
        elif mode_results:
            best = min(mode_results, key=lambda result: abs(result["overhead_pct"]))
            print(
                f"\nNo negligible-overhead page found for {page_table_mode}; "
                f"best page {best['page_block_size']} overhead {best['overhead_pct']:.1f}%"
            )

    print(f"{'='*170}\n")


# === TEST 5: PERFORMANCE CHECK (CI-gated by SDPA_PERF_CHECKS=1) ===
# Symmetric +/- band — catches both regressions and unexpected speedups.
RING_JOINT_PERF_MARGIN = 0.005

RING_JOINT_PERF_CHECK_CONFIGS = [
    # (model_name, q_chunk_size, k_chunk_size, ring_size, expected_util)
    # 4-device ring (QuietBox)
    ("wan2_2_1xGLX", 288, 512, 4, 68.9),
    ("mla_100k", 160, 320, 4, 62.9),
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

    # Match perf-table effective_cores rounding.
    effective_cores = compute_effective_ring_joint_compute_cores(measured_core_count, MESH_CONFIG)
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
