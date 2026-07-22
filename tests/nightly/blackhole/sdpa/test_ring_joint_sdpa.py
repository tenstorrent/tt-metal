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
- Minimax3 GQA: nhk == nhv < nhq, d_q == d_k == d_v == 128, bfloat16 for Q, bfloat8_b for K/V

BH adaptation: uses init_device_compute_kernel_config instead of WormholeComputeKernelConfig.
"""

import math
import os
from dataclasses import dataclass, replace
from itertools import product
from typing import Dict, List, Sequence, Tuple
from unittest import mock

import pytest
import torch
from loguru import logger
from ttnn.operations.ccl import Topology

import ttnn
from models.common.utility_functions import skip_with_llk_assert

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
from tests.nightly.sdpa_perf_utils import MeshConfig
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)

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

    # Dtypes
    q_dtype: ttnn.DataType
    kv_dtype: ttnn.DataType

    # Sweep parameters (chunk sizes to test)
    q_chunk_sizes: List[int]
    k_chunk_sizes: List[int]

    # Can be hardware-dependent to keep per-core work balanced between Galaxy and Quiet Box
    seq_len: int

    is_balanced: bool = False


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
            q_dtype=ttnn.bfloat16,
            kv_dtype=ttnn.bfloat16,
            q_chunk_sizes=[224, 256, 288],
            k_chunk_sizes=[128, 256, 512],
            seq_len=9472 if mesh_config.is_galaxy else 8544,  # Tuned for each platform to match per-core work
        )
    )

    configs.append(
        ModelConfig(
            name="minimax3_gqa_smoke",
            nhq=16,
            nhk=1,
            nhv=1,
            d_q=128,
            d_k=128,
            d_v=128,
            is_causal=True,
            is_balanced=True,
            q_dtype=ttnn.bfloat16,
            kv_dtype=ttnn.bfloat8_b,
            # q_chunk=256 (Sq_chunk_t > 2*qktv_h) exercises the full-size cb_out streaming path on the
            # production row-wide multicast transport — guards the streaming cb_out shrink fix.
            q_chunk_sizes=[128, 256],
            k_chunk_sizes=[512],
            seq_len=1024,
        )
    )

    configs.append(
        ModelConfig(
            name="minimax3_gqa_unicast_smoke",
            nhq=16,
            nhk=2,
            nhv=2,
            d_q=128,
            d_k=128,
            d_v=128,
            is_causal=True,
            is_balanced=True,
            q_dtype=ttnn.bfloat16,
            kv_dtype=ttnn.bfloat8_b,
            # Grouped-unicast fallback (nhk>1) is otherwise only nominally exercised, so sweep a
            # small q/k chunk cross-product over a longer prefix instead of a single shape.
            q_chunk_sizes=[128, 256],
            k_chunk_sizes=[256, 512],
            seq_len=1024,
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


def generate_ring_joint_perf_model_configs(
    mesh_config: MeshConfig, model_configs: Dict[str, ModelConfig]
) -> Dict[str, ModelConfig]:
    perf_configs = dict(model_configs)

    perf_configs["minimax3_gqa_causal_perf"] = ModelConfig(
        name="minimax3_gqa_causal_perf",
        nhq=16,
        nhk=1,
        nhv=1,
        d_q=128,
        d_k=128,
        d_v=128,
        is_causal=True,
        is_balanced=True,
        q_dtype=ttnn.bfloat16,
        kv_dtype=ttnn.bfloat8_b,
        q_chunk_sizes=[128],
        k_chunk_sizes=[512],
        seq_len=4096,
    )

    return perf_configs


RING_JOINT_PERF_MODEL_CONFIGS = generate_ring_joint_perf_model_configs(MESH_CONFIG, MODEL_CONFIGS)


def model_uses_tp_replicated_shared_k(model: ModelConfig) -> bool:
    return model.nhk == 1 and model.nhv == model.nhq


def scaled_model_heads_for_mesh(model: ModelConfig, mesh_config: MeshConfig) -> Tuple[int, int, int]:
    nhq = model.nhq * mesh_config.tp_size
    nhk = model.nhk if model_uses_tp_replicated_shared_k(model) else model.nhk * mesh_config.tp_size
    nhv = model.nhv * mesh_config.tp_size
    return nhq, nhk, nhv


# Accuracy threshold constants
DEFAULT_PCC_THRESHOLD = 0.994
DEFAULT_RMSE_THRESHOLD = 0.05
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32

from tests.nightly.sdpa_perf_utils import (
    ARCH_CONSTANTS,
)
from tests.nightly.sdpa_perf_utils import compute_math_utilization as compute_ring_joint_utilization
from tests.nightly.sdpa_perf_utils import (
    compute_sdpa_flops,
    create_balanced_chunk_order,
    post_process_ops_log,
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


def deterministic_input_tensor(*shape, offset=0.0):
    """Generate cheap finite inputs for determinism tests, where accuracy is checked elsewhere."""
    _, h, s, d = shape
    seq_pattern = (torch.arange(s, dtype=torch.float32).view(1, 1, s, 1) % 251) * 0.001
    head_pattern = (torch.arange(h, dtype=torch.float32).view(1, h, 1, 1) % 31) * 0.01
    dim_pattern = (torch.arange(d, dtype=torch.float32).view(1, 1, 1, d) % 37) * 0.0001
    return (seq_pattern + head_pattern + dim_pattern + offset).expand(shape).contiguous()


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
        if t.shape[1] == H:
            return t[:, h_start:h_end]
        assert H % t.shape[1] == 0, f"Q heads must be divisible by KV heads, got H={H}, KV={t.shape[1]}"
        heads_per_kv = H // t.shape[1]
        kv_indices = torch.arange(h_start, h_end, device=t.device) // heads_per_kv
        return t[:, kv_indices]

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


def compute_pcc_rmse(expected, actual):
    if expected.dtype != actual.dtype:
        actual = actual.type(expected.dtype)

    expected_flat = expected.reshape(-1).float()
    actual_flat = actual.reshape(-1).float()
    diff = expected_flat - actual_flat
    rmse = torch.linalg.vector_norm(diff).item() / math.sqrt(diff.numel())

    if not math.isfinite(rmse):
        return 0.0, rmse
    if torch.equal(expected_flat, actual_flat):
        return 1.0, rmse
    if expected_flat.numel() == 1:
        return float(torch.equal(expected_flat, actual_flat)), rmse

    expected_centered = expected_flat.double()
    actual_centered = actual_flat.double()
    expected_centered = expected_centered - expected_centered.mean()
    actual_centered = actual_centered - actual_centered.mean()
    denom = torch.linalg.vector_norm(expected_centered) * torch.linalg.vector_norm(actual_centered)
    if denom == 0:
        return float(torch.isclose(torch.max(expected_flat), torch.max(actual_flat)).item()), rmse

    pcc = (torch.dot(expected_centered, actual_centered) / denom).clamp(-1.0, 1.0).item()
    return pcc, rmse


def assert_pcc_rmse(expected, actual, pcc_threshold, rmse_threshold, label):
    pcc, rmse = compute_pcc_rmse(expected, actual)
    logger.info(f"{label} - PCC: {pcc}, RMSE: {rmse:.6f}")

    if rmse_threshold is not None:
        assert rmse < rmse_threshold, f"{label} RMSE {rmse:.6f} exceeds threshold {rmse_threshold}"
    assert pcc >= pcc_threshold, f"{label} PCC {pcc} below threshold {pcc_threshold}"
    return pcc, rmse


def is_supported_ring_joint_head_mode(nhq, nhk, nhv):
    is_mha = nhq == nhk == nhv
    is_separate_v_shared_k = nhk == 1 and nhv == nhq
    is_gqa_grouped_kv = nhk == nhv < nhq and nhk > 0 and nhq % nhk == 0
    return is_mha or is_separate_v_shared_k or is_gqa_grouped_kv


def device_tensors_mismatch_marker(reference_tensor, actual_tensor):
    """Return a reduced device tensor that is non-zero when any mesh shard has an exact mismatch."""
    mismatch = ttnn.ne(reference_tensor, actual_tensor, dtype=ttnn.bfloat16)
    return ttnn.max(mismatch)


def merge_device_mismatch_markers(current_marker, new_marker):
    if current_marker is None:
        return new_marker
    return ttnn.maximum(current_marker, new_marker)


def device_mismatch_marker_is_set(mismatch_marker):
    mismatch_marker_host = ttnn.from_device(mismatch_marker)
    return any(float(ttnn.to_torch(shard).item()) != 0.0 for shard in ttnn.get_device_tensors(mismatch_marker_host))


@dataclass
class RingJointSDPARuntime:
    mesh_device: object
    topology: object
    sp_axis: int
    tp_axis: int
    num_links: int
    sdpa_compute_grid: Tuple[int, int]
    ccl_column: int
    worker_sub_device_id: object
    ccl_semaphore_handles: object
    compute_kernel_config: object


def open_ring_joint_sdpa_runtime(mesh_config):
    use_ring = mesh_config.sp_size > 2
    fabric_config = ttnn.FabricConfig.FABRIC_1D_RING if use_ring else ttnn.FabricConfig.FABRIC_1D
    topology = Topology.Ring if use_ring else Topology.Linear

    sp_axis = 1
    tp_axis = 0

    if mesh_config.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    mesh_device = None

    try:
        ttnn.set_fabric_config(
            fabric_config,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
        )

        mesh_shape = ttnn.MeshShape(mesh_config.tp_size, mesh_config.sp_size)
        mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

        full_compute_grid = mesh_device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
        )
        worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
        mesh_device.set_sub_device_stall_group([worker_sub_device_id])

        return RingJointSDPARuntime(
            mesh_device=mesh_device,
            topology=topology,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            num_links=2,
            sdpa_compute_grid=(mesh_config.sdpa_cols, mesh_config.grid_rows),
            ccl_column=mesh_config.ccl_column,
            worker_sub_device_id=worker_sub_device_id,
            ccl_semaphore_handles=create_global_semaphores(mesh_device, ccl_sub_device_crs, 0),
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
        )
    except Exception:
        try:
            if mesh_device is not None:
                ttnn.close_mesh_device(mesh_device)
        finally:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise


def close_ring_joint_sdpa_runtime(runtime: RingJointSDPARuntime, *, clear_program_cache=False):
    try:
        if clear_program_cache:
            runtime.mesh_device.disable_and_clear_program_cache()
    finally:
        try:
            ttnn.close_mesh_device(runtime.mesh_device)
        finally:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


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


def get_model_qk_configs(config: ModelConfig) -> List[Tuple[int, int]]:
    """Return all Q/K chunk-size combinations covered by a model config."""
    return list(product(config.q_chunk_sizes, config.k_chunk_sizes))


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
        nhq, nhk, nhv = scaled_model_heads_for_mesh(model, mesh_config)
        for q_chunk, k_chunk in product(model.q_chunk_sizes, model.k_chunk_sizes):
            configs.append(
                (
                    BATCH_SIZE,
                    model.seq_len * mesh_config.sp_size,  # Global sequence length across all devices in the ring
                    nhq,  # Total query heads across all TP shards
                    nhk,  # Total key heads across all TP shards
                    nhv,  # Total value heads across all TP shards
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
    kv_cache_batch_idx=None,
    kv_actual_isl=None,
    is_cross=False,
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
        is_cross=is_cross,
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
        kv_cache_batch_idx=kv_cache_batch_idx,
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

    if not is_supported_ring_joint_head_mode(nhq, nhk, nhv):
        pytest.skip(f"Unsupported ring joint attention heads: nhq={nhq}, nhk={nhk}, nhv={nhv}")

    runtime = open_ring_joint_sdpa_runtime(mesh_config)
    mesh_device = runtime.mesh_device
    topology = runtime.topology
    sp_axis = runtime.sp_axis
    tp_axis = runtime.tp_axis
    num_links = runtime.num_links
    sdpa_compute_grid = runtime.sdpa_compute_grid
    ccl_column = runtime.ccl_column
    worker_sub_device_id = runtime.worker_sub_device_id
    ccl_semaphore_handles = runtime.ccl_semaphore_handles
    compute_kernel_config = runtime.compute_kernel_config

    try:
        if mesh_config.tp_size > 1 and nhq % mesh_config.tp_size != 0:
            pytest.skip(
                f"num_heads ({nhq}) must be divisible by TP size ({mesh_config.tp_size}) for multi-ring architecture"
            )

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
        close_ring_joint_sdpa_runtime(runtime)


def run_ring_joint_sdpa_model_configs(
    mesh_config,
    model: ModelConfig,
    qk_configs: Sequence[Tuple[int, int]],
    *,
    pcc_threshold=DEFAULT_PCC_THRESHOLD,
    rmse_threshold=None,
    do_check=True,
    num_iterations=1,
    runtime: RingJointSDPARuntime = None,
):
    """Run all q/k configs for one model while reusing shared inputs, mesh setup, and reference data."""
    qk_configs = list(qk_configs)
    assert qk_configs, f"No q/k configs provided for {model.name}"

    torch.manual_seed(1234)

    b = BATCH_SIZE
    nhq, nhk, nhv = scaled_model_heads_for_mesh(model, mesh_config)
    sq = model.seq_len * mesh_config.sp_size
    d_q, d_k, d_v = model.d_q, model.d_k, model.d_v
    q_dtype, kv_dtype = model.q_dtype, model.kv_dtype
    is_causal, is_balanced = model.is_causal, model.is_balanced

    logger.debug(
        f"run_ring_joint_sdpa_model_configs params: model={model.name}, b={b}, "
        f"nhq={nhq}, nhk={nhk}, nhv={nhv}, sq={sq}, d_q={d_q}, d_k={d_k}, d_v={d_v}, "
        f"qk_configs={qk_configs}, q_dtype={q_dtype}, kv_dtype={kv_dtype}, "
        f"is_causal={is_causal}, is_balanced={is_balanced}, "
        f"pcc_threshold={pcc_threshold}, rmse_threshold={rmse_threshold}, "
        f"do_check={do_check}, num_iterations={num_iterations}"
    )

    if not is_supported_ring_joint_head_mode(nhq, nhk, nhv):
        pytest.skip(f"Unsupported ring joint attention heads: nhq={nhq}, nhk={nhk}, nhv={nhv}")

    owns_runtime = runtime is None
    if runtime is None:
        runtime = open_ring_joint_sdpa_runtime(mesh_config)

    mesh_device = runtime.mesh_device
    topology = runtime.topology
    sp_axis = runtime.sp_axis
    tp_axis = runtime.tp_axis
    num_links = runtime.num_links
    sdpa_compute_grid = runtime.sdpa_compute_grid
    ccl_column = runtime.ccl_column
    ccl_semaphore_handles = runtime.ccl_semaphore_handles
    compute_kernel_config = runtime.compute_kernel_config
    worker_sub_device_id = runtime.worker_sub_device_id

    try:
        if mesh_config.tp_size > 1 and nhq % mesh_config.tp_size != 0:
            pytest.skip(
                f"num_heads ({nhq}) must be divisible by TP size ({mesh_config.tp_size}) for multi-ring architecture"
            )

        if num_iterations > 1:
            Q = deterministic_input_tensor(b, nhq, sq, d_q, offset=0.125)
            K = deterministic_input_tensor(b, nhk, sq, d_k, offset=0.25)
            V = deterministic_input_tensor(b, nhv, sq, d_v, offset=0.375)
        else:
            Q = fa_rand(b, nhq, sq, d_q)
            K = fa_rand(b, nhk, sq, d_k)
            V = fa_rand(b, nhv, sq, d_v)
        Q_original, K_original, V_original = Q, K, V

        chunk_order = None
        if is_balanced:
            chunk_order = create_balanced_chunk_order(mesh_config.sp_size)
            Q = reorder_tensor_chunks(Q, chunk_order)
            K = reorder_tensor_chunks(K, chunk_order)
            V = reorder_tensor_chunks(V, chunk_order)

        kv_shard_dims = [None, None]
        kv_shard_dims[sp_axis] = None
        if mesh_config.tp_size > 1:
            kv_shard_dims[tp_axis] = 1

        persistent_k_shard_dims = [None, None]
        persistent_k_shard_dims[sp_axis] = None
        if mesh_config.tp_size > 1 and nhk != 1:
            persistent_k_shard_dims[tp_axis] = 1

        sdpa_input_shard_dims = [None, None]
        sdpa_input_shard_dims[sp_axis] = 2
        if mesh_config.tp_size > 1:
            sdpa_input_shard_dims[tp_axis] = 1

        sdpa_k_shard_dims = [None, None]
        sdpa_k_shard_dims[sp_axis] = 2
        if mesh_config.tp_size > 1 and nhk != 1:
            sdpa_k_shard_dims[tp_axis] = 1

        tt_Q = ttnn.from_torch(
            Q,
            dtype=q_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
            ),
        )
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

        main_row_dim = sdpa_input_shard_dims[0] if sdpa_input_shard_dims[0] is not None else -1
        main_col_dim = sdpa_input_shard_dims[1] if sdpa_input_shard_dims[1] is not None else -1
        mesh_composer = ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(main_row_dim, main_col_dim))

        gt_main = None
        if do_check and num_iterations == 1:
            gt_main = torch_sdpa_reference(Q_original, K_original, V_original, is_causal=is_causal)

        use_device_determinism_compare = num_iterations > 1 and sq % ttnn.TILE_SIZE == 0 and d_v % ttnn.TILE_SIZE == 0

        def create_persistent_buffers():
            persistent_output_buffer_k = ttnn.from_torch(
                torch.zeros(b, nhk, sq, d_k),
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_k_shard_dims
                ),
            )
            persistent_output_buffer_v = ttnn.from_torch(
                torch.zeros(b, nhv, sq, d_v),
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kv_shard_dims
                ),
            )
            return persistent_output_buffer_k, persistent_output_buffer_v

        persistent_output_buffer_k, persistent_output_buffer_v = create_persistent_buffers()
        model_mismatch_marker = None
        model_mismatch_markers_by_config = []
        for q_chunk_size, k_chunk_size in qk_configs:
            config_id = get_test_case_id(model, q_chunk_size, k_chunk_size)
            logger.info(f"Running ring joint SDPA {config_id}")

            program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=sdpa_compute_grid,
                q_chunk_size=q_chunk_size,
                k_chunk_size=k_chunk_size,
                exp_approx_mode=False,
            )

            reference_output = None
            reference_tt_out = None
            mismatch_marker = None
            tt_out_torch = None
            for i in range(num_iterations):
                tt_out = call_sdpa(
                    tt_Q,
                    tt_K,
                    tt_V,
                    sq,
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

                if use_device_determinism_compare:
                    if reference_tt_out is None:
                        reference_tt_out = tt_out
                    else:
                        mismatch_marker = merge_device_mismatch_markers(
                            mismatch_marker,
                            device_tensors_mismatch_marker(reference_tt_out, tt_out),
                        )
                    continue

                tt_out_torch = ttnn.to_torch(tt_out, mesh_composer=mesh_composer)
                tt_out_torch = tt_out_torch[:, :, :sq, :]

                if do_check and num_iterations == 1 and is_balanced and chunk_order is not None:
                    tt_out_torch = reverse_reorder_tensor_chunks(tt_out_torch, chunk_order)

                if num_iterations > 1:
                    if reference_output is None:
                        reference_output = tt_out_torch
                    elif not torch.equal(reference_output, tt_out_torch):
                        diff_mask = reference_output != tt_out_torch
                        num_diffs = diff_mask.sum().item()
                        max_diff = (reference_output - tt_out_torch).abs().max().item()
                        pytest.fail(
                            f"Ring joint SDPA {config_id} output at iteration {i} differs from iteration 0: "
                            f"{num_diffs} differing elements, max diff = {max_diff}"
                        )

            if num_iterations > 1:
                if use_device_determinism_compare and mismatch_marker is not None:
                    model_mismatch_marker = merge_device_mismatch_markers(model_mismatch_marker, mismatch_marker)
                    model_mismatch_markers_by_config.append((config_id, mismatch_marker))
                elif not use_device_determinism_compare:
                    logger.info(f"Ring joint SDPA {config_id} determinism verified: all {num_iterations} outputs match")
            elif do_check:
                assert_pcc_rmse(gt_main, tt_out_torch, pcc_threshold, rmse_threshold, f"Main output {config_id}")

        if num_iterations > 1 and use_device_determinism_compare and model_mismatch_marker is not None:
            if device_mismatch_marker_is_set(model_mismatch_marker):
                for config_id, mismatch_marker in model_mismatch_markers_by_config:
                    if device_mismatch_marker_is_set(mismatch_marker):
                        pytest.fail(f"Ring joint SDPA {config_id} produced output that differs from iteration 0")
                pytest.fail(f"Ring joint SDPA {model.name} produced output that differs from iteration 0")
            for config_id, _ in model_mismatch_markers_by_config:
                logger.info(f"Ring joint SDPA {config_id} determinism verified: all {num_iterations} outputs match")

    finally:
        if owns_runtime:
            close_ring_joint_sdpa_runtime(runtime)


def run_ring_mla_sdpa(
    mesh_config,
    b,
    nhq,
    nhk,
    sq,
    d_q,
    d_k,
    d_v,
    q_chunk_size,
    k_chunk_size,
    q_dtype,
    kv_dtype,
    *,
    is_balanced=True,
    pcc_threshold=DEFAULT_PCC_THRESHOLD,
    rmse_threshold=DEFAULT_RMSE_THRESHOLD,
    do_check=True,
    num_iterations=1,
    kv_cache_batch_idx=None,
    cache_batch=2,
):
    """Run ring_mla where V is the first d_v columns of the single KV tensor."""
    if mesh_config.sp_size < 2:
        pytest.skip(f"ring_mla requires at least 2 devices in ring, got SP={mesh_config.sp_size}")
    if nhk != 1:
        pytest.skip(f"Focused ring_mla test covers MLA shared-KV-head shapes, got nhk={nhk}")

    torch.manual_seed(1234)

    runtime = open_ring_joint_sdpa_runtime(mesh_config)
    mesh_device = runtime.mesh_device
    topology = runtime.topology
    sp_axis = runtime.sp_axis
    tp_axis = runtime.tp_axis
    num_links = runtime.num_links
    sdpa_compute_grid = runtime.sdpa_compute_grid
    ccl_column = runtime.ccl_column
    worker_sub_device_id = runtime.worker_sub_device_id
    ccl_semaphore_handles = runtime.ccl_semaphore_handles
    compute_kernel_config = runtime.compute_kernel_config

    try:
        Q = fa_rand(b, nhq, sq, d_q)
        KV = fa_rand(b, nhk, sq, d_k)
        V_prefix = KV[:, :, :, :d_v]
        Q_original, KV_original, V_original = Q, KV, V_prefix
        if kv_cache_batch_idx is not None:
            assert b == 1, f"ring_mla indexed K/V cache requires Q batch 1, got {b}"
            assert (
                0 <= kv_cache_batch_idx < cache_batch
            ), f"kv_cache_batch_idx {kv_cache_batch_idx} must be in [0, {cache_batch})"

        chunk_order = None
        if is_balanced:
            chunk_order = create_balanced_chunk_order(mesh_config.sp_size)
            Q = reorder_tensor_chunks(Q, chunk_order)
            KV = reorder_tensor_chunks(KV, chunk_order)

        kv_cache_batch = b
        KV_input = KV
        if kv_cache_batch_idx is not None:
            kv_cache_batch = cache_batch
            KV_input = fa_rand(cache_batch, nhk, sq, d_k)
            KV_input[kv_cache_batch_idx : kv_cache_batch_idx + 1] = KV

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_compute_grid,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        sdpa_input_shard_dims = [None, None]
        sdpa_input_shard_dims[sp_axis] = 2
        if mesh_config.tp_size > 1:
            sdpa_input_shard_dims[tp_axis] = 1

        sdpa_kv_shard_dims = [None, None]
        sdpa_kv_shard_dims[sp_axis] = 2

        persistent_kv_shard_dims = [None, None]
        persistent_kv_shard_dims[sp_axis] = None

        tt_Q = ttnn.from_torch(
            Q,
            dtype=q_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
            ),
        )
        tt_KV = ttnn.from_torch(
            KV_input,
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_kv_shard_dims
            ),
        )
        persistent_output_buffer_kv = ttnn.from_torch(
            torch.zeros(kv_cache_batch, nhk, sq, d_k),
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_kv_shard_dims
            ),
        )

        main_row_dim = sdpa_input_shard_dims[0] if sdpa_input_shard_dims[0] is not None else -1
        main_col_dim = sdpa_input_shard_dims[1] if sdpa_input_shard_dims[1] is not None else -1
        reference_output = None
        for i in range(num_iterations):
            tt_out, _ = ttnn.transformer.ring_mla(
                tt_Q,
                tt_KV,
                persistent_output_buffer_kv=persistent_output_buffer_kv,
                head_dim_v=d_v,
                logical_n=sq,
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
                kv_cache_batch_idx=kv_cache_batch_idx,
            )

            tt_out_torch = ttnn.to_torch(
                tt_out,
                mesh_composer=ttnn.create_mesh_composer(
                    mesh_device, ttnn.MeshComposerConfig(main_row_dim, main_col_dim)
                ),
            )
            tt_out_torch = tt_out_torch[:, :, :sq, :d_v]
            if is_balanced and chunk_order is not None:
                tt_out_torch = reverse_reorder_tensor_chunks(tt_out_torch, chunk_order)

            if num_iterations > 1:
                if reference_output is None:
                    reference_output = tt_out_torch
                elif not torch.equal(reference_output, tt_out_torch):
                    max_diff = (reference_output - tt_out_torch).abs().max().item()
                    pytest.fail(f"ring_mla output at iteration {i} differs from iteration 0, max diff={max_diff}")

        if num_iterations > 1 or not do_check:
            return

        gt_main = torch_sdpa_reference(Q_original, KV_original, V_original, is_causal=True)
        out_pass, out_pcc = comp_pcc(gt_main, tt_out_torch, pcc_threshold)
        rmse = torch.sqrt(((gt_main - tt_out_torch) ** 2).mean()).item()
        logger.info(f"ring_mla output - PCC: {out_pcc}, RMSE: {rmse:.6f}")
        assert rmse < rmse_threshold, f"ring_mla RMSE {rmse:.6f} exceeds threshold {rmse_threshold}"
        assert out_pass, f"ring_mla PCC {out_pcc} below threshold {pcc_threshold}"

    finally:
        close_ring_joint_sdpa_runtime(runtime)


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
# Hardware-dependent to keep per-core work balanced: galaxy has 110 SDPA cores, the
# non-galaxy single ring has 100 (10x10), so it carries fewer heads per ring.
CHUNKED_PREFILL_HEADS_PER_RING = 16 if MESH_CONFIG.is_galaxy else 14
CHUNKED_PREFILL_SEED = 1234
# Set to a chunk index to run/profile only that chunk in isolation instead of the
# full chunked-prefill sequence. Honored by both the accuracy and perf-table tests.
CHUNKED_PREFILL_CHUNK_ID_ENV = "RING_JOINT_CHUNKED_CHUNK_ID"


def get_chunked_only_chunk_id(n_chunks: int):
    """Parse RING_JOINT_CHUNKED_CHUNK_ID; return the chunk index to run, or None for all chunks."""
    raw = os.environ.get(CHUNKED_PREFILL_CHUNK_ID_ENV)
    if raw is None or raw == "":
        return None
    try:
        only_chunk = int(raw)
    except ValueError as e:
        raise AssertionError(f"{CHUNKED_PREFILL_CHUNK_ID_ENV}={raw!r} is not an integer") from e
    assert 0 <= only_chunk < n_chunks, f"{CHUNKED_PREFILL_CHUNK_ID_ENV}={only_chunk} out of range [0, {n_chunks})"
    return only_chunk


def run_ring_joint_sdpa_chunked(
    mesh_config,
    model: ModelConfig,
    chunk_size: int = CHUNKED_PREFILL_CHUNK_SIZE,
    total_seq: int = CHUNKED_PREFILL_TOTAL_SEQ,
    pcc_threshold: float = CHUNKED_PREFILL_PCC_THRESHOLD,
    rmse_threshold: float = None,
    q_chunk_size: int = None,
    k_chunk_size: int = None,
    qk_configs: Sequence[Tuple[int, int]] = None,
    num_iterations: int = 1,
    indexed_nd_sharded_kv_cache: bool = False,
    kv_cache_batch_idx: int = 1,
    cache_batch: int = 2,
    persistent_buffer_mode: str = "exact_per_chunk",
    use_ring_mla: bool = False,
    do_check: bool = True,
    reuse_kv_buffer: bool = False,
):
    """
    Validate ring joint SDPA chunked-prefill, or verify deterministic replay.

    SUT: n_chunks calls; each call passes a short Q chunk at absolute positions
    [i*c, (i+1)*c) against a K/V cache holding the first (i+1)*c rows.
    If persistent_buffer_mode="reuse_max", one max-length persistent K/V buffer
    pair is allocated once and reused across all chunks and iterations.

    num_iterations > 1 switches to determinism mode: each chunk's uploaded
    device inputs are reused for num_iterations calls, and later outputs must
    be bit-exact with iteration 0. PCC/RMSE coverage is handled by accuracy tests.

    use_ring_mla=True drives ttnn.transformer.ring_mla instead of the classic
    separate-K/V op: K and V live in a single latent K/V tensor (width d_k) and V
    is its first d_v columns. This is the MLA latent-V deployment shape.
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
    if use_ring_mla:
        assert model.nhk == 1, f"ring_mla requires one shared latent K/V head, got nhk={model.nhk}"
        assert not indexed_nd_sharded_kv_cache, "ring_mla chunked path here does not use the indexed ND-sharded cache"
        assert model.d_v <= model.d_k, f"latent V (d_v={model.d_v}) must fit within the K/V latent (d_k={model.d_k})"

    # reuse_kv_buffer: reuse one fixed, oversized KV cache across all chunks (logical_n / kv_actual_isl
    # grow per chunk) instead of a fresh right-sized input each chunk. Perf-only (pad-rotation permutes
    # the output); covers both ring_mla and classic separate-K/V.
    if reuse_kv_buffer:
        assert persistent_buffer_mode == "reuse_max", "reuse_kv_buffer requires reuse_max"
        assert not do_check, "reuse_kv_buffer is perf-only (pad-rotation permutes output; no PCC check)"
        assert num_iterations == 1, "reuse_kv_buffer is a single-pass perf path"
        assert not indexed_nd_sharded_kv_cache, "reuse_kv_buffer uses its own oversized cache layout"
    if indexed_nd_sharded_kv_cache:
        assert BATCH_SIZE == 1, "Indexed K/V cache test path assumes query batch is 1"
        assert (
            0 <= kv_cache_batch_idx < cache_batch
        ), f"kv_cache_batch_idx {kv_cache_batch_idx} must be in [0, {cache_batch})"
        assert (
            persistent_buffer_mode == "exact_per_chunk"
        ), "Reusable max buffers are not combined with indexed K/V cache"

    n_chunks = total_seq // chunk_size

    # Optional single-chunk mode: run only the chunk whose index matches the env var.
    # Each chunk rebuilds its K/V cache from scratch (to_balanced_growing_cache_layout),
    # so chunk i is independent of chunks 0..i-1 and its device kernel duration is
    # identical whether run in isolation or as part of the full sequence.
    only_chunk = get_chunked_only_chunk_id(n_chunks)

    if qk_configs is None:
        if q_chunk_size is None:
            q_chunk_size = model.q_chunk_sizes[0]
        if k_chunk_size is None:
            k_chunk_size = model.k_chunk_sizes[0]
        qk_configs = [(q_chunk_size, k_chunk_size)]
    else:
        qk_configs = list(qk_configs)
        assert qk_configs, f"No q/k configs provided for chunked model {model.name}"

    # model.nhq/nhk/nhv are PER RING; scale to total head counts across all TP shards.
    # Only the MLA-style shared-K path keeps nhk=1 replicated across TP.
    nhq, nhk, nhv = scaled_model_heads_for_mesh(model, mesh_config)
    d_q, d_k, d_v = model.d_q, model.d_k, model.d_v
    q_dtype, kv_dtype = model.q_dtype, model.kv_dtype
    # Chunked cache growth is linear in sequence order; balanced zigzag applies only to full prefill.
    is_balanced = False

    b = BATCH_SIZE

    runtime = open_ring_joint_sdpa_runtime(mesh_config)
    mesh_device = runtime.mesh_device
    topology = runtime.topology
    sp_axis = runtime.sp_axis
    tp_axis = runtime.tp_axis
    num_links = runtime.num_links
    sdpa_compute_grid = runtime.sdpa_compute_grid
    ccl_column = runtime.ccl_column
    worker_sub_device_id = runtime.worker_sub_device_id
    ccl_semaphore_handles = runtime.ccl_semaphore_handles
    compute_kernel_config = runtime.compute_kernel_config

    try:
        torch.manual_seed(CHUNKED_PREFILL_SEED)
        if num_iterations > 1:
            Q_full = deterministic_input_tensor(b, nhq, total_seq, d_q, offset=0.125)
            K_full = deterministic_input_tensor(b, nhk, total_seq, d_k, offset=0.25)
        else:
            Q_full = fa_rand(b, nhq, total_seq, d_q)
            K_full = fa_rand(b, nhk, total_seq, d_k)

        if use_ring_mla:
            # MLA latent: a single shared K/V tensor; V is its first d_v columns.
            V_full = K_full[:, :, :, :d_v]
        elif num_iterations > 1:
            V_full = deterministic_input_tensor(b, nhv, total_seq, d_v, offset=0.375)
        else:
            V_full = fa_rand(b, nhv, total_seq, d_v)

        # do_check=False (perf/profiling) skips the full-sequence CPU torch reference: it is an
        # O(total_seq^2) host SDPA that dominates wall-clock, while the device kernel — the only
        # thing the profiler measures — still runs below. Mirrors run_ring_joint_sdpa's do_check.
        ref_full = None
        if num_iterations == 1 and do_check:
            ref_full = torch_sdpa_reference(Q_full, K_full, V_full, is_causal=True)

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

        program_configs = {
            (q_chunk, k_chunk): ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=sdpa_compute_grid,
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            )
            for q_chunk, k_chunk in qk_configs
        }

        use_device_determinism_compare = (
            num_iterations > 1 and chunk_size % ttnn.TILE_SIZE == 0 and d_v % ttnn.TILE_SIZE == 0
        )

        main_row_dim = sdpa_input_shard_dims[0] if sdpa_input_shard_dims[0] is not None else -1
        main_col_dim = sdpa_input_shard_dims[1] if sdpa_input_shard_dims[1] is not None else -1
        mesh_composer = ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(main_row_dim, main_col_dim))

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
            out = ttnn.to_torch(tt_out, mesh_composer=mesh_composer)
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

        # ring_mla uses one shared latent K/V tensor (width d_k); V is its first d_v columns.
        ring_mla_kv_shard_dims = [None, None]
        ring_mla_kv_shard_dims[sp_axis] = 2  # input KV sharded along seq across the ring
        ring_mla_persistent_shard_dims = [None, None]  # gathered KV is replicated (full seq, single head)

        def upload_kv(kv_host):
            return ttnn.from_torch(
                kv_host,
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=ring_mla_kv_shard_dims
                ),
            )

        def create_ring_mla_kv_buffer(seq_len, kv_buffer_batch):
            return ttnn.from_torch(
                torch.zeros(kv_buffer_batch, nhk, seq_len, d_k),
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=ring_mla_persistent_shard_dims
                ),
            )

        logger.info(
            f"Chunked prefill: model={model.name}, use_ring_mla={use_ring_mla}, total_seq={total_seq}, "
            f"sp_size={sp_size}, per-device Q seq_len={total_seq // sp_size}, "
            f"qk_configs={qk_configs}, "
            f"indexed_nd_sharded_kv_cache={indexed_nd_sharded_kv_cache}, "
            f"persistent_buffer_mode={persistent_buffer_mode}"
        )

        # Chunked K/V cache layout: device d's local slab c holds global rows
        # [c*chunk_size + d*slab_rows, c*chunk_size + (d+1)*slab_rows). This packing is
        # independent of the is_balanced causal zigzag mode; the cache grows one chunk per call.
        slab_rows = chunk_size // sp_size
        assert chunk_size % sp_size == 0, f"chunk_size {chunk_size} not divisible by sp_size {sp_size}"
        assert slab_rows % 32 == 0, f"slab_rows {slab_rows} not tile-aligned (TILE_HEIGHT=32)"

        # Oversized cache reused across chunks: strictly larger than any chunk's valid prefix, so its
        # stable shape hits one cached program.
        reuse_kv_stable_seq = (math.ceil(total_seq / chunk_size) + 1) * chunk_size

        k_memory_config = nd_sharded_dram_memory_config(mesh_device, d_k) if indexed_nd_sharded_kv_cache else None
        v_memory_config = nd_sharded_dram_memory_config(mesh_device, d_v) if indexed_nd_sharded_kv_cache else None

        def prepare_chunk_inputs(i):
            s, e = i * chunk_size, (i + 1) * chunk_size

            if reuse_kv_buffer:
                # Pad-rotation layout for the [0, s) prefix + new [s, e) chunk, then grow each device's
                # slab to reuse_kv_stable_seq with a garbage tail. The tail is never read iff the gather
                # honours logical_n=e / kv_actual_isl=s (set in run_chunk_call).
                Q_chunk, K_chunk = Q_full[:, :, s:e, :].contiguous(), K_full[:, :, s:e, :].contiguous()
                stable_per_dev = reuse_kv_stable_seq // sp_size

                def oversize(host, nh, head_dim):
                    seq_per_dev = host.shape[2] // sp_size
                    assert stable_per_dev >= seq_per_dev, "reuse_kv_stable_seq too small for chunk's valid prefix"
                    out = torch.randn(host.shape[0], nh, sp_size, stable_per_dev, head_dim, dtype=host.dtype) * 100
                    out[:, :, :, :seq_per_dev, :] = host.reshape(host.shape[0], nh, sp_size, seq_per_dev, head_dim)
                    return out.reshape(host.shape[0], nh, reuse_kv_stable_seq, head_dim)

                if use_ring_mla:
                    q_host, kv_host, *_ = build_kv_pad_rotation_mla_inputs(
                        K_full[:, :, :s, :], Q_chunk, K_chunk, s, sp_size, slab_rows
                    )
                    return (s, e, b, None, upload_q(q_host), upload_kv(oversize(kv_host, nhk, d_k)), None)

                q_host, k_host, v_host, *_ = build_kv_pad_rotation_inputs(
                    K_full[:, :, :s, :],
                    V_full[:, :, :s, :],
                    Q_chunk,
                    K_chunk,
                    V_full[:, :, s:e, :].contiguous(),
                    s,
                    sp_size,
                    slab_rows,
                )
                return (
                    s,
                    e,
                    b,
                    None,
                    upload_q(q_host),
                    upload_k(oversize(k_host, nhk, d_k)),
                    upload_v(oversize(v_host, nhv, d_v)),
                )

            Q_chunk = Q_full[:, :, s:e, :].contiguous()
            K_balanced = to_balanced_growing_cache_layout(K_full, sp_size, chunk_size, i)
            kv_buffer_batch = b
            kv_cache_batch_idx_arg = None

            if use_ring_mla:
                return (
                    s,
                    e,
                    kv_buffer_batch,
                    kv_cache_batch_idx_arg,
                    upload_q(Q_chunk),
                    upload_kv(K_balanced),
                    None,
                )

            V_balanced = to_balanced_growing_cache_layout(V_full, sp_size, chunk_size, i)
            if indexed_nd_sharded_kv_cache:
                kv_buffer_batch = cache_batch
                kv_cache_batch_idx_arg = kv_cache_batch_idx
                K_input = torch.randn(cache_batch, nhk, e, d_k, dtype=K_balanced.dtype) * 100
                V_input = torch.randn(cache_batch, nhv, e, d_v, dtype=V_balanced.dtype) * 100
                K_input[kv_cache_batch_idx : kv_cache_batch_idx + 1] = K_balanced
                V_input[kv_cache_batch_idx : kv_cache_batch_idx + 1] = V_balanced
            else:
                K_input = K_balanced
                V_input = V_balanced

            return (
                s,
                e,
                kv_buffer_batch,
                kv_cache_batch_idx_arg,
                upload_q(Q_chunk),
                upload_k(K_input, memory_config=k_memory_config),
                upload_v(V_input, memory_config=v_memory_config),
            )

        def get_persistent_buffers(shared_persistent_buffers, chunk_persistent_buffers):
            if persistent_buffer_mode == "reuse_max":
                return shared_persistent_buffers
            return chunk_persistent_buffers

        def run_chunk_call(
            config_id,
            program_config,
            it,
            i,
            s,
            e,
            tt_Q,
            tt_K,
            tt_V,
            persistent_output_buffer_k,
            persistent_output_buffer_v,
            kv_cache_batch_idx_arg,
        ):
            try:
                if use_ring_mla:
                    tt_out, _ = ttnn.transformer.ring_mla(
                        tt_Q,
                        tt_K,
                        persistent_output_buffer_kv=persistent_output_buffer_k,
                        head_dim_v=d_v,
                        logical_n=e,
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
                        kv_actual_isl=s if reuse_kv_buffer else None,
                    )
                    return tt_out

                return call_sdpa(
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
                    kv_cache_batch_idx=kv_cache_batch_idx_arg,
                    kv_actual_isl=s if reuse_kv_buffer else None,
                )
            except Exception as exc:
                op_name = "ring_mla" if use_ring_mla else "SDPA"
                pytest.fail(
                    f"Chunked prefill {op_name} {config_id} call raised on iter {it}, chunk {i} "
                    f"(Q rows [{s}, {e}), logical_n={e}): {type(exc).__name__}: {exc}"
                )
                raise

        def record_chunk_accuracy(config_id, per_chunk_results, i, s, e, out_i):
            expected_i = ref_full[:, :, s:e, :]
            pcc, rmse = compute_pcc_rmse(expected_i, out_i)
            pcc_passed = pcc >= pcc_threshold
            rmse_passed = rmse_threshold is None or rmse < rmse_threshold
            logger.info(
                f"{config_id} chunk {i:2d} [{s:6d}, {e:6d}) logical_n={e}: PCC={pcc} RMSE={rmse:.6f} "
                f"-> {'PASS' if pcc_passed and rmse_passed else 'FAIL'}"
            )
            per_chunk_results.append((i, e, pcc_passed, pcc, rmse_passed, rmse))

        config_ids = {(q, k): get_test_case_id(model, q, k) for q, k in qk_configs}
        per_chunk_results_by_config = {config_id: [] for config_id in config_ids.values()}
        determinism_mismatch_marker = None
        shared_persistent_buffers = None
        if persistent_buffer_mode == "reuse_max":
            # The gather buffer must hold the full physical input — the oversized cache under reuse_kv_buffer.
            reuse_seq = reuse_kv_stable_seq if reuse_kv_buffer else total_seq
            shared_persistent_buffers = (
                create_ring_mla_kv_buffer(reuse_seq, b) if use_ring_mla else create_persistent_buffers(reuse_seq, b)
            )

        for i in range(n_chunks):
            if only_chunk is not None and i != only_chunk:
                continue

            (
                s,
                e,
                kv_buffer_batch,
                kv_cache_batch_idx_arg,
                tt_Q,
                tt_K,
                tt_V,
            ) = prepare_chunk_inputs(i)
            chunk_persistent_buffers = None
            if persistent_buffer_mode != "reuse_max":
                chunk_persistent_buffers = (
                    create_ring_mla_kv_buffer(e, kv_buffer_batch)
                    if use_ring_mla
                    else create_persistent_buffers(e, kv_buffer_batch)
                )

            for q_chunk_size, k_chunk_size in qk_configs:
                config_id = config_ids[(q_chunk_size, k_chunk_size)]
                program_config = program_configs[(q_chunk_size, k_chunk_size)]
                persistent_buffers = get_persistent_buffers(shared_persistent_buffers, chunk_persistent_buffers)
                if use_ring_mla:
                    persistent_output_buffer_k = persistent_buffers
                    persistent_output_buffer_v = None
                else:
                    persistent_output_buffer_k, persistent_output_buffer_v = persistent_buffers

                if num_iterations > 1:
                    reference_output = None
                    reference_tt_out = None
                    mismatch_marker = None
                    for it in range(num_iterations):
                        tt_out = run_chunk_call(
                            config_id,
                            program_config,
                            it,
                            i,
                            s,
                            e,
                            tt_Q,
                            tt_K,
                            tt_V,
                            persistent_output_buffer_k,
                            persistent_output_buffer_v,
                            kv_cache_batch_idx_arg,
                        )

                        if use_device_determinism_compare:
                            if reference_tt_out is None:
                                reference_tt_out = tt_out
                            else:
                                mismatch_marker = merge_device_mismatch_markers(
                                    mismatch_marker,
                                    device_tensors_mismatch_marker(reference_tt_out, tt_out),
                                )
                            continue

                        out_i = to_host(tt_out, chunk_size)
                        if use_ring_mla:
                            out_i = out_i[:, :, :, :d_v]

                        if reference_output is None:
                            reference_output = out_i
                        elif not torch.equal(reference_output, out_i):
                            num_diffs = (reference_output != out_i).sum().item()
                            max_diff = (reference_output - out_i).abs().max().item()
                            pytest.fail(
                                f"Chunked prefill {config_id} determinism failed at iter {it}, chunk {i}: "
                                f"{num_diffs} differing elements, max diff = {max_diff}"
                            )
                    if use_device_determinism_compare and mismatch_marker is not None:
                        determinism_mismatch_marker = merge_device_mismatch_markers(
                            determinism_mismatch_marker, mismatch_marker
                        )
                else:
                    tt_out = run_chunk_call(
                        config_id,
                        program_config,
                        0,
                        i,
                        s,
                        e,
                        tt_Q,
                        tt_K,
                        tt_V,
                        persistent_output_buffer_k,
                        persistent_output_buffer_v,
                        kv_cache_batch_idx_arg,
                    )
                    # do_check=False (perf/profiling): the device op above already ran for the
                    # profiler; skip the host readback and PCC comparison against ref_full.
                    if do_check:
                        out_i = to_host(tt_out, chunk_size)
                        if use_ring_mla:
                            out_i = out_i[:, :, :, :d_v]
                        record_chunk_accuracy(
                            config_id,
                            per_chunk_results_by_config[config_id],
                            i,
                            s,
                            e,
                            out_i,
                        )

        if num_iterations > 1 and use_device_determinism_compare and determinism_mismatch_marker is not None:
            assert not device_mismatch_marker_is_set(
                determinism_mismatch_marker
            ), "Chunked prefill produced output that differs from iteration 0"

        for config_id, per_chunk_results in per_chunk_results_by_config.items():
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
                    f"Chunked prefill {config_id} PCC/RMSE failures "
                    f"(pcc_threshold={pcc_threshold}, rmse_threshold={rmse_threshold}): {details}"
                )
            if num_iterations > 1:
                logger.info(
                    f"Chunked prefill {config_id} determinism verified: "
                    f"all {num_iterations} runs of {n_chunks if only_chunk is None else 1} chunks are exactly equal"
                )

    finally:
        close_ring_joint_sdpa_runtime(runtime)


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


def build_kv_pad_rotation_mla_inputs(
    old_cache_kv,
    new_tokens_q,
    new_tokens_kv,
    kv_actual_isl,
    sp_size,
    chunk_size_local,
):
    """Build padded Q/KV host tensors for one KV-pad-aware ring_mla call."""
    assert old_cache_kv.shape[0] == 1
    assert new_tokens_q.shape[0] == 1

    nhq = new_tokens_q.shape[1]
    nhk = new_tokens_kv.shape[1]
    d_q = new_tokens_q.shape[3]
    d_k = new_tokens_kv.shape[3]
    new_actual_isl = new_tokens_q.shape[2]
    chunk_size_global = sp_size * chunk_size_local
    logical_n = kv_actual_isl + new_actual_isl
    num_cache_slabs = max(2, math.ceil(logical_n / chunk_size_global))
    cache_seq_per_dev = num_cache_slabs * chunk_size_local

    kv_per_dev = torch.zeros(sp_size, nhk, cache_seq_per_dev, d_k, dtype=torch.bfloat16)
    kv_valid_per_dev = torch.zeros(sp_size, cache_seq_per_dev, dtype=torch.bool)
    for global_pos in range(kv_actual_isl):
        group = global_pos // chunk_size_global
        within_group = global_pos % chunk_size_global
        dev = within_group // chunk_size_local
        cell = within_group % chunk_size_local
        cache_row = group * chunk_size_local + cell
        kv_per_dev[dev, :, cache_row, :] = old_cache_kv[0, :, global_pos, :]
        kv_valid_per_dev[dev, cache_row] = True

    q_per_dev = torch.zeros(sp_size, nhq, chunk_size_local, d_q, dtype=torch.bfloat16)
    q_global_pos_per_dev = [[None] * chunk_size_local for _ in range(sp_size)]
    destinations = kv_pad_rotation_destinations(kv_actual_isl, new_actual_isl, sp_size, chunk_size_local)
    assert len(destinations) == new_actual_isl
    for token_idx, dev, cache_row, q_row, global_pos in destinations:
        kv_per_dev[dev, :, cache_row, :] = new_tokens_kv[0, :, token_idx, :]
        kv_valid_per_dev[dev, cache_row] = True
        q_per_dev[dev, :, q_row, :] = new_tokens_q[0, :, token_idx, :]
        q_global_pos_per_dev[dev][q_row] = global_pos

    q_host = q_per_dev.permute(1, 0, 2, 3).reshape(1, nhq, chunk_size_global, d_q)
    kv_host = kv_per_dev.permute(1, 0, 2, 3).reshape(1, nhk, sp_size * cache_seq_per_dev, d_k)

    combined_q_global_pos = []
    for dev in range(sp_size):
        combined_q_global_pos.extend(q_global_pos_per_dev[dev])

    valid_rows = [None] * new_actual_isl
    for row, q_pos in enumerate(combined_q_global_pos):
        if q_pos is not None:
            valid_rows[q_pos - kv_actual_isl] = row
    assert all(row is not None for row in valid_rows)

    return q_host, kv_host, valid_rows, kv_valid_per_dev, num_cache_slabs


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

    runtime = open_ring_joint_sdpa_runtime(mesh_config)
    mesh_device = runtime.mesh_device
    topology = runtime.topology
    sp_axis = runtime.sp_axis
    tp_axis = runtime.tp_axis
    num_links = runtime.num_links
    sdpa_compute_grid = runtime.sdpa_compute_grid
    ccl_column = runtime.ccl_column
    worker_sub_device_id = runtime.worker_sub_device_id
    ccl_semaphore_handles = runtime.ccl_semaphore_handles
    compute_kernel_config = runtime.compute_kernel_config

    try:
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
        close_ring_joint_sdpa_runtime(runtime)


def run_ring_mla_sdpa_chunked_kv_actual_isl_reuse_max_case(
    mesh_config,
    chunk_size_local=32,
    num_chunks=5,
    num_iterations=2,
    kv_cache_batch_idx=1,
    cache_batch=2,
    pcc_threshold=CHUNKED_PREFILL_PCC_THRESHOLD,
    rmse_threshold=DEFAULT_RMSE_THRESHOLD,
):
    sp_size = mesh_config.sp_size
    if sp_size < 2:
        pytest.skip(f"ring_mla chunked kv_actual_isl requires at least 2 devices in ring, got SP={sp_size}")

    tile_height = 32
    assert chunk_size_local % tile_height == 0
    assert BATCH_SIZE == 1, "Indexed K/V cache test path assumes query batch is 1"
    assert (
        0 <= kv_cache_batch_idx < cache_batch
    ), f"kv_cache_batch_idx {kv_cache_batch_idx} must be in [0, {cache_batch})"

    b = BATCH_SIZE
    local_heads = 4
    nhq = local_heads * mesh_config.tp_size
    nhk = 1
    d_q = 64
    d_k = 64
    d_v = 32
    q_dtype = ttnn.bfloat16
    kv_dtype = ttnn.bfloat16
    chunk_size_global = chunk_size_local * sp_size
    # Drift chunk starts across cache-slab boundaries while keeping each chunk within one fixed Q slab.
    new_actual_isl = chunk_size_global // 2 + tile_height
    total_seq = new_actual_isl * num_chunks
    assert new_actual_isl % tile_height == 0
    assert new_actual_isl <= chunk_size_global
    assert total_seq % tile_height == 0

    max_cache_slabs = max(2, math.ceil(total_seq / chunk_size_global) + 1)
    max_cache_seq_per_dev = max_cache_slabs * chunk_size_local
    persistent_seq_len = sp_size * max_cache_seq_per_dev
    max_input_cache_slabs = max(
        max(2, math.ceil(((i + 1) * new_actual_isl) / chunk_size_global)) for i in range(num_chunks)
    )
    stable_kv_input_seq_len = sp_size * max_input_cache_slabs * chunk_size_local

    torch.manual_seed(CHUNKED_PREFILL_SEED)
    q_full = fa_rand(b, nhq, total_seq, d_q)
    kv_full = fa_rand(b, nhk, total_seq, d_k)

    runtime = open_ring_joint_sdpa_runtime(mesh_config)
    mesh_device = runtime.mesh_device
    topology = runtime.topology
    sp_axis = runtime.sp_axis
    tp_axis = runtime.tp_axis
    num_links = runtime.num_links
    sdpa_compute_grid = runtime.sdpa_compute_grid
    ccl_column = runtime.ccl_column
    worker_sub_device_id = runtime.worker_sub_device_id
    ccl_semaphore_handles = runtime.ccl_semaphore_handles
    compute_kernel_config = runtime.compute_kernel_config

    try:
        sdpa_input_shard_dims = [None, None]
        sdpa_input_shard_dims[sp_axis] = 2
        if mesh_config.tp_size > 1:
            sdpa_input_shard_dims[tp_axis] = 1

        sdpa_kv_shard_dims = [None, None]
        sdpa_kv_shard_dims[sp_axis] = 2

        persistent_kv_shard_dims = [None, None]
        main_row_dim = sdpa_input_shard_dims[0] if sdpa_input_shard_dims[0] is not None else -1
        main_col_dim = sdpa_input_shard_dims[1] if sdpa_input_shard_dims[1] is not None else -1

        def upload(host_tensor, dtype, shard_dims):
            return ttnn.from_torch(
                host_tensor,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
            )

        persistent_output_buffer_kv = ttnn.from_torch(
            torch.zeros(cache_batch, nhk, persistent_seq_len, d_k),
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_kv_shard_dims
            ),
        )
        mesh_device.enable_program_cache()
        mesh_device.clear_program_cache()

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_compute_grid,
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
        )

        logger.info(
            f"ring_mla chunked kv_actual_isl reuse-max: sp_size={sp_size}, "
            f"chunk_size_global={chunk_size_global}, new_actual_isl={new_actual_isl}, "
            f"num_chunks={num_chunks}, num_iterations={num_iterations}, "
            f"kv_cache_batch_idx={kv_cache_batch_idx}, cache_batch={cache_batch}, "
            f"stable_kv_input_seq_len={stable_kv_input_seq_len}, persistent_seq_len={persistent_seq_len}"
        )

        reference_outputs = None
        per_chunk_results = []
        cache_entries_after_first_call = None
        for it in range(num_iterations):
            iter_outputs = []
            for i in range(num_chunks):
                kv_actual_isl = i * new_actual_isl
                logical_n = kv_actual_isl + new_actual_isl
                new_tokens_q = q_full[:, :, kv_actual_isl:logical_n, :].contiguous()
                new_tokens_kv = kv_full[:, :, kv_actual_isl:logical_n, :].contiguous()

                q_host, kv_host, valid_rows, kv_valid_per_dev, num_cache_slabs = build_kv_pad_rotation_mla_inputs(
                    kv_full[:, :, :kv_actual_isl, :],
                    new_tokens_q,
                    new_tokens_kv,
                    kv_actual_isl,
                    sp_size,
                    chunk_size_local,
                )
                assert persistent_seq_len > kv_host.shape[2], (
                    f"test requires oversized persistent buffer, got persistent_seq_len={persistent_seq_len}, "
                    f"input KV seq={kv_host.shape[2]}"
                )
                assert stable_kv_input_seq_len >= kv_host.shape[2], (
                    f"stable physical KV input is too small, got stable_kv_input_seq_len={stable_kv_input_seq_len}, "
                    f"input KV seq={kv_host.shape[2]}"
                )
                cache_seq_per_dev = kv_host.shape[2] // sp_size
                stable_cache_seq_per_dev = stable_kv_input_seq_len // sp_size
                assert kv_host.shape[2] % sp_size == 0
                assert stable_kv_input_seq_len % sp_size == 0
                kv_input_per_dev = (
                    torch.randn(cache_batch, nhk, sp_size, stable_cache_seq_per_dev, d_k, dtype=kv_host.dtype) * 100
                )
                active_kv_input = kv_input_per_dev[
                    kv_cache_batch_idx : kv_cache_batch_idx + 1, :, :, :cache_seq_per_dev, :
                ]
                active_kv_input.copy_(
                    torch.where(
                        kv_valid_per_dev.reshape(1, 1, sp_size, cache_seq_per_dev, 1),
                        kv_host.reshape(1, nhk, sp_size, cache_seq_per_dev, d_k),
                        active_kv_input,
                    )
                )
                kv_input = kv_input_per_dev.reshape(cache_batch, nhk, stable_kv_input_seq_len, d_k)

                tt_q = upload(q_host, q_dtype, sdpa_input_shard_dims)
                tt_kv = upload(kv_input, kv_dtype, sdpa_kv_shard_dims)

                try:
                    tt_out, _ = ttnn.transformer.ring_mla(
                        tt_q,
                        tt_kv,
                        persistent_output_buffer_kv=persistent_output_buffer_kv,
                        head_dim_v=d_v,
                        logical_n=logical_n,
                        is_balanced=False,
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
                        kv_cache_batch_idx=kv_cache_batch_idx,
                        kv_actual_isl=kv_actual_isl,
                    )
                except Exception as exc:
                    pytest.fail(
                        f"ring_mla chunked kv_actual_isl call raised on iter {it}, chunk {i} "
                        f"(kv_actual_isl={kv_actual_isl}, logical_n={logical_n}, "
                        f"input_kv_seq={kv_host.shape[2]}, kv_cache_batch_idx={kv_cache_batch_idx}, "
                        f"persistent_seq={persistent_seq_len}): "
                        f"{type(exc).__name__}: {exc}"
                    )

                cache_entries = mesh_device.num_program_cache_entries()
                if cache_entries_after_first_call is None:
                    assert cache_entries > 0, "ring_mla kv_actual_isl test expected at least one program-cache entry"
                    cache_entries_after_first_call = cache_entries
                else:
                    assert cache_entries == cache_entries_after_first_call, (
                        f"ring_mla kv_actual_isl should reuse program cache for stable physical KV shape; "
                        f"got {cache_entries} entries after iter {it}, chunk {i}, "
                        f"expected {cache_entries_after_first_call} "
                        f"(kv_actual_isl={kv_actual_isl}, logical_n={logical_n})"
                    )

                out_host = ttnn.to_torch(
                    tt_out,
                    mesh_composer=ttnn.create_mesh_composer(
                        mesh_device, ttnn.MeshComposerConfig(main_row_dim, main_col_dim)
                    ),
                )
                out_valid_natural = out_host[:, :, valid_rows, :]
                iter_outputs.append(out_valid_natural)

                if it == 0:
                    ref_out = torch_chunked_causal_sdpa_reference(
                        new_tokens_q, kv_full[:, :, :logical_n, :], kv_full[:, :, :logical_n, :d_v], kv_actual_isl
                    )
                    passed, pcc = comp_pcc(ref_out, out_valid_natural, pcc_threshold)
                    rmse = torch.sqrt(((ref_out - out_valid_natural) ** 2).mean()).item()
                    rmse_passed = rmse_threshold is None or rmse < rmse_threshold
                    logger.info(
                        f"ring_mla kv_actual_isl chunk {i:2d}: kv_actual_isl={kv_actual_isl}, "
                        f"logical_n={logical_n}, input_kv_seq={kv_host.shape[2]}, "
                        f"kv_cache_batch_idx={kv_cache_batch_idx}, persistent_seq={persistent_seq_len}, "
                        f"cache_slabs={num_cache_slabs}, "
                        f"PCC={pcc}, RMSE={rmse:.6f} -> {'PASS' if passed and rmse_passed else 'FAIL'}"
                    )
                    per_chunk_results.append((i, kv_actual_isl, logical_n, passed, pcc, rmse_passed, rmse))

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
                    pytest.fail(f"ring_mla chunked kv_actual_isl determinism failed at iter {it}: {details}")

        failures = [
            (i, kv_actual_isl, logical_n, pcc, rmse)
            for i, kv_actual_isl, logical_n, passed, pcc, rmse_passed, rmse in per_chunk_results
            if not passed or not rmse_passed
        ]
        if failures:
            details = "; ".join(
                f"chunk {i} (kv_actual_isl={kv_actual_isl}, logical_n={logical_n}): PCC={pcc}, RMSE={rmse:.6f}"
                for i, kv_actual_isl, logical_n, pcc, rmse in failures
            )
            pytest.fail(
                f"ring_mla chunked kv_actual_isl PCC/RMSE failures "
                f"(pcc_threshold={pcc_threshold}, rmse_threshold={rmse_threshold}): {details}"
            )
        logger.info(
            f"ring_mla chunked kv_actual_isl verified: {num_chunks} chunks replayed {num_iterations} times with "
            f"kv_cache_batch_idx={kv_cache_batch_idx}, persistent_seq_len={persistent_seq_len}"
        )

    finally:
        close_ring_joint_sdpa_runtime(runtime, clear_program_cache=True)


def run_ring_mla_sdpa_chunked_indexed_kv_cache(
    mesh_config,
    model: ModelConfig,
    chunk_size: int,
    total_seq: int,
    pcc_threshold: float = DEFAULT_PCC_THRESHOLD,
    rmse_threshold: float = DEFAULT_RMSE_THRESHOLD,
    q_chunk_size: int = None,
    k_chunk_size: int = None,
    num_iterations: int = 2,
    kv_cache_batch_idx: int = 1,
    cache_batch: int = 2,
):
    """
    Validate ring_mla chunked-prefill with an indexed shared K/V cache.

    The SUT replays a sequence of short Q chunks. For chunk i, the indexed cache
    slot contains K/V rows [0, (i + 1) * chunk_size), packed in the same growing
    cache layout used by ring_joint_sdpa chunked prefill. The unselected cache
    slots contain random data and must not affect output.
    """
    torch.manual_seed(CHUNKED_PREFILL_SEED)

    sp_size = mesh_config.sp_size
    if sp_size < 2:
        pytest.skip(f"ring_mla chunked prefill requires at least 2 devices in ring, got SP={sp_size}")
    if model.nhk != 1:
        pytest.skip(f"ring_mla chunked prefill requires one shared KV head, got nhk={model.nhk}")

    assert BATCH_SIZE == 1, "Indexed K/V cache test path assumes query batch is 1"
    assert total_seq % sp_size == 0, f"total_seq {total_seq} must divide sp_size {sp_size}"
    assert chunk_size % sp_size == 0, f"chunk_size {chunk_size} must divide sp_size {sp_size}"
    assert total_seq % chunk_size == 0, f"total_seq {total_seq} must be a multiple of chunk_size {chunk_size}"
    assert (
        0 <= kv_cache_batch_idx < cache_batch
    ), f"kv_cache_batch_idx {kv_cache_batch_idx} must be in [0, {cache_batch})"

    n_chunks = total_seq // chunk_size
    if q_chunk_size is None:
        q_chunk_size = model.q_chunk_sizes[0]
    if k_chunk_size is None:
        k_chunk_size = model.k_chunk_sizes[0]

    b = BATCH_SIZE
    nhq = model.nhq * mesh_config.tp_size
    nhk = model.nhk
    d_q, d_k, d_v = model.d_q, model.d_k, model.d_v
    q_dtype, kv_dtype = model.q_dtype, model.kv_dtype

    runtime = open_ring_joint_sdpa_runtime(mesh_config)
    mesh_device = runtime.mesh_device
    topology = runtime.topology
    sp_axis = runtime.sp_axis
    tp_axis = runtime.tp_axis
    num_links = runtime.num_links
    sdpa_compute_grid = runtime.sdpa_compute_grid
    ccl_column = runtime.ccl_column
    worker_sub_device_id = runtime.worker_sub_device_id
    ccl_semaphore_handles = runtime.ccl_semaphore_handles
    compute_kernel_config = runtime.compute_kernel_config

    try:
        Q_full = fa_rand(b, nhq, total_seq, d_q)
        KV_full = fa_rand(b, nhk, total_seq, d_k)
        V_full = KV_full[:, :, :, :d_v]
        ref_full = torch_sdpa_reference(Q_full, KV_full, V_full, is_causal=True)

        sdpa_input_shard_dims = [None, None]
        sdpa_input_shard_dims[sp_axis] = 2
        if mesh_config.tp_size > 1:
            sdpa_input_shard_dims[tp_axis] = 1

        sdpa_kv_shard_dims = [None, None]
        sdpa_kv_shard_dims[sp_axis] = 2

        persistent_kv_shard_dims = [None, None]
        main_row_dim = sdpa_input_shard_dims[0] if sdpa_input_shard_dims[0] is not None else -1
        main_col_dim = sdpa_input_shard_dims[1] if sdpa_input_shard_dims[1] is not None else -1

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_compute_grid,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        kv_memory_config = nd_sharded_dram_memory_config(mesh_device, d_k)

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

        def upload_kv(kv_host):
            return ttnn.from_torch(
                kv_host,
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=kv_memory_config,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_kv_shard_dims
                ),
            )

        def to_host(tt_out, expected_q_len):
            out = ttnn.to_torch(
                tt_out,
                mesh_composer=ttnn.create_mesh_composer(
                    mesh_device, ttnn.MeshComposerConfig(main_row_dim, main_col_dim)
                ),
            )
            return out[:, :, :expected_q_len, :d_v]

        logger.info(
            f"ring_mla chunked indexed cache: model={model.name}, total_seq={total_seq}, "
            f"chunk_size={chunk_size}, n_chunks={n_chunks}, q_chunk={q_chunk_size}, k_chunk={k_chunk_size}, "
            f"kv_cache_batch_idx={kv_cache_batch_idx}, cache_batch={cache_batch}"
        )

        reference_outputs = None
        per_chunk_results = []
        for it in range(num_iterations):
            iter_outputs = []
            for i in range(n_chunks):
                s, e = i * chunk_size, (i + 1) * chunk_size
                KV_balanced = to_balanced_growing_cache_layout(KV_full, sp_size, chunk_size, i)
                Q_chunk = Q_full[:, :, s:e, :].contiguous()
                KV_input = torch.randn(cache_batch, nhk, e, d_k, dtype=KV_balanced.dtype) * 100
                KV_input[kv_cache_batch_idx : kv_cache_batch_idx + 1] = KV_balanced

                tt_Q = upload_q(Q_chunk)
                tt_KV = upload_kv(KV_input)
                persistent_output_buffer_kv = ttnn.from_torch(
                    torch.zeros(cache_batch, nhk, e, d_k),
                    dtype=kv_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_kv_shard_dims
                    ),
                )

                try:
                    tt_out, _ = ttnn.transformer.ring_mla(
                        tt_Q,
                        tt_KV,
                        persistent_output_buffer_kv=persistent_output_buffer_kv,
                        head_dim_v=d_v,
                        logical_n=e,
                        is_balanced=False,
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
                        kv_cache_batch_idx=kv_cache_batch_idx,
                    )
                except Exception as exc:
                    pytest.fail(
                        f"ring_mla chunked indexed-cache call raised on iter {it}, chunk {i} "
                        f"(Q rows [{s}, {e}), logical_n={e}): {type(exc).__name__}: {exc}"
                    )

                out_i = to_host(tt_out, chunk_size)
                iter_outputs.append(out_i)

                if it == 0:
                    expected_i = ref_full[:, :, s:e, :]
                    passed, pcc = comp_pcc(expected_i, out_i, pcc_threshold)
                    rmse = torch.sqrt(((expected_i - out_i) ** 2).mean()).item()
                    rmse_passed = rmse < rmse_threshold
                    logger.info(
                        f"ring_mla chunk {i:2d} [{s:6d}, {e:6d}) logical_n={e}: "
                        f"PCC={pcc} RMSE={rmse:.6f} -> {'PASS' if passed and rmse_passed else 'FAIL'}"
                    )
                    per_chunk_results.append((i, e, passed, pcc, rmse_passed, rmse))

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
                    pytest.fail(f"ring_mla chunked indexed-cache determinism failed at iter {it}: {details}")

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
                f"ring_mla chunked indexed-cache PCC/RMSE failures "
                f"(pcc_threshold={pcc_threshold}, rmse_threshold={rmse_threshold}): {details}"
            )
        logger.info(f"ring_mla chunked indexed-cache verified: {n_chunks} chunks replayed {num_iterations} times")

    finally:
        close_ring_joint_sdpa_runtime(runtime)


def test_ring_joint_attention_chunked_nd_sharded_indexed_kv_cache_accuracy():
    """Validate chunked direct ND-sharded K/V cache inputs selected by kv_cache_batch_idx."""
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


def test_ring_mla_chunked_kv_actual_isl_indexed_reuse_max_accuracy_and_determinism():
    """Validate chunked ring_mla with kv_cache_batch_idx, kv_actual_isl, and oversized reusable persistent KV cache."""
    mesh_config = MESH_CONFIG
    run_ring_mla_sdpa_chunked_kv_actual_isl_reuse_max_case(
        mesh_config,
    )


def test_ring_mla_chunked_nd_sharded_indexed_kv_cache_accuracy_and_determinism():
    """Validate ring_mla chunked prefill with indexed ND-sharded shared K/V cache."""
    mesh_config = MESH_CONFIG
    local_heads = 4
    chunk_seq_len = 64
    total_seq_len = 128
    model = ModelConfig(
        name="mla_chunked_indexed_nd_sharded_kv",
        nhq=local_heads,
        nhk=1,
        nhv=local_heads,
        d_q=64,
        d_k=64,
        d_v=32,
        is_causal=True,
        q_dtype=ttnn.bfloat16,
        kv_dtype=ttnn.bfloat16,
        q_chunk_sizes=[32],
        k_chunk_sizes=[32],
        seq_len=total_seq_len,
    )

    run_ring_mla_sdpa_chunked_indexed_kv_cache(
        mesh_config,
        model,
        chunk_size=chunk_seq_len * mesh_config.sp_size,
        total_seq=total_seq_len * mesh_config.sp_size,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        q_chunk_size=model.q_chunk_sizes[0],
        k_chunk_size=model.k_chunk_sizes[0],
        num_iterations=2,
        kv_cache_batch_idx=1,
        cache_batch=2,
    )


def test_ring_mla_nd_sharded_indexed_kv_cache_accuracy():
    """Validate ring_mla selects the requested batch slot from an indexed K/V cache."""
    mesh_config = MESH_CONFIG
    local_heads = 4
    per_device_seq_len = 128

    run_ring_mla_sdpa(
        mesh_config,
        b=1,
        nhq=local_heads * mesh_config.tp_size,
        nhk=1,
        sq=per_device_seq_len * mesh_config.sp_size,
        d_q=64,
        d_k=64,
        d_v=32,
        q_chunk_size=32,
        k_chunk_size=32,
        q_dtype=ttnn.bfloat16,
        kv_dtype=ttnn.bfloat16,
        is_balanced=False,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        rmse_threshold=DEFAULT_RMSE_THRESHOLD,
        kv_cache_batch_idx=1,
        cache_batch=2,
    )


# ============================================================================
# TRACE-SAFE METADATA PATH: metadata-path == scalar-path (bit-exact)
# ============================================================================
# ring_mla gains an opt-in `metadata` tensor (the runner's canonical h2d_socket_sync payload
# [slot_id, actual_start, actual_end], replicated uint32 DRAM). When set, the per-chunk scalars
# (kv_cache_batch_idx = slot_id; kv_actual_isl = actual_start; logical_n = actual_start +
# chunk_size_global) are read ON-DEVICE from this tensor instead of being baked into the program by
# the host, so a single captured ttnn trace replays across chunks. These tests assert the metadata
# path is BIT-IDENTICAL to the classic host-scalar path on every supported case (the op is the same
# kernel math; only where the scalars come from differs).
#
# The migration is incremental (see TRACEABLE_METADATA_PATH.md): each scalar is moved on-device one
# at a time. `META_PATH_HOST_SCALARS` lists the scalars whose on-device read has NOT landed yet -- a
# listed scalar is still passed as a host arg on the metadata path so the comparison stays exact.
# Drop an entry the moment its on-device read is implemented; the bit-exact assert then proves the
# new on-device computation matches the host one.
# kv_cache_batch_idx is now read on-device from metadata[0] (all-gather reader + SDPA reader), so the
# metadata path no longer passes it as a host scalar -- dropping it makes the test discriminating.
META_PATH_HOST_SCALARS = {"kv_actual_isl"}


def _make_ring_mla_scalar_tensor(mesh_device, value):
    """1-element uint32 replicated DRAM tensor ([1,1,1,1]) holding a single per-chunk scalar that the
    trace-safe ring_mla reads on-device. Mirrors the update_padded_kv_cache / rotary metadata layout."""
    payload = torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1)
    return ttnn.from_torch(
        payload,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _make_ring_mla_metadata(mesh_device, slot_id, actual_start, actual_end):
    """Build the two 1-element uint32 DRAM tensors the trace-safe ring_mla reads on-device: slot_id
    (was metadata[0]) and kv_actual_isl == actual_start (was metadata[1]). actual_end is unused by the
    op (logical_n is still passed as a host arg). Returns (slot_id_tensor, kv_actual_isl_tensor)."""
    return (
        _make_ring_mla_scalar_tensor(mesh_device, slot_id),
        _make_ring_mla_scalar_tensor(mesh_device, actual_start),
    )


@pytest.mark.parametrize("kv_cache_batch_idx", [0, 1], ids=["slot0", "slot1"])
def test_ring_mla_metadata_matches_scalar_indexed(kv_cache_batch_idx):
    """Indexed K/V cache (no rotation): the metadata path (slot read on-device from metadata[0])
    must produce a bit-identical output to the scalar path (kv_cache_batch_idx host arg).

    This is the first migration increment -- it exercises kv_cache_batch_idx, which the fused
    all-gather reader will read from metadata[0]. While 'kv_cache_batch_idx' is still in
    META_PATH_HOST_SCALARS the metadata path also passes the host scalar, so the only thing under
    test today is the metadata plumbing (nanobind kwarg + tensor_args + program hash). Once the
    on-device read lands, drop it from the set and this asserts the on-device slot select."""
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"ring_mla requires at least 2 devices in ring, got SP={mesh_config.sp_size}")

    b, local_heads, per_device_seq_len = 1, 4, 128
    nhq = local_heads * mesh_config.tp_size
    nhk = 1
    sq = per_device_seq_len * mesh_config.sp_size
    d_q, d_k, d_v = 64, 64, 32
    cache_batch = 2
    assert 0 <= kv_cache_batch_idx < cache_batch

    torch.manual_seed(1234)
    runtime = open_ring_joint_sdpa_runtime(mesh_config)
    mesh_device = runtime.mesh_device
    sp_axis, tp_axis = runtime.sp_axis, runtime.tp_axis
    try:
        Q = fa_rand(b, nhq, sq, d_q)
        KV = fa_rand(b, nhk, sq, d_k)
        # Embed the real K/V at the requested cache slot; other slots are garbage that must not leak.
        KV_input = fa_rand(cache_batch, nhk, sq, d_k) * 100
        KV_input[kv_cache_batch_idx : kv_cache_batch_idx + 1] = KV

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=runtime.sdpa_compute_grid,
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
        )

        q_shard_dims = [None, None]
        q_shard_dims[sp_axis] = 2
        if mesh_config.tp_size > 1:
            q_shard_dims[tp_axis] = 1
        kv_shard_dims = [None, None]
        kv_shard_dims[sp_axis] = 2
        persistent_shard_dims = [None, None]  # gathered KV replicated (single latent head)

        tt_Q = ttnn.from_torch(
            Q,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=q_shard_dims),
        )
        tt_KV = ttnn.from_torch(
            KV_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kv_shard_dims),
        )
        persistent_output_buffer_kv = ttnn.from_torch(
            torch.zeros(cache_batch, nhk, sq, d_k),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_shard_dims
            ),
        )

        # No rotation here: the whole sq is valid, so actual_start=0, actual_end=logical_n=sq.
        tt_slot_id, tt_kv_actual_isl = _make_ring_mla_metadata(
            mesh_device, slot_id=kv_cache_batch_idx, actual_start=0, actual_end=sq
        )

        main_row_dim = q_shard_dims[0] if q_shard_dims[0] is not None else -1
        main_col_dim = q_shard_dims[1] if q_shard_dims[1] is not None else -1
        composer = ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(main_row_dim, main_col_dim))

        def run(use_metadata):
            # On the metadata path, a scalar still in META_PATH_HOST_SCALARS is also passed as a host
            # arg (its on-device read has not landed yet) so the result stays bit-exact.
            pass_kv_idx = (not use_metadata) or ("kv_cache_batch_idx" in META_PATH_HOST_SCALARS)
            tt_out, _ = ttnn.transformer.ring_mla(
                tt_Q,
                tt_KV,
                persistent_output_buffer_kv=persistent_output_buffer_kv,
                head_dim_v=d_v,
                logical_n=sq,
                is_balanced=False,
                program_config=program_config,
                compute_kernel_config=runtime.compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=runtime.ccl_semaphore_handles,
                num_links=runtime.num_links,
                cluster_axis=sp_axis,
                mesh_device=mesh_device,
                topology=runtime.topology,
                subdevice_id=runtime.worker_sub_device_id,
                ccl_core_grid_offset=(runtime.ccl_column, 0),
                use_column_major_ccl=True,
                kv_cache_batch_idx=kv_cache_batch_idx if pass_kv_idx else None,
                slot_id=tt_slot_id if use_metadata else None,
                kv_actual_isl_tensor=tt_kv_actual_isl if use_metadata else None,
            )
            return ttnn.to_torch(tt_out, mesh_composer=composer)[:, :, :sq, :d_v]

        out_scalar = run(use_metadata=False)
        out_meta = run(use_metadata=True)

        assert torch.equal(out_scalar, out_meta), (
            f"slot {kv_cache_batch_idx}: metadata-path ring_mla output differs from scalar-path "
            f"(max abs diff {(out_scalar - out_meta).abs().max().item()})"
        )
        logger.success(f"ring_mla slot {kv_cache_batch_idx}: metadata path == scalar path (bit-exact)")
    finally:
        close_ring_joint_sdpa_runtime(runtime)


@pytest.mark.parametrize("kv_actual_isl", [64, 256, 320], ids=["kv64", "kv256", "kv320"])
def test_ring_mla_metadata_matches_scalar_rotation(kv_actual_isl):
    """KV-pad rotation: the metadata path (kv_actual_isl read on-device from metadata[1], with logical_nt
    / q-mapping / ring masks derived in the reader and handed to compute via cb_kv_pad_derived) must be
    bit-identical to the scalar path (host kv_actual_isl). This is the discriminating test for the task-4
    on-device derivation: on the metadata path kv_actual_isl is dropped, so the host CANNOT compute the
    q-mapping -- it comes solely from the reader's metadata-driven derivation. Both paths run indexed
    (single-slot) mode at slot 0 so the only difference under test is where kv_actual_isl comes from."""
    mesh_config = MESH_CONFIG
    if mesh_config.sp_size < 2:
        pytest.skip(f"ring_mla requires at least 2 devices in ring, got SP={mesh_config.sp_size}")
    sp_size = mesh_config.sp_size
    tile = 32
    chunk_size_local = 64
    chunk_size_global = chunk_size_local * sp_size
    new_actual_isl = chunk_size_global  # one full new chunk
    assert kv_actual_isl % tile == 0

    b, local_heads = 1, 4
    nhq = local_heads * mesh_config.tp_size
    nhk = 1
    d_q, d_k, d_v = 64, 64, 32
    logical_n = kv_actual_isl + new_actual_isl

    torch.manual_seed(1234)
    old_cache_kv = fa_rand(b, nhk, kv_actual_isl, d_k)
    new_tokens_q = fa_rand(b, nhq, new_actual_isl, d_q)
    new_tokens_kv = fa_rand(b, nhk, new_actual_isl, d_k)
    q_host, kv_host, valid_rows, _, num_cache_slabs = build_kv_pad_rotation_mla_inputs(
        old_cache_kv, new_tokens_q, new_tokens_kv, kv_actual_isl, sp_size, chunk_size_local
    )
    cache_seq_per_dev = num_cache_slabs * chunk_size_local

    runtime = open_ring_joint_sdpa_runtime(mesh_config)
    mesh_device = runtime.mesh_device
    sp_axis, tp_axis = runtime.sp_axis, runtime.tp_axis
    try:
        q_shard_dims = [None, None]
        q_shard_dims[sp_axis] = 2
        if mesh_config.tp_size > 1:
            q_shard_dims[tp_axis] = 1
        kv_shard_dims = [None, None]
        kv_shard_dims[sp_axis] = 2  # latent K/V sharded along seq across the ring
        persistent_shard_dims = [None, None]  # gathered KV replicated

        tt_q = ttnn.from_torch(
            q_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=q_shard_dims),
        )
        tt_kv = ttnn.from_torch(
            kv_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kv_shard_dims),
        )
        persistent_output_buffer_kv = ttnn.from_torch(
            torch.zeros(b, nhk, sp_size * cache_seq_per_dev, d_k),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_shard_dims
            ),
        )

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=runtime.sdpa_compute_grid,
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
        )
        # metadata tensors: slot_id=0, kv_actual_isl tensor = kv_actual_isl (actual_end=logical_n unused).
        tt_slot_id, tt_kv_actual_isl = _make_ring_mla_metadata(
            mesh_device, slot_id=0, actual_start=kv_actual_isl, actual_end=logical_n
        )

        main_row_dim = q_shard_dims[0] if q_shard_dims[0] is not None else -1
        main_col_dim = q_shard_dims[1] if q_shard_dims[1] is not None else -1
        composer = ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(main_row_dim, main_col_dim))

        def run(use_metadata):
            tt_out, _ = ttnn.transformer.ring_mla(
                tt_q,
                tt_kv,
                persistent_output_buffer_kv=persistent_output_buffer_kv,
                head_dim_v=d_v,
                logical_n=logical_n,
                is_balanced=False,
                program_config=program_config,
                compute_kernel_config=runtime.compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=runtime.ccl_semaphore_handles,
                num_links=runtime.num_links,
                cluster_axis=sp_axis,
                mesh_device=mesh_device,
                topology=runtime.topology,
                subdevice_id=runtime.worker_sub_device_id,
                ccl_core_grid_offset=(runtime.ccl_column, 0),
                use_column_major_ccl=True,
                # Both paths run indexed at slot 0; the metadata path additionally drops kv_actual_isl so
                # the q-mapping must be derived on-device from metadata[1].
                kv_cache_batch_idx=None if use_metadata else 0,
                kv_actual_isl=None if use_metadata else kv_actual_isl,
                slot_id=tt_slot_id if use_metadata else None,
                kv_actual_isl_tensor=tt_kv_actual_isl if use_metadata else None,
            )
            return ttnn.to_torch(tt_out, mesh_composer=composer)[:, :, valid_rows, :d_v]

        out_scalar = run(use_metadata=False)
        out_meta = run(use_metadata=True)
        assert torch.equal(out_scalar, out_meta), (
            f"kv_actual_isl={kv_actual_isl}: metadata-path ring_mla output differs from scalar-path "
            f"(max abs diff {(out_scalar - out_meta).abs().max().item()})"
        )
        logger.success(f"ring_mla rotation kv_actual_isl={kv_actual_isl}: metadata path == scalar path (bit-exact)")
    finally:
        close_ring_joint_sdpa_runtime(runtime)


# Generate perf test parameters dynamically based on detected hardware for different models (WAN, MLA, VideGen...)
TEST_CONFIGS, TEST_CONFIG_IDS = generate_test_configs(MESH_CONFIG, RING_JOINT_PERF_MODEL_CONFIGS)
TEST_CONFIG_MODELS = list(MODEL_CONFIGS.keys())
PERF_TEST_CONFIG_MODELS = list(RING_JOINT_PERF_MODEL_CONFIGS.keys())


def generate_ring_mla_test_configs(mesh_config: MeshConfig, model_configs: Dict[str, ModelConfig]):
    if mesh_config.sp_size < 2 or "mla_100k" not in model_configs:
        return [], []
    model = model_configs["mla_100k"]
    q_chunk_size = 160
    k_chunk_size = 320
    config = (
        BATCH_SIZE,
        model.seq_len * mesh_config.sp_size,
        model.nhq * mesh_config.tp_size,
        model.nhk,
        model.d_q,
        model.d_k,
        model.d_v,
        q_chunk_size,
        k_chunk_size,
        model.is_balanced,
        model.q_dtype,
        model.kv_dtype,
    )
    return [config], [f"ring_mla-{model.name}-q{q_chunk_size}-k{k_chunk_size}"]


RING_MLA_TEST_CONFIGS, RING_MLA_TEST_CONFIG_IDS = generate_ring_mla_test_configs(MESH_CONFIG, MODEL_CONFIGS)


def _generate_standard_model_groups():
    if os.environ.get("CI") == "true":
        return [tuple(TEST_CONFIG_MODELS)], ["all-models"]
    return [(model_name,) for model_name in TEST_CONFIG_MODELS], TEST_CONFIG_MODELS


STANDARD_MODEL_GROUPS, STANDARD_MODEL_GROUP_IDS = _generate_standard_model_groups()


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


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.parametrize(
    "b,sq,nhq,nhk,d_q,d_k,d_v,q_chunk_size,k_chunk_size,is_balanced,q_dtype,kv_dtype",
    RING_MLA_TEST_CONFIGS,
    ids=RING_MLA_TEST_CONFIG_IDS,
)
def test_ring_mla_sweep_perf_impl(
    b,
    sq,
    nhq,
    nhk,
    d_q,
    d_k,
    d_v,
    q_chunk_size,
    k_chunk_size,
    is_balanced,
    q_dtype,
    kv_dtype,
):
    run_ring_mla_sdpa(
        MESH_CONFIG,
        b,
        nhq,
        nhk,
        sq,
        d_q,
        d_k,
        d_v,
        q_chunk_size,
        k_chunk_size,
        q_dtype,
        kv_dtype,
        is_balanced=is_balanced,
        do_check=False,
    )


# === TEST 2: ACCURACY VERIFICATION ===
@pytest.mark.parametrize("model_name", TEST_CONFIG_MODELS)
def test_ring_joint_attention_sdpa_accuracy(model_name):
    """
    Accuracy verification for every q/k chunk-size config in a model.

    ACCURACY METRICS:
    - PCC (Pearson Correlation Coefficient): Measures linear correlation
    - RMSE (Root Mean Square Error): Measures absolute error magnitude

    THRESHOLD RATIONALE:
    - PCC = 0.994: Relaxed for joint attention complexity
    """
    mesh_config = MESH_CONFIG
    model = MODEL_CONFIGS[model_name]

    pcc_threshold = DEFAULT_PCC_THRESHOLD
    rmse_threshold = DEFAULT_RMSE_THRESHOLD
    run_ring_joint_sdpa_model_configs(
        mesh_config,
        model,
        get_model_qk_configs(model),
        pcc_threshold=pcc_threshold,
        rmse_threshold=rmse_threshold,
    )


@pytest.mark.parametrize(
    "b,sq,nhq,nhk,d_q,d_k,d_v,q_chunk_size,k_chunk_size,is_balanced,q_dtype,kv_dtype",
    RING_MLA_TEST_CONFIGS,
    ids=RING_MLA_TEST_CONFIG_IDS,
)
def test_ring_mla_accuracy(
    b,
    sq,
    nhq,
    nhk,
    d_q,
    d_k,
    d_v,
    q_chunk_size,
    k_chunk_size,
    is_balanced,
    q_dtype,
    kv_dtype,
):
    run_ring_mla_sdpa(
        MESH_CONFIG,
        b,
        nhq,
        nhk,
        sq,
        d_q,
        d_k,
        d_v,
        q_chunk_size,
        k_chunk_size,
        q_dtype,
        kv_dtype,
        is_balanced=is_balanced,
    )


# === TEST 3: DETERMINISM VERIFICATION ===
@pytest.mark.parametrize(
    "model_names",
    STANDARD_MODEL_GROUPS,
    ids=STANDARD_MODEL_GROUP_IDS,
)
def test_ring_joint_attention_sdpa_determinism(model_names):
    """
    Test determinism for every q/k chunk-size config in a model.
    """
    mesh_config = MESH_CONFIG
    num_iterations = 10

    runtime = open_ring_joint_sdpa_runtime(mesh_config)
    try:
        for model_name in model_names:
            model = MODEL_CONFIGS[model_name]
            run_ring_joint_sdpa_model_configs(
                mesh_config,
                model,
                get_model_qk_configs(model),
                num_iterations=num_iterations,
                runtime=runtime,
            )
    finally:
        close_ring_joint_sdpa_runtime(runtime)


@pytest.mark.parametrize(
    "b,sq,nhq,nhk,d_q,d_k,d_v,q_chunk_size,k_chunk_size,is_balanced,q_dtype,kv_dtype",
    RING_MLA_TEST_CONFIGS,
    ids=RING_MLA_TEST_CONFIG_IDS,
)
def test_ring_mla_determinism(
    b,
    sq,
    nhq,
    nhk,
    d_q,
    d_k,
    d_v,
    q_chunk_size,
    k_chunk_size,
    is_balanced,
    q_dtype,
    kv_dtype,
):
    run_ring_mla_sdpa(
        MESH_CONFIG,
        b,
        nhq,
        nhk,
        sq,
        d_q,
        d_k,
        d_v,
        q_chunk_size,
        k_chunk_size,
        q_dtype,
        kv_dtype,
        is_balanced=is_balanced,
        do_check=False,
        num_iterations=10,
    )


# === TEST 4: PERFORMANCE TABLE GENERATOR (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.parametrize("model_name", PERF_TEST_CONFIG_MODELS)
def test_ring_joint_attention_create_perf_table(model_name):
    """
    Sweep chunk sizes for ring joint attention SDPA and print a performance table.
    Skipped on CI - run locally with tracy profiler.
    """
    from tracy.process_model_log import run_device_profiler

    mesh_config = MESH_CONFIG
    model_configs = RING_JOINT_PERF_MODEL_CONFIGS

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
RING_JOINT_PERF_MARGIN = 0.01

# Ring/TP geometry and per-device shapes are auto-selected by MeshConfig.detect():
#   QuietBox -> 4-device ring (sp=4, tp=1);  Galaxy -> 8-device ring x 4 TP shards (sp=8, tp=4).
if MESH_CONFIG.is_galaxy:
    RING_JOINT_PERF_CHECK_CONFIGS = [
        # (model_name, q_chunk_size, k_chunk_size, ring_size, expected_util, margin)
        # 8-device ring (Galaxy, sp=8 tp=4)
        ("wan2_2_1xGLX", 288, 512, 8, 70.7, RING_JOINT_PERF_MARGIN),
        # mla_100k on Galaxy is noisier than the other cases: observed run-to-run util spans
        # ~64.8-67.8% (midpoint ~66.3%, ~+/-2.3%), well beyond the default +/-1% band. Widen to
        # +/-3% so the gate tracks regressions without flagging this case's normal variance.
        ("mla_100k", 160, 320, 8, 66.3, 0.03),
    ]
else:
    RING_JOINT_PERF_CHECK_CONFIGS = [
        # (model_name, q_chunk_size, k_chunk_size, ring_size, expected_util, margin)
        # 4-device ring (QuietBox, sp=4 tp=1)
        ("wan2_2_1xGLX", 288, 512, 4, 68.9, RING_JOINT_PERF_MARGIN),
        ("mla_100k", 160, 320, 4, 63.2, RING_JOINT_PERF_MARGIN),
    ]


@pytest.mark.skipif(
    os.environ.get("SDPA_PERF_CHECKS") != "1",
    reason="Set SDPA_PERF_CHECKS=1 to run (CI: sdpa perf tests job)",
)
@pytest.mark.parametrize(
    "model_name, q_chunk_size, k_chunk_size, ring_size_expected, expected_util, margin",
    RING_JOINT_PERF_CHECK_CONFIGS,
    ids=[f"{cfg[0]}-q{cfg[1]}-k{cfg[2]}-ring{cfg[3]}" for cfg in RING_JOINT_PERF_CHECK_CONFIGS],
)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
def test_ring_joint_attention_perf_check(
    model_name, q_chunk_size, k_chunk_size, ring_size_expected, expected_util, margin
):
    """Measure ring joint SDPA math utilization via tracy and assert within the config's +/- margin."""
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
    ), "profiler returned no SDPA ops - inner test was skipped or did not produce a kernel run"

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

    lower = expected_util * (1 - margin)
    upper = expected_util * (1 + margin)

    logger.info(
        f"Ring joint SDPA perf check {config_id}: "
        f"duration={duration_ns/1e6:.3f} ms, math_util={utilization:.2f}% "
        f"(expected {expected_util:.2f}%, band [{lower:.2f}, {upper:.2f}])"
    )

    assert lower <= utilization <= upper, (
        f"Math utilization {utilization:.2f}% outside band [{lower:.2f}, {upper:.2f}] "
        f"(expected {expected_util:.2f}%, margin +/- {margin*100:.1f}%)"
    )


@pytest.mark.skipif(
    os.environ.get("SDPA_PERF_CHECKS") != "1",
    reason="Set SDPA_PERF_CHECKS=1 to run (CI: sdpa perf tests job)",
)
def test_ring_mla_perf_better_than_separate_v_ring_joint():
    """Profile ring_mla against the existing separate-V MLA ring joint path and require a speedup."""
    from tracy.process_model_log import run_device_profiler

    model_name = "mla_100k"
    q_chunk_size = 160
    k_chunk_size = 320
    ring_size_expected = 4

    if MESH_CONFIG.sp_size != ring_size_expected:
        pytest.skip(f"Expected ring size {ring_size_expected}, current topology has ring size {MESH_CONFIG.sp_size}")
    if model_name not in MODEL_CONFIGS or not RING_MLA_TEST_CONFIG_IDS:
        pytest.skip("ring_mla perf config unavailable for current mesh")

    model = MODEL_CONFIGS[model_name]
    joint_config_id = get_test_case_id(model, q_chunk_size, k_chunk_size)
    mla_config_id = RING_MLA_TEST_CONFIG_IDS[0]

    float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
    cols = ["ATTRIBUTES"]

    def profile_duration(command, subdir):
        with mock.patch.dict(os.environ, {"CI": "false"}):
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
        result = post_process_ops_log(
            subdir,
            float_columns=float_cols,
            columns=cols,
            op_name="RingJointSDPADeviceOperation",
            sum_vals=False,
            has_signposts=False,
        )
        assert len(result["DEVICE KERNEL DURATION [ns]"]) > 0, f"profiler returned no SDPA ops for {command}"
        return int(result["DEVICE KERNEL DURATION [ns]"].max())

    ring_mla_duration_ns = profile_duration(
        (
            "pytest tests/nightly/blackhole/sdpa/"
            f"test_ring_joint_sdpa.py::test_ring_mla_sweep_perf_impl[{mla_config_id}]"
        ),
        "ttnn_ring_mla_perf_check",
    )
    separate_v_duration_ns = profile_duration(
        (
            "pytest tests/nightly/blackhole/sdpa/"
            f"test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_sweep_perf_impl[{joint_config_id}]"
        ),
        "ttnn_ring_mla_baseline_perf_check",
    )

    logger.info(
        f"ring_mla perf: {ring_mla_duration_ns/1e6:.3f} ms, "
        f"separate-V ring_joint: {separate_v_duration_ns/1e6:.3f} ms"
    )
    assert ring_mla_duration_ns < separate_v_duration_ns, (
        f"ring_mla must beat separate-V ring_joint for {model_name} q={q_chunk_size} k={k_chunk_size}: "
        f"{ring_mla_duration_ns/1e6:.3f} ms >= {separate_v_duration_ns/1e6:.3f} ms"
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
        q_dtype=ttnn.bfloat16,
        kv_dtype=ttnn.bfloat8_b,
        q_chunk_sizes=[32],
        k_chunk_sizes=[512, 640],
        seq_len=CHUNKED_PREFILL_CHUNK_SIZE,  # unused by chunked path
    ),
}
CHUNKED_PREFILL_MODELS = list(CHUNKED_PREFILL_MODEL_CONFIGS.keys())

# ring_mla (latent-V) chunked-prefill configs are identical to the classic separate-V configs
# except V lives in the first d_v columns of the shared K/V latent (the MLA deployment shape):
# d_v widens to the latent V dimension. Derive them so the two paths can't drift apart.
RING_MLA_CHUNKED_LATENT_D_V = 512
RING_MLA_CHUNKED_MODEL_CONFIGS = {
    name: replace(cfg, d_v=RING_MLA_CHUNKED_LATENT_D_V) for name, cfg in CHUNKED_PREFILL_MODEL_CONFIGS.items()
}

# Minimax3 production chunked-prefill GQA shape. The full model is 64 Q heads and 4 K/V
# heads. With TP=4, each chip sees one KV head; keep the config per-ring so the generic
# TP scaling produces 64Q/4KV globally on Galaxy and still exercises one-KV-head GQA locally.
MINIMAX3_GQA_CHUNKED_MODEL_CONFIGS = {
    "minimax3_55k": ModelConfig(
        name="minimax3_55k",
        nhq=16,
        nhk=1,
        nhv=1,
        d_q=128,
        d_k=128,
        d_v=128,
        is_causal=True,
        q_dtype=ttnn.bfloat16,
        kv_dtype=ttnn.bfloat8_b,
        q_chunk_sizes=[128],
        k_chunk_sizes=[512],
        seq_len=CHUNKED_PREFILL_CHUNK_SIZE,  # unused by chunked path
    ),
}


def _generate_chunked_configs(model_configs):
    configs = []
    ids = []
    for model_name, model in model_configs.items():
        for q, k in product(model.q_chunk_sizes, model.k_chunk_sizes):
            configs.append((model_name, q, k))
            ids.append(f"{model_name}-q{q}-k{k}")
    return configs, ids


CHUNKED_CONFIGS, CHUNKED_CONFIG_IDS = _generate_chunked_configs(CHUNKED_PREFILL_MODEL_CONFIGS)
RING_MLA_CHUNKED_CONFIGS, RING_MLA_CHUNKED_CONFIG_IDS = _generate_chunked_configs(RING_MLA_CHUNKED_MODEL_CONFIGS)
MINIMAX3_GQA_CHUNKED_CONFIGS, MINIMAX3_GQA_CHUNKED_CONFIG_IDS = _generate_chunked_configs(
    MINIMAX3_GQA_CHUNKED_MODEL_CONFIGS
)


def _generate_chunked_test_configs(model_configs, chunked_configs, chunked_config_ids):
    if os.environ.get("CI") == "true":
        configs = []
        ids = []
        for model_name, model in model_configs.items():
            configs.append((model_name, get_model_qk_configs(model)))
            ids.append(f"{model_name}-all-qk")
        return configs, ids

    configs = []
    for model_name, q_chunk_size, k_chunk_size in chunked_configs:
        configs.append((model_name, [(q_chunk_size, k_chunk_size)]))
    return configs, chunked_config_ids


CHUNKED_TEST_CONFIGS, CHUNKED_TEST_CONFIG_IDS = _generate_chunked_test_configs(
    CHUNKED_PREFILL_MODEL_CONFIGS,
    CHUNKED_CONFIGS,
    CHUNKED_CONFIG_IDS,
)
RING_MLA_CHUNKED_TEST_CONFIGS, RING_MLA_CHUNKED_TEST_CONFIG_IDS = _generate_chunked_test_configs(
    RING_MLA_CHUNKED_MODEL_CONFIGS,
    RING_MLA_CHUNKED_CONFIGS,
    RING_MLA_CHUNKED_CONFIG_IDS,
)
MINIMAX3_GQA_CHUNKED_TEST_CONFIGS, MINIMAX3_GQA_CHUNKED_TEST_CONFIG_IDS = _generate_chunked_test_configs(
    MINIMAX3_GQA_CHUNKED_MODEL_CONFIGS,
    MINIMAX3_GQA_CHUNKED_CONFIGS,
    MINIMAX3_GQA_CHUNKED_CONFIG_IDS,
)

MINIMAX3_GQA_CHUNKED_ACCURACY_CHUNK_SIZE = CHUNKED_PREFILL_CHUNK_SIZE
MINIMAX3_GQA_CHUNKED_ACCURACY_TOTAL_SEQ = 3 * MINIMAX3_GQA_CHUNKED_ACCURACY_CHUNK_SIZE


@pytest.mark.parametrize("chunk_size", [CHUNKED_PREFILL_CHUNK_SIZE], ids=[f"chunk{CHUNKED_PREFILL_CHUNK_SIZE}"])
@pytest.mark.parametrize(
    "model_name,qk_configs",
    CHUNKED_TEST_CONFIGS,
    ids=CHUNKED_TEST_CONFIG_IDS,
)
def test_ring_joint_attention_sdpa_chunked_accuracy(model_name, qk_configs, chunk_size):
    """Validate ring joint SDPA chunked prefill with reusable max-sized K/V buffers."""
    mesh_config = MESH_CONFIG

    run_ring_joint_sdpa_chunked(
        mesh_config,
        CHUNKED_PREFILL_MODEL_CONFIGS[model_name],
        chunk_size=chunk_size,
        qk_configs=qk_configs,
        persistent_buffer_mode="reuse_max",
    )


@pytest.mark.parametrize(
    "nhq, nhk, nhv, supported",
    [
        # Supported head modes (mirror the device-op classification in ring_joint_sdpa_device_operation.cpp).
        (16, 16, 16, True),  # MHA
        (16, 1, 16, True),  # separate-V shared-K
        (16, 1, 1, True),  # GQA, one local KV head (production multicast case)
        (16, 2, 2, True),  # GQA, grouped KV (unicast-fallback case)
        (16, 4, 4, True),  # GQA, larger group count
        # Rejected: GQA ratio must divide evenly and K/V head counts must match.
        (16, 3, 3, False),  # nhq % nhk != 0
        (16, 5, 5, False),  # nhq % nhk != 0
        (16, 2, 4, False),  # nhk != nhv
        (16, 16, 1, False),  # NVH must equal NQH for shared-K, or NQH for GQA
        (16, 32, 32, False),  # nhk > nhq
    ],
)
def test_is_supported_ring_joint_head_mode(nhq, nhk, nhv, supported):
    """Lock the GQA head-mode acceptance contract that the test harness uses to skip and that the
    device op enforces via TT_FATAL (ring_joint_sdpa_device_operation.cpp head-relationship check).
    Pure host-side check — no device required."""
    assert is_supported_ring_joint_head_mode(nhq, nhk, nhv) == supported


def test_ring_joint_attention_gqa_with_joint_tensors_rejected(expect_error):
    """GQA grouped-K/V is unsupported with joint tensors; the device op must reject it (validation-only,
    so no kernel runs). Covers ring_joint_sdpa_device_operation.cpp's GQA-with-joint TT_FATAL, which the
    host head-mode guard does not catch."""
    mesh_config = MESH_CONFIG
    b, nhq, nhk, nhv, d = 1, 16, 2, 2, 128  # GQA grouped K/V (nhk == nhv < nhq)
    sq = 512 * mesh_config.sp_size
    joint_l = 32  # tile-aligned joint length

    runtime = open_ring_joint_sdpa_runtime(mesh_config)
    try:
        mesh_device = runtime.mesh_device
        sp_axis, tp_axis = runtime.sp_axis, runtime.tp_axis
        mesh_shape = tuple(mesh_device.shape)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        # Mirror the run helper's sharding so the gathered-buffer shape check (input_seq * ring_size)
        # passes and validation proceeds to the GQA-with-joint check.
        def sharded(t, shard_heads):
            dims = [None, None]
            dims[sp_axis] = 2  # shard inputs on the sequence dim
            if mesh_config.tp_size > 1 and shard_heads:
                dims[tp_axis] = 1
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
            )

        def persistent(t):
            dims = [None, None]  # seq NOT sharded: holds the gathered full sequence
            if mesh_config.tp_size > 1:
                dims[tp_axis] = 1
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
            )

        def replicated(t):
            return ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=replicate
            )

        tt_q = sharded(fa_rand(b, nhq, sq, d), shard_heads=True)
        tt_k = sharded(fa_rand(b, nhk, sq, d), shard_heads=True)
        tt_v = sharded(fa_rand(b, nhv, sq, d), shard_heads=True)
        # Joint tensors only need to exist with valid shapes to reach the GQA-with-joint check.
        joint_q = replicated(fa_rand(b, nhq, joint_l, d))
        joint_k = replicated(fa_rand(b, nhk, joint_l, d))
        joint_v = replicated(fa_rand(b, nhv, joint_l, d))
        p_buf_k, p_buf_v = persistent(torch.zeros(b, nhk, sq, d)), persistent(torch.zeros(b, nhv, sq, d))

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=runtime.sdpa_compute_grid,
            q_chunk_size=128,
            k_chunk_size=512,
            exp_approx_mode=False,
        )

        # Joint tensors require non-causal (causal+joint is rejected earlier), so this is the
        # non-causal GQA + joint case that the GQA-with-joint TT_FATAL is meant to catch.
        with expect_error(RuntimeError, "GQA with joint tensors"):
            ttnn.transformer.ring_joint_scaled_dot_product_attention(
                tt_q,
                tt_k,
                tt_v,
                joint_q,
                joint_k,
                joint_v,
                persistent_output_buffer_k=p_buf_k,
                persistent_output_buffer_v=p_buf_v,
                joint_strategy="rear",
                logical_n=sq,
                is_causal=False,
                is_balanced=False,
                program_config=program_config,
                compute_kernel_config=runtime.compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=runtime.ccl_semaphore_handles,
                num_links=runtime.num_links,
                cluster_axis=runtime.sp_axis,
                mesh_device=mesh_device,
                topology=runtime.topology,
                subdevice_id=runtime.worker_sub_device_id,
                ccl_core_grid_offset=(runtime.ccl_column, 0),
                use_column_major_ccl=True,
            )
    finally:
        close_ring_joint_sdpa_runtime(runtime)


def test_ring_joint_attention_minimax3_gqa_chunked_accuracy():
    """Small causal GQA chunked-prefill PCC gate for Minimax3 dims without the full 55k CPU reference."""
    run_ring_joint_sdpa_chunked(
        MESH_CONFIG,
        MINIMAX3_GQA_CHUNKED_MODEL_CONFIGS["minimax3_55k"],
        chunk_size=MINIMAX3_GQA_CHUNKED_ACCURACY_CHUNK_SIZE,
        total_seq=MINIMAX3_GQA_CHUNKED_ACCURACY_TOTAL_SEQ,
        qk_configs=[(128, 512)],
        persistent_buffer_mode="reuse_max",
    )


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.parametrize("reuse_kv_buffer", [False, True], ids=["fresh_kv", "reuse_kv"])
@pytest.mark.parametrize("chunk_size", [CHUNKED_PREFILL_CHUNK_SIZE], ids=[f"chunk{CHUNKED_PREFILL_CHUNK_SIZE}"])
@pytest.mark.parametrize(
    "model_name,qk_configs",
    CHUNKED_TEST_CONFIGS,
    ids=CHUNKED_TEST_CONFIG_IDS,
)
def test_ring_joint_attention_chunked_perf_impl(model_name, qk_configs, chunk_size, reuse_kv_buffer):
    """Classic separate-K/V ring joint SDPA chunked prefill without the CPU reference (profiled by
    test_ring_joint_attention_create_chunked_perf_table). reuse_kv: one oversized cache reused across
    chunks; fresh_kv: a per-chunk right-sized input."""
    mesh_config = MESH_CONFIG

    run_ring_joint_sdpa_chunked(
        mesh_config,
        CHUNKED_PREFILL_MODEL_CONFIGS[model_name],
        chunk_size=chunk_size,
        qk_configs=qk_configs,
        persistent_buffer_mode="reuse_max",
        do_check=False,
        reuse_kv_buffer=reuse_kv_buffer,
    )


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.parametrize("reuse_kv_buffer", [False, True], ids=["fresh_kv", "reuse_kv"])
@pytest.mark.parametrize("chunk_size", [CHUNKED_PREFILL_CHUNK_SIZE], ids=[f"chunk{CHUNKED_PREFILL_CHUNK_SIZE}"])
@pytest.mark.parametrize(
    "model_name,qk_configs",
    MINIMAX3_GQA_CHUNKED_TEST_CONFIGS,
    ids=MINIMAX3_GQA_CHUNKED_TEST_CONFIG_IDS,
)
def test_ring_joint_attention_minimax3_gqa_chunked_perf_impl(model_name, qk_configs, chunk_size, reuse_kv_buffer):
    """Minimax3 GQA chunked prefill without the CPU reference. This mirrors the Kimi chunked
    perf harness but uses one KV head per TP shard."""
    mesh_config = MESH_CONFIG

    run_ring_joint_sdpa_chunked(
        mesh_config,
        MINIMAX3_GQA_CHUNKED_MODEL_CONFIGS[model_name],
        chunk_size=chunk_size,
        qk_configs=qk_configs,
        persistent_buffer_mode="reuse_max",
        do_check=False,
        reuse_kv_buffer=reuse_kv_buffer,
    )


@pytest.mark.parametrize("chunk_size", [CHUNKED_PREFILL_CHUNK_SIZE], ids=[f"chunk{CHUNKED_PREFILL_CHUNK_SIZE}"])
@pytest.mark.parametrize(
    "model_name,qk_configs",
    RING_MLA_CHUNKED_TEST_CONFIGS,
    ids=RING_MLA_CHUNKED_TEST_CONFIG_IDS,
)
def test_ring_mla_chunked_accuracy(model_name, qk_configs, chunk_size):
    """Validate ring_mla (latent V) chunked prefill with reusable max-sized K/V buffers."""
    mesh_config = MESH_CONFIG

    run_ring_joint_sdpa_chunked(
        mesh_config,
        RING_MLA_CHUNKED_MODEL_CONFIGS[model_name],
        chunk_size=chunk_size,
        qk_configs=qk_configs,
        persistent_buffer_mode="reuse_max",
        use_ring_mla=True,
    )


# Perf-profiling twin of test_ring_mla_chunked_accuracy: identical device work, but do_check=False
# skips the O(total_seq^2) CPU torch reference so the run fits the profiler timeout. Skipped on CI
# directly; it is driven by test_ring_mla_chunked_perf_check via run_device_profiler (which sets
# CI=false in the subprocess). Mirrors the test_ring_joint_attention_sdpa_sweep_perf_impl pattern.
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.parametrize("reuse_kv_buffer", [False, True], ids=["fresh_kv", "reuse_kv"])
@pytest.mark.parametrize("chunk_size", [CHUNKED_PREFILL_CHUNK_SIZE], ids=[f"chunk{CHUNKED_PREFILL_CHUNK_SIZE}"])
@pytest.mark.parametrize(
    "model_name,qk_configs",
    RING_MLA_CHUNKED_TEST_CONFIGS,
    ids=RING_MLA_CHUNKED_TEST_CONFIG_IDS,
)
def test_ring_mla_chunked_perf_impl(model_name, qk_configs, chunk_size, reuse_kv_buffer):
    """ring_mla chunked prefill without the CPU reference (profiled by the perf table and check).
    reuse_kv: one oversized cache reused across chunks; fresh_kv (what the CI perf check profiles):
    a per-chunk right-sized input."""
    mesh_config = MESH_CONFIG

    run_ring_joint_sdpa_chunked(
        mesh_config,
        RING_MLA_CHUNKED_MODEL_CONFIGS[model_name],
        chunk_size=chunk_size,
        qk_configs=qk_configs,
        persistent_buffer_mode="reuse_max",
        use_ring_mla=True,
        do_check=False,
        reuse_kv_buffer=reuse_kv_buffer,
    )


# === TEST 7: CHUNKED-PREFILL DETERMINISM ===
@pytest.mark.parametrize("chunk_size", [CHUNKED_PREFILL_CHUNK_SIZE], ids=[f"chunk{CHUNKED_PREFILL_CHUNK_SIZE}"])
@pytest.mark.parametrize(
    "model_name,qk_configs",
    CHUNKED_TEST_CONFIGS,
    ids=CHUNKED_TEST_CONFIG_IDS,
)
def test_ring_joint_attention_sdpa_chunked_determinism(model_name, qk_configs, chunk_size):
    """Run each chunked prefill step 3 times and require bit-exact per-chunk outputs."""
    mesh_config = MESH_CONFIG

    run_ring_joint_sdpa_chunked(
        mesh_config,
        CHUNKED_PREFILL_MODEL_CONFIGS[model_name],
        chunk_size=chunk_size,
        qk_configs=qk_configs,
        num_iterations=3,
    )


@pytest.mark.parametrize("chunk_size", [CHUNKED_PREFILL_CHUNK_SIZE], ids=[f"chunk{CHUNKED_PREFILL_CHUNK_SIZE}"])
@pytest.mark.parametrize(
    "model_name,qk_configs",
    MINIMAX3_GQA_CHUNKED_TEST_CONFIGS,
    ids=MINIMAX3_GQA_CHUNKED_TEST_CONFIG_IDS,
)
def test_ring_joint_attention_minimax3_gqa_chunked_determinism(model_name, qk_configs, chunk_size):
    """Run the Minimax3 final chunk three times and require bit-exact output. The final chunk is
    the production-shaped 5k Q chunk attending to the full 55k K/V cache on Galaxy."""
    mesh_config = MESH_CONFIG
    n_chunks = CHUNKED_PREFILL_TOTAL_SEQ // chunk_size
    final_chunk = n_chunks - 1

    with mock.patch.dict(os.environ, {CHUNKED_PREFILL_CHUNK_ID_ENV: str(final_chunk)}):
        run_ring_joint_sdpa_chunked(
            mesh_config,
            MINIMAX3_GQA_CHUNKED_MODEL_CONFIGS[model_name],
            chunk_size=chunk_size,
            qk_configs=qk_configs,
            num_iterations=3,
        )


@pytest.mark.parametrize("chunk_size", [CHUNKED_PREFILL_CHUNK_SIZE], ids=[f"chunk{CHUNKED_PREFILL_CHUNK_SIZE}"])
@pytest.mark.parametrize(
    "model_name,qk_configs",
    RING_MLA_CHUNKED_TEST_CONFIGS,
    ids=RING_MLA_CHUNKED_TEST_CONFIG_IDS,
)
def test_ring_mla_chunked_determinism(model_name, qk_configs, chunk_size):
    """Run each ring_mla chunked prefill step 3 times and require bit-exact per-chunk outputs."""
    mesh_config = MESH_CONFIG

    run_ring_joint_sdpa_chunked(
        mesh_config,
        RING_MLA_CHUNKED_MODEL_CONFIGS[model_name],
        chunk_size=chunk_size,
        qk_configs=qk_configs,
        num_iterations=3,
        use_ring_mla=True,
    )


# === TEST 8: CHUNKED-PREFILL PERF TABLE (skipped on CI) ===
def _run_chunked_perf_table(
    mesh_config,
    model,
    model_name,
    q_chunk_size,
    k_chunk_size,
    chunk_size,
    accuracy_test_name,
    subdir,
    label,
    id_suffix="",
):
    """Run chunked prefill once with tracy and print a per-chunk math-util table.

    Per-chunk work is rectangle (Q_chunk vs prefix K/V, non-causal) + triangle (Q_chunk vs
    current K/V, causal half), so later chunks have a larger prefix and should reach higher
    math utilization than chunk 0 (which is only the triangle).

    accuracy_test_name selects which chunked-accuracy test the profiler subprocess drives
    (classic separate-K/V vs the ring_mla latent-V fork); everything else is identical.
    """
    from tracy.process_model_log import run_device_profiler

    ring_size = mesh_config.sp_size

    if ring_size < 2:
        pytest.skip(f"Ring joint chunked prefill requires at least 2 devices, got {ring_size}")

    total_seq = CHUNKED_PREFILL_TOTAL_SEQ
    n_chunks = total_seq // chunk_size

    # Single-chunk mode: when RING_JOINT_CHUNKED_CHUNK_ID is set, the accuracy subprocess
    # runs only that chunk, so we profile/report just that one. The env var is inherited by
    # the run_device_profiler subprocess.
    only_chunk = get_chunked_only_chunk_id(n_chunks)
    chunk_indices = [only_chunk] if only_chunk is not None else list(range(n_chunks))
    num_profiled = len(chunk_indices)

    config_id = f"{get_test_case_id(model, q_chunk_size, k_chunk_size)}-chunk{chunk_size}{id_suffix}"
    command = f"pytest tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::{accuracy_test_name}[{config_id}]"

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
        len(durations) % num_profiled == 0
    ), f"RingJointSDPADeviceOperation entry count ({len(durations)}) is not a multiple of profiled chunks ({num_profiled})"
    devs_per_chunk = len(durations) // num_profiled
    expected_devs = mesh_config.tp_size * mesh_config.sp_size
    assert (
        devs_per_chunk == expected_devs
    ), f"Expected {expected_devs} entries per chunk (tp_size * sp_size), got {devs_per_chunk}"

    chunk_durations = [max(durations[s * devs_per_chunk : (s + 1) * devs_per_chunk]) for s in range(num_profiled)]
    chunk_core_counts = [
        max(int(c) for c in core_counts[s * devs_per_chunk : (s + 1) * devs_per_chunk]) for s in range(num_profiled)
    ]

    q_per_dev = chunk_size // ring_size
    # model.nhq is PER RING, and heads-per-ring == heads-per-device (heads shard across tp_axis).
    nh_per_dev = model.nhq
    d_q, d_v = model.d_q, model.d_v
    constants = ARCH_CONSTANTS["blackhole"]
    clock_ghz = constants["clock_ghz"]
    flops_per_cycle_per_core = constants["mm_flops_per_cycle_per_core"]

    per_chunk_rows = []
    for slot, (dur_ns, ccount) in enumerate(zip(chunk_durations, chunk_core_counts)):
        i = chunk_indices[slot]
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
        f"{label} Per-Chunk Math Util: model={model_name}, "
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
    if only_chunk is None:
        assert utils[-1] > utils[0], (
            f"Expected last chunk util ({utils[-1]:.1f}%) > first chunk util ({utils[0]:.1f}%) "
            f"— prefix grows with chunk index, so util should increase."
        )


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("chunk_size", [CHUNKED_PREFILL_CHUNK_SIZE], ids=[f"chunk{CHUNKED_PREFILL_CHUNK_SIZE}"])
@pytest.mark.parametrize(
    "model_name,q_chunk_size,k_chunk_size",
    CHUNKED_CONFIGS,
    ids=CHUNKED_CONFIG_IDS,
)
def test_ring_joint_attention_create_chunked_perf_table(model_name, q_chunk_size, k_chunk_size, chunk_size):
    """Per-chunk math-util + duration table for the classic separate-K/V chunked-prefill path,
    profiling the reuse_kv variant (one oversized cache reused across chunks). Per-chunk device time
    tracks the logical_n-bounded gather, exposing whether the gather honours that bound."""
    _run_chunked_perf_table(
        MESH_CONFIG,
        CHUNKED_PREFILL_MODEL_CONFIGS[model_name],
        model_name,
        q_chunk_size,
        k_chunk_size,
        chunk_size,
        accuracy_test_name="test_ring_joint_attention_chunked_perf_impl",
        subdir="ttnn_ring_joint_sdpa_chunked_performance",
        label="Ring Joint Chunked-Prefill (reuse KV buffer)",
        id_suffix="-reuse_kv",
    )


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("chunk_size", [CHUNKED_PREFILL_CHUNK_SIZE], ids=[f"chunk{CHUNKED_PREFILL_CHUNK_SIZE}"])
@pytest.mark.parametrize(
    "model_name,q_chunk_size,k_chunk_size",
    MINIMAX3_GQA_CHUNKED_CONFIGS,
    ids=MINIMAX3_GQA_CHUNKED_CONFIG_IDS,
)
def test_ring_joint_attention_minimax3_gqa_create_chunked_perf_table(
    model_name, q_chunk_size, k_chunk_size, chunk_size
):
    """Per-chunk math-util + duration table for Minimax3 GQA chunked prefill."""
    _run_chunked_perf_table(
        MESH_CONFIG,
        MINIMAX3_GQA_CHUNKED_MODEL_CONFIGS[model_name],
        model_name,
        q_chunk_size,
        k_chunk_size,
        chunk_size,
        accuracy_test_name="test_ring_joint_attention_minimax3_gqa_chunked_perf_impl",
        subdir="ttnn_ring_joint_sdpa_minimax3_gqa_chunked_performance",
        label="Ring Joint Minimax3 GQA Chunked-Prefill (reuse KV buffer)",
        id_suffix="-reuse_kv",
    )


def compute_chunked_prefill_perf_check_utilization(
    mesh_config, model, chunk_size, perf_chunk, duration_ns, measured_core_count
):
    # Chunk geometry: q_per_dev Q rows attend to the full prefix (non-causal rectangle) plus
    # the current chunk's causal triangle. Folding the triangle into an effective K/V length
    # (prefix + chunk_size/2) makes compute_ring_joint_utilization's non-causal FLOPs exact.
    q_per_dev = chunk_size // mesh_config.sp_size
    prefix_k = perf_chunk * chunk_size
    effective_kv = prefix_k + chunk_size // 2

    # Match perf-table effective_cores rounding (ignore non-multiple-of-grid-row CCL strays).
    effective_cores = measured_core_count - measured_core_count % mesh_config.grid_rows
    assert (
        effective_cores > 0
    ), f"effective_cores=0 (measured_core_count={measured_core_count}) - profiler output incomplete"

    utilization = compute_ring_joint_utilization(
        q_per_dev, effective_kv, model.d_q, model.d_v, model.nhq, duration_ns, effective_cores, is_causal=False
    )
    return utilization, effective_cores


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("chunk_size", [CHUNKED_PREFILL_CHUNK_SIZE], ids=[f"chunk{CHUNKED_PREFILL_CHUNK_SIZE}"])
@pytest.mark.parametrize(
    "model_name,q_chunk_size,k_chunk_size",
    RING_MLA_CHUNKED_CONFIGS,
    ids=RING_MLA_CHUNKED_CONFIG_IDS,
)
def test_ring_mla_create_chunked_perf_table(model_name, q_chunk_size, k_chunk_size, chunk_size):
    """Per-chunk math-util + duration table for the ring_mla chunked-prefill path, profiling the
    reuse_kv variant (one oversized cache reused across chunks). Per-chunk device time tracks the
    logical_n-bounded gather, exposing whether the gather honours that bound."""
    _run_chunked_perf_table(
        MESH_CONFIG,
        RING_MLA_CHUNKED_MODEL_CONFIGS[model_name],
        model_name,
        q_chunk_size,
        k_chunk_size,
        chunk_size,
        accuracy_test_name="test_ring_mla_chunked_perf_impl",
        subdir="ttnn_ring_mla_chunked_performance",
        label="Ring MLA Chunked-Prefill (reuse KV buffer)",
        id_suffix="-reuse_kv",
    )


# === TEST 9: CHUNKED-PREFILL ring_mla PERF CHECK (CI-gated by SDPA_PERF_CHECKS=1) ===
# Profiles the kimi 50k+5k galaxy chunk (final, most compute-bound chunk of the kimi50k chunked
# prefill): natively on Galaxy (sp=8, tp=4) and simulated on the 4-device QuietBox (sp=4, tp=1).
# Symmetric +/- band, same as the ring joint perf check.
if MESH_CONFIG.is_galaxy:
    RING_MLA_CHUNKED_PERF_CHECK_CONFIGS = [
        # (model_name, q_chunk_size, k_chunk_size, ring_size, expected_util)
        # 8-device ring (Galaxy, sp=8 tp=4, 100 SDPA cores)
        ("kimi50k", 32, 640, 8, 68.5),
    ]
else:
    RING_MLA_CHUNKED_PERF_CHECK_CONFIGS = [
        # (model_name, q_chunk_size, k_chunk_size, ring_size, expected_util)
        # 4-device ring (QuietBox, 100 SDPA cores)
        ("kimi50k", 32, 640, 4, 66.05),
    ]


if MESH_CONFIG.is_galaxy:
    MINIMAX3_GQA_CHUNKED_PERF_CHECK_CONFIGS = [
        # (model_name, q_chunk_size, k_chunk_size, ring_size, expected_util)
        # 8-device ring (Galaxy, sp=8 tp=4): production 5k Q chunk attending to 50k K/V prefix.
        ("minimax3_55k", 128, 512, 8, 47.64),
    ]
else:
    MINIMAX3_GQA_CHUNKED_PERF_CHECK_CONFIGS = [
        # (model_name, q_chunk_size, k_chunk_size, ring_size, expected_util)
        # 4-device ring (QuietBox): same per-device Q rows and 16Q/1KV local GQA shape.
        ("minimax3_55k", 128, 512, 4, 47.64),
    ]


@pytest.mark.skipif(
    os.environ.get("SDPA_PERF_CHECKS") != "1",
    reason="Set SDPA_PERF_CHECKS=1 to run (CI: sdpa perf tests job)",
)
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "model_name, q_chunk_size, k_chunk_size, ring_size_expected, expected_util",
    RING_MLA_CHUNKED_PERF_CHECK_CONFIGS,
    ids=[f"{cfg[0]}-q{cfg[1]}-k{cfg[2]}-ring{cfg[3]}" for cfg in RING_MLA_CHUNKED_PERF_CHECK_CONFIGS],
)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
def test_ring_mla_chunked_perf_check(model_name, q_chunk_size, k_chunk_size, ring_size_expected, expected_util):
    """Measure ring_mla chunked-prefill math utilization for the kimi 50k+5k galaxy chunk (a 5k Q
    chunk against a 50k K/V prefix), simulated on the 4-device QuietBox, via tracy and assert
    within +/- RING_JOINT_PERF_MARGIN.

    RING_JOINT_CHUNKED_CHUNK_ID isolates the final chunk so only its iteration is profiled;
    each chunk rebuilds its K/V cache from scratch, so it reproduces the full-sequence kernel.
    """
    from tracy.process_model_log import run_device_profiler

    if MESH_CONFIG.sp_size != ring_size_expected:
        pytest.skip(f"Expected ring size {ring_size_expected}, current topology has ring size {MESH_CONFIG.sp_size}")

    model = RING_MLA_CHUNKED_MODEL_CONFIGS[model_name]
    chunk_size = CHUNKED_PREFILL_CHUNK_SIZE
    n_chunks = CHUNKED_PREFILL_TOTAL_SEQ // chunk_size
    perf_chunk = n_chunks - 1  # final chunk: largest K/V prefix + current chunk (simulates galaxy 50k+5k)

    config_id = f"{get_test_case_id(model, q_chunk_size, k_chunk_size)}-chunk{chunk_size}-fresh_kv"
    subdir = "ttnn_ring_mla_chunked_perf_check"
    command = (
        f"pytest tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_mla_chunked_perf_impl[{config_id}]"
    )

    float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
    cols = ["ATTRIBUTES"]

    with mock.patch.dict(os.environ, {"CI": "false", CHUNKED_PREFILL_CHUNK_ID_ENV: str(perf_chunk)}):
        run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log(
        subdir,
        float_columns=float_cols,
        columns=cols,
        op_name="RingJointSDPADeviceOperation",
        sum_vals=False,
        has_signposts=False,
    )

    assert (
        len(r["CORE COUNT"]) > 0 and len(r["DEVICE KERNEL DURATION [ns]"]) > 0
    ), "profiler returned no SDPA ops - inner test was skipped or did not produce a kernel run"

    measured_core_count = int(r["CORE COUNT"][0])
    duration_ns = int(r["DEVICE KERNEL DURATION [ns]"].max())

    utilization, _ = compute_chunked_prefill_perf_check_utilization(
        MESH_CONFIG, model, chunk_size, perf_chunk, duration_ns, measured_core_count
    )

    lower = expected_util * (1 - RING_JOINT_PERF_MARGIN)
    upper = expected_util * (1 + RING_JOINT_PERF_MARGIN)

    logger.info(
        f"ring_mla chunked 50k+5k perf check {config_id}: "
        f"duration={duration_ns/1e6:.3f} ms, math_util={utilization:.2f}% "
        f"(expected {expected_util:.2f}%, band [{lower:.2f}, {upper:.2f}])"
    )

    assert lower <= utilization <= upper, (
        f"Math utilization {utilization:.2f}% outside band [{lower:.2f}, {upper:.2f}] "
        f"(expected {expected_util:.2f}%, margin +/- {RING_JOINT_PERF_MARGIN*100:.1f}%)"
    )


@pytest.mark.skipif(
    os.environ.get("SDPA_PERF_CHECKS") != "1",
    reason="Set SDPA_PERF_CHECKS=1 to run (CI: sdpa perf tests job)",
)
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "model_name, q_chunk_size, k_chunk_size, ring_size_expected, expected_util",
    MINIMAX3_GQA_CHUNKED_PERF_CHECK_CONFIGS,
    ids=[f"{cfg[0]}-q{cfg[1]}-k{cfg[2]}-ring{cfg[3]}" for cfg in MINIMAX3_GQA_CHUNKED_PERF_CHECK_CONFIGS],
)
def test_ring_joint_attention_minimax3_gqa_chunked_perf_check(
    model_name, q_chunk_size, k_chunk_size, ring_size_expected, expected_util
):
    """Measure Minimax3 GQA chunked-prefill math utilization for the production-style final chunk.

    RING_JOINT_CHUNKED_CHUNK_ID isolates the final chunk so only its iteration is profiled; the
    inner perf_impl uses reuse_kv mode so kv_actual_isl models the long prefix with a stable cache shape.
    """
    from tracy.process_model_log import run_device_profiler

    if MESH_CONFIG.sp_size != ring_size_expected:
        pytest.skip(f"Expected ring size {ring_size_expected}, current topology has ring size {MESH_CONFIG.sp_size}")

    model = MINIMAX3_GQA_CHUNKED_MODEL_CONFIGS[model_name]
    chunk_size = CHUNKED_PREFILL_CHUNK_SIZE
    n_chunks = CHUNKED_PREFILL_TOTAL_SEQ // chunk_size
    perf_chunk = n_chunks - 1

    config_id = f"{get_test_case_id(model, q_chunk_size, k_chunk_size)}-chunk{chunk_size}-reuse_kv"
    subdir = "ttnn_ring_joint_sdpa_minimax3_gqa_chunked_perf_check"
    command = (
        "pytest tests/nightly/blackhole/sdpa/"
        f"test_ring_joint_sdpa.py::test_ring_joint_attention_minimax3_gqa_chunked_perf_impl[{config_id}]"
    )

    float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
    cols = ["ATTRIBUTES"]

    with mock.patch.dict(os.environ, {"CI": "false", CHUNKED_PREFILL_CHUNK_ID_ENV: str(perf_chunk)}):
        run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log(
        subdir,
        float_columns=float_cols,
        columns=cols,
        op_name="RingJointSDPADeviceOperation",
        sum_vals=False,
        has_signposts=False,
    )

    assert (
        len(r["CORE COUNT"]) > 0 and len(r["DEVICE KERNEL DURATION [ns]"]) > 0
    ), "profiler returned no SDPA ops - inner test was skipped or did not produce a kernel run"

    measured_core_count = int(r["CORE COUNT"][0])
    duration_ns = int(r["DEVICE KERNEL DURATION [ns]"].max())
    utilization, _ = compute_chunked_prefill_perf_check_utilization(
        MESH_CONFIG, model, chunk_size, perf_chunk, duration_ns, measured_core_count
    )

    lower = expected_util * (1 - RING_JOINT_PERF_MARGIN)
    upper = expected_util * (1 + RING_JOINT_PERF_MARGIN)

    logger.info(
        f"Minimax3 GQA chunked final-chunk perf check {config_id}: "
        f"duration={duration_ns/1e6:.3f} ms, math_util={utilization:.2f}% "
        f"(expected {expected_util:.2f}%, band [{lower:.2f}, {upper:.2f}])"
    )

    assert lower <= utilization <= upper, (
        f"Math utilization {utilization:.2f}% outside band [{lower:.2f}, {upper:.2f}] "
        f"(expected {expected_util:.2f}%, margin +/- {RING_JOINT_PERF_MARGIN*100:.1f}%)"
    )


# ============================================================================
# NON-CAUSAL CROSS-ATTENTION VALIDATION (is_cross)
# ============================================================================
CROSS_PCC_THRESHOLD = 0.99
CROSS_RMSE_THRESHOLD = DEFAULT_RMSE_THRESHOLD

# (nhq_total, head_dim, q_global, kv_global, logical_n) — LTX-2 V2A production shapes.
CROSS_SHAPE_CONFIGS = [
    pytest.param(32, 64, 256, 9728, 9690, id="ltx_v2a_stage1_6s"),
    pytest.param(32, 64, 256, 38784, 38760, id="ltx_v2a_stage2_1080p_6s"),
    pytest.param(32, 64, 384, 87808, 87720, id="ltx_v2a_stage2_1080p_14s"),
]
CROSS_MESH_PARAMS = [
    pytest.param(2, 4, Topology.Linear, id="bh_2x4sp1tp0"),
    pytest.param(4, 8, Topology.Ring, id="bh_4x8sp1tp0_ring"),
]


def run_ring_joint_sdpa_cross(
    mesh_config,
    nhq_total,
    head_dim,
    q_global,
    kv_global,
    logical_n,
    q_chunk_sizes,
    k_chunk_size=512,
    tp_size=2,
    sp_size=4,
    topology=Topology.Linear,
    pcc_threshold=CROSS_PCC_THRESHOLD,
    rmse_threshold=CROSS_RMSE_THRESHOLD,
):
    """Validate is_cross ring SDPA (short Q, long K/V) against a non-causal torch oracle over the
    first logical_n keys."""
    if mesh_config.num_devices < tp_size * sp_size:
        pytest.skip(f"cross ring SDPA needs a {tp_size}x{sp_size} mesh ({tp_size * sp_size} chips)")
    assert nhq_total % tp_size == 0, f"nhq_total {nhq_total} must divide TP {tp_size}"
    for nm, g in (("q_global", q_global), ("kv_global", kv_global)):
        if g % (sp_size * 32) != 0:
            pytest.skip(f"{nm}={g} not SP/tile-aligned for SP={sp_size}")

    torch.manual_seed(1234)

    sp_axis = 1
    tp_axis = 0
    num_links = 2
    fabric_config = ttnn.FabricConfig.FABRIC_1D_RING if topology == Topology.Ring else ttnn.FabricConfig.FABRIC_1D

    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )

    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(tp_size, sp_size))
    try:
        full_compute_grid = mesh_device.compute_with_storage_grid_size()
        sdpa_compute_grid = (full_compute_grid.x - 1, full_compute_grid.y)
        ccl_column = full_compute_grid.x - 1

        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
        )
        worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
        mesh_device.set_sub_device_stall_group([worker_sub_device_id])
        ccl_semaphore_handles = create_global_semaphores(mesh_device, ccl_sub_device_crs, 0)

        Q = fa_rand(BATCH_SIZE, nhq_total, q_global, head_dim)
        K = fa_rand(BATCH_SIZE, nhq_total, kv_global, head_dim)
        V = fa_rand(BATCH_SIZE, nhq_total, kv_global, head_dim)

        sdpa_input_shard_dims = [None, None]
        sdpa_input_shard_dims[sp_axis] = 2
        sdpa_input_shard_dims[tp_axis] = 1
        persistent_kv_shard_dims = [None, None]
        persistent_kv_shard_dims[tp_axis] = 1

        def shard_to_device(tensor, dims):
            return ttnn.from_torch(
                tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
            )

        tt_Q = shard_to_device(Q, sdpa_input_shard_dims)
        tt_K = shard_to_device(K, sdpa_input_shard_dims)
        tt_V = shard_to_device(V, sdpa_input_shard_dims)
        persistent_output_buffer_k = shard_to_device(
            torch.zeros(BATCH_SIZE, nhq_total, kv_global, head_dim), persistent_kv_shard_dims
        )
        persistent_output_buffer_v = shard_to_device(
            torch.zeros(BATCH_SIZE, nhq_total, kv_global, head_dim), persistent_kv_shard_dims
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
        gt_main = torch_sdpa_reference(Q, K[:, :, :logical_n, :], V[:, :, :logical_n, :], is_causal=False)

        # Run every q_chunk config against the same mesh, uploads, and reference (one CI item).
        for q_chunk_size in q_chunk_sizes:
            program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=sdpa_compute_grid,
                q_chunk_size=q_chunk_size,
                k_chunk_size=k_chunk_size,
                exp_approx_mode=False,
            )
            tt_out = call_sdpa(
                tt_Q,
                tt_K,
                tt_V,
                logical_n,
                is_causal=False,
                is_balanced=False,
                is_cross=True,
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
            tt_out_torch = ttnn.to_torch(
                tt_out,
                mesh_composer=ttnn.create_mesh_composer(
                    mesh_device, ttnn.MeshComposerConfig(main_row_dim, main_col_dim)
                ),
            )[:, :, :q_global, :]

            out_pass_main, out_pcc_main = comp_pcc(gt_main, tt_out_torch, pcc_threshold)
            rmse_main = torch.sqrt(((gt_main - tt_out_torch) ** 2).mean()).item()
            logger.info(f"Cross output (q_chunk={q_chunk_size}) - PCC: {out_pcc_main}, RMSE: {rmse_main:.6f}")
            assert (
                rmse_main < rmse_threshold
            ), f"Cross RMSE {rmse_main:.6f} >= {rmse_threshold} (q_chunk={q_chunk_size})"
            assert out_pass_main, f"Cross PCC {out_pcc_main} below {pcc_threshold} (q_chunk={q_chunk_size})"

    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# === TEST: NON-CAUSAL CROSS-ATTENTION ACCURACY ===
@pytest.mark.timeout(600)
@pytest.mark.parametrize("tp_size, sp_size, topology", CROSS_MESH_PARAMS)
@pytest.mark.parametrize("nhq_total, head_dim, q_global, kv_global, logical_n", CROSS_SHAPE_CONFIGS)
def test_ring_joint_attention_sdpa_cross_accuracy(
    nhq_total, head_dim, q_global, kv_global, logical_n, tp_size, sp_size, topology
):
    """is_cross=True accuracy at the LTX-2 V2A shapes; must match the non-causal cross reference."""
    run_ring_joint_sdpa_cross(
        MESH_CONFIG,
        nhq_total,
        head_dim,
        q_global,
        kv_global,
        logical_n,
        q_chunk_sizes=[64, 128],
        tp_size=tp_size,
        sp_size=sp_size,
        topology=topology,
    )
