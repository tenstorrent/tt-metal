# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the Attention1D module (1D mesh topology: N150, N300, T3K).

This test suite verifies:
1. Unit tests for config dataclasses (no device needed)
2. Attention1D class matches HuggingFace/Meta reference model
3. Attention1D correctly rejects TG/Galaxy devices
4. Sliding window attention works correctly (seq_len > window_size)

Test coverage notes:
- Paged attention: Tested via (page_block_size, chunk_size) parameter combinations.
- Chunked prefill: Tested via paged-chunked variant. Requires paged=True and mode="prefill".
- Variants: non-paged, paged, paged-chunked (3 combinations per test case).
"""

import os
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig, _resolve_attention1d_config
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1DConfig
from models.common.modules.tt_ccl import TT_CCL
from models.common.tensor_utils import get_rot_transformation_mat, zeros_like_kv_cache, zeros_like_paged_cache
from models.common.tests.utils import stable_model_seed
from models.common.utility_functions import comp_allclose, comp_pcc, nearest_32

# =============================================================================
# RoPE Helper Functions (replaces TTTv1 rope imports)
# =============================================================================


def _permute_to_meta_format(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert HuggingFace RoPE format to Meta format for TTNN compatibility.

    HF stores cos/sin with shape [batch, seq_len, head_dim] where head_dim is interleaved.
    Meta format expects [1, 1, seq_len, head_dim] with doubled values.
    """
    # Handle different HF output shapes
    if len(cos.shape) == 3:
        cos = cos.squeeze(0)  # [seq_len, head_dim]
        sin = sin.squeeze(0)

    # Undo the HF permute: take first half and duplicate
    cos = cos[:, : cos.shape[1] // 2]
    cos = torch.stack((cos, cos), dim=-1).flatten(-2)

    sin = sin[:, : sin.shape[1] // 2]
    sin = torch.stack((sin, sin), dim=-1).flatten(-2)

    # Add batch dimensions: [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return cos, sin


def get_cos_sin_from_hf(
    rotary_emb,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract cos/sin rotation matrices from HuggingFace rotary_emb module.

    Args:
        rotary_emb: HuggingFace rotary embedding module (e.g., LlamaRotaryEmbedding)
        seq_len: Maximum sequence length
        head_dim: Head dimension
        dtype: Output dtype

    Returns:
        (cos, sin) tensors in Meta format with shape [1, 1, seq_len, head_dim]
    """
    # Create dummy input for HF's rotary_emb forward
    x = torch.zeros(1, 1, seq_len, head_dim, dtype=dtype)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    # HF rotary_emb.forward() returns (cos, sin)
    with torch.no_grad():
        cos_hf, sin_hf = rotary_emb(x, position_ids)

    # Convert to Meta format
    cos_meta, sin_meta = _permute_to_meta_format(cos_hf.float(), sin_hf.float())

    return cos_meta.to(dtype), sin_meta.to(dtype)


def get_rot_mats_from_hf(
    rotary_emb,
    seq_len: int,
    head_dim: int,
    device: ttnn.MeshDevice,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> list[ttnn.Tensor]:
    """
    Create TTNN rotation matrices from HuggingFace rotary_emb.

    Replaces `get_rot_mats` from models.tt_transformers.tt.rope.
    """
    cos_meta, sin_meta = get_cos_sin_from_hf(rotary_emb, seq_len * 2, head_dim)

    cos_tt = ttnn.from_torch(
        cos_meta,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    sin_tt = ttnn.from_torch(
        sin_meta,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    return [cos_tt, sin_tt]


class RotarySetupHelper:
    """
    Simplified RotarySetup for testing - extracts rotation matrices from HuggingFace's
    rotary_emb instead of computing from scratch. Replaces TTTv1's RotarySetup class.
    """

    def __init__(
        self,
        device: ttnn.MeshDevice,
        batch_size: int,
        head_dim: int,
        max_seq_len: int,
        rotary_emb,  # HuggingFace rotary embedding module
        use_qk_fused: bool = False,
        datatype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.head_dim = head_dim
        self.use_qk_fused = use_qk_fused
        self.batch_size = batch_size
        self.doubled_batch_size = batch_size * 2 if use_qk_fused else batch_size

        is_mesh = isinstance(device, ttnn.MeshDevice)
        num_devices = device.get_num_devices() if is_mesh else 1

        if num_devices == 32:
            self.batch_size_per_device_group = max(self.doubled_batch_size // device.shape[1], 1)
        else:
            self.batch_size_per_device_group = self.doubled_batch_size

        self.core_grid = device.compute_with_storage_grid_size()
        self.batch_grid = ttnn.num_cores_to_corerangeset(self.doubled_batch_size, self.core_grid, row_wise=True)

        # Get cos/sin from HuggingFace rotary_emb
        self.cos_matrix, self.sin_matrix = get_rot_mats_from_hf(rotary_emb, max_seq_len, head_dim, device, datatype)

        # Create transformation matrices
        trans_mat = get_rot_transformation_mat().repeat(1, 1, self.doubled_batch_size, 1)
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.transformation_mat = ttnn.from_torch(
            trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=trans_mat_mem_config,
            mesh_mapper=(
                ttnn.ShardTensor2dMesh(
                    device,
                    dims=(None, 2) if (num_devices == 32 and batch_size > 1) else (None, None),
                    mesh_shape=list(device.shape),
                )
                if is_mesh
                else None
            ),
        )

        # Prefill transformation matrix
        prefill_trans_mat = get_rot_transformation_mat()
        if head_dim != ttnn.TILE_SIZE:
            prefill_trans_mat = torch.zeros(1, 1, head_dim, head_dim)
            base_mat = get_rot_transformation_mat()
            prefill_trans_mat[:, :, : ttnn.TILE_SIZE, : ttnn.TILE_SIZE] = base_mat

        self.transformation_mat_prefill = ttnn.from_torch(
            prefill_trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh else None,
        )

    def get_both_trans_mats(self) -> dict[str, ttnn.Tensor]:
        """Return transformation matrices for decode and prefill."""
        return {"decode": self.transformation_mat, "prefill": self.transformation_mat_prefill}

    def get_rot_idxs(self, position_idxs: torch.Tensor, on_host: bool = False) -> ttnn.Tensor:
        """Convert position indices to TTNN tensor."""

        if self.use_qk_fused:
            position_idxs = position_idxs.repeat(2)

        batch = position_idxs.shape[0]
        position_idxs = position_idxs.reshape(1, batch)

        # Pad to tile boundary
        pad_size = nearest_32(batch) - batch
        position_idxs = torch.nn.functional.pad(position_idxs, (0, pad_size), "constant", 0)

        rot_idxs = ttnn.as_tensor(
            position_idxs,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None if on_host else self.device,
            memory_config=None if on_host else ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        return rot_idxs

    def get_rot_mats(self, position_idxs: torch.Tensor) -> list[ttnn.Tensor]:
        """Get rotation matrices for given position indices."""
        rot_idxs = self.get_rot_idxs(position_idxs)

        if rot_idxs.device != self.device:
            rot_idxs = ttnn.to_device(rot_idxs, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)

        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)

        cos = ttnn.transpose(cos, 1, 2)
        sin = ttnn.transpose(sin, 1, 2)

        if self.batch_size_per_device_group % ttnn.TILE_SIZE != 0:
            cos = cos[:, : self.batch_size_per_device_group, :, :]
            sin = sin[:, : self.batch_size_per_device_group, :, :]

        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        cos = ttnn.interleaved_to_sharded(cos, mem_config)
        sin = ttnn.interleaved_to_sharded(sin, mem_config)

        return [cos, sin]


# =============================================================================
# HfAttentionWrapper (replaces TTTv1 model_config import)
# =============================================================================


class HfAttentionWrapper:
    """
    Wrapper for HuggingFace attention modules with KV cache support.
    Provides a consistent interface for running HF attention as reference.
    """

    def __init__(self, attention, head_dim: int, rotary_emb):
        from transformers import DynamicCache

        self.attention = attention
        self.past_key_value = DynamicCache()
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb

    def forward(self, x: torch.Tensor, start_pos: int, mask=None):
        """Run attention forward pass using rotary_emb directly."""
        position_ids = torch.tensor([list(range(start_pos, start_pos + x.shape[1]))] * x.shape[0])

        if mask is not None:
            while len(mask.shape) < 4:
                mask = mask.unsqueeze(0)

        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(x, position_ids)
            output, *_ = self.attention(
                x,
                position_embeddings=position_embeddings,
                past_key_value=self.past_key_value,
                use_cache=True,
                attention_mask=mask,
            )
        else:
            output, _, self.past_key_value = self.attention(
                x,
                past_key_value=self.past_key_value,
                use_cache=True,
                position_ids=position_ids,
                attention_mask=mask,
            )
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def reset_cache(self):
        """Reset KV cache for new sequence."""
        from transformers import DynamicCache

        self.past_key_value = DynamicCache()

    @property
    def cache_k(self) -> torch.Tensor:
        """Get key cache in shape [batch, seq_len, n_kv_heads, head_dim]."""
        if len(self.past_key_value.key_cache) == 0:
            return torch.zeros(0)
        # DynamicCache stores as [batch, n_heads, seq_len, head_dim]
        # Transpose to [batch, seq_len, n_heads, head_dim]
        return self.past_key_value.key_cache[0].transpose(1, 2)

    @property
    def cache_v(self) -> torch.Tensor:
        """Get value cache in shape [batch, seq_len, n_kv_heads, head_dim]."""
        if len(self.past_key_value.value_cache) == 0:
            return torch.zeros(0)
        # DynamicCache stores as [batch, n_heads, seq_len, head_dim]
        # Transpose to [batch, seq_len, n_heads, head_dim]
        return self.past_key_value.value_cache[0].transpose(1, 2)


# =============================================================================
# PagedAttentionConfig (replaces TTTv1 common import)
# =============================================================================


@dataclass
class PagedAttentionConfig:
    """Configuration for paged attention."""

    block_size: int = 64
    max_num_blocks: int = 2048


# =============================================================================
# Weight extraction helpers
# =============================================================================


def _reverse_permute(tensor, n_heads, dim1, dim2):
    """Convert HuggingFace Q/K weights to Meta format for RoPE compatibility.

    HuggingFace stores Q/K weights in a format optimized for their attention implementation,
    while Meta format is required for TTNN's RoPE implementation.
    """
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def _reverse_permute_1d(tensor):
    """Convert the last dim from separate real/imaginary (r1,r2,i1,i2,...) to interleaved (r1,i1,r2,i2,...)"""
    shape = tensor.shape
    dim = shape[-1]
    assert dim % 2 == 0, "Last dimension must be even"
    reals = tensor[..., : dim // 2]
    imags = tensor[..., dim // 2 :]
    interleaved = torch.stack((reals, imags), dim=-1).flatten(start_dim=len(shape) - 1)
    return interleaved


def get_attention_weights_from_ref_model(
    reference_attn, num_devices: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    Extract attention weights from a reference attention module in TTNN layout.

    Applies reverse_permute to Q and K weights to convert from HuggingFace format
    to Meta format, which is required for TTNN's RoPE implementation.

    Returns:
        (wqkv, wo, q_norm, k_norm, wqkv_bias) tensors in TTNN layout
    """
    # Get raw weights from HF module
    wq_raw = reference_attn.q_proj.weight  # (n_heads * head_dim, dim)
    wk_raw = reference_attn.k_proj.weight  # (n_kv_heads * head_dim, dim)
    wv_raw = reference_attn.v_proj.weight  # (n_kv_heads * head_dim, dim)
    wo_raw = reference_attn.o_proj.weight  # (dim, n_heads * head_dim)

    # Compute head_dim from weight shapes
    dim = wq_raw.shape[1]
    n_heads_times_head_dim = wq_raw.shape[0]
    n_kv_heads_times_head_dim = wk_raw.shape[0]

    # For head_dim calculation, we need n_heads. Use the ratio of Q/K sizes.
    # Q: (n_heads * head_dim, dim), K: (n_kv_heads * head_dim, dim)
    # If n_heads == n_kv_heads (no GQA), just use q shape
    # Otherwise, we need to infer from config or assume head_dim from common values
    if hasattr(reference_attn, "head_dim"):
        head_dim = reference_attn.head_dim
    elif hasattr(reference_attn, "config") and hasattr(reference_attn.config, "head_dim"):
        head_dim = reference_attn.config.head_dim
    else:
        # Common head_dim values for LLaMA models
        head_dim = 128 if n_heads_times_head_dim >= 4096 else 64

    n_heads = n_heads_times_head_dim // head_dim
    n_kv_heads = n_kv_heads_times_head_dim // head_dim

    # Apply reverse_permute to convert HF format to Meta format for RoPE compatibility
    # This transformation is critical for Q and K weights
    wq_meta = _reverse_permute(wq_raw, n_heads, n_heads_times_head_dim, dim)
    wk_meta = _reverse_permute(wk_raw, n_kv_heads, n_kv_heads_times_head_dim, dim)
    # V and O don't need permutation
    wv_meta = wv_raw
    wo_meta = wo_raw

    # Transpose to TTNN layout: (dim, out_features)
    wq = wq_meta.T  # (dim, n_heads * head_dim)
    wk = wk_meta.T  # (dim, n_kv_heads * head_dim)
    wv = wv_meta.T  # (dim, n_kv_heads * head_dim)
    wo = wo_meta.T  # (n_heads * head_dim, dim)

    # Build combined QKV weight
    # Shape: (1, 1, dim, qkv_size_per_device * num_devices)
    qkv_list = []
    for i in range(num_devices):
        wq_chunk = torch.chunk(wq, num_devices, dim=1)[i]
        wk_chunk = torch.chunk(wk, num_devices, dim=1)[i]
        wv_chunk = torch.chunk(wv, num_devices, dim=1)[i]
        qkv = torch.cat([wq_chunk, wk_chunk, wv_chunk], dim=-1)
        qkv_list.append(qkv)

    wqkv = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

    # WO weight: (1, 1, n_heads * head_dim, dim)
    wo = wo.unsqueeze(0).unsqueeze(0)

    # Q/K norm weights (optional, e.g., for Qwen models)
    # These also need reverse_permute_1d transformation
    q_norm = None
    k_norm = None
    if hasattr(reference_attn, "q_norm") and reference_attn.q_norm is not None:
        q_norm = _reverse_permute_1d(reference_attn.q_norm.weight)
    if hasattr(reference_attn, "k_norm") and reference_attn.k_norm is not None:
        k_norm = _reverse_permute_1d(reference_attn.k_norm.weight)

    # QKV bias (optional, e.g., for Qwen2/Qwen2.5 models)
    # Bias also needs the same chunking/concat pattern as weights
    wqkv_bias = None
    if hasattr(reference_attn.q_proj, "bias") and reference_attn.q_proj.bias is not None:
        bq_raw = reference_attn.q_proj.bias  # (n_heads * head_dim,)
        bk_raw = reference_attn.k_proj.bias  # (n_kv_heads * head_dim,)
        bv_raw = reference_attn.v_proj.bias  # (n_kv_heads * head_dim,)

        # Apply reverse_permute to Q and K biases (same as weights)
        bq_meta = _reverse_permute_1d(bq_raw.view(n_heads, head_dim)).view(-1)
        bk_meta = _reverse_permute_1d(bk_raw.view(n_kv_heads, head_dim)).view(-1)
        bv_meta = bv_raw  # V doesn't need permutation

        # Build combined QKV bias with chunking for multi-device
        qkv_bias_list = []
        for i in range(num_devices):
            bq_chunk = torch.chunk(bq_meta, num_devices, dim=0)[i]
            bk_chunk = torch.chunk(bk_meta, num_devices, dim=0)[i]
            bv_chunk = torch.chunk(bv_meta, num_devices, dim=0)[i]
            qkv_bias = torch.cat([bq_chunk, bk_chunk, bv_chunk], dim=-1)
            qkv_bias_list.append(qkv_bias)

        wqkv_bias = torch.cat(qkv_bias_list, dim=-1)

    return wqkv, wo, q_norm, k_norm, wqkv_bias


# ============================================================================
# Weight Caching - Avoid expensive torch.randn_like() per test
# ============================================================================

_CACHED_ATTN_WEIGHTS: dict[str, dict[str, torch.Tensor]] = {}


def _init_weight_scaled_normal(param: torch.Tensor, name: str) -> torch.Tensor:
    """Initialize a weight tensor using scaled normal distribution.

    Uses a scaling factor based on typical pretrained LLM weight statistics
    (std ~0.02) rather than pure randn (std=1.0) which can cause
    extreme activations and numerical issues in softmax.

    For 2D weights (linear layers): Uses normal with std = 0.02
    For 1D weights (biases, norms): Uses appropriate initialization

    NOTE: Uses torch.randn() instead of torch.randn_like() to ensure
    the global RNG state (set via torch.manual_seed) is respected.
    torch.randn_like() may not use the global RNG consistently.
    """
    if param.dim() >= 2:
        # Linear layer weights: scaled normal initialization
        # std=0.02 matches typical pretrained transformer weights
        return torch.randn(param.shape, dtype=param.dtype, device=param.device) * 0.02
    elif "norm" in name.lower() or "weight" in name.lower():
        # Norm weights (e.g., q_norm.weight, k_norm.weight): use ones
        return torch.ones_like(param)
    else:
        # Biases: use small random values
        return torch.randn(param.shape, dtype=param.dtype, device=param.device) * 0.01


def _get_or_init_attn_weights(model_name: str, reference_attn) -> None:
    """Initialize attention weights once per model, cache and reuse across tests.

    Uses scaled normal initialization (std=0.02) for better numerical conditioning
    compared to pure random noise (std=1.0). This helps maintain reasonable
    activation magnitudes through the attention computation.

    NOTE: Uses a deterministic seed per model to ensure reproducible weights
    regardless of test execution order or Python process hash randomization.
    This prevents flaky tests where PCC varies based on which tests ran first.
    """
    if model_name not in _CACHED_ATTN_WEIGHTS:
        logger.info(f"\033[33m[cache miss]\033[0m Initializing weights for {model_name}")
        _CACHED_ATTN_WEIGHTS[model_name] = {}
        # Use deterministic seed based on model name to ensure reproducible weights
        # regardless of test execution order
        seed = stable_model_seed(model_name)
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        with torch.no_grad():
            for name, param in reference_attn.named_parameters():
                _CACHED_ATTN_WEIGHTS[model_name][name] = _init_weight_scaled_normal(param, name)
        torch.set_rng_state(rng_state)  # Restore original RNG state
    else:
        logger.info(f"\033[32m[cache hit]\033[0m Reusing cached weights for {model_name}")

    # Load cached weights into model
    with torch.no_grad():
        for name, param in reference_attn.named_parameters():
            param.copy_(_CACHED_ATTN_WEIGHTS[model_name][name])


# ============================================================================
# Unit Tests - No device required
# ============================================================================


def test_attention_1d_config_creation():
    """Test that Attention1DConfig dataclass can be created with explicit values."""
    mock_mesh_device = MagicMock()
    mock_tt_ccl = MagicMock()
    mock_wqkv = MagicMock()
    mock_wo = MagicMock()

    # Create config with explicit values
    config = Attention1DConfig(
        wqkv=mock_wqkv,
        wo=mock_wo,
        mesh_device=mock_mesh_device,
        tt_ccl=mock_tt_ccl,
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        max_batch_size=64,
        topology=ttnn.Topology.Ring,
    )

    # Verify explicit values are preserved
    assert config.wqkv is mock_wqkv
    assert config.wo is mock_wo
    assert config.mesh_device is mock_mesh_device
    assert config.tt_ccl is mock_tt_ccl
    assert config.dim == 4096
    assert config.n_heads == 32
    assert config.n_kv_heads == 8
    assert config.head_dim == 128
    assert config.max_batch_size == 64
    assert config.topology == ttnn.Topology.Ring


def test_attention_1d_config_defaults():
    """Test that Attention1DConfig has sensible defaults."""
    # Minimal creation - only required fields
    config = Attention1DConfig(wqkv=MagicMock(), wo=MagicMock())

    # Check defaults
    assert config.max_batch_size == 32
    assert config.max_seq_len == 128 * 1024
    assert config.use_vllm_paged_kv_cache is False
    assert config.kv_cache_dtype == ttnn.bfloat8_b
    assert config.use_qk_fused is False
    assert config.num_reduce_scatter_links is None
    assert config.num_all_gather_links is None

    # Optional fields default to None
    assert config.mesh_device is None
    assert config.tt_ccl is None
    assert config.dim is None
    assert config.n_heads is None
    assert config.n_kv_heads is None
    assert config.q_norm_config is None
    assert config.k_norm_config is None


def test_attention_1d_config_power_user_overrides():
    """Test that Attention1DConfig accepts power-user overrides for program configs."""
    mock_prg_config = MagicMock()
    mock_mem_config = MagicMock()
    mock_compute_kernel = MagicMock()

    config = Attention1DConfig(
        wqkv=MagicMock(),
        wo=MagicMock(),
        decode_xqkv_prg_config=mock_prg_config,
        decode_sdpa_prg_config=mock_prg_config,
        decode_attn_output_prg_config=mock_prg_config,
        decode_residual_memcfg=mock_mem_config,
        li_qkv_decode_compute_kernel_cfg=mock_compute_kernel,
        activation_dtype=ttnn.bfloat16,
        scale=0.08838834764831843,  # 1/sqrt(128)
    )

    # User-provided overrides should be preserved
    assert config.decode_xqkv_prg_config is mock_prg_config
    assert config.decode_sdpa_prg_config is mock_prg_config
    assert config.decode_attn_output_prg_config is mock_prg_config
    assert config.decode_residual_memcfg is mock_mem_config
    assert config.li_qkv_decode_compute_kernel_cfg is mock_compute_kernel
    assert config.activation_dtype == ttnn.bfloat16
    assert config.scale == pytest.approx(0.08838834764831843)


def test_attention_1d_happy_path_signature():
    """Test that Attention1D.__init__ accepts the happy path signature (weights + dimensions).

    Note: This is a unit test that verifies the API signature without a device.
    Full integration testing of Attention1D creation happens in device tests.
    """
    import inspect

    # Verify the __init__ signature has the expected parameters
    sig = inspect.signature(Attention1D.__init__)
    params = list(sig.parameters.keys())

    # Expected: self, wqkv, wo, n_heads, n_kv_heads, head_dim
    assert "wqkv" in params, "wqkv should be a parameter"
    assert "wo" in params, "wo should be a parameter"
    assert "n_heads" in params, "n_heads should be a required parameter"
    assert "n_kv_heads" in params, "n_kv_heads should be a required parameter"
    assert "head_dim" in params, "head_dim should be a required parameter"

    # Verify n_heads, n_kv_heads, head_dim have no defaults (are required)
    for param_name in ["n_heads", "n_kv_heads", "head_dim"]:
        param = sig.parameters[param_name]
        assert param.default is inspect.Parameter.empty, f"{param_name} should be required (no default)"


def test_attention_1d_resolve_requires_n_heads():
    """Test that _resolve_attention1d_config raises ValueError when n_heads is missing."""
    mock_source = MagicMock()
    mock_source.shape = (4096, 1536)

    mock_wqkv = MagicMock(spec=LazyWeight)
    mock_wqkv.source = mock_source
    mock_wqkv.device = None

    config = Attention1DConfig(
        wqkv=mock_wqkv,
        wo=MagicMock(spec=LazyWeight),
        n_heads=None,  # Missing!
        n_kv_heads=8,
        head_dim=128,
    )

    with pytest.raises(ValueError, match="n_heads must be provided"):
        _resolve_attention1d_config(config)


def test_attention_1d_resolve_requires_n_kv_heads():
    """Test that _resolve_attention1d_config raises ValueError when n_kv_heads is missing."""
    mock_source = MagicMock()
    mock_source.shape = (4096, 1536)

    mock_wqkv = MagicMock(spec=LazyWeight)
    mock_wqkv.source = mock_source
    mock_wqkv.device = None

    config = Attention1DConfig(
        wqkv=mock_wqkv,
        wo=MagicMock(spec=LazyWeight),
        n_heads=32,
        n_kv_heads=None,  # Missing!
        head_dim=128,
    )

    with pytest.raises(ValueError, match="n_kv_heads must be provided"):
        _resolve_attention1d_config(config)


def test_attention_1d_resolve_requires_head_dim():
    """Test that _resolve_attention1d_config raises ValueError when head_dim is missing."""
    mock_source = MagicMock()
    mock_source.shape = (4096, 1536)

    mock_wqkv = MagicMock(spec=LazyWeight)
    mock_wqkv.source = mock_source
    mock_wqkv.device = None

    config = Attention1DConfig(
        wqkv=mock_wqkv,
        wo=MagicMock(spec=LazyWeight),
        n_heads=32,
        n_kv_heads=8,
        head_dim=None,  # Missing!
    )

    with pytest.raises(ValueError, match="head_dim must be provided"):
        _resolve_attention1d_config(config)


def test_attention_1d_resolve_validates_token_budget():
    """Test that _resolve_attention1d_config raises ValueError when token budget is exceeded."""
    mock_source = MagicMock()
    mock_source.shape = (4096, 1536)

    mock_wqkv = MagicMock(spec=LazyWeight)
    mock_wqkv.source = mock_source
    mock_wqkv.device = None

    # 32 batch × 8192 seq = 262,144 tokens > 128K limit
    config = Attention1DConfig(
        wqkv=mock_wqkv,
        wo=MagicMock(spec=LazyWeight),
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        max_batch_size=32,
        max_seq_len=8192,  # 32 × 8192 = 262K > 128K
    )

    with pytest.raises(ValueError, match="Total token budget exceeded"):
        _resolve_attention1d_config(config)


def test_attention_1d_resolve_validates_token_budget_edge_case():
    """Test that exactly 128K tokens is allowed but 128K+1 is not."""
    mock_source = MagicMock()
    mock_source.shape = (4096, 1536)

    mock_wqkv = MagicMock(spec=LazyWeight)
    mock_wqkv.source = mock_source
    mock_wqkv.device = None

    # Exactly 128K tokens should be allowed (boundary case)
    # 32 batch × 4096 seq = 131,072 tokens = 128K (should pass token budget check)
    config_ok = Attention1DConfig(
        wqkv=mock_wqkv,
        wo=MagicMock(spec=LazyWeight),
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        max_batch_size=32,
        max_seq_len=4096,  # 32 × 4096 = 131,072 = 128K exactly
    )
    # Should not raise token budget error (may fail later due to missing device)
    try:
        _resolve_attention1d_config(config_ok)
    except (ValueError, AssertionError) as e:
        # Token budget errors should NOT occur - only device-related errors are acceptable
        assert "Total token budget exceeded" not in str(e), f"Unexpected token budget error: {e}"

    # 128K + 1 token should fail with token budget error
    config_fail = Attention1DConfig(
        wqkv=mock_wqkv,
        wo=MagicMock(spec=LazyWeight),
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        max_batch_size=1,
        max_seq_len=128 * 1024 + 1,  # 128K + 1 tokens
    )

    with pytest.raises(ValueError, match="Total token budget exceeded"):
        _resolve_attention1d_config(config_fail)


def test_attention_1d_resolve_kv_cache_tensor_passthrough():
    """Test that _resolve_attention1d_config passes through raw ttnn.Tensor KV cache entries."""
    mock_source = MagicMock()
    mock_source.shape = (4096, 1536)

    mock_wqkv = MagicMock(spec=LazyWeight)
    mock_wqkv.source = mock_source
    mock_wqkv.device = None

    # Simulate pre-allocated ttnn.Tensor KV cache (e.g., from vLLM).
    # Use plain MagicMock (NOT spec=LazyWeight) so isinstance(_, LazyWeight) is False.
    mock_cache_k = MagicMock()
    mock_cache_v = MagicMock()

    config = Attention1DConfig(
        wqkv=mock_wqkv,
        wo=MagicMock(spec=LazyWeight),
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        max_batch_size=1,
        max_seq_len=128,
        kv_cache=(mock_cache_k, mock_cache_v),
    )

    # Should not crash — previously would fail with dataclasses.replace on non-dataclass
    try:
        resolved = _resolve_attention1d_config(config)
    except (ValueError, AssertionError) as e:
        # Only device-related errors are acceptable, not kv_cache errors
        assert "kv_cache" not in str(e).lower(), f"Unexpected kv_cache error: {e}"
        return

    # If resolution succeeded fully, verify the tensors were passed through as-is
    assert resolved.kv_cache[0] is mock_cache_k
    assert resolved.kv_cache[1] is mock_cache_v


def test_attention_1d_resolve_rejects_sliding_window_with_paged():
    """Test that _resolve_attention1d_config rejects sliding_window + paged_attention_config."""
    mock_source = MagicMock()
    mock_source.shape = (4096, 1536)

    mock_wqkv = MagicMock(spec=LazyWeight)
    mock_wqkv.source = mock_source
    mock_wqkv.device = None

    config = Attention1DConfig(
        wqkv=mock_wqkv,
        wo=MagicMock(spec=LazyWeight),
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        max_batch_size=1,
        max_seq_len=128,
        sliding_window=4096,
        paged_attention_config=PagedAttentionConfig(block_size=64, max_num_blocks=2048),
    )

    with pytest.raises(ValueError, match="sliding_window"):
        _resolve_attention1d_config(config)


# ============================================================================
# Model name constants
# ============================================================================

LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"
LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_11B = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LLAMA_90B = "meta-llama/Llama-3.2-90B-Vision-Instruct"
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-v0.1"
QWEN2_7B = "Qwen/Qwen2-7B-Instruct"
QWEN25_7B = "Qwen/Qwen2.5-7B-Instruct"
QWEN25_72B = "Qwen/Qwen2.5-72B-Instruct"
QWEN25_CODER_32B = "Qwen/Qwen2.5-Coder-32B-Instruct"
DEEPSEEK_R1_14B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
QWEN3_32B = "Qwen/Qwen3-32B"

_slow = pytest.mark.slow


# ============================================================================
# Test cases from attn_1d_performance.csv - hardcoded as pytest parameters
# ============================================================================


def _list_test_cases() -> list[pytest.param]:
    """
    Hardcoded test cases from attn_1d_performance.csv.

    Parameters: mesh_shape, seq_len, batch_size, mode, x_dtype, wqkv_dtype, hf_model_name, pcc

    batch_size semantics:
    - Prefill: batch_size=1 (single user prefill), input shape (1, 1, seq_len, dim)
    - Decode: batch_size=32 (continuous batching), input shape (1, 1, batch_size, dim)
    """
    # fmt: off
    return [
        # === Fast tests (minimal coverage set) ===
        # Single device (1x1)
        pytest.param((1, 1), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-128-1B"),
        pytest.param((1, 1), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-decode-32-1B"),
        pytest.param((1, 1), 8192, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-8192-1B"),
        # Dual device (1x2)
        pytest.param((1, 2), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-128-8B"),
        pytest.param((1, 2), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-decode-32-8B"),
        pytest.param((1, 2), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x2-decode-32-11B"),
        pytest.param((1, 2), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat16, QWEN25_7B, 0.95, id="1x2-decode-32-Qwen2.5-7B"),
        # Multi-device (1x8)
        pytest.param((1, 8), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-128-8B"),
        pytest.param((1, 8), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-decode-32-8B"),

        # === Slow tests (full coverage from models sweep) ===
        # --- Llama-3.2-1B on N150 (1x1) ---
        pytest.param((1, 1), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-1024-1B", marks=_slow),
        pytest.param((1, 1), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-2048-1B", marks=_slow),
        pytest.param((1, 1), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-4096-1B", marks=_slow),
        pytest.param((1, 1), 16384, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-16384-1B", marks=_slow),
        pytest.param((1, 1), 32768, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-32768-1B", marks=_slow),

        # --- Llama-3.2-3B on N150 (1x1) ---
        pytest.param((1, 1), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x1-prefill-128-3B", marks=_slow),
        pytest.param((1, 1), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x1-prefill-1024-3B", marks=_slow),
        pytest.param((1, 1), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x1-prefill-2048-3B", marks=_slow),
        pytest.param((1, 1), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x1-prefill-4096-3B", marks=_slow),
        pytest.param((1, 1), 8192, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x1-prefill-8192-3B", marks=_slow),
        pytest.param((1, 1), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x1-decode-32-3B", marks=_slow),

        # --- Llama-3.1-8B on N150 (1x1) ---
        pytest.param((1, 1), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x1-prefill-128-8B", marks=_slow),
        pytest.param((1, 1), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x1-prefill-1024-8B", marks=_slow),
        pytest.param((1, 1), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x1-prefill-2048-8B", marks=_slow),
        pytest.param((1, 1), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x1-prefill-4096-8B", marks=_slow),
        pytest.param((1, 1), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x1-decode-32-8B", marks=_slow),

        # --- Mistral-7B on N150 (1x1) ---
        pytest.param((1, 1), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x1-prefill-128-Mistral-7B", marks=_slow),
        pytest.param((1, 1), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x1-prefill-1024-Mistral-7B", marks=_slow),
        pytest.param((1, 1), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x1-prefill-2048-Mistral-7B", marks=_slow),
        pytest.param((1, 1), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x1-prefill-4096-Mistral-7B", marks=_slow),
        pytest.param((1, 1), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x1-decode-32-Mistral-7B", marks=_slow),

        # --- Llama-3.2-1B on N300 (1x2) ---
        pytest.param((1, 2), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-prefill-128-1B", marks=_slow),
        pytest.param((1, 2), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-prefill-1024-1B", marks=_slow),
        pytest.param((1, 2), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-prefill-2048-1B", marks=_slow),
        pytest.param((1, 2), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-prefill-4096-1B", marks=_slow),
        pytest.param((1, 2), 8192, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-prefill-8192-1B", marks=_slow),
        pytest.param((1, 2), 16384, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-prefill-16384-1B", marks=_slow),
        pytest.param((1, 2), 32768, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-prefill-32768-1B", marks=_slow),
        pytest.param((1, 2), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-decode-32-1B", marks=_slow),

        # --- Llama-3.2-3B on N300 (1x2) ---
        pytest.param((1, 2), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-prefill-128-3B", marks=_slow),
        pytest.param((1, 2), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-prefill-1024-3B", marks=_slow),
        pytest.param((1, 2), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-prefill-2048-3B", marks=_slow),
        pytest.param((1, 2), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-prefill-4096-3B", marks=_slow),
        pytest.param((1, 2), 8192, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-prefill-8192-3B", marks=_slow),
        pytest.param((1, 2), 16384, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-prefill-16384-3B", marks=_slow),
        pytest.param((1, 2), 32768, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-prefill-32768-3B", marks=_slow),
        pytest.param((1, 2), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-decode-32-3B", marks=_slow),

        # --- Llama-3.1-8B on N300 (1x2) ---
        pytest.param((1, 2), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-128-8B", marks=_slow),
        pytest.param((1, 2), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-1024-8B", marks=_slow),
        pytest.param((1, 2), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-2048-8B", marks=_slow),
        pytest.param((1, 2), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-4096-8B", marks=_slow),
        pytest.param((1, 2), 8192, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-8192-8B", marks=_slow),
        pytest.param((1, 2), 16384, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-16384-8B", marks=_slow),
        pytest.param((1, 2), 32768, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-32768-8B", marks=_slow),
        pytest.param((1, 2), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-decode-32-8B", marks=_slow),

        # --- Llama-3.2-11B on N300 (1x2) ---
        pytest.param((1, 2), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x2-prefill-128-11B", marks=_slow),
        # NOTE: 11B 1024+ prefill has lower PCC (0.9845) due to vision model complexity
        pytest.param((1, 2), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.98, id="1x2-prefill-1024-11B", marks=_slow),
        pytest.param((1, 2), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.98, id="1x2-prefill-2048-11B", marks=_slow),
        pytest.param((1, 2), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.98, id="1x2-prefill-4096-11B", marks=_slow),
        pytest.param((1, 2), 8192, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.98, id="1x2-prefill-8192-11B", marks=_slow),
        pytest.param((1, 2), 16384, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.98, id="1x2-prefill-16384-11B", marks=_slow),
        pytest.param((1, 2), 32768, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.98, id="1x2-prefill-32768-11B", marks=_slow),

        # --- Mistral-7B on N300 (1x2) ---
        pytest.param((1, 2), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x2-prefill-128-Mistral-7B", marks=_slow),
        pytest.param((1, 2), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x2-prefill-1024-Mistral-7B", marks=_slow),
        pytest.param((1, 2), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x2-prefill-2048-Mistral-7B", marks=_slow),
        pytest.param((1, 2), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x2-prefill-4096-Mistral-7B", marks=_slow),
        pytest.param((1, 2), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x2-decode-32-Mistral-7B", marks=_slow),

        # --- Qwen2-7B on N300 (1x2) ---
        # NOTE: Qwen2-7B has Q/K biases causing numerical precision issues
        pytest.param((1, 2), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN2_7B, 0.98, id="1x2-prefill-128-Qwen2-7B", marks=_slow),
        pytest.param((1, 2), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN2_7B, 0.97, id="1x2-prefill-1024-Qwen2-7B", marks=_slow),
        pytest.param((1, 2), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN2_7B, 0.97, id="1x2-prefill-2048-Qwen2-7B", marks=_slow),
        pytest.param((1, 2), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN2_7B, 0.97, id="1x2-prefill-4096-Qwen2-7B", marks=_slow),
        pytest.param((1, 2), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, QWEN2_7B, 0.99, id="1x2-decode-32-Qwen2-7B", marks=_slow),

        # --- Qwen2.5-7B on N300 (1x2) ---
        # NOTE: Qwen2.5-7B has large Q/K biases causing numerical precision issues
        pytest.param((1, 2), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat16, QWEN25_7B, 0.98, id="1x2-prefill-128-Qwen2.5-7B", marks=_slow),
        pytest.param((1, 2), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat16, QWEN25_7B, 0.97, id="1x2-prefill-1024-Qwen2.5-7B", marks=_slow),
        pytest.param((1, 2), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat16, QWEN25_7B, 0.97, id="1x2-prefill-2048-Qwen2.5-7B", marks=_slow),
        pytest.param((1, 2), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat16, QWEN25_7B, 0.97, id="1x2-prefill-4096-Qwen2.5-7B", marks=_slow),
        pytest.param((1, 2), 8192, 1, "prefill", ttnn.bfloat16, ttnn.bfloat16, QWEN25_7B, 0.97, id="1x2-prefill-8192-Qwen2.5-7B", marks=_slow),
        pytest.param((1, 2), 16384, 1, "prefill", ttnn.bfloat16, ttnn.bfloat16, QWEN25_7B, 0.97, id="1x2-prefill-16384-Qwen2.5-7B", marks=_slow),
        pytest.param((1, 2), 32768, 1, "prefill", ttnn.bfloat16, ttnn.bfloat16, QWEN25_7B, 0.97, id="1x2-prefill-32768-Qwen2.5-7B", marks=_slow),
        # NOTE: Qwen2.5-7B has lower PCC for prefill+decode due to Q/K biases + RoPE interaction.
        # TTTv1's test_attention.py also shows ~0.984 min PCC. With 128-token prefill, accumulated
        # numerical error in SDPA over the larger KV cache causes further degradation.
        # See models/common/tests/modules/attention/low_pcc_notes.md for detailed analysis

        # --- DeepSeek-R1-14B on N300 (1x2) ---
        pytest.param((1, 2), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, DEEPSEEK_R1_14B, 0.99, id="1x2-prefill-128-DeepSeek-R1-14B", marks=_slow),
        pytest.param((1, 2), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, DEEPSEEK_R1_14B, 0.99, id="1x2-prefill-1024-DeepSeek-R1-14B", marks=_slow),
        pytest.param((1, 2), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, DEEPSEEK_R1_14B, 0.99, id="1x2-prefill-2048-DeepSeek-R1-14B", marks=_slow),
        pytest.param((1, 2), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, DEEPSEEK_R1_14B, 0.99, id="1x2-prefill-4096-DeepSeek-R1-14B", marks=_slow),
        pytest.param((1, 2), 8192, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, DEEPSEEK_R1_14B, 0.99, id="1x2-prefill-8192-DeepSeek-R1-14B", marks=_slow),
        pytest.param((1, 2), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, DEEPSEEK_R1_14B, 0.99, id="1x2-decode-32-DeepSeek-R1-14B", marks=_slow),

        # --- Llama-3.2-1B on T3K (1x8) ---
        pytest.param((1, 8), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-prefill-128-1B", marks=_slow),
        pytest.param((1, 8), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-prefill-1024-1B", marks=_slow),
        pytest.param((1, 8), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-prefill-2048-1B", marks=_slow),
        pytest.param((1, 8), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-prefill-4096-1B", marks=_slow),
        pytest.param((1, 8), 8192, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-prefill-8192-1B", marks=_slow),
        pytest.param((1, 8), 16384, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-prefill-16384-1B", marks=_slow),
        pytest.param((1, 8), 32768, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-prefill-32768-1B", marks=_slow),
        pytest.param((1, 8), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-decode-32-1B", marks=_slow),

        # --- Llama-3.2-3B on T3K (1x8) ---
        pytest.param((1, 8), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-prefill-128-3B", marks=_slow),
        pytest.param((1, 8), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-prefill-1024-3B", marks=_slow),
        pytest.param((1, 8), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-prefill-2048-3B", marks=_slow),
        pytest.param((1, 8), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-prefill-4096-3B", marks=_slow),
        pytest.param((1, 8), 8192, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-prefill-8192-3B", marks=_slow),
        pytest.param((1, 8), 16384, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-prefill-16384-3B", marks=_slow),
        pytest.param((1, 8), 32768, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-prefill-32768-3B", marks=_slow),
        pytest.param((1, 8), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-decode-32-3B", marks=_slow),

        # --- Llama-3.1-8B on T3K (1x8) ---
        pytest.param((1, 8), 256, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-256-8B", marks=_slow),
        pytest.param((1, 8), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-1024-8B", marks=_slow),
        pytest.param((1, 8), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-2048-8B", marks=_slow),
        pytest.param((1, 8), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-4096-8B", marks=_slow),
        pytest.param((1, 8), 8192, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-8192-8B", marks=_slow),
        pytest.param((1, 8), 16384, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-16384-8B", marks=_slow),
        pytest.param((1, 8), 32768, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-32768-8B", marks=_slow),
        pytest.param((1, 8), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-decode-32-8B", marks=_slow),

        # --- Llama-3.2-11B on T3K (1x8) ---
        pytest.param((1, 8), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x8-prefill-128-11B", marks=_slow),
        # NOTE: 11B 1024+ prefill has lower PCC (0.9844) due to vision model complexity
        pytest.param((1, 8), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.98, id="1x8-prefill-1024-11B", marks=_slow),
        pytest.param((1, 8), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.98, id="1x8-prefill-2048-11B", marks=_slow),
        pytest.param((1, 8), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.98, id="1x8-prefill-4096-11B", marks=_slow),
        pytest.param((1, 8), 8192, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.98, id="1x8-prefill-8192-11B", marks=_slow),
        pytest.param((1, 8), 16384, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.98, id="1x8-prefill-16384-11B", marks=_slow),
        pytest.param((1, 8), 32768, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.98, id="1x8-prefill-32768-11B", marks=_slow),
        pytest.param((1, 8), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x8-decode-32-11B", marks=_slow),

        # --- Llama-3.3-70B on T3K (1x8) ---
        # NOTE: 70B has slightly lower PCC (0.997) due to model size and multi-device communication
        pytest.param((1, 8), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_70B, 0.99, id="1x8-prefill-128-70B", marks=_slow),
        pytest.param((1, 8), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_70B, 0.99, id="1x8-prefill-1024-70B", marks=_slow),
        pytest.param((1, 8), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_70B, 0.99, id="1x8-prefill-2048-70B", marks=_slow),
        pytest.param((1, 8), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_70B, 0.99, id="1x8-prefill-4096-70B", marks=_slow),
        pytest.param((1, 8), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_70B, 0.97, id="1x8-decode-32-70B", marks=_slow),

        # --- Llama-3.2-90B on T3K (1x8) ---
        # NOTE: 90B has slightly lower PCC (0.995-0.996) due to model size
        pytest.param((1, 8), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_90B, 0.99, id="1x8-prefill-128-90B", marks=_slow),
        pytest.param((1, 8), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, LLAMA_90B, 0.99, id="1x8-decode-32-90B", marks=_slow),

        # --- Mistral-7B on T3K (1x8) ---
        pytest.param((1, 8), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x8-prefill-128-Mistral-7B", marks=_slow),
        pytest.param((1, 8), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x8-prefill-1024-Mistral-7B", marks=_slow),
        pytest.param((1, 8), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x8-prefill-2048-Mistral-7B", marks=_slow),
        pytest.param((1, 8), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x8-prefill-4096-Mistral-7B", marks=_slow),
        pytest.param((1, 8), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x8-decode-32-Mistral-7B", marks=_slow),

        # --- Mixtral-8x7B on T3K (1x8) ---
        pytest.param((1, 8), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MIXTRAL_8X7B, 0.99, id="1x8-prefill-128-Mixtral-8x7B", marks=_slow),
        pytest.param((1, 8), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MIXTRAL_8X7B, 0.99, id="1x8-prefill-1024-Mixtral-8x7B", marks=_slow),
        pytest.param((1, 8), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MIXTRAL_8X7B, 0.99, id="1x8-prefill-2048-Mixtral-8x7B", marks=_slow),
        pytest.param((1, 8), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, MIXTRAL_8X7B, 0.99, id="1x8-prefill-4096-Mixtral-8x7B", marks=_slow),
        pytest.param((1, 8), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, MIXTRAL_8X7B, 0.99, id="1x8-decode-32-Mixtral-8x7B", marks=_slow),

        # --- Qwen2.5-72B on T3K (1x8) ---
        pytest.param((1, 8), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_72B, 0.99, id="1x8-prefill-128-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_72B, 0.99, id="1x8-prefill-1024-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_72B, 0.99, id="1x8-prefill-2048-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_72B, 0.99, id="1x8-prefill-4096-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 8192, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_72B, 0.99, id="1x8-prefill-8192-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 16384, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_72B, 0.99, id="1x8-prefill-16384-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_72B, 0.99, id="1x8-decode-32-Qwen2.5-72B", marks=_slow),

        # --- Qwen2.5-Coder-32B on T3K (1x8) ---
        pytest.param((1, 8), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_CODER_32B, 0.99, id="1x8-prefill-128-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_CODER_32B, 0.99, id="1x8-prefill-1024-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_CODER_32B, 0.99, id="1x8-prefill-2048-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_CODER_32B, 0.99, id="1x8-prefill-4096-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, QWEN25_CODER_32B, 0.99, id="1x8-decode-32-Qwen2.5-Coder-32B", marks=_slow),
        # BF16 weights
        pytest.param((1, 8), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat16, QWEN25_CODER_32B, 0.99, id="1x8-prefill-128-Qwen2.5-Coder-32B-bf16", marks=_slow),
        pytest.param((1, 8), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat16, QWEN25_CODER_32B, 0.99, id="1x8-prefill-1024-Qwen2.5-Coder-32B-bf16", marks=_slow),
        pytest.param((1, 8), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat16, QWEN25_CODER_32B, 0.99, id="1x8-decode-32-Qwen2.5-Coder-32B-bf16", marks=_slow),

        # --- Qwen3-32B on T3K (1x8) ---
        pytest.param((1, 8), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN3_32B, 0.99, id="1x8-prefill-128-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN3_32B, 0.99, id="1x8-prefill-1024-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 2048, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN3_32B, 0.99, id="1x8-prefill-2048-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 4096, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN3_32B, 0.99, id="1x8-prefill-4096-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 8192, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN3_32B, 0.99, id="1x8-prefill-8192-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 16384, 1, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, QWEN3_32B, 0.99, id="1x8-prefill-16384-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, QWEN3_32B, 0.99, id="1x8-decode-32-Qwen3-32B", marks=_slow),
        # BF16 weights
        pytest.param((1, 8), 128, 1, "prefill", ttnn.bfloat16, ttnn.bfloat16, QWEN3_32B, 0.99, id="1x8-prefill-128-Qwen3-32B-bf16", marks=_slow),
        pytest.param((1, 8), 1024, 1, "prefill", ttnn.bfloat16, ttnn.bfloat16, QWEN3_32B, 0.99, id="1x8-prefill-1024-Qwen3-32B-bf16", marks=_slow),
        pytest.param((1, 8), 32, 32, "decode", ttnn.bfloat16, ttnn.bfloat16, QWEN3_32B, 0.99, id="1x8-decode-32-Qwen3-32B-bf16", marks=_slow),
    ]
    # fmt: on


# ============================================================================
# Integration Tests - Require device
# ============================================================================


@torch.no_grad()
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2), (1, 8)],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape,seq_len,batch_size,mode,act_dtype,wqkv_dtype,hf_model_name,pcc",
    _list_test_cases(),
)
@pytest.mark.parametrize(
    "page_block_size,chunk_size",
    [
        (None, None),  # standard (non-paged, non-chunked)
        (64, None),  # paged only (non-chunked)
        (64, 4096),  # paged + chunked
    ],
    ids=["standard", "paged", "paged-chunked"],
)
@pytest.mark.parametrize("num_decode_iterations", [10])
def test_attention_1d_vs_reference(
    ttnn_mesh_device: ttnn.MeshDevice,
    mesh_shape,
    seq_len,
    batch_size,
    mode,
    act_dtype,
    wqkv_dtype,
    hf_model_name,
    pcc,
    page_block_size,
    chunk_size,
    num_decode_iterations,
):
    """
    Test Attention1D constructed via from_config (direct API) executes successfully.

    This test uses the TTTv2 pattern:
    1. Load HF config directly via AutoConfig.from_pretrained
    2. Extract weights from HF reference model
    3. Create LazyWeight objects
    4. Build Attention1DConfig with explicit parameters
    5. Create Attention1D via from_config (NOT from_model_args)
    6. Run forward pass and verify execution

    Note: Full numerical comparison against HF reference is done in
    test_attention_1d_vs_reference_from_model_args which uses existing infrastructure
    that handles HF API differences.
    """
    # Skip if mesh_shape doesn't match device
    device_shape = (ttnn_mesh_device.shape[0], ttnn_mesh_device.shape[1])
    if device_shape != mesh_shape:
        pytest.skip(f"Test requires {mesh_shape} mesh, got {device_shape}")

    # Chunked prefill only applies to prefill mode
    chunked_prefill = chunk_size is not None
    if chunked_prefill and mode != "prefill":
        pytest.skip("Chunked prefill only applies to prefill mode")

    # Chunked prefill requires seq_len > chunk_size
    # TTTv1 uses chunk_size = N * 1024 where N ranges from 4-128 depending on model/device
    # Default of 4096 matches TTTv1's minimum (4 * 1024)
    if chunked_prefill and seq_len <= chunk_size:
        pytest.skip(f"Chunked prefill requires seq_len > chunk_size ({chunk_size})")

    # batch_size is now a test parameter:
    # - Prefill: typically batch_size=1, input shape (1, 1, seq_len, dim)
    # - Decode: typically batch_size=32 (continuous batching), input shape (1, 1, batch, dim)
    #   current_pos has shape (batch_size,) - one position per user
    # Note: For decode tests, seq_len parameter is unused (kept for parameterization compatibility)

    # max_seq_len: minimum allocation for KV cache
    # - Prefill: exactly seq_len tokens written to cache
    # - Decode: num_decode_iterations tokens (starts from position 0, no prefill)
    # Round up to alignment (SDPA kernel requires multiples of 32, paged attention requires page_block_size)
    if mode == "prefill":
        max_seq_len = seq_len
    else:
        max_seq_len = num_decode_iterations
    # Round up: use page_block_size if paged, otherwise 32 (SDPA tile alignment)
    alignment = page_block_size if page_block_size is not None else 32
    max_seq_len = ((max_seq_len + alignment - 1) // alignment) * alignment
    num_devices = ttnn_mesh_device.get_num_devices()

    # Seed for reproducibility
    seed = 1234
    torch.manual_seed(seed)

    # Load HF config directly (no ModelArgs)
    hf_config = AutoConfig.from_pretrained(hf_model_name)

    # Handle multimodal models (Mllama, LLaVA, etc.) which nest text config under .text_config
    is_multimodal = hasattr(hf_config, "text_config") and hf_config.text_config is not None
    cfg = hf_config.text_config if is_multimodal else hf_config
    cfg.num_hidden_layers = 1  # Only need 1 layer for testing

    dim = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    # Use explicit head_dim if available (e.g., Qwen3 models), else calculate from dim/n_heads
    # Note: some configs have head_dim=None explicitly, so we use `or` to fallback
    head_dim = getattr(cfg, "head_dim", None) or (dim // n_heads)
    sliding_window = getattr(cfg, "sliding_window", None)

    # Load HF model structure without weights, then initialize with random weights
    # This avoids slow network downloads and ensures reproducible deterministic testing
    if is_multimodal:
        # For multimodal models, import and use the specific model class
        from transformers import MllamaForConditionalGeneration

        with no_init_weights():
            # MllamaForConditionalGeneration uses _from_config (internal method) instead of from_config
            hf_model = MllamaForConditionalGeneration._from_config(hf_config, torch_dtype=torch.bfloat16)
        # Mllama has layers directly at language_model.layers (not language_model.model.layers)
        first_layer = hf_model.language_model.layers[0]
        rotary_emb = getattr(hf_model.language_model, "rotary_emb", None)
    else:
        with no_init_weights():
            hf_model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.bfloat16)
        first_layer = hf_model.model.layers[0]
        rotary_emb = getattr(hf_model.model, "rotary_emb", None)

    # Get reference attention from first layer
    reference_attn = first_layer.self_attn

    # Initialize attention weights deterministically (cached for speed across test cases)
    _get_or_init_attn_weights(hf_model_name, reference_attn)

    # Wrap in HfAttentionWrapper for consistent KV cache and RoPE handling (local class)
    reference_wrapper = HfAttentionWrapper(reference_attn, head_dim, rotary_emb)

    # Extract attention weights in TTNN layout
    wqkv_torch, wo_torch, q_norm_torch, k_norm_torch, wqkv_bias_torch = get_attention_weights_from_ref_model(
        reference_attn, num_devices
    )

    # Create LazyWeights with caching enabled
    # Cache keys include model name + seed to avoid mismatched cached weights across runs
    ttnn.SetDefaultDevice(ttnn_mesh_device)
    cache_dir = Path(os.getenv("TT_CACHE_PATH", "model_cache/attention_1d"))
    model_seed = stable_model_seed(hf_model_name)
    model_cache_prefix = f"{hf_model_name.replace('/', '_').replace('-', '_')}_seed{model_seed:08x}"

    # QKV weight: shard on dim=-1
    lazy_wqkv = LazyWeight(
        source=wqkv_torch,
        dtype=wqkv_dtype,
        cache_dir_weight_name=(cache_dir, f"{model_cache_prefix}_layer0_wqkv"),
    )

    # WO weight: shard on dim=-2
    lazy_wo = LazyWeight(
        source=wo_torch,
        dtype=wqkv_dtype,
        cache_dir_weight_name=(cache_dir, f"{model_cache_prefix}_layer0_wo"),
    )

    # Q/K norm configs (optional) - using RMSNorm1DConfig composition pattern
    q_norm_config = None
    k_norm_config = None
    if q_norm_torch is not None:
        lazy_q_norm = LazyWeight(
            source=q_norm_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            cache_dir_weight_name=(cache_dir, f"{model_cache_prefix}_layer0_q_norm"),
        )
        q_norm_config = RMSNorm1DConfig(
            weight=lazy_q_norm,
            mesh_device=ttnn_mesh_device,
            eps=1e-5,
            decode_in_sharded=False,  # Q/K heads are interleaved
            decode_out_sharded=False,
            prefill_distributed=False,
        )
    if k_norm_torch is not None:
        lazy_k_norm = LazyWeight(
            source=k_norm_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            cache_dir_weight_name=(cache_dir, f"{model_cache_prefix}_layer0_k_norm"),
        )
        k_norm_config = RMSNorm1DConfig(
            weight=lazy_k_norm,
            mesh_device=ttnn_mesh_device,
            eps=1e-5,
            decode_in_sharded=False,  # Q/K heads are interleaved
            decode_out_sharded=False,
            prefill_distributed=False,
        )

    # Create TT_CCL for multi-device
    tt_ccl = TT_CCL(ttnn_mesh_device) if num_devices > 1 else None

    # Determine topology
    if num_devices == 1:
        topology = None
    elif num_devices == 2:
        topology = ttnn.Topology.Linear
    else:
        topology = ttnn.Topology.Ring

    # Setup paged attention config and page table if enabled
    paged_attention_config = None
    page_table_tt = None
    page_table = None
    reverse_permutation = None

    if page_block_size is not None:
        # Paged attention parameters (use local PagedAttentionConfig)
        # Each user needs ceil(max_seq_len / block_size) blocks
        blocks_per_user = (max_seq_len + page_block_size - 1) // page_block_size
        max_num_blocks = max(128, blocks_per_user * batch_size)

        paged_attention_config = PagedAttentionConfig(
            block_size=page_block_size,
            max_num_blocks=max_num_blocks,
        )

        # Create page table: random permutation simulates block allocation
        permutation = torch.randperm(max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(batch_size, max_num_blocks // batch_size)
        page_table_tt = ttnn.from_torch(
            page_table,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
        )

    # Determine if we need the power user path (from_config) or can use the simple API
    # Power user path is needed for models with special features that aren't in the simple API
    has_special_features = (
        q_norm_config is not None
        or k_norm_config is not None
        or wqkv_bias_torch is not None
        or sliding_window is not None
        or paged_attention_config is not None
    )

    if has_special_features:
        # Power user path: use from_config() for models with special features
        config = Attention1DConfig(
            wqkv=lazy_wqkv,
            wo=lazy_wo,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            q_norm_config=q_norm_config,
            k_norm_config=k_norm_config,
            wqkv_bias=LazyWeight(source=wqkv_bias_torch) if wqkv_bias_torch is not None else None,
            sliding_window=sliding_window,
            paged_attention_config=paged_attention_config,
        )
        tt_model = Attention1D.from_config(config)
    else:
        # Happy path: simple API for basic Llama-style models
        tt_model = Attention1D(
            wqkv=lazy_wqkv,
            wo=lazy_wo,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
        )

    # Verify config is properly resolved (works for both happy path and from_config)
    assert tt_model.config.is_resolved(), "Config should be resolved after Attention1D creation"
    assert tt_model.config.dim == dim
    assert tt_model.config.n_heads == n_heads
    assert tt_model.config.n_kv_heads == n_kv_heads
    assert tt_model.config.head_dim == head_dim

    if mode == "prefill":
        _run_prefill_test(
            tt_model=tt_model,
            reference_wrapper=reference_wrapper,
            ttnn_mesh_device=ttnn_mesh_device,
            mesh_shape=mesh_shape,
            batch_size=batch_size,
            seq_len=seq_len,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            act_dtype=act_dtype,
            page_block_size=page_block_size,
            chunked_prefill=chunked_prefill,
            chunk_size=chunk_size,
            paged_attention_config=paged_attention_config,
            page_table=page_table,
            page_table_tt=page_table_tt,
            pcc=pcc,
        )
    else:
        _run_decode_test(
            tt_model=tt_model,
            reference_wrapper=reference_wrapper,
            ttnn_mesh_device=ttnn_mesh_device,
            mesh_shape=mesh_shape,
            batch_size=batch_size,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            act_dtype=act_dtype,
            page_block_size=page_block_size,
            page_table_tt=page_table_tt,
            num_decode_iterations=num_decode_iterations,
            pcc=pcc,
        )


def _run_prefill_test(
    tt_model,
    reference_wrapper,
    ttnn_mesh_device,
    mesh_shape,
    batch_size,
    seq_len,
    dim,
    n_heads,
    n_kv_heads,
    head_dim,
    act_dtype,
    page_block_size,
    chunked_prefill,
    chunk_size,
    paged_attention_config,
    page_table,
    page_table_tt,
    pcc,
):
    """Run prefill test and compare against HuggingFace reference."""
    pt_attention_input = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)

    # Prepare TT input for prefill
    tt_input = ttnn.from_torch(
        pt_attention_input.unsqueeze(0),  # [1, batch, seq_len, dim]
        device=ttnn_mesh_device,
        dtype=act_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    # Get rot_mats for prefill (cos/sin matrices) - extract from HF rotary_emb
    if chunked_prefill:
        # Chunked prefill: process sequence in chunks
        # Compute full rotation matrices once (covering all positions 0 to seq_len)
        full_cos, full_sin = get_cos_sin_from_hf(
            reference_wrapper.rotary_emb,
            seq_len=seq_len,
            head_dim=head_dim,
        )

        num_chunks = seq_len // chunk_size
        tt_outputs_chunked = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = chunk_start + chunk_size

            # Extract chunk input
            pt_chunk = pt_attention_input[:, chunk_start:chunk_end, :]

            # Prepare TT input for this chunk
            tt_chunk = ttnn.from_torch(
                pt_chunk.unsqueeze(0),
                device=ttnn_mesh_device,
                dtype=act_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
            )

            # Chunk page table: subset of pages for this chunk
            block_size = paged_attention_config.block_size
            chunk_page_table_pt = page_table[:, chunk_start // block_size : chunk_end // block_size]
            chunk_page_table_tt = ttnn.from_torch(
                chunk_page_table_pt,
                device=ttnn_mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
            )

            # Slice rotation matrices for this chunk's positions
            chunk_cos = full_cos[:, :, chunk_start:chunk_end, :]
            chunk_sin = full_sin[:, :, chunk_start:chunk_end, :]

            chunk_cos_tt = ttnn.from_torch(
                chunk_cos,
                device=ttnn_mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=act_dtype,
                mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_mesh_device),
            )
            chunk_sin_tt = ttnn.from_torch(
                chunk_sin,
                device=ttnn_mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=act_dtype,
                mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_mesh_device),
            )
            chunk_rot_mats = [chunk_cos_tt, chunk_sin_tt]

            # Run chunked prefill
            tt_chunk_out = tt_model.forward(
                tt_chunk,
                None,
                chunk_rot_mats,
                mode="prefill",
                page_table=page_table_tt,
                chunk_page_table=chunk_page_table_tt,
                chunk_start_idx=chunk_start,
            )

            # Collect output
            tt_chunk_out_torch = to_torch_auto_compose(tt_chunk_out)
            tt_chunk_output = tt_chunk_out_torch[:, 0:1, :chunk_size, :dim].view(batch_size, chunk_size, dim)
            tt_outputs_chunked.append(tt_chunk_output)

        # Concatenate all chunk outputs
        tt_output_torch = torch.cat(tt_outputs_chunked, dim=1)
    else:
        # Standard prefill: single forward pass - use HF rotary_emb
        rot_mats = get_rot_mats_from_hf(
            reference_wrapper.rotary_emb,
            seq_len=seq_len,
            head_dim=head_dim,
            device=ttnn_mesh_device,
        )

        # Run TT model - verify forward pass executes without error
        tt_out = tt_model.forward(
            tt_input,
            None,  # current_pos not used in prefill
            rot_mats,
            mode="prefill",
            page_table=page_table_tt,
        )

        # Convert output to torch and verify shape
        tt_out = to_torch_auto_compose(tt_out)
        tt_output_torch = tt_out[:, 0:1, :seq_len, :dim].view(batch_size, seq_len, dim)
    ttnn.SetDefaultDevice(None)

    # Verify output shape and content
    assert tt_output_torch.shape == (
        batch_size,
        seq_len,
        dim,
    ), f"Expected shape {(batch_size, seq_len, dim)}, got {tt_output_torch.shape}"
    assert not torch.isnan(tt_output_torch).any(), "Output contains NaN values"
    assert not torch.isinf(tt_output_torch).any(), "Output contains Inf values"

    # Run reference HuggingFace attention using HfAttentionWrapper
    # Note: freqs_cis_i is None because HfAttentionWrapper uses rotary_emb directly
    with torch.no_grad():
        reference_output = reference_wrapper(pt_attention_input, start_pos=0, mask=None)

    # Compare TT output with reference using PCC
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch.to(reference_output.dtype), pcc)
    logger.info(f"  PCC comparison: {pcc_message}")
    logger.info(comp_allclose(reference_output, tt_output_torch.to(reference_output.dtype)))
    assert passing, f"Prefill PCC failed: {pcc_message} (expected >= {pcc})"

    # Note: KV cache comparison is skipped because:
    # - HF's DynamicCache stores K/V after applying HF-format RoPE
    # - TT's Attention1D stores K/V after applying Meta-format RoPE
    # These are different rotary embedding formats, so cached values won't match.
    # The output comparison above is the meaningful correctness check.
    logger.info("  KV cache validation: SKIPPED (HF/TT use different RoPE formats in cache)")

    paged_str = f"paged(block_size={page_block_size})" if page_block_size is not None else "non-paged"
    chunked_str = "chunked" if chunked_prefill else "non-chunked"
    logger.info(
        f"test_attention_1d_vs_reference (from_config): PASSED for mode=prefill, seq_len={seq_len}, {paged_str}, {chunked_str}"
    )
    logger.info(f"  Config: dim={dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    logger.info(f"  Output shape: {tt_output_torch.shape}, dtype: {tt_output_torch.dtype}")


def _run_decode_test(
    tt_model,
    reference_wrapper,
    ttnn_mesh_device,
    mesh_shape,
    batch_size,
    dim,
    n_heads,
    n_kv_heads,
    head_dim,
    max_seq_len,
    act_dtype,
    page_block_size,
    page_table_tt,
    num_decode_iterations,
    pcc,
):
    """
    Run decode-only test starting from position 0 (TTTv1 style).

    This is a pure unit test for decode_forward:
    - No prefill step - KV cache builds incrementally during decode
    - Runs num_decode_iterations decode steps at positions 0, 1, 2, ...
    - Compares each iteration against HuggingFace reference

    For batch_size > 1 (continuous batching):
    - All users decode with identical inputs at the same position
    - Verifies batching mechanism doesn't introduce numerical errors

    Note: prefill→decode transition is tested separately in test_attention_1d_prefill_decode_transition.
    """
    # Create decode-specific RotarySetupHelper using HF rotary_emb (reused across iterations)
    decode_rope_setup = RotarySetupHelper(
        ttnn_mesh_device,
        batch_size,
        head_dim,
        max_seq_len,
        reference_wrapper.rotary_emb,
        use_qk_fused=False,
    )

    # Decode iterations starting from position 0
    # KV cache builds incrementally: position 0, 1, 2, ... (TTTv1 style)
    all_iterations_passing = True
    min_pcc_across_iterations = 1.0

    for decode_iter in range(num_decode_iterations):
        current_pos_value = decode_iter  # Start from 0, not after prefill

        # Create identical input for all users (for comparison against single HF reference)
        # TTNN decode expects shape (1, 1, batch_size, dim) - seq_len=1, batch in 3rd dim
        pt_decode_single = torch.randn(1, 1, dim, dtype=torch.bfloat16)
        pt_decode_batched = pt_decode_single.unsqueeze(2).expand(1, 1, batch_size, dim).contiguous()

        tt_decode_input = ttnn.from_torch(
            pt_decode_batched,
            device=ttnn_mesh_device,
            dtype=act_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
        )

        # Convert to decode input memory config
        tt_decode_input = ttnn.to_memory_config(tt_decode_input, tt_model.config.decode_input_memcfg)

        # Position for decode: all users at the same position
        position_idxs = torch.full((batch_size,), current_pos_value, dtype=torch.long)
        decode_rot_mats = decode_rope_setup.get_rot_mats(position_idxs)

        current_pos = ttnn.from_torch(
            position_idxs,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(ttnn_mesh_device),
        )

        # Run TT decode (batched)
        tt_out = tt_model.forward(
            tt_decode_input,
            current_pos,
            decode_rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )

        # Convert output to torch
        tt_out = to_torch_auto_compose(tt_out)

        # Extract TT output: shape (1, 1, batch_size, dim) -> (batch_size, 1, dim)
        tt_output_torch = tt_out[:, 0:1, :batch_size, :dim].view(batch_size, 1, dim)

        # Run HuggingFace decode using wrapper (single user reference)
        with torch.no_grad():
            reference_output = reference_wrapper(pt_decode_single, start_pos=current_pos_value, mask=None)

        # Verify output content
        assert tt_output_torch.numel() > 0, f"Output is empty at iteration {decode_iter}"
        assert not torch.isnan(tt_output_torch).any(), f"Output contains NaN at iteration {decode_iter}"
        assert not torch.isinf(tt_output_torch).any(), f"Output contains Inf at iteration {decode_iter}"

        # Compare EACH user's TT output with the single HF reference
        for user_idx in range(batch_size):
            user_output = tt_output_torch[user_idx : user_idx + 1]
            passing, pcc_value = comp_pcc(reference_output, user_output.to(reference_output.dtype), pcc)
            if isinstance(pcc_value, (int, float)):
                min_pcc_across_iterations = min(min_pcc_across_iterations, float(pcc_value))
            if not passing:
                logger.warning(f"  Iteration {decode_iter}, User {user_idx} PCC failed: {pcc_value}")
                all_iterations_passing = False

    ttnn.SetDefaultDevice(None)

    logger.info(
        f"  Decode iterations: {num_decode_iterations}, min_pcc={min_pcc_across_iterations:.6f} across all iterations"
    )
    assert all_iterations_passing, f"Decode PCC failed (min_pcc={min_pcc_across_iterations:.6f}, expected >= {pcc})"

    paged_str = f"paged(block_size={page_block_size})" if page_block_size is not None else "non-paged"
    logger.info(
        f"test_attention_1d_vs_reference (from_config): PASSED for mode=decode, "
        f"batch_size={batch_size}, {paged_str}, iterations={num_decode_iterations}"
    )
    logger.info(f"  Config: dim={dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")


# =============================================================================
# Integration Test: Prefill → Decode Transition
# =============================================================================


@torch.no_grad()
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        (1, 1),  # single device
        (1, 2),  # 1D mesh, 2 devices
        (1, 8),  # 1D mesh, 8 devices
    ],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize(
    "page_block_size",
    [None, 64],  # Test both non-paged and paged attention
    ids=["non-paged", "paged"],
)
def test_attention_1d_prefill_decode_transition(ttnn_mesh_device: ttnn.MeshDevice, page_block_size):
    """
    Integration test for prefill→decode transition.

    This test verifies that KV cache state is correctly maintained across the
    prefill→decode boundary. Unlike the unit tests which test prefill and decode
    in isolation, this test verifies the handoff between modes.

    **Integration Point: KV Cache**
    The KV cache is explicitly created and passed to the model config, making
    the integration point crystal clear:
    - Prefill writes keys/values to positions 0..prefill_seq_len-1
    - Decode reads from those positions and writes to prefill_seq_len+

    Test flow:
    1. Create explicit KV cache tensors (the integration point)
    2. Run prefill → populates KV cache positions 0..N-1
    3. Run decode at positions N, N+1, ... → reads from cache, writes new positions
    4. Compare outputs against HuggingFace reference
    """
    # Minimal test parameters - we're testing transition, not model variations
    hf_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    prefill_seq_len = 128  # Must be divisible by 128
    num_decode_after_prefill = 5
    batch_size = 32  # Realistic continuous batching scenario
    pcc = 0.98

    seed = 42
    torch.manual_seed(seed)

    # Load HF config
    hf_config = AutoConfig.from_pretrained(hf_model_name)
    cfg = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
    cfg.num_hidden_layers = 1

    dim = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", None) or (dim // n_heads)

    # Calculate max_seq_len with proper alignment
    max_seq_len = prefill_seq_len + num_decode_after_prefill
    alignment = page_block_size if page_block_size is not None else 32
    max_seq_len = ((max_seq_len + alignment - 1) // alignment) * alignment

    mesh_shape = ttnn_mesh_device.shape
    num_devices = ttnn_mesh_device.get_num_devices()
    n_local_kv_heads = n_kv_heads // num_devices

    # Load HF model
    with no_init_weights():
        hf_model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.bfloat16)
    first_layer = hf_model.model.layers[0]
    rotary_emb = getattr(hf_model.model, "rotary_emb", None)

    reference_attn = first_layer.self_attn
    _get_or_init_attn_weights(hf_model_name, reference_attn)
    reference_wrapper = HfAttentionWrapper(reference_attn, head_dim, rotary_emb)

    # Extract weights
    wqkv_torch, wo_torch, q_norm_torch, k_norm_torch, wqkv_bias_torch = get_attention_weights_from_ref_model(
        reference_attn, num_devices
    )

    act_dtype = ttnn.bfloat16
    wqkv_dtype = ttnn.bfloat8_b
    lazy_wqkv = LazyWeight(source=wqkv_torch, dtype=wqkv_dtype, cache_dir_weight_name=None)
    lazy_wo = LazyWeight(source=wo_torch, dtype=wqkv_dtype, cache_dir_weight_name=None)

    # Setup paged attention if enabled
    paged_attention_config = None
    page_table_tt = None
    if page_block_size is not None:
        # Each user needs ceil(max_seq_len / block_size) blocks
        blocks_per_user = (max_seq_len + page_block_size - 1) // page_block_size
        max_num_blocks = max(128, blocks_per_user * batch_size)
        paged_attention_config = PagedAttentionConfig(block_size=page_block_size, max_num_blocks=max_num_blocks)

        permutation = torch.randperm(max_num_blocks)
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(batch_size, max_num_blocks // batch_size)
        page_table_tt = ttnn.from_torch(
            page_table,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
        )

    # =========================================================================
    # INTEGRATION POINT: Explicitly create KV cache tensors
    # =========================================================================
    # The KV cache is the shared state between prefill and decode.
    # By creating it explicitly here, we make the integration point visible:
    # - Prefill will WRITE to this cache (positions 0..prefill_seq_len-1)
    # - Decode will READ from this cache and WRITE new positions
    # =========================================================================
    if paged_attention_config is not None:
        cache_k = zeros_like_paged_cache(paged_attention_config, n_local_kv_heads, head_dim)
        cache_v = zeros_like_paged_cache(paged_attention_config, n_local_kv_heads, head_dim)
    else:
        cache_k = zeros_like_kv_cache(batch_size, n_local_kv_heads, max_seq_len, head_dim)
        cache_v = zeros_like_kv_cache(batch_size, n_local_kv_heads, max_seq_len, head_dim)

    # Wrap as LazyWeight for config
    kv_cache = (LazyWeight(source=cache_k), LazyWeight(source=cache_v))

    # Build config with explicit KV cache
    topology = ttnn.Topology.Ring if num_devices > 1 else None
    tt_ccl = TT_CCL(ttnn_mesh_device) if num_devices > 1 else None

    config = Attention1DConfig(
        wqkv=lazy_wqkv,
        wo=lazy_wo,
        mesh_device=ttnn_mesh_device,
        tt_ccl=tt_ccl,
        topology=topology,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        scale=head_dim**-0.5,
        use_vllm_paged_kv_cache=False,
        paged_attention_config=paged_attention_config,
        kv_cache=kv_cache,  # <-- EXPLICIT KV CACHE: the integration point
        wqkv_dtype=wqkv_dtype,
        wo_dtype=wqkv_dtype,
        activation_dtype=act_dtype,
    )

    tt_model = Attention1D.from_config(config)

    # =========================================================================
    # Step 1: PREFILL - populates KV cache positions 0..prefill_seq_len-1
    # =========================================================================
    # Run prefill per-user (continuous batching style) to avoid compute grid limits.
    # Each user gets the same input for easy comparison.
    # =========================================================================
    pt_prefill_input_single = torch.randn(1, prefill_seq_len, dim, dtype=torch.bfloat16)
    prefill_rot_mats = get_rot_mats_from_hf(rotary_emb, prefill_seq_len, head_dim, ttnn_mesh_device)

    # Collect outputs for all users
    tt_prefill_outputs = []
    for user_id in range(batch_size):
        tt_prefill_input = ttnn.from_torch(
            pt_prefill_input_single.unsqueeze(0),  # (1, 1, prefill_seq_len, dim)
            device=ttnn_mesh_device,
            dtype=act_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
        )
        tt_prefill_out = tt_model.forward(
            tt_prefill_input, None, prefill_rot_mats, user_id=user_id, mode="prefill", page_table=page_table_tt
        )
        tt_prefill_out = to_torch_auto_compose(tt_prefill_out)
        tt_prefill_outputs.append(tt_prefill_out[:, 0:1, :prefill_seq_len, :dim].view(1, prefill_seq_len, dim))

    # Stack outputs: (batch_size, prefill_seq_len, dim)
    tt_prefill_output = torch.cat(tt_prefill_outputs, dim=0)

    # HF reference prefill (single user, same input)
    with torch.no_grad():
        ref_prefill_output = reference_wrapper(pt_prefill_input_single, start_pos=0, mask=None)

    # Compare first user's output (all users have same input, should match)
    passing, pcc_msg = comp_pcc(ref_prefill_output, tt_prefill_output[0:1].to(ref_prefill_output.dtype), pcc)
    logger.info(f"  Prefill PCC: {pcc_msg}")
    assert passing, f"Prefill failed: {pcc_msg}"

    # =========================================================================
    # Step 2: DECODE - uses prefill output as input (realistic autoregressive flow)
    # =========================================================================
    # In real autoregressive generation:
    # - First decode input = last position of prefill output
    # - Subsequent decode inputs = previous decode output
    # This tests both KV cache integration AND data flow between modes.
    # =========================================================================
    decode_rope_setup = RotarySetupHelper(
        ttnn_mesh_device, batch_size, head_dim, max_seq_len, rotary_emb, use_qk_fused=False
    )

    # First decode input: last position of prefill output (shape: batch, 1, dim)
    pt_decode_input = tt_prefill_output[:, -1:, :].clone()  # (batch_size, 1, dim)
    ref_decode_input = ref_prefill_output[:, -1:, :].clone()  # For HF reference

    for decode_iter in range(num_decode_after_prefill):
        current_pos_value = prefill_seq_len + decode_iter

        # Prepare TT input: (batch, 1, dim) -> (1, 1, batch, dim) for TTNN decode format
        pt_decode_batched = pt_decode_input.transpose(0, 1).unsqueeze(0)  # (1, 1, batch_size, dim)

        tt_decode_input = ttnn.from_torch(
            pt_decode_batched,
            device=ttnn_mesh_device,
            dtype=act_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
        )
        tt_decode_input = ttnn.to_memory_config(tt_decode_input, tt_model.config.decode_input_memcfg)

        position_idxs = torch.full((batch_size,), current_pos_value, dtype=torch.long)
        decode_rot_mats = decode_rope_setup.get_rot_mats(position_idxs)

        current_pos = ttnn.from_torch(
            position_idxs,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(ttnn_mesh_device),
        )

        tt_decode_out = tt_model.forward(
            tt_decode_input, current_pos, decode_rot_mats, mode="decode", page_table=page_table_tt
        )
        tt_decode_out = to_torch_auto_compose(tt_decode_out)
        tt_decode_output = tt_decode_out[:, 0:1, :batch_size, :dim].view(batch_size, 1, dim)

        # HF reference decode (using same input flow)
        with torch.no_grad():
            ref_decode_output = reference_wrapper(ref_decode_input, start_pos=current_pos_value, mask=None)

        # Compare first user's output (all users have identical input, should produce identical output)
        passing, pcc_msg = comp_pcc(ref_decode_output, tt_decode_output[0:1].to(ref_decode_output.dtype), pcc)
        logger.info(f"  Decode[pos={current_pos_value}] PCC: {pcc_msg}")
        assert passing, f"Decode at position {current_pos_value} failed: {pcc_msg}"

        # Next decode input = this decode output (autoregressive flow)
        pt_decode_input = tt_decode_output.clone()
        ref_decode_input = ref_decode_output.clone()

    ttnn.SetDefaultDevice(None)

    paged_str = f"paged(block_size={page_block_size})" if page_block_size is not None else "non-paged"
    logger.info(
        f"test_attention_1d_prefill_decode_transition: PASSED ({paged_str}, "
        f"prefill={prefill_seq_len}, decode={num_decode_after_prefill})"
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        (1, 1),  # single device
        (1, 2),  # 1D mesh, 2 devices
        (1, 8),  # 1D mesh, 8 devices
    ],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (512, 32))
def test_attention_1d_vs_reference_from_model_args(ttnn_mesh_device: ttnn.MeshDevice, seq_len):
    """
    Test that Attention1D class created via from_model_args matches reference model.

    This test validates backward compatibility with the ModelArgs factory method
    and performs numerical PCC comparison against the reference attention.
    """
    from models.tt_transformers.tests.test_utils import get_ref_model_dype
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.common import precompute_freqs
    from models.tt_transformers.tt.model_config import Mode, ModelArgs
    from models.tt_transformers.tt.rope import RotarySetup, get_rot_mats

    # Use HF_MODEL env var if set, otherwise use appropriate default based on device count
    # Multi-device requires larger models (dim >= 4096) for proper sharding
    num_devices = ttnn_mesh_device.get_num_devices()
    env_model = os.environ.get("HF_MODEL")

    if not env_model:
        # Set default model based on device configuration
        if num_devices == 1:
            # Small model works for single device
            default_model = "meta-llama/Llama-3.2-1B"
        else:
            # Multi-device needs larger model with dim >= 4096
            default_model = "meta-llama/Llama-3.1-8B-Instruct"
        os.environ["HF_MODEL"] = default_model
        logger.info(f"HF_MODEL not set, using default: {default_model}")

    dtype = ttnn.bfloat8_b
    pcc = 0.98
    batch_size = 1
    mode = "decode" if seq_len <= 32 else "prefill"

    model_args = ModelArgs(ttnn_mesh_device, max_batch_size=batch_size, max_seq_len=2048, cache_hf=True)
    model_args.n_layers = 1

    if model_args.is_galaxy:
        pytest.skip("Attention1D test only runs on non-TG devices")

    # Verify model dimensions are sufficient for multi-device
    if num_devices > 1 and model_args.dim < 4096:
        pytest.skip(f"Model dim={model_args.dim} too small for {num_devices} devices - use 8B+ models")

    state_dict = model_args.load_state_dict()

    # Load reference attention model
    first_layer_prefix = model_args.get_state_dict_prefix("Attention", 0) + "."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }
    reference_model = model_args.reference_attention()
    reference_model.load_state_dict(partial_state_dict)

    # Setup RoPE transformation matrices
    rope_setup = RotarySetup(
        ttnn_mesh_device,
        batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling,
        model_args.use_qk_fused,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    # Precompute freqs_cis for reference model
    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.rope_scaling.factor if model_args.rope_scaling else None,
        model_args.rope_scaling.original_max_position_embeddings if model_args.rope_scaling else None,
        model_args.rope_scaling.rope_type.value if model_args.rope_scaling else "llama3",
    )
    freqs_cis = torch.complex(cos, sin)

    # Cache path
    def topology_aware_cache_path(dtype):
        if model_args.instruct:
            return model_args.model_cache_path / {
                ttnn.bfloat16: f"tensor_cache_instruct_bf16_{ttnn_mesh_device.shape}",
                ttnn.bfloat8_b: f"tensor_cache_instruct_bfp8_{ttnn_mesh_device.shape}",
            }.get(dtype, f"tensor_cache_instruct_{ttnn_mesh_device.shape}")
        return model_args.model_cache_path / {
            ttnn.bfloat16: f"tensor_cache_bf16_{ttnn_mesh_device.shape}",
            ttnn.bfloat8_b: f"tensor_cache_bfp8_{ttnn_mesh_device.shape}",
        }.get(dtype, f"tensor_cache_{ttnn_mesh_device.shape}")

    weight_cache_path = topology_aware_cache_path(dtype)

    # Create TT_CCL for multi-device
    tt_ccl = TT_CCL(ttnn_mesh_device) if num_devices > 1 else None

    # Create Attention1D via from_model_args
    tt_model = Attention1D.from_model_args(
        mesh_device=ttnn_mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        layer_num=0,
        transformation_mats=transformation_mats,
        use_paged_kv_cache=False,
    )

    # Verify the model was created successfully
    assert tt_model is not None
    assert tt_model.config.is_resolved()
    assert tt_model.config.dim == model_args.dim
    assert tt_model.config.n_heads == model_args.n_heads
    assert tt_model.config.n_kv_heads == model_args.n_kv_heads

    logger.info(f"test_attention_1d_vs_reference_from_model_args: Testing mode={mode}, seq_len={seq_len}")

    if mode == "prefill":
        # Prefill mode test
        pt_attention_input = torch.randn(
            batch_size, seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
        )

        # Prepare TT input
        tt_input = ttnn.from_torch(
            pt_attention_input.unsqueeze(0),  # [1, batch, seq_len, dim]
            device=ttnn_mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                ttnn_mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape
            ),
        )

        # Get rot_mats for prefill
        rot_mats = get_rot_mats(
            head_dim=model_args.head_dim,
            device=ttnn_mesh_device,
            seq_len=seq_len,
            theta=model_args.rope_theta,
            rope_scaling=model_args.rope_scaling,
        )

        # Run TT model
        tt_out = tt_model.forward(
            tt_input,
            None,  # current_pos not used in prefill
            rot_mats,
            mode="prefill",
        )

        # Convert TT output to torch
        tt_out = to_torch_auto_compose(tt_out)
        tt_output_torch = tt_out[:, 0:1, :seq_len, : model_args.dim].view(batch_size, seq_len, model_args.dim)

        # Run reference model
        freqs_cis_slice = freqs_cis[:seq_len]
        reference_output = reference_model(pt_attention_input, start_pos=0, freqs_cis_i=freqs_cis_slice, mask=None)

        # Compare with PCC
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch.to(reference_output.dtype), pcc)
        logger.info(f"  PCC comparison: {pcc_message}")
        logger.info(comp_allclose(reference_output, tt_output_torch.to(reference_output.dtype)))
        assert passing, f"Prefill PCC failed: {pcc_message} (expected >= {pcc})"

        logger.info(f"test_attention_1d_vs_reference_from_model_args: PASSED for mode={mode}, seq_len={seq_len}")

    else:
        # Decode mode test - requires prefill first to populate KV cache
        prefill_seq_len = 128
        pt_prefill_input = torch.randn(
            batch_size,
            prefill_seq_len,
            model_args.dim,
            dtype=get_ref_model_dype(reference_model, model_args.model_name),
        )

        # Prefill TT model
        tt_prefill_input = ttnn.from_torch(
            pt_prefill_input.unsqueeze(0),
            device=ttnn_mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                ttnn_mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape
            ),
        )

        prefill_rot_mats = get_rot_mats(
            head_dim=model_args.head_dim,
            device=ttnn_mesh_device,
            seq_len=prefill_seq_len,
            theta=model_args.rope_theta,
            rope_scaling=model_args.rope_scaling,
        )

        _ = tt_model.forward(
            tt_prefill_input,
            None,
            prefill_rot_mats,
            user_id=0,
            mode="prefill",
        )

        # Prefill reference model (to populate its KV cache state)
        freqs_cis_prefill = freqs_cis[:prefill_seq_len]
        _ = reference_model(pt_prefill_input, start_pos=0, freqs_cis_i=freqs_cis_prefill, mask=None)

        # Decode pass
        current_pos = torch.tensor([prefill_seq_len])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_mesh_device),
        )

        pt_decode_input = torch.randn(
            batch_size, 1, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
        )

        # Prepare decode input for TT
        attention_input = model_args.prepare_residual_tensor_decode(
            pt_decode_input.clone(),
            model_args.get_attn_input_mem_config(Mode.DECODE),
            force_replicated=True,
        )

        decode_rot_mats = rope_setup.get_rot_mats(current_pos)

        # Run TT decode
        tt_out = tt_model.forward(
            attention_input,
            current_pos_tensor,
            decode_rot_mats,
            mode="decode",
        )

        # Convert TT output
        tt_out = to_torch_auto_compose(tt_out)
        tt_output_torch = tt_out[:, 0:1, :batch_size, : model_args.dim].view(batch_size, 1, model_args.dim)

        # Run reference decode
        freqs_cis_i = freqs_cis[prefill_seq_len, :].unsqueeze(0)
        reference_output = reference_model(
            pt_decode_input, start_pos=prefill_seq_len, freqs_cis_i=freqs_cis_i, mask=None
        )

        # Compare with PCC
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch.to(reference_output.dtype), pcc)
        logger.info(f"  PCC comparison: {pcc_message}")
        logger.info(comp_allclose(reference_output, tt_output_torch.to(reference_output.dtype)))
        assert passing, f"Decode PCC failed: {pcc_message} (expected >= {pcc})"

        logger.info(f"test_attention_1d_vs_reference_from_model_args: PASSED for mode={mode}, seq_len={seq_len}")


def test_attention_1d_rejects_galaxy():
    """Test that Attention1D.from_model_args rejects Galaxy/TG devices."""
    # Mock args with is_galaxy = True
    mock_args = MagicMock()
    mock_args.is_galaxy = True

    with pytest.raises(ValueError, match="cannot be used for Galaxy"):
        Attention1D.from_model_args(
            mesh_device=MagicMock(),
            tt_ccl=MagicMock(),
            args=mock_args,
            state_dict={},
            weight_cache_path=None,
            layer_num=0,
            transformation_mats={},
        )


@torch.no_grad()
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1)],
    ids=["1x1"],
    indirect=True,
)
@pytest.mark.parametrize(
    "sliding_window,seq_len,pcc",
    [
        pytest.param(64, 128, 0.99, id="sw64-seq128"),
        pytest.param(64, 256, 0.99, id="sw64-seq256"),
        pytest.param(128, 256, 0.99, id="sw128-seq256"),
    ],
)
def test_attention_1d_sliding_window(
    ttnn_mesh_device: ttnn.MeshDevice,
    sliding_window: int,
    seq_len: int,
    pcc: float,
):
    """
    Test Attention1D with sliding window attention.

    This test explicitly verifies that sliding window masking works correctly by:
    1. Using seq_len > sliding_window to ensure the window is actually applied
    2. Creating a reference implementation with sliding window causal mask
    3. Comparing TT output against the masked reference

    The sliding window limits attention to the last `sliding_window` tokens,
    which affects which KV entries are attended to during SDPA.
    """
    # Use Llama-3.2-1B as base model (fast to load, no sliding window by default)
    hf_model_name = LLAMA_1B

    mesh_shape = tuple(ttnn_mesh_device.shape)
    num_devices = ttnn_mesh_device.get_num_devices()
    batch_size = 1
    max_seq_len = max(512, seq_len * 2)

    torch.manual_seed(42)

    # Load HuggingFace model
    hf_config = AutoConfig.from_pretrained(hf_model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=torch.bfloat16)
    reference_attn = hf_model.model.layers[0].self_attn
    rotary_emb = getattr(hf_model.model, "rotary_emb", None)

    dim = hf_config.hidden_size
    n_heads = hf_config.num_attention_heads
    n_kv_heads = hf_config.num_key_value_heads
    head_dim = dim // n_heads

    # Get weights from reference model
    wqkv_torch, wo_torch, _, _, _ = get_attention_weights_from_ref_model(reference_attn, num_devices)

    # Create TT model with sliding window
    ttnn.SetDefaultDevice(ttnn_mesh_device)
    lazy_wqkv = LazyWeight(source=wqkv_torch, dtype=ttnn.bfloat8_b, cache_dir_weight_name=None)
    lazy_wo = LazyWeight(source=wo_torch, dtype=ttnn.bfloat8_b, cache_dir_weight_name=None)

    # Note: kv_cache is auto-created by config resolution
    config = Attention1DConfig(
        wqkv=lazy_wqkv,
        wo=lazy_wo,
        mesh_device=ttnn_mesh_device,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        scale=head_dim**-0.5,
        sliding_window=sliding_window,  # Enable sliding window
        use_vllm_paged_kv_cache=False,
        activation_dtype=ttnn.bfloat16,
    )

    tt_model = Attention1D.from_config(config)

    rot_mats = get_rot_mats_from_hf(
        rotary_emb,
        seq_len=seq_len,
        head_dim=head_dim,
        device=ttnn_mesh_device,
    )

    # Prepare input
    pt_input = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        pt_input.unsqueeze(0),  # [1, batch, seq_len, dim]
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    # Run TT model with sliding window
    tt_out = tt_model.forward(
        tt_input,
        None,  # current_pos not used in prefill
        rot_mats,
        mode="prefill",
    )

    tt_out_torch = to_torch_auto_compose(tt_out)
    tt_output = tt_out_torch[:, 0:1, :seq_len, :dim].view(batch_size, seq_len, dim)

    # Create reference with sliding window causal mask
    # Sliding window mask: position i can only attend to positions max(0, i - sliding_window + 1) to i
    def create_sliding_window_mask(seq_len: int, sliding_window: int) -> torch.Tensor:
        """Create a sliding window causal attention mask for HuggingFace attention."""
        # HF expects mask shape: (batch, 1, seq_len, seq_len) where 0 = attend, -inf = mask
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bfloat16)

        for i in range(seq_len):
            # Position i can attend to [max(0, i - sliding_window + 1), i]
            for j in range(seq_len):
                if j > i:
                    # Causal: can't attend to future
                    mask[i, j] = torch.finfo(torch.bfloat16).min
                elif j < i - sliding_window + 1:
                    # Sliding window: can't attend beyond window
                    mask[i, j] = torch.finfo(torch.bfloat16).min

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    sliding_mask = create_sliding_window_mask(seq_len, sliding_window)

    # Setup reference attention using local HfAttentionWrapper
    reference_wrapper = HfAttentionWrapper(reference_attn, head_dim, rotary_emb)

    # Run reference WITH sliding window mask
    ref_output_sw = reference_wrapper(pt_input, start_pos=0, mask=sliding_mask)

    # Also run reference WITHOUT sliding window (full attention) for comparison
    # Need a fresh wrapper since it caches KV
    reference_wrapper_full = HfAttentionWrapper(reference_attn, head_dim, rotary_emb)
    ref_output_full = reference_wrapper_full(pt_input, start_pos=0, mask=None)

    ttnn.SetDefaultDevice(None)

    # Basic sanity checks
    assert not torch.isnan(tt_output).any(), "TT output contains NaN"
    assert not torch.isinf(tt_output).any(), "TT output contains Inf"
    assert tt_output.shape == (batch_size, seq_len, dim), f"Shape mismatch: {tt_output.shape}"

    logger.info(f"Sliding window test: sw={sliding_window}, seq_len={seq_len}")

    # Compare TT output (with sliding window) against HF reference (with sliding window mask)
    passing_sw, pcc_sw = comp_pcc(ref_output_sw, tt_output.to(ref_output_sw.dtype), pcc)
    logger.info(f"  PCC TT vs HF (both with sliding window): {pcc_sw}")
    assert passing_sw, f"Sliding window PCC failed: {pcc_sw} (expected >= {pcc})"

    # Verify sliding window actually has an effect by comparing to full attention
    if seq_len > sliding_window:
        # The sliding window reference should differ from full attention
        ref_diff = (ref_output_sw - ref_output_full).abs().mean()
        logger.info(f"  HF sliding window vs full attention diff: {ref_diff:.6f}")
        assert ref_diff > 1e-4, "HF sliding window mask had no effect on reference"

        # TT output should also differ from full attention
        tt_diff = (tt_output - ref_output_full).abs().mean()
        logger.info(f"  TT sliding window vs HF full attention diff: {tt_diff:.6f}")
        assert tt_diff > 1e-4, "TT sliding window had no effect"

        # The TT-vs-reference-sliding-window PCC should be higher than TT-vs-full-attention
        _, pcc_full = comp_pcc(ref_output_full, tt_output.to(ref_output_full.dtype), 0.0)
        logger.info(f"  PCC TT (sliding) vs HF (full): {pcc_full}")

        # Compare PCC values
        pcc_sw_val = float(pcc_sw)
        pcc_full_val = float(pcc_full)
        assert pcc_sw_val > pcc_full_val, (
            f"TT sliding window should match HF sliding window better than HF full attention. "
            f"PCC(TT,HF_sw)={pcc_sw_val:.4f} should be > PCC(TT,HF_full)={pcc_full_val:.4f}"
        )

    logger.info(f"test_attention_1d_sliding_window: PASSED (sw={sliding_window}, seq_len={seq_len})")
