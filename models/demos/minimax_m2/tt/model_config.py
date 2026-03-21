# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2.5 TTNN model configuration.

Architecture reference: models/demos/minimax_m2/ARCHITECTURE.md
Galaxy parallelism: 1×Galaxy = mesh (8,4) = 32 chips
  Decode: EP=8 (rows), TP=4 (cols)
  Prefill: SP=8 (rows), TP=4 (cols), EP=1
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ttnn
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.tt_transformers.tt.common import PagedAttentionConfig


@dataclass
class MiniMaxM2TTConfig:
    """TTNN-specific configuration for MiniMax-M2.5."""

    # Model dimensions (from config.json)
    hidden_size: int = 3072
    head_dim: int = 128
    num_attention_heads: int = 48  # Q heads
    num_key_value_heads: int = 8  # KV heads (GQA 6:1)
    num_hidden_layers: int = 62
    intermediate_size: int = 1536  # per-expert FFN hidden
    num_local_experts: int = 256
    num_experts_per_tok: int = 8
    rotary_dim: int = 64  # partial RoPE: first 64 of 128 head dims
    rope_theta: float = 5_000_000.0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 200_064
    max_position_embeddings: int = 196_608

    # TTNN dtypes
    # 228B params / 32 devices = 7.125B params/device
    # At BF16: 14.25GB (doesn't fit in 12GB DRAM)
    # At BF8:  7.125GB (fits!)
    # At BF4:  3.56GB  (fits, lower accuracy)
    weight_dtype: ttnn.DataType = field(default=ttnn.bfloat8_b)
    act_dtype: ttnn.DataType = field(default=ttnn.bfloat16)

    # Memory configs
    dram_mem: ttnn.MemoryConfig = field(default=ttnn.DRAM_MEMORY_CONFIG)
    l1_mem: ttnn.MemoryConfig = field(default=ttnn.L1_MEMORY_CONFIG)

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


def make_tt_config(num_hidden_layers: Optional[int] = None, **kwargs) -> MiniMaxM2TTConfig:
    """Build a MiniMaxM2TTConfig, optionally overriding num_hidden_layers for fast tests."""
    cfg = MiniMaxM2TTConfig(**kwargs)
    if num_hidden_layers is not None:
        cfg.num_hidden_layers = num_hidden_layers
    return cfg


def make_mesh_config(mesh_device) -> MeshConfig:
    """
    Create MeshConfig for MiniMax-M2.5 on Galaxy (8,4) mesh.

    Galaxy mesh_device.shape = (8, 4):
      - rows=8  → EP axis (expert parallelism)
      - cols=4  → TP axis (tensor parallelism)

    Decode:  EP=8, TP=4, SP=1
    Prefill: EP=1, TP=4, SP=8
    """
    rows, cols = mesh_device.shape
    return MeshConfig(
        mesh_shape=mesh_device.shape,
        decode=ModeConfig(tp=cols, ep=rows, sp=1),
    )


def make_paged_attention_config(
    max_seq_len: int = 32768,
    block_size: int = 64,
) -> PagedAttentionConfig:
    """
    Create PagedAttentionConfig for MiniMax-M2.5.

    Args:
        max_seq_len: Maximum sequence length to support (default 32k)
        block_size: Tokens per KV cache block (default 64, must be tile-aligned)

    Returns:
        PagedAttentionConfig with computed max_num_blocks
    """
    # Compute blocks needed for max_seq_len
    max_num_blocks = (max_seq_len + block_size - 1) // block_size
    return PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)


def get_weight_cache_path(
    model_path: str = None,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
) -> Path:
    """
    Get the weight cache path for MiniMax-M2.5.

    Weight caching stores pre-converted TTNN tensors on disk to speed up
    model loading on subsequent runs (first run: ~50 min, cached: ~5 min).

    The cache path is determined by:
    1. TT_CACHE_PATH env var (if set)
    2. ~/.cache/ttnn/minimax_m2/<dtype>/

    Args:
        model_path: Path to model weights (used to create unique cache dir)
        dtype: Weight dtype (bfloat8_b or bfloat16)

    Returns:
        Path to weight cache directory
    """
    dtype_str = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}.get(dtype, "bf16")

    # Check for user-specified cache path
    cache_root = os.environ.get("TT_CACHE_PATH")
    if cache_root:
        cache_path = Path(cache_root) / "minimax_m2" / dtype_str
    else:
        cache_path = Path.home() / ".cache" / "ttnn" / "minimax_m2" / dtype_str

    # Create cache directory if it doesn't exist
    cache_path.mkdir(parents=True, exist_ok=True)

    return cache_path
