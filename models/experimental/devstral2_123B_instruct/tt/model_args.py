# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Devstral-2 / Ministral3 TT stack (shapes, mesh, dtypes, RoPE, KV paging)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import ttnn

DEVSTRAL2_LARGE_L1_SMALL_SIZE = 24576


@dataclass
class RopeParameters:
    """Subset of HF ``rope_parameters`` used by the TT RoPE module."""

    rope_theta: float = 1_000_000.0
    rope_type: str = "yarn"
    factor: float = 16.0
    original_max_position_embeddings: int = 16_384
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 1.0
    llama_4_scaling_beta: float = 0.1

    @classmethod
    def from_hf_dict(cls, d: dict) -> "RopeParameters":
        if d is None:
            return cls()
        # HF uses ``type`` (and some snapshots ``rope_type``); accept either.
        rt = d.get("rope_type", d.get("type", "yarn"))
        return cls(
            rope_theta=float(d.get("rope_theta", 1_000_000.0)),
            rope_type=str(rt),
            factor=float(d.get("factor", 16.0)),
            original_max_position_embeddings=int(d.get("original_max_position_embeddings", 16_384)),
            beta_fast=float(d.get("beta_fast", 32.0)),
            beta_slow=float(d.get("beta_slow", 1.0)),
            mscale=float(d.get("mscale", 1.0)),
            mscale_all_dim=float(d.get("mscale_all_dim", 1.0)),
            llama_4_scaling_beta=float(d.get("llama_4_scaling_beta", 0.0)),
        )


@dataclass
class Devstral2Args:
    """Devstral-2 / Ministral3 args for the TT rewrite. Mesh-aware (TP along ``cluster_axis``)."""

    # Architecture (HF Ministral3Config field names where applicable).
    hidden_size: int = 12_288
    num_hidden_layers: int = 88
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 28_672
    vocab_size: int = 131_072
    max_position_embeddings: int = 262_144
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = False
    pad_token_id: Optional[int] = 11

    # RoPE + Llama-4 query scaling.
    rope: RopeParameters = field(default_factory=RopeParameters)

    # Runtime / mesh.
    mesh_shape: tuple[int, int] = (1, 4)  # Quietbox 4-Blackhole default.
    cluster_axis: int = 1  # axis along which we tensor-parallelize (the N in 1xN).
    max_batch_size: int = 1
    max_seq_len: int = 4096  # Working KV-cache budget (separate from max_position_embeddings).

    # Dtypes.
    weight_dtype: ttnn.DataType = ttnn.bfloat16
    activation_dtype: ttnn.DataType = ttnn.bfloat16
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat16
    ccl_dtype: ttnn.DataType = ttnn.bfloat16

    # CCL topology for ttnn.experimental.* collectives.
    ccl_topology: ttnn.Topology = ttnn.Topology.Linear

    # When True, route prefill activations through DRAM (see ``mem_config.get_activation_mem_config``).
    prefill_activations_dram: bool = False

    # Paged KV cache: physical block size in tokens. ``max_seq_len`` must be a multiple.
    # 128 keeps each chunked-prefill dispatch within the sharded-LayerNorm L1 budget; smaller
    # blocks reduce wasted slots when sequences don't fill the last block but raise page_table size.
    kv_block_size: int = 128

    def __post_init__(self) -> None:
        if self.kv_block_size <= 0:
            raise ValueError(f"kv_block_size must be positive, got {self.kv_block_size}")
        if self.max_seq_len % self.kv_block_size != 0:
            raise ValueError(
                f"max_seq_len ({self.max_seq_len}) must be a multiple of kv_block_size " f"({self.kv_block_size})"
            )

    @classmethod
    def from_hf_config(
        cls,
        hf_config,
        *,
        mesh_shape: tuple[int, int] = (1, 4),
        cluster_axis: int = 1,
        max_batch_size: int = 1,
        max_seq_len: int = 4096,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        kv_cache_dtype: ttnn.DataType = ttnn.bfloat16,
        ccl_dtype: ttnn.DataType = ttnn.bfloat16,
        prefill_activations_dram: bool = False,
        kv_block_size: int = 128,
    ) -> "Devstral2Args":
        """Build from a HuggingFace ``Ministral3Config`` (or the inner ``text_config`` of a wrapper)."""
        text = getattr(hf_config, "text_config", None) or hf_config
        rope_params = getattr(text, "rope_parameters", None)
        if rope_params is None:
            rope_params = {}
        # Some HF configs surface a class instance; both behave as dicts via getattr.
        if not isinstance(rope_params, dict):
            rope_params = {
                k: getattr(rope_params, k)
                for k in (
                    "rope_type",
                    "rope_theta",
                    "factor",
                    "original_max_position_embeddings",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                    "llama_4_scaling_beta",
                )
                if hasattr(rope_params, k)
            }
        return cls(
            hidden_size=int(text.hidden_size),
            num_hidden_layers=int(text.num_hidden_layers),
            num_attention_heads=int(text.num_attention_heads),
            num_key_value_heads=int(getattr(text, "num_key_value_heads", text.num_attention_heads)),
            head_dim=int(getattr(text, "head_dim", text.hidden_size // text.num_attention_heads)),
            intermediate_size=int(text.intermediate_size),
            vocab_size=int(text.vocab_size),
            max_position_embeddings=int(getattr(text, "max_position_embeddings", 262_144)),
            rms_norm_eps=float(getattr(text, "rms_norm_eps", 1e-5)),
            hidden_act=str(getattr(text, "hidden_act", "silu")),
            sliding_window=getattr(text, "sliding_window", None),
            tie_word_embeddings=bool(getattr(text, "tie_word_embeddings", False)),
            pad_token_id=getattr(text, "pad_token_id", None),
            rope=RopeParameters.from_hf_dict(rope_params),
            mesh_shape=mesh_shape,
            cluster_axis=cluster_axis,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            kv_cache_dtype=kv_cache_dtype,
            ccl_dtype=ccl_dtype,
            prefill_activations_dram=prefill_activations_dram,
            kv_block_size=kv_block_size,
        )

    @property
    def num_devices(self) -> int:
        return self.mesh_shape[0] * self.mesh_shape[1]

    @property
    def tp(self) -> int:
        """Tensor-parallel degree along ``cluster_axis``."""
        return self.mesh_shape[self.cluster_axis]

    @property
    def n_local_heads(self) -> int:
        if self.num_attention_heads % self.tp != 0:
            raise ValueError(f"num_attention_heads={self.num_attention_heads} not divisible by TP={self.tp}")
        return self.num_attention_heads // self.tp

    @property
    def n_local_kv_heads(self) -> int:
        if self.num_key_value_heads % self.tp != 0:
            raise ValueError(f"num_key_value_heads={self.num_key_value_heads} not divisible by TP={self.tp}")
        return self.num_key_value_heads // self.tp

    @property
    def n_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def kv_num_blocks_per_user(self) -> int:
        """Logical blocks per user (``max_seq_len // kv_block_size``)."""
        return self.max_seq_len // self.kv_block_size

    @property
    def kv_page_table_blocks_per_user(self) -> int:
        """Page-table columns and per-user physical KV blocks.

        Flexible chunked SDPA requires the page-table row width (int32 columns) to be a
        multiple of 8 so the stick size is divisible by 32 bytes. ``paged_update_cache`` also
        requires ``page_table.shape[1] <= k_cache.shape[0]``, so the KV cache must allocate at
        least this many physical blocks (extra slots stay unused for typical ``max_seq_len``).
        """
        n = self.kv_num_blocks_per_user
        return ((n + 7) // 8) * 8

    @property
    def kv_num_total_blocks(self) -> int:
        """Total physical blocks in the paged KV cache (``batch * page_table_blocks``)."""
        return self.max_batch_size * self.kv_page_table_blocks_per_user

    @property
    def supports_flexible_chunked_sdpa(self) -> bool:
        """Whether ``chunk_start_idx_tensor`` (flexible chunked SDPA) is safe to use.

        Flexible SDPA sets the attention span to ``Sq + page_table_cols * block_size``. If the
        page table was widened only for 32-byte stick alignment (cols > logical blocks), that
        span is too large and reads uninitialized KV — use legacy ``start_pos`` instead.
        """
        return self.kv_page_table_blocks_per_user == self.kv_num_blocks_per_user

    @property
    def q_proj_in_features(self) -> int:
        return self.hidden_size

    @property
    def q_proj_out_features(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_out_features(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def attn_scale(self) -> float:
        return self.head_dim**-0.5

    def get_activation_mem_config(self, mode: str, mesh_device) -> ttnn.MemoryConfig:
        """Prefill: L1 interleaved (DRAM on BH). Decode: L1 width-sharded (see ``mem_config``)."""
        from models.experimental.devstral2_123B_instruct.tt.mem_config import get_activation_mem_config

        return get_activation_mem_config(self, mode, mesh_device)

    def get_ccl_output_mem_config(self, mode: str, mesh_device) -> ttnn.MemoryConfig:
        """Where all-reduce should leave activations."""
        return self.get_activation_mem_config(mode, mesh_device)

    def get_weight_cache_path(self, *, num_layers: Optional[int] = None) -> str:
        """Default on-disk cache for tiled TTNN weights (see ``weight_loading.resolve_weight_cache_path``)."""
        from models.experimental.devstral2_123B_instruct.tt.weight_loading import resolve_weight_cache_path

        path = resolve_weight_cache_path(None, self, num_layers=num_layers)
        if path is None:
            raise RuntimeError("Weight cache disabled (DEVSTRAL2_DISABLE_WEIGHT_CACHE=1)")
        return path

    def state_dict_prefix(self, module: str = "", layer_idx: Optional[int] = None) -> str:
        """Mirror HF state dict layout: ``model.layers.<i>.<module>.`` (or ``model.<module>.``)."""
        parts = ["model"]
        if layer_idx is not None:
            parts.append(f"layers.{layer_idx}")
        if module:
            parts.append(module)
        return ".".join(parts) + "."


def is_blackhole_mesh(mesh_device) -> bool:
    """True on Blackhole / BH Galaxy meshes (used for kernel and program-config selection)."""
    try:
        if mesh_device is not None and hasattr(mesh_device, "arch"):
            return mesh_device.arch() == ttnn.device.Arch.BLACKHOLE
    except Exception:
        pass
    try:
        return ttnn.device.is_blackhole(mesh_device)
    except Exception:
        pass
    from models.common.utility_functions import is_blackhole

    return is_blackhole()


def torch_default_dtype_for(t_dtype: ttnn.DataType) -> torch.dtype:
    """Pick a torch dtype that round-trips through ``ttnn.from_torch`` cleanly."""
    if t_dtype == ttnn.bfloat16:
        return torch.bfloat16
    if t_dtype == ttnn.float32:
        return torch.float32
    # BFP8 / BFP4 / etc. are packed device-side from a bf16 host tensor.
    return torch.bfloat16
