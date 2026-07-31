# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Functional TTNN decoder layer for meta-llama/Llama-3.2-1B-Instruct.

Runtime contract
----------------
This module implements one HuggingFace ``LlamaDecoderLayer`` kind:

* hidden size 2048, 32 query heads, 8 KV heads, head dim 64
* RMSNorm epsilon 1e-5
* attention order: norm -> self-attention -> residual
* MLP order: norm -> down_proj(silu(gate_proj(x)) * up_proj(x)) -> residual
* no attention or MLP biases
* paged KV cache for both prefill and decode

``from_state_dict`` is the weight-loading boundary. It accepts HF-format layer
weights, converts Q/K weights to the Meta RoPE layout expected by TTNN, creates
the paged KV cache, and materializes all TTNN tensors. The hot ``prefill_forward``
and ``decode_forward`` paths must receive TTNN tensors and do not use torch,
``ttnn.from_torch``, or ``ttnn.to_torch``.

Forward signatures
------------------
``prefill_forward(hidden_states, *, rot_mats, page_table, user_id=0,
chunk_page_table=None, chunk_start_idx=None)``

* ``hidden_states``: TTNN tensor, shape ``[1, 1, seq_len, 2048]``, tile layout.
  ``seq_len`` must be positive and divisible by 128.
* ``rot_mats``: ``(cos, sin)`` TTNN tensors in Meta/Llama format for positions
  covered by the prefill chunk.
* ``page_table``: int32 TTNN tensor, shape
  ``[batch, max_blocks_per_sequence]``. For this single-layer functional
  contract, prefill normally uses ``batch=1``.
* ``chunk_page_table`` and ``chunk_start_idx`` select a paged prefill chunk when
  a long prompt is split by the caller.

``decode_forward(hidden_states, *, current_pos, rot_mats, page_table)``

* ``hidden_states``: TTNN tensor, shape ``[1, 1, batch, 2048]``. The logical
  batch can be smaller than 32; TTNN decode kernels tile-pad the batch dimension.
* ``current_pos``: int32 TTNN tensor with one current token position per user.
  This tensor is part of the traced decode state and is passed directly to
  ``paged_update_cache`` and paged SDPA.
* ``rot_mats``: ``(cos, sin)`` TTNN tensors for ``current_pos``.
* ``page_table``: int32 TTNN tensor used for both the decode KV update and SDPA
  cache lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


@dataclass(frozen=True)
class PagedAttentionConfig:
    """Minimal config object consumed by ``Attention1DConfig``."""

    block_size: int = 64
    max_num_blocks: int = 2048


def _reverse_permute(tensor: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
    """Convert HuggingFace Q/K weights to the Meta RoPE layout used by TTNN."""

    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def _layer_prefix(state_dict: dict[str, torch.Tensor], layer_idx: int) -> str:
    prefix = f"model.layers.{layer_idx}."
    if any(key.startswith(prefix) for key in state_dict):
        return prefix
    return ""


def _state_tensor(state_dict: dict[str, torch.Tensor], prefix: str, name: str) -> torch.Tensor:
    key = prefix + name
    if key not in state_dict:
        raise KeyError(f"missing required decoder weight: {key}")
    return state_dict[key].detach().to(torch.bfloat16).cpu()


def _as_replicated_tt(
    tensor: torch.Tensor,
    *,
    mesh_device: ttnn.MeshDevice,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _mesh_shape_tuple(mesh_device: ttnn.MeshDevice) -> tuple[int, int]:
    shape = tuple(mesh_device.shape)
    if len(shape) != 2:
        raise ValueError(f"expected a 2D mesh shape, got {shape}")
    return int(shape[0]), int(shape[1])


class _LlamaMLP(LightweightModule):
    """Local Llama MLP path kept independent from missing legacy imports."""

    def __init__(
        self,
        *,
        gate_weight: ttnn.Tensor,
        up_weight: ttnn.Tensor,
        down_weight: ttnn.Tensor,
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()
        self.gate_weight = gate_weight
        self.up_weight = up_weight
        self.down_weight = down_weight
        self.mesh_device = mesh_device

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        prefix: str,
        mesh_device: ttnn.MeshDevice,
    ) -> "_LlamaMLP":
        gate = _state_tensor(state_dict, prefix, "mlp.gate_proj.weight").transpose(0, 1).contiguous()
        up = _state_tensor(state_dict, prefix, "mlp.up_proj.weight").transpose(0, 1).contiguous()
        down = _state_tensor(state_dict, prefix, "mlp.down_proj.weight").transpose(0, 1).contiguous()

        return cls(
            gate_weight=_as_replicated_tt(gate.unsqueeze(0).unsqueeze(0), mesh_device=mesh_device),
            up_weight=_as_replicated_tt(up.unsqueeze(0).unsqueeze(0), mesh_device=mesh_device),
            down_weight=_as_replicated_tt(down.unsqueeze(0).unsqueeze(0), mesh_device=mesh_device),
            mesh_device=mesh_device,
        )

    def _forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        if hidden_states.is_sharded():
            hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        else:
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

        gate = ttnn.linear(hidden_states, self.gate_weight, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.linear(hidden_states, self.up_weight, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        activated = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        fused = ttnn.mul(activated, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.linear(fused, self.down_weight, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out

    def prefill_forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        return self._forward(hidden_states)

    def decode_forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        return self._forward(hidden_states)


class FunctionalDecoder(LightweightModule):
    """Single Llama-3.2-1B decoder layer with paged prefill/decode TTNN paths."""

    def __init__(
        self,
        *,
        attention_norm: RMSNorm1D,
        attention: Attention1D,
        post_attention_norm: RMSNorm1D,
        mlp: _LlamaMLP,
        decode_residual_memcfg: ttnn.MemoryConfig,
        mesh_device: ttnn.MeshDevice,
        hf_config: Any,
        layer_idx: int,
        page_block_size: int,
        max_seq_len: int,
        max_batch_size: int,
    ) -> None:
        super().__init__()
        self.attention_norm = attention_norm
        self.attention = attention
        self.post_attention_norm = post_attention_norm
        self.mlp = mlp
        self.decode_residual_memcfg = decode_residual_memcfg
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.page_block_size = page_block_size
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        page_block_size: int = 64,
        max_seq_len: int | None = None,
        max_batch_size: int = 1,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        kv_cache_dtype: ttnn.DataType = ttnn.bfloat16,
        cache_path: str | Path | None = None,
        materialize: bool = True,
        **_: Any,
    ) -> "FunctionalDecoder":
        """Create the TTNN decoder from canonical HF layer weights.

        ``state_dict`` may be either a full HF model state dict with
        ``model.layers.{layer_idx}.`` keys or a layer-local dict whose keys start
        at ``self_attn.*`` / ``mlp.*``.
        """

        hidden_size = int(hf_config.hidden_size)
        n_heads = int(hf_config.num_attention_heads)
        n_kv_heads = int(getattr(hf_config, "num_key_value_heads", n_heads))
        head_dim = int(getattr(hf_config, "head_dim", hidden_size // n_heads) or (hidden_size // n_heads))
        if hidden_size != 2048 or n_heads != 32 or n_kv_heads != 8 or head_dim != 64:
            raise ValueError(
                f"{MODEL_ID} functional decoder expected hidden=2048 heads=32 kv_heads=8 head_dim=64, "
                f"got hidden={hidden_size} heads={n_heads} kv_heads={n_kv_heads} head_dim={head_dim}"
            )
        if bool(getattr(hf_config, "attention_bias", False)) or bool(getattr(hf_config, "mlp_bias", False)):
            raise ValueError(f"{MODEL_ID} functional decoder only supports bias-free Llama layers")

        max_seq_len = int(max_seq_len or hf_config.max_position_embeddings)
        prefix = _layer_prefix(state_dict, layer_idx)
        cache_dir = Path(cache_path) if cache_path is not None else None
        cache_prefix = f"layer{layer_idx}"

        q_raw = _state_tensor(state_dict, prefix, "self_attn.q_proj.weight")
        k_raw = _state_tensor(state_dict, prefix, "self_attn.k_proj.weight")
        v_raw = _state_tensor(state_dict, prefix, "self_attn.v_proj.weight")
        o_raw = _state_tensor(state_dict, prefix, "self_attn.o_proj.weight")

        q_meta = _reverse_permute(q_raw, n_heads, n_heads * head_dim, hidden_size).transpose(0, 1).contiguous()
        k_meta = _reverse_permute(k_raw, n_kv_heads, n_kv_heads * head_dim, hidden_size).transpose(0, 1).contiguous()
        v_meta = v_raw.transpose(0, 1).contiguous()
        o_tt = o_raw.transpose(0, 1).contiguous()
        wqkv = torch.cat([q_meta, k_meta, v_meta], dim=-1).unsqueeze(0).unsqueeze(0)
        wo = o_tt.unsqueeze(0).unsqueeze(0)

        def lazy_weight(name: str, tensor: torch.Tensor, dtype: ttnn.DataType) -> LazyWeight:
            return LazyWeight(
                source=tensor,
                dtype=dtype,
                cache_dir_weight_name=(cache_dir, f"{cache_prefix}_{name}") if cache_dir is not None else None,
            )

        norm_eps = float(hf_config.rms_norm_eps)
        attention_norm = RMSNorm1D.from_config(
            RMSNorm1DConfig(
                weight=lazy_weight("input_layernorm", _state_tensor(state_dict, prefix, "input_layernorm.weight"), ttnn.bfloat16),
                mesh_device=mesh_device,
                eps=norm_eps,
                max_batch_size=max_batch_size,
                prefill_distributed=False,
            )
        )
        post_attention_norm = RMSNorm1D.from_config(
            RMSNorm1DConfig(
                weight=lazy_weight(
                    "post_attention_layernorm",
                    _state_tensor(state_dict, prefix, "post_attention_layernorm.weight"),
                    ttnn.bfloat16,
                ),
                mesh_device=mesh_device,
                eps=norm_eps,
                max_batch_size=max_batch_size,
                prefill_distributed=False,
            )
        )

        blocks_per_user = (max_seq_len + page_block_size - 1) // page_block_size
        paged_attention_config = PagedAttentionConfig(
            block_size=page_block_size,
            max_num_blocks=blocks_per_user * max_batch_size,
        )
        attention = Attention1D.from_config(
            Attention1DConfig(
                wqkv=lazy_weight("wqkv", wqkv, weight_dtype),
                wo=lazy_weight("wo", wo, weight_dtype),
                mesh_device=mesh_device,
                dim=hidden_size,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                paged_attention_config=paged_attention_config,
                kv_cache_dtype=kv_cache_dtype,
                activation_dtype=ttnn.bfloat16,
            )
        )

        mlp = _LlamaMLP.from_state_dict(state_dict, prefix=prefix, mesh_device=mesh_device)
        decode_residual_memcfg = attention.config.decode_residual_memcfg
        decoder = cls(
            attention_norm=attention_norm,
            attention=attention,
            post_attention_norm=post_attention_norm,
            mlp=mlp,
            decode_residual_memcfg=decode_residual_memcfg,
            mesh_device=mesh_device,
            hf_config=hf_config,
            layer_idx=layer_idx,
            page_block_size=page_block_size,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        if materialize:
            decoder.load_device_weights()
        return decoder

    @property
    def kv_cache(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        self.attention.load_device_weights()
        return self.attention.kv_cache

    @property
    def decode_input_memcfg(self) -> ttnn.MemoryConfig:
        return self.attention.config.decode_input_memcfg

    def load_device_weights(self) -> None:
        self.attention_norm.load_device_weights()
        self.attention.load_device_weights()
        self.post_attention_norm.load_device_weights()

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor,
        user_id: int = 0,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
    ) -> ttnn.Tensor:
        residual = hidden_states
        normed = self.attention_norm.prefill_forward(hidden_states)
        attn_out = self.attention.prefill_forward(
            normed,
            rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
        )
        hidden_states = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        residual = hidden_states
        normed = self.post_attention_norm.prefill_forward(hidden_states)
        mlp_out = self.mlp.prefill_forward(normed)
        return ttnn.add(residual, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        hidden_states = ttnn.to_memory_config(hidden_states, self.decode_residual_memcfg)
        residual = hidden_states
        normed = self.attention_norm.decode_forward(hidden_states)
        attn_out = self.attention.decode_forward(normed, current_pos, rot_mats, page_table=page_table)
        hidden_states = ttnn.add(residual, attn_out, memory_config=self.decode_residual_memcfg, dtype=ttnn.bfloat16)

        residual = hidden_states
        normed = self.post_attention_norm.decode_forward(hidden_states)
        mlp_out = self.mlp.decode_forward(normed)
        mlp_out = ttnn.to_memory_config(mlp_out, self.decode_residual_memcfg)
        return ttnn.add(residual, mlp_out, memory_config=self.decode_residual_memcfg, dtype=ttnn.bfloat16)

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str, **kwargs: Any) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")

