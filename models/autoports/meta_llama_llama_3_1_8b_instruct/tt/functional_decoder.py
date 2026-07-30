# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functional TTNN decoder layer for meta-llama/Llama-3.1-8B-Instruct.

This module implements one HuggingFace ``LlamaDecoderLayer`` on a single 1x1
``ttnn.MeshDevice``. Setup-time helpers may consume PyTorch tensors from a
HuggingFace state dict, but ``prefill_forward`` and ``decode_forward`` are
TTNN-only hot paths: no torch, no ``ttnn.from_torch``, and no ``ttnn.to_torch``.

Forward contract
----------------

``prefill_forward(hidden_states, rot_mats, page_table, user_id=0,
chunk_page_table=None, chunk_start_idx=None)``
    ``hidden_states`` is a TTNN tensor shaped ``[1, 1, seq_len, 4096]`` in tile
    layout. ``rot_mats`` is ``(cos, sin)`` for the same sequence positions,
    each shaped ``[1, 1, seq_len, 128]``. ``page_table`` maps virtual KV blocks
    to physical blocks for paged cache fill.

``decode_forward(hidden_states, current_pos, rot_mats, page_table)``
    ``hidden_states`` is a TTNN tensor shaped ``[1, 1, batch, 4096]`` in tile
    layout. ``current_pos`` is a TTNN tensor shaped ``[batch]`` containing the
    absolute decode position for each user. ``rot_mats`` is ``(cos, sin)`` for
    those positions, each shaped ``[1, batch, 1, 128]`` as expected by TTNN's
    Llama rotary decode kernel. ``page_table`` has the same paged-cache mapping
    used during prefill.

Both methods return a TTNN tensor shaped like ``hidden_states``.
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
from models.common.tensor_utils import TILE_SIZE


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_DIR_NAME = "meta_llama_llama_3_1_8b_instruct"


@dataclass(frozen=True)
class PagedAttentionConfig:
    """Minimal paged-attention config consumed by TTNN SDPA/cache kernels."""

    block_size: int = 64
    max_num_blocks: int = 2048


def _require_llama31_8b_config(hf_config: Any) -> None:
    expected = {
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-5,
        "attention_bias": False,
        "mlp_bias": False,
        "hidden_act": "silu",
    }
    for name, expected_value in expected.items():
        actual = getattr(hf_config, name, None)
        if actual != expected_value:
            raise ValueError(f"{MODEL_ID} functional decoder expects {name}={expected_value!r}, got {actual!r}")


def _layer_prefix(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}"


def _get_layer_tensor(state_dict: dict[str, torch.Tensor], layer_idx: int, suffix: str) -> torch.Tensor:
    canonical = f"{_layer_prefix(layer_idx)}.{suffix}"
    if canonical in state_dict:
        return state_dict[canonical]
    if suffix in state_dict:
        return state_dict[suffix]
    raise KeyError(f"Missing HF decoder tensor {canonical!r} or layer-local key {suffix!r}")


def _reverse_permute(tensor: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
    """Convert HuggingFace Q/K projection weights to Meta/TTNN RoPE head order."""

    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def _as_lazy_weight(
    source: torch.Tensor,
    *,
    mesh_device: ttnn.MeshDevice,
    dtype: ttnn.DataType,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    cache_dir_weight_name: tuple[Path, str] | None = None,
) -> LazyWeight:
    return LazyWeight(
        source=source,
        dtype=dtype,
        device=mesh_device,
        layout=layout,
        memory_config=memory_config,
        cache_dir_weight_name=cache_dir_weight_name,
    )


def _norm_weight(
    state_dict: dict[str, torch.Tensor],
    *,
    hf_config: Any,
    layer_idx: int,
    name: str,
    mesh_device: ttnn.MeshDevice,
    cache_dir: Path | None,
) -> RMSNorm1D:
    weight = _get_layer_tensor(state_dict, layer_idx, f"{name}.weight")
    source = weight.reshape(1, 1, hf_config.hidden_size // TILE_SIZE, TILE_SIZE)
    lazy_weight = _as_lazy_weight(
        source,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        cache_dir_weight_name=(cache_dir, name) if cache_dir is not None else None,
    )
    return RMSNorm1D.from_config(
        RMSNorm1DConfig(
            weight=lazy_weight,
            mesh_device=mesh_device,
            eps=hf_config.rms_norm_eps,
            decode_in_sharded=False,
            decode_out_sharded=False,
            prefill_distributed=False,
        )
    )


class _FunctionalMLP(LightweightModule):
    """Autoport-local Llama SwiGLU MLP using direct TTNN ops."""

    def __init__(
        self,
        *,
        gate: LazyWeight,
        up: LazyWeight,
        down: LazyWeight,
        activation_dtype: ttnn.DataType,
    ) -> None:
        super().__init__()
        self.gate_lazy = gate
        self.up_lazy = up
        self.down_lazy = down
        self.activation_dtype = activation_dtype
        self._loaded = False

    def load_device_weights(self) -> None:
        if self._loaded:
            return
        self.gate = self.gate_lazy.get_device_weight()
        self.up = self.up_lazy.get_device_weight()
        self.down = self.down_lazy.get_device_weight()
        self._loaded = True

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self.load_device_weights()

        gate = ttnn.linear(
            x,
            self.gate,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self.up,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.mul(gate, up, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        out = ttnn.linear(
            hidden,
            self.down,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)
        return out


class FunctionalDecoder(LightweightModule):
    """Single-layer TTNN implementation of the target HF Llama decoder."""

    def __init__(
        self,
        *,
        input_layernorm: RMSNorm1D,
        self_attn: Attention1D,
        post_attention_layernorm: RMSNorm1D,
        mlp: _FunctionalMLP,
    ) -> None:
        super().__init__()
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        page_block_size: int = 64,
        max_num_blocks: int | None = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        kv_cache_dtype: ttnn.DataType = ttnn.bfloat16,
        cache_dir: str | Path | None = None,
        **kwargs,
    ) -> "FunctionalDecoder":
        """Build a functional decoder from canonical HF decoder-layer weights.

        ``state_dict`` may be a full HuggingFace model state dict or a layer-local
        dict whose keys omit ``model.layers.<layer_idx>.``. Weight conversion and
        all ``ttnn.from_torch`` work are setup-time only.
        """

        if kwargs:
            raise TypeError(f"Unexpected FunctionalDecoder.from_state_dict kwargs: {sorted(kwargs)}")
        _require_llama31_8b_config(hf_config)
        if mesh_device.get_num_devices() != 1:
            raise ValueError("FunctionalDecoder is the single-chip stage; use a 1x1 MeshDevice.")

        max_seq_len = int(max_seq_len or hf_config.max_position_embeddings)
        if max_num_blocks is None:
            max_num_blocks = max(1, (max_batch_size * max_seq_len + page_block_size - 1) // page_block_size)
        paged_attention_config = PagedAttentionConfig(
            block_size=page_block_size,
            max_num_blocks=max_num_blocks,
        )
        cache_path = Path(cache_dir) if cache_dir is not None else None

        dim = hf_config.hidden_size
        head_dim = hf_config.head_dim
        n_heads = hf_config.num_attention_heads
        n_kv_heads = hf_config.num_key_value_heads
        q_size = n_heads * head_dim
        kv_size = n_kv_heads * head_dim
        qkv_size = q_size + 2 * kv_size

        wq_raw = _get_layer_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        wk_raw = _get_layer_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        wv_raw = _get_layer_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        wo_raw = _get_layer_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")

        wq = _reverse_permute(wq_raw, n_heads, q_size, dim).transpose(-2, -1)
        wk = _reverse_permute(wk_raw, n_kv_heads, kv_size, dim).transpose(-2, -1)
        wv = wv_raw.transpose(-2, -1)
        wqkv = torch.cat([wq, wk, wv], dim=-1).unsqueeze(0).unsqueeze(0)
        wo = wo_raw.transpose(-2, -1).unsqueeze(0).unsqueeze(0)

        lazy_wqkv = LazyWeight(
            source=wqkv,
            dtype=weight_dtype,
            cache_dir_weight_name=(cache_path, "self_attn_wqkv") if cache_path is not None else None,
        )
        lazy_wo = LazyWeight(
            source=wo,
            dtype=weight_dtype,
            cache_dir_weight_name=(cache_path, "self_attn_wo") if cache_path is not None else None,
        )
        self_attn = Attention1D.from_config(
            Attention1DConfig(
                wqkv=lazy_wqkv,
                wo=lazy_wo,
                mesh_device=mesh_device,
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                qkv_size=qkv_size,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                paged_attention_config=paged_attention_config,
                kv_cache_dtype=kv_cache_dtype,
                wqkv_dtype=weight_dtype,
                wo_dtype=weight_dtype,
                activation_dtype=activation_dtype,
                scale=head_dim**-0.5,
            )
        )

        input_layernorm = _norm_weight(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            name="input_layernorm",
            mesh_device=mesh_device,
            cache_dir=cache_path,
        )
        post_attention_layernorm = _norm_weight(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            name="post_attention_layernorm",
            mesh_device=mesh_device,
            cache_dir=cache_path,
        )

        gate = _get_layer_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        up = _get_layer_tensor(state_dict, layer_idx, "mlp.up_proj.weight").transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        down = _get_layer_tensor(state_dict, layer_idx, "mlp.down_proj.weight").transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        mlp = _FunctionalMLP(
            gate=LazyWeight(
                source=gate,
                dtype=weight_dtype,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_dir_weight_name=(cache_path, "mlp_gate") if cache_path is not None else None,
            ),
            up=LazyWeight(
                source=up,
                dtype=weight_dtype,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_dir_weight_name=(cache_path, "mlp_up") if cache_path is not None else None,
            ),
            down=LazyWeight(
                source=down,
                dtype=weight_dtype,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_dir_weight_name=(cache_path, "mlp_down") if cache_path is not None else None,
            ),
            activation_dtype=activation_dtype,
        )

        return cls(
            input_layernorm=input_layernorm,
            self_attn=self_attn,
            post_attention_layernorm=post_attention_layernorm,
            mlp=mlp,
        )

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
        residual = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self.input_layernorm.prefill_forward(residual)
        hidden_states = self.self_attn.prefill_forward(
            hidden_states,
            rot_mats=rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
        )
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm.prefill_forward(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        residual = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self.input_layernorm.decode_forward(residual)
        hidden_states = ttnn.to_memory_config(hidden_states, self.self_attn.config.decode_input_memcfg)
        hidden_states = self.self_attn.decode_forward(
            hidden_states,
            current_pos=current_pos,
            rot_mats=rot_mats,
            page_table=page_table,
        )
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm.decode_forward(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str, **kwargs) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"Unknown decoder mode {mode!r}; expected 'prefill' or 'decode'.")
