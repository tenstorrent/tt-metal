# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Dense 1x1 translation of the pre-generated Qwen3-32B EmitPy graphs.

The source TTNN->EmitPy packages are tensor-parallel over a 1x4 mesh. This
correctness-first functional layer preserves their prefill and decode math while
collapsing sharded projections and collectives to dense matmuls over canonical,
unsharded Hugging Face weights.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

import ttnn
from models.common.lightweightmodule import LightweightModule

HF_MODEL = "Qwen/Qwen3-32B"
EMITTED_BATCH = 32
EMITTED_PREFILL_SEQUENCE = 17
EMITTED_CACHE_LENGTH = 128
REPRESENTATIVE_LAYER = 32
TP_DEGREE = 4


def _config_value(config, name: str):
    value = getattr(config, name, None)
    if value is None and getattr(config, "text_config", None) is not None:
        value = getattr(config.text_config, name, None)
    return value


def _state_tensor(state_dict: Mapping[str, torch.Tensor], layer_idx: int, suffix: str) -> torch.Tensor:
    candidates = (
        f"model.layers.{layer_idx}.{suffix}",
        f"model.language_model.layers.{layer_idx}.{suffix}",
        f"language_model.model.layers.{layer_idx}.{suffix}",
        f"layers.{layer_idx}.{suffix}",
        suffix,
    )
    for key in candidates:
        if key in state_dict:
            return state_dict[key]
    raise KeyError(f"Missing Qwen3 decoder weight {suffix!r}; tried {candidates}")


def _to_device_tensor(tensor: torch.Tensor, mesh_device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class FunctionalDecoder(LightweightModule):
    """One Qwen3-32B decoder layer translated from the flat EmitPy graphs."""

    def __init__(
        self,
        *,
        mesh_device,
        layer_idx: int,
        batch: int,
        max_cache_len: int,
        hidden_size: int,
        attention_width: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        input_norm,
        post_attention_norm,
        q_norm,
        k_norm,
        qkv_weight,
        output_weight,
        gate_weight,
        up_weight,
        down_weight,
        rotary_cos,
        rotary_sin,
        position_indices,
    ):
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.batch = batch
        self.max_cache_len = max_cache_len
        self.hidden_size = hidden_size
        self.attention_width = attention_width
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.scale = 1.0 / math.sqrt(head_dim)

        self.input_norm = input_norm
        self.post_attention_norm = post_attention_norm
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.qkv_weight = qkv_weight
        self.output_weight = output_weight
        self.gate_weight = gate_weight
        self.up_weight = up_weight
        self.down_weight = down_weight
        self.rotary_cos = rotary_cos
        self.rotary_sin = rotary_sin
        self.position_indices = position_indices

        # Decode cache-update, SDPA, and head-concat kernels need a minimal L1
        # head layout. One 32-row shard holds one Q head across the emitted
        # batch; the compiler-selected grids/program configs are not retained.
        device_grid = mesh_device.compute_with_storage_grid_size()
        decode_grid = ttnn.num_cores_to_corerangeset(num_heads, device_grid, row_wise=True)
        cache_update_grid = ttnn.num_cores_to_corerangeset(batch, device_grid, row_wise=True)
        self.decode_compute_core_grid = decode_grid
        self.decode_heads_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(decode_grid, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.decode_kv_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cache_update_grid, [32, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.decode_concat_input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cache_update_grid, [num_heads, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int = EMITTED_BATCH,
        max_cache_len: int = EMITTED_CACHE_LENGTH,
        **kwargs,
    ) -> "FunctionalDecoder":
        """Load full HF weights and perform every host-side constant transform."""

        if kwargs:
            raise TypeError(f"Unsupported FunctionalDecoder kwargs: {sorted(kwargs)}")
        num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
        if num_devices != 1:
            raise ValueError(f"FunctionalDecoder requires a 1x1 mesh, got {num_devices} devices")
        if batch < 1 or batch > EMITTED_BATCH:
            raise ValueError(f"batch must be in [1, {EMITTED_BATCH}], got {batch}")
        if max_cache_len < 1:
            raise ValueError(f"max_cache_len must be positive, got {max_cache_len}")

        hidden_size = int(_config_value(hf_config, "hidden_size"))
        num_layers = int(_config_value(hf_config, "num_hidden_layers"))
        num_heads = int(_config_value(hf_config, "num_attention_heads"))
        num_kv_heads = int(_config_value(hf_config, "num_key_value_heads"))
        head_dim = int(_config_value(hf_config, "head_dim") or hidden_size // num_heads)
        intermediate_size = int(_config_value(hf_config, "intermediate_size"))
        rms_norm_eps = float(_config_value(hf_config, "rms_norm_eps"))
        advertised_context = int(_config_value(hf_config, "max_position_embeddings"))
        attention_width = num_heads * head_dim

        expected_contract = {
            "hidden_size": (hidden_size, 5120),
            "num_hidden_layers": (num_layers, 64),
            "num_attention_heads": (num_heads, 64),
            "num_key_value_heads": (num_kv_heads, 8),
            "head_dim": (head_dim, 128),
            "intermediate_size": (intermediate_size, 25600),
        }
        mismatches = [
            f"{name}={actual} (expected {expected})"
            for name, (actual, expected) in expected_contract.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError(f"{HF_MODEL} config does not match the translated emit: {', '.join(mismatches)}")
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"layer_idx must be in [0, {num_layers}), got {layer_idx}")
        if max_cache_len > advertised_context:
            raise ValueError(f"max_cache_len={max_cache_len} exceeds max_position_embeddings={advertised_context}")
        if str(_config_value(hf_config, "hidden_act")) != "silu":
            raise ValueError(
                f"The translated EmitPy graph requires hidden_act='silu', "
                f"got {_config_value(hf_config, 'hidden_act')!r}"
            )

        q = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.weight").to(torch.bfloat16)
        k = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.weight").to(torch.bfloat16)
        v = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.weight").to(torch.bfloat16)
        o = _state_tensor(state_dict, layer_idx, "self_attn.o_proj.weight").to(torch.bfloat16)
        gate = _state_tensor(state_dict, layer_idx, "mlp.gate_proj.weight").to(torch.bfloat16)
        up = _state_tensor(state_dict, layer_idx, "mlp.up_proj.weight").to(torch.bfloat16)
        down = _state_tensor(state_dict, layer_idx, "mlp.down_proj.weight").to(torch.bfloat16)
        q_norm = _state_tensor(state_dict, layer_idx, "self_attn.q_norm.weight").to(torch.bfloat16)
        k_norm = _state_tensor(state_dict, layer_idx, "self_attn.k_norm.weight").to(torch.bfloat16)
        input_norm = _state_tensor(state_dict, layer_idx, "input_layernorm.weight").to(torch.bfloat16)
        post_attention_norm = _state_tensor(state_dict, layer_idx, "post_attention_layernorm.weight").to(torch.bfloat16)

        expected_shapes = {
            "q": (attention_width, hidden_size),
            "k": (num_kv_heads * head_dim, hidden_size),
            "v": (num_kv_heads * head_dim, hidden_size),
            "o": (hidden_size, attention_width),
            "gate": (intermediate_size, hidden_size),
            "up": (intermediate_size, hidden_size),
            "down": (hidden_size, intermediate_size),
            "q_norm": (head_dim,),
            "k_norm": (head_dim,),
            "input_norm": (hidden_size,),
            "post_attention_norm": (hidden_size,),
        }
        tensors = {
            "q": q,
            "k": k,
            "v": v,
            "o": o,
            "gate": gate,
            "up": up,
            "down": down,
            "q_norm": q_norm,
            "k_norm": k_norm,
            "input_norm": input_norm,
            "post_attention_norm": post_attention_norm,
        }
        for name, expected in expected_shapes.items():
            if tuple(tensors[name].shape) != expected:
                raise ValueError(f"{name} weight has shape {tuple(tensors[name].shape)}, expected {expected}")

        # Both selected emits feed const-eval with [V, K, Q], reverse that list,
        # transpose each operand, and concatenate Q -> K -> V.
        qkv = torch.cat((q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)), dim=-1)

        rotary = Qwen3RotaryEmbedding(hf_config)
        positions = torch.arange(max_cache_len, dtype=torch.long).unsqueeze(0)
        rope_probe = torch.zeros((1, 1, max_cache_len, head_dim), dtype=torch.bfloat16)
        cos, sin = rotary(rope_probe, positions)
        cos = cos.to(torch.bfloat16).unsqueeze(1)
        sin = sin.to(torch.bfloat16).unsqueeze(1)

        return cls(
            mesh_device=mesh_device,
            layer_idx=layer_idx,
            batch=batch,
            max_cache_len=max_cache_len,
            hidden_size=hidden_size,
            attention_width=attention_width,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            input_norm=_to_device_tensor(input_norm, mesh_device),
            post_attention_norm=_to_device_tensor(post_attention_norm, mesh_device),
            q_norm=_to_device_tensor(q_norm, mesh_device),
            k_norm=_to_device_tensor(k_norm, mesh_device),
            qkv_weight=_to_device_tensor(qkv, mesh_device),
            output_weight=_to_device_tensor(o.transpose(0, 1), mesh_device),
            gate_weight=_to_device_tensor(gate.transpose(0, 1), mesh_device),
            up_weight=_to_device_tensor(up.transpose(0, 1), mesh_device),
            down_weight=_to_device_tensor(down.transpose(0, 1), mesh_device),
            rotary_cos=_to_device_tensor(cos, mesh_device),
            rotary_sin=_to_device_tensor(sin, mesh_device),
            position_indices=_to_device_tensor(
                torch.arange(max_cache_len, dtype=torch.int32),
                mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
            ),
        )

    def _validate_hidden_states(self, hidden_states, expected_seq_len: int | None = None) -> int:
        shape = tuple(hidden_states.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != self.batch or shape[3] != self.hidden_size:
            raise ValueError(f"hidden_states must have shape [1, {self.batch}, seq, {self.hidden_size}], got {shape}")
        seq_len = int(shape[2])
        if expected_seq_len is not None and seq_len != expected_seq_len:
            raise ValueError(f"expected seq_len={expected_seq_len}, got {seq_len}")
        if seq_len < 1 or seq_len > self.max_cache_len:
            raise ValueError(f"seq_len must be in [1, {self.max_cache_len}], got {seq_len}")
        return seq_len

    def _validate_caches(self, key_cache, value_cache) -> None:
        expected = (self.batch, self.num_kv_heads, self.max_cache_len, self.head_dim)
        if tuple(key_cache.shape) != expected or tuple(value_cache.shape) != expected:
            raise ValueError(
                f"key/value caches must both have shape {expected}; got "
                f"{tuple(key_cache.shape)} and {tuple(value_cache.shape)}"
            )

    def _mlp_forward(self, hidden_states):
        gate = ttnn.matmul(
            hidden_states,
            self.gate_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.matmul(
            hidden_states,
            self.up_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gated = ttnn.multiply(gate, up, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.matmul(
            gated,
            self.down_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def prefill_forward(self, hidden_states, key_cache, value_cache):
        seq_len = self._validate_hidden_states(hidden_states)
        self._validate_caches(key_cache, value_cache)

        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_qkv = ttnn.matmul(
            normed,
            self.qkv_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [self.batch, seq_len, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        query = ttnn.rms_norm(
            query,
            epsilon=self.rms_norm_eps,
            weight=self.q_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.rms_norm(
            key,
            epsilon=self.rms_norm_eps,
            weight=self.k_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cos = ttnn.slice(self.rotary_cos, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        sin = ttnn.slice(self.rotary_sin, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        query = ttnn.experimental.rotary_embedding(
            query,
            cos,
            sin,
            None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.experimental.rotary_embedding(
            key,
            cos,
            sin,
            None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        query = ttnn.slice(
            query,
            [0, 0, 0, 0],
            [self.batch, self.num_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.slice(
            key,
            [0, 0, 0, 0],
            [self.batch, self.num_kv_heads, seq_len, self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        for user_id in range(self.batch):
            key_user = ttnn.slice(
                key,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                [1, 1, 1, 1],
            )
            value_user = ttnn.slice(
                value,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                [1, 1, 1, 1],
            )
            ttnn.fill_cache(key_cache, key_user, user_id)
            ttnn.fill_cache(value_cache, value_user, user_id)

        attention = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.transformer.concatenate_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.reshape(attention, [1, self.batch, seq_len, self.attention_width])
        attention = ttnn.matmul(
            attention,
            self.output_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states = ttnn.add(
            residual,
            attention,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states = self._mlp_forward(hidden_states)
        return ttnn.add(
            residual,
            hidden_states,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def decode_forward(self, hidden_states, key_cache, value_cache, *, current_pos: int):
        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        self._validate_caches(key_cache, value_cache)
        if current_pos < 0 or current_pos >= self.max_cache_len:
            raise ValueError(f"current_pos must be in [0, {self.max_cache_len}), got {current_pos}")

        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_qkv = ttnn.matmul(
            normed,
            self.qkv_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_qkv = ttnn.permute(fused_qkv, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [1, 1, self.batch, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            overlap_qk_coregrid=None,
            memory_config=self.decode_heads_mem_config,
        )
        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.to_memory_config(value, self.decode_kv_mem_config)
        query = ttnn.rms_norm(
            query,
            epsilon=self.rms_norm_eps,
            weight=self.q_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.rms_norm(
            key,
            epsilon=self.rms_norm_eps,
            weight=self.k_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cos = ttnn.slice(
            self.rotary_cos,
            [0, 0, current_pos, 0],
            [1, 1, current_pos + 1, self.head_dim],
            [1, 1, 1, 1],
        )
        sin = ttnn.slice(
            self.rotary_sin,
            [0, 0, current_pos, 0],
            [1, 1, current_pos + 1, self.head_dim],
            [1, 1, 1, 1],
        )
        query = ttnn.experimental.rotary_embedding(
            query,
            cos,
            sin,
            0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.experimental.rotary_embedding(
            key,
            cos,
            sin,
            0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key = ttnn.to_memory_config(key, self.decode_kv_mem_config)

        position = ttnn.slice(self.position_indices, [current_pos], [current_pos + 1], [1])
        update_indices = ttnn.repeat(position, ttnn.Shape([self.batch]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=None,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=update_indices,
            share_cache=False,
            page_table=None,
        )

        attention = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            is_causal=True,
            cur_pos_tensor=update_indices,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.to_memory_config(attention, self.decode_concat_input_mem_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(
            attention,
            num_heads=self.num_heads,
            sub_core_grids=self.decode_compute_core_grid,
        )
        attention = ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.slice(
            attention,
            [0, 0, 0, 0],
            [1, 1, self.batch, self.attention_width],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.permute(attention, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.matmul(
            attention,
            self.output_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states = ttnn.add(
            residual,
            attention,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        residual = hidden_states
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states = self._mlp_forward(hidden_states)
        return ttnn.add(
            residual,
            hidden_states,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, hidden_states, key_cache, value_cache, *, mode: str, current_pos: int | None = None):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, key_cache, value_cache)
        if mode == "decode":
            if current_pos is None:
                raise ValueError("decode mode requires current_pos")
            return self.decode_forward(hidden_states, key_cache, value_cache, current_pos=current_pos)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
