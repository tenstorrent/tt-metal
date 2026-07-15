# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-device Falcon3 decoder translated from the supplied TTNN IR graphs.

The compiler capture was tensor-parallel over a 1x4 mesh.  This functional
version deliberately loads the full Hugging Face weights and removes the
collectives, compiler shard specs, and static program configurations.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

EMITTED_BATCH = 32
EMITTED_CACHE_LENGTH = 128
EMITTED_PREFILL_SEQUENCE = 17
IR_TP_DEGREE = 4
IR_REPRESENTATIVE_LAYER = 14


def _config_value(config, name: str):
    value = getattr(config, name, None)
    if value is None:
        raise ValueError(f"Falcon3 config is missing required field {name!r}")
    return value


def _resolve_layer_tensor(state_dict: Mapping[str, torch.Tensor], layer_idx: int, suffix: str) -> torch.Tensor:
    candidates = (
        f"model.layers.{layer_idx}.{suffix}",
        f"model.language_model.layers.{layer_idx}.{suffix}",
        f"layers.{layer_idx}.{suffix}",
        suffix,
    )
    for key in candidates:
        if key in state_dict:
            return state_dict[key]
    raise KeyError(f"Missing Falcon3 layer-{layer_idx} tensor {suffix!r}; tried {candidates}")


def _as_bf16_tile(tensor: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.detach().to(dtype=torch.bfloat16).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class FunctionalDecoder(LightweightModule):
    """Correctness-first dense translation of one Falcon3 decoder layer."""

    def __init__(
        self,
        *,
        mesh_device,
        layer_idx: int,
        batch: int,
        max_cache_len: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        qkv_weight: ttnn.Tensor,
        o_weight: ttnn.Tensor,
        gate_weight: ttnn.Tensor,
        up_weight: ttnn.Tensor,
        down_weight: ttnn.Tensor,
        input_norm_weight: ttnn.Tensor,
        post_attention_norm_weight: ttnn.Tensor,
        cos_cache: ttnn.Tensor,
        sin_cache: ttnn.Tensor,
        decode_positions: ttnn.Tensor,
        decode_head_memory_config: ttnn.MemoryConfig,
    ):
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.batch = batch
        self.max_cache_len = max_cache_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.scale = 1.0 / math.sqrt(head_dim)

        self.qkv_weight = qkv_weight
        self.o_weight = o_weight
        self.gate_weight = gate_weight
        self.up_weight = up_weight
        self.down_weight = down_weight
        self.input_norm_weight = input_norm_weight
        self.post_attention_norm_weight = post_attention_norm_weight
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache
        self.decode_positions = decode_positions
        self.decode_head_memory_config = decode_head_memory_config

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
        """Load full HF weights and reproduce the IR's load-time transforms."""
        if kwargs:
            raise TypeError(f"Unsupported FunctionalDecoder options: {sorted(kwargs)}")
        if batch <= 0:
            raise ValueError(f"batch must be positive, got {batch}")
        if max_cache_len <= 0:
            raise ValueError(f"max_cache_len must be positive, got {max_cache_len}")

        hidden_size = int(_config_value(hf_config, "hidden_size"))
        num_layers = int(_config_value(hf_config, "num_hidden_layers"))
        num_heads = int(_config_value(hf_config, "num_attention_heads"))
        num_kv_heads = int(_config_value(hf_config, "num_key_value_heads"))
        head_dim = int(_config_value(hf_config, "head_dim"))
        intermediate_size = int(_config_value(hf_config, "intermediate_size"))
        rms_norm_eps = float(_config_value(hf_config, "rms_norm_eps"))
        hidden_act = str(_config_value(hf_config, "hidden_act"))
        max_position_embeddings = int(_config_value(hf_config, "max_position_embeddings"))
        ir_config = {
            "hidden_size": (hidden_size, 3072),
            "num_hidden_layers": (num_layers, 28),
            "num_attention_heads": (num_heads, 12),
            "num_key_value_heads": (num_kv_heads, 4),
            "head_dim": (head_dim, 256),
            "intermediate_size": (intermediate_size, 23040),
        }
        mismatches = {name: values for name, values in ir_config.items() if values[0] != values[1]}
        if mismatches:
            raise ValueError(f"HF config does not match the supplied Falcon3-7B IR: {mismatches}")
        if hidden_act != "silu":
            raise ValueError(f"The supplied IR uses SwiGLU/SILU, got hidden_act={hidden_act!r}")
        if not math.isclose(rms_norm_eps, 1e-6, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"The supplied IR uses rms_norm_eps=1e-6, got {rms_norm_eps}")
        if bool(getattr(hf_config, "attention_bias", False)) or bool(getattr(hf_config, "mlp_bias", False)):
            raise ValueError("The supplied IR does not contain attention or MLP biases")
        if max_cache_len > max_position_embeddings:
            raise ValueError(f"max_cache_len={max_cache_len} exceeds the HF context limit {max_position_embeddings}")
        if not 0 <= layer_idx < num_layers:
            raise ValueError(f"layer_idx={layer_idx} is outside [0, {num_layers})")
        if hidden_size != num_heads * head_dim:
            raise ValueError(
                f"Falcon3 hidden/head contract is inconsistent: {hidden_size=} vs {num_heads=} * {head_dim=}"
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"GQA requires num_heads divisible by num_kv_heads, got {num_heads}/{num_kv_heads}")

        q_weight = _resolve_layer_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k_weight = _resolve_layer_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v_weight = _resolve_layer_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        o_weight = _resolve_layer_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")
        gate_weight = _resolve_layer_tensor(state_dict, layer_idx, "mlp.gate_proj.weight")
        up_weight = _resolve_layer_tensor(state_dict, layer_idx, "mlp.up_proj.weight")
        down_weight = _resolve_layer_tensor(state_dict, layer_idx, "mlp.down_proj.weight")
        input_norm_weight = _resolve_layer_tensor(state_dict, layer_idx, "input_layernorm.weight")
        post_attention_norm_weight = _resolve_layer_tensor(state_dict, layer_idx, "post_attention_layernorm.weight")

        expected_shapes = {
            "q_proj": (num_heads * head_dim, hidden_size),
            "k_proj": (num_kv_heads * head_dim, hidden_size),
            "v_proj": (num_kv_heads * head_dim, hidden_size),
            "o_proj": (hidden_size, hidden_size),
            "gate_proj": (intermediate_size, hidden_size),
            "up_proj": (intermediate_size, hidden_size),
            "down_proj": (hidden_size, intermediate_size),
            "input_layernorm": (hidden_size,),
            "post_attention_layernorm": (hidden_size,),
        }
        actual_tensors = {
            "q_proj": q_weight,
            "k_proj": k_weight,
            "v_proj": v_weight,
            "o_proj": o_weight,
            "gate_proj": gate_weight,
            "up_proj": up_weight,
            "down_proj": down_weight,
            "input_layernorm": input_norm_weight,
            "post_attention_layernorm": post_attention_norm_weight,
        }
        for name, expected in expected_shapes.items():
            actual = tuple(actual_tensors[name].shape)
            if actual != expected:
                raise ValueError(f"{name} has shape {actual}, expected {expected}")

        # The IR's const-eval receives [V, K, Q], reverses it, transposes each
        # matrix, and concatenates columns.  The resulting fused order is Q,K,V.
        qkv_host = torch.cat(
            [q_weight.transpose(-2, -1), k_weight.transpose(-2, -1), v_weight.transpose(-2, -1)],
            dim=-1,
        )

        rope_parameters = getattr(hf_config, "rope_parameters", None) or {}
        rope_theta = getattr(hf_config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is None:
            raise ValueError("Falcon3 config does not provide rope_theta")
        if float(rope_theta) != 1000042.0:
            raise ValueError(f"The supplied IR uses rope_theta=1000042, got {rope_theta}")
        rope_type = rope_parameters.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(f"The supplied IR uses default RoPE, got rope_type={rope_type!r}")
        positions = torch.arange(max_cache_len, dtype=torch.float32)
        inv_freq = 1.0 / (float(rope_theta) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim)))
        frequencies = torch.outer(positions, inv_freq)
        rotary_angles = torch.cat([frequencies, frequencies], dim=-1)
        cos_host = rotary_angles.cos().unsqueeze(0).unsqueeze(0)
        sin_host = rotary_angles.sin().unsqueeze(0).unsqueeze(0)

        grid_size = mesh_device.compute_with_storage_grid_size()
        grid_x = min(batch, grid_size.x)
        while grid_x > 0 and (batch % grid_x != 0 or batch // grid_x > grid_size.y):
            grid_x -= 1
        if grid_x == 0:
            raise ValueError(f"batch={batch} cannot form a rectangular decode grid within {grid_size}")
        batch_grid = ttnn.CoreGrid(y=batch // grid_x, x=grid_x)
        decode_head_memory_config = ttnn.create_sharded_memory_config(
            shape=(32, head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        return cls(
            mesh_device=mesh_device,
            layer_idx=layer_idx,
            batch=batch,
            max_cache_len=max_cache_len,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            qkv_weight=_as_bf16_tile(qkv_host, mesh_device),
            o_weight=_as_bf16_tile(o_weight.transpose(-2, -1), mesh_device),
            gate_weight=_as_bf16_tile(gate_weight.transpose(-2, -1), mesh_device),
            up_weight=_as_bf16_tile(up_weight.transpose(-2, -1), mesh_device),
            down_weight=_as_bf16_tile(down_weight.transpose(-2, -1), mesh_device),
            input_norm_weight=_as_bf16_tile(input_norm_weight, mesh_device),
            post_attention_norm_weight=_as_bf16_tile(post_attention_norm_weight, mesh_device),
            cos_cache=_as_bf16_tile(cos_host, mesh_device),
            sin_cache=_as_bf16_tile(sin_host, mesh_device),
            decode_positions=ttnn.from_torch(
                torch.arange(max_cache_len, dtype=torch.int32).reshape(1, 1, 1, max_cache_len),
                dtype=ttnn.int32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            decode_head_memory_config=decode_head_memory_config,
        )

    def allocate_kv_cache(self, max_cache_len: int | None = None) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Allocate the dense single-device form of the IR's linear KV cache."""
        cache_len = self.max_cache_len if max_cache_len is None else max_cache_len
        if cache_len <= 0:
            raise ValueError(f"max_cache_len must be positive, got {cache_len}")
        shape = (self.batch, self.num_kv_heads, cache_len, self.head_dim)
        key_cache = ttnn.zeros(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        value_cache = ttnn.zeros(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return key_cache, value_cache

    def _validate_hidden(self, hidden_states: ttnn.Tensor, *, decode: bool) -> tuple[int, int]:
        shape = tuple(int(value) for value in hidden_states.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[3] != self.hidden_size:
            raise ValueError(f"hidden_states must be [1,batch,seq,{self.hidden_size}], got {shape}")
        batch, seq_len = shape[1], shape[2]
        if batch != self.batch:
            raise ValueError(f"runtime batch {batch} does not match configured batch {self.batch}")
        if decode and seq_len != 1:
            raise ValueError(f"decode requires seq_len=1, got {seq_len}")
        if not decode and not 0 < seq_len <= self.max_cache_len:
            raise ValueError(f"prefill seq_len must be in [1, {self.max_cache_len}], got {seq_len}")
        return batch, seq_len

    def _validate_caches(self, key_cache: ttnn.Tensor, value_cache: ttnn.Tensor) -> None:
        expected_prefix = (self.batch, self.num_kv_heads)
        for name, cache in (("key_cache", key_cache), ("value_cache", value_cache)):
            shape = tuple(int(value) for value in cache.shape)
            if len(shape) != 4 or shape[:2] != expected_prefix or shape[3] != self.head_dim:
                raise ValueError(
                    f"{name} must be [batch={self.batch},kv_heads={self.num_kv_heads},cache,head_dim={self.head_dim}], "
                    f"got {shape}"
                )
        if int(key_cache.shape[2]) != int(value_cache.shape[2]):
            raise ValueError("key_cache and value_cache must have the same cache length")

    def _validate_cache_position(self, cache_position: ttnn.Tensor) -> None:
        shape = tuple(int(value) for value in cache_position.shape)
        if shape != (self.batch,):
            raise ValueError(f"cache_position must have shape [{self.batch}], got {shape}")
        if cache_position.dtype != ttnn.int32:
            raise ValueError(f"cache_position must use ttnn.int32, got {cache_position.dtype}")

    def _rotary_slice(self, start: int, length: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        end = start + length
        if start < 0 or end > self.max_cache_len:
            raise ValueError(f"rotary range [{start}, {end}) exceeds cache length {self.max_cache_len}")
        starts = [0, 0, start, 0]
        ends = [1, 1, end, self.head_dim]
        return ttnn.slice(self.cos_cache, starts, ends), ttnn.slice(self.sin_cache, starts, ends)

    def _unpad_prefill_sequence(self, tensor: ttnn.Tensor, *, batch: int, heads: int, seq_len: int) -> ttnn.Tensor:
        """Reproduce the IR slice after RoPE rounds a short sequence to a tile."""
        if int(tensor.shape[2]) == seq_len:
            return tensor
        padded = tensor
        tensor = ttnn.slice(
            padded,
            [0, 0, 0, 0],
            [batch, heads, seq_len, self.head_dim],
        )
        padded.deallocate(True)
        return tensor

    def _prefill_attention(
        self,
        residual: ttnn.Tensor,
        *,
        batch: int,
        seq_len: int,
        key_cache: ttnn.Tensor,
        value_cache: ttnn.Tensor,
    ) -> ttnn.Tensor:
        normed = ttnn.rms_norm(
            residual,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_qkv = ttnn.matmul(
            normed,
            self.qkv_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        normed.deallocate(True)
        fused_qkv_matmul = fused_qkv
        fused_qkv = ttnn.reshape(
            fused_qkv_matmul,
            (batch, seq_len, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim),
        )
        fused_qkv_matmul.deallocate(False)
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused_qkv,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_qkv.deallocate(True)

        cos, sin = self._rotary_slice(0, seq_len)
        key_rotated = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        query_rotated = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key.deallocate(True)
        query.deallocate(True)
        # A tile-aligned slice may alias the persistent RoPE table allocation.
        cos.deallocate(False)
        sin.deallocate(False)
        key_rotated = self._unpad_prefill_sequence(
            key_rotated,
            batch=batch,
            heads=self.num_kv_heads,
            seq_len=seq_len,
        )
        query_rotated = self._unpad_prefill_sequence(
            query_rotated,
            batch=batch,
            heads=self.num_heads,
            seq_len=seq_len,
        )

        for user_id in range(batch):
            value_user = ttnn.slice(
                value,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
            )
            key_user = ttnn.slice(
                key_rotated,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
            )
            ttnn.fill_cache(value_cache, value_user, batch_idx=user_id)
            ttnn.fill_cache(key_cache, key_user, batch_idx=user_id)
            value_user.deallocate(True)
            key_user.deallocate(True)

        attention = ttnn.transformer.scaled_dot_product_attention(
            query_rotated,
            key_rotated,
            value,
            is_causal=True,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        query_rotated.deallocate(True)
        key_rotated.deallocate(True)
        value.deallocate(True)
        attention_heads = attention
        attention = ttnn.transformer.concatenate_heads(attention_heads, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention_heads.deallocate(True)
        concatenated = attention
        attention = ttnn.reshape(concatenated, (batch * seq_len, self.hidden_size))
        concatenated.deallocate(False)
        projected = ttnn.matmul(
            attention,
            self.o_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention.deallocate(True)
        projected_matmul = projected
        projected = ttnn.reshape(projected_matmul, (1, 1, batch * seq_len, self.hidden_size))
        projected_matmul.deallocate(False)
        output = ttnn.add(residual, projected, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        projected.deallocate(True)
        return output

    def _prepare_decode_heads(
        self,
        tensor: ttnn.Tensor,
        num_heads: int,
        *,
        memory_config: ttnn.MemoryConfig,
    ) -> ttnn.Tensor:
        """Reproduce the emitted interleave, exact head slice, and target layout."""
        interleaved = ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG)
        tensor.deallocate(True)
        unpadded = ttnn.slice(
            interleaved,
            [0, 0, 0, 0],
            [1, self.batch, num_heads, self.head_dim],
        )
        interleaved.deallocate(True)
        prepared = ttnn.to_memory_config(unpadded, memory_config)
        unpadded.deallocate(True)
        return prepared

    def _decode_attention_mask(self, cache_position: ttnn.Tensor) -> ttnn.Tensor:
        """Translate the emitted position >= cache-index mask on device."""
        scalar_position = ttnn.slice(cache_position, [0], [1])
        valid_positions = ttnn.le(
            self.decode_positions,
            scalar_position,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        scalar_position.deallocate(False)
        single_head_mask = ttnn.where(
            valid_positions,
            0.0,
            -3.3895313892515355e38,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        valid_positions.deallocate(True)
        attention_mask = ttnn.repeat(
            single_head_mask,
            ttnn.Shape([1, 1, self.num_heads, 1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        single_head_mask.deallocate(True)
        return attention_mask

    def _decode_qkv(
        self, residual: ttnn.Tensor, *, position_index: int
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Produce the emitted decode Q/K/V head tensors after RoPE and unpadding."""
        normed = ttnn.rms_norm(
            residual,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_qkv = ttnn.matmul(
            normed,
            self.qkv_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        normed.deallocate(True)
        fused_qkv_dram = fused_qkv
        fused_qkv = ttnn.to_memory_config(fused_qkv_dram, ttnn.L1_MEMORY_CONFIG)
        fused_qkv_dram.deallocate(True)
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        fused_qkv.deallocate(True)

        cos_dram, sin_dram = self._rotary_slice(position_index, 1)
        cos = ttnn.to_memory_config(cos_dram, ttnn.L1_MEMORY_CONFIG)
        sin = ttnn.to_memory_config(sin_dram, ttnn.L1_MEMORY_CONFIG)
        # The DRAM slices may alias the persistent RoPE table allocation.
        cos_dram.deallocate(False)
        sin_dram.deallocate(False)
        key_rotated = ttnn.experimental.rotary_embedding(
            key,
            cos,
            sin,
            0,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        query_rotated = ttnn.experimental.rotary_embedding(
            query,
            cos,
            sin,
            0,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        key.deallocate(True)
        query.deallocate(True)
        cos.deallocate(True)
        sin.deallocate(True)
        key_rotated = self._prepare_decode_heads(
            key_rotated,
            self.num_kv_heads,
            memory_config=self.decode_head_memory_config,
        )
        query_rotated = self._prepare_decode_heads(
            query_rotated,
            self.num_heads,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return query_rotated, key_rotated, value

    def _decode_attention(
        self,
        residual: ttnn.Tensor,
        *,
        key_cache: ttnn.Tensor,
        value_cache: ttnn.Tensor,
        cache_position: ttnn.Tensor,
        position_index: int,
    ) -> ttnn.Tensor:
        query_rotated, key_rotated, value = self._decode_qkv(residual, position_index=position_index)

        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=cache_position,
            share_cache=False,
            page_table=None,
        )
        ttnn.experimental.paged_update_cache(
            key_cache,
            key_rotated,
            update_idxs_tensor=cache_position,
            share_cache=False,
            page_table=None,
        )
        value.deallocate(True)
        key_rotated.deallocate(True)
        attention_mask = self._decode_attention_mask(cache_position)
        attention = ttnn.transformer.scaled_dot_product_attention_decode(
            query_rotated,
            key_cache,
            value_cache,
            attn_mask=attention_mask,
            cur_pos_tensor=None,
            is_causal=False,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention_mask.deallocate(True)
        query_rotated.deallocate(True)
        attention_dram = attention
        attention = ttnn.to_memory_config(attention_dram, self.decode_head_memory_config)
        attention_dram.deallocate(True)
        attention_heads = attention
        attention = ttnn.experimental.nlp_concat_heads_decode(attention_heads, num_heads=self.num_heads)
        attention_heads.deallocate(True)
        attention_sharded = attention
        attention = ttnn.to_memory_config(attention_sharded, ttnn.DRAM_MEMORY_CONFIG)
        attention_sharded.deallocate(True)
        concatenated = attention
        attention = ttnn.reshape(concatenated, (self.batch, self.hidden_size))
        concatenated.deallocate(False)
        projected = ttnn.matmul(
            attention,
            self.o_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention.deallocate(True)
        projected_matmul = projected
        projected = ttnn.reshape(projected_matmul, (1, 1, self.batch, self.hidden_size))
        projected_matmul.deallocate(False)
        output = ttnn.add(residual, projected, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        projected.deallocate(True)
        return output

    def _mlp(self, residual: ttnn.Tensor) -> ttnn.Tensor:
        normed = ttnn.rms_norm(
            residual,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.matmul(
            normed,
            self.gate_weight,
            activation="silu",
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.matmul(
            normed,
            self.up_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        normed.deallocate(True)
        gated = ttnn.multiply(gate, up, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate.deallocate(True)
        up.deallocate(True)
        down = ttnn.matmul(
            gated,
            self.down_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gated.deallocate(True)
        output = ttnn.add(residual, down, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        residual.deallocate(True)
        down.deallocate(True)
        return output

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        key_cache: ttnn.Tensor,
        value_cache: ttnn.Tensor,
    ) -> ttnn.Tensor:
        batch, seq_len = self._validate_hidden(hidden_states, decode=False)
        self._validate_caches(key_cache, value_cache)
        residual = ttnn.reshape(hidden_states, (1, 1, batch * seq_len, self.hidden_size))
        residual = self._prefill_attention(
            residual,
            batch=batch,
            seq_len=seq_len,
            key_cache=key_cache,
            value_cache=value_cache,
        )
        residual = self._mlp(residual)
        output = ttnn.reshape(residual, (1, batch, seq_len, self.hidden_size))
        residual.deallocate(False)
        return output

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        key_cache: ttnn.Tensor,
        value_cache: ttnn.Tensor,
        cache_position: ttnn.Tensor,
        position_index: int,
    ) -> ttnn.Tensor:
        batch, _ = self._validate_hidden(hidden_states, decode=True)
        self._validate_caches(key_cache, value_cache)
        self._validate_cache_position(cache_position)
        if not 0 <= position_index < int(key_cache.shape[2]):
            raise ValueError(f"position_index={position_index} is outside the cache")
        residual = ttnn.reshape(hidden_states, (1, 1, batch, self.hidden_size))
        residual = self._decode_attention(
            residual,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=cache_position,
            position_index=position_index,
        )
        residual = self._mlp(residual)
        output = ttnn.reshape(residual, (1, batch, 1, self.hidden_size))
        residual.deallocate(False)
        return output
