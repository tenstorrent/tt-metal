# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-device GPT-OSS decoder layer translated from the supplied TTNN IR.

The compiler graphs were captured with tensor parallelism over a 1x4 mesh.  This
functional baseline keeps their load-time QKV fusion and decoder math, but uses
the canonical full Hugging Face weights and therefore needs no collectives.
"""

from __future__ import annotations

from typing import Mapping

import ttnn
from models.common.lightweightmodule import LightweightModule

EMITTED_BATCH = 1
EMITTED_PREFILL_SEQUENCE = 17
EMITTED_CACHE_LENGTH = 128
EMITTED_TP_DEGREE = 4
SUPPORTED_CONTEXT = 21_248


def _candidate_keys(layer_idx: int, suffix: str) -> tuple[str, ...]:
    return (
        f"model.layers.{layer_idx}.{suffix}",
        f"model.model.layers.{layer_idx}.{suffix}",
        f"model.language_model.layers.{layer_idx}.{suffix}",
        f"layers.{layer_idx}.{suffix}",
        suffix,
    )


def _find_tensor(state_dict: Mapping[str, object], layer_idx: int, suffix: str):
    for key in _candidate_keys(layer_idx, suffix):
        if key in state_dict:
            return state_dict[key]
    return None


def _require_tensor(state_dict: Mapping[str, object], layer_idx: int, suffix: str):
    value = _find_tensor(state_dict, layer_idx, suffix)
    if value is None:
        candidates = ", ".join(_candidate_keys(layer_idx, suffix))
        raise KeyError(f"Missing GPT-OSS layer tensor {suffix!r}; tried: {candidates}")
    return value


def _dense_expert_weight(state_dict: Mapping[str, object], layer_idx: int, projection: str):
    dense = _find_tensor(state_dict, layer_idx, f"mlp.experts.{projection}")
    if dense is not None:
        return dense

    blocks = _require_tensor(state_dict, layer_idx, f"mlp.experts.{projection}_blocks")
    scales = _require_tensor(state_dict, layer_idx, f"mlp.experts.{projection}_scales")
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    return convert_moe_packed_tensors(blocks, scales)


def _as_replicated_tensor(
    tensor,
    *,
    mesh_device,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
    )


class FunctionalDecoder(LightweightModule):
    """Correctness-first dense translation of one GPT-OSS decoder layer."""

    def __init__(
        self,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int,
        max_cache_len: int,
        weights: dict[str, ttnn.Tensor],
        cos_cache: ttnn.Tensor,
        sin_cache: ttnn.Tensor,
    ):
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.batch = batch
        self.max_cache_len = max_cache_len
        self.weights = weights
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

        self.hidden_size = int(hf_config.hidden_size)
        self.num_heads = int(hf_config.num_attention_heads)
        self.num_kv_heads = int(hf_config.num_key_value_heads)
        self.head_dim = int(hf_config.head_dim)
        self.intermediate_size = int(hf_config.intermediate_size)
        self.num_experts = int(hf_config.num_local_experts)
        self.top_k = int(hf_config.num_experts_per_tok)
        self.eps = float(hf_config.rms_norm_eps)
        self.scale = self.head_dim**-0.5
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        layer_types = getattr(hf_config, "layer_types", None)
        self.layer_type = layer_types[layer_idx] if layer_types is not None else None
        self.sliding_window = int(hf_config.sliding_window) if self.layer_type == "sliding_attention" else None

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch=EMITTED_BATCH,
        max_cache_len=EMITTED_CACHE_LENGTH,
        **_kwargs,
    ):
        """Run all host-side const-eval and transfer full dense BF16 weights."""
        import torch
        import torch.nn.functional as F
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

        if batch != EMITTED_BATCH:
            raise ValueError(f"The emitted workload batch is {EMITTED_BATCH}, got {batch}")
        if not 0 <= layer_idx < int(hf_config.num_hidden_layers):
            raise ValueError(f"layer_idx={layer_idx} is outside the configured layer range")
        if not 1 <= max_cache_len <= SUPPORTED_CONTEXT:
            raise ValueError(f"max_cache_len must be in [1, {SUPPORTED_CONTEXT}], got {max_cache_len}")
        if not isinstance(mesh_device, ttnn.MeshDevice):
            raise TypeError("FunctionalDecoder requires a ttnn.MeshDevice")
        if tuple(mesh_device.shape) != (1, 1):
            raise ValueError(f"FunctionalDecoder requires a 1x1 mesh, got {tuple(mesh_device.shape)}")

        hidden_size = int(hf_config.hidden_size)
        num_heads = int(hf_config.num_attention_heads)
        num_kv_heads = int(hf_config.num_key_value_heads)
        head_dim = int(hf_config.head_dim)
        intermediate_size = int(hf_config.intermediate_size)
        num_experts = int(hf_config.num_local_experts)
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        if q_dim != 4096 or kv_dim != 512 or hidden_size != 2880:
            raise ValueError(
                "The translated IR expects GPT-OSS-20B dimensions "
                f"hidden=2880, q=4096, kv=512; got hidden={hidden_size}, q={q_dim}, kv={kv_dim}"
            )
        if intermediate_size != 2880 or num_experts != 32:
            raise ValueError(
                "The translated IR expects intermediate_size=2880 and 32 experts; "
                f"got {intermediate_size} and {num_experts}"
            )

        q_weight = _require_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k_weight = _require_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v_weight = _require_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        q_bias = _require_tensor(state_dict, layer_idx, "self_attn.q_proj.bias")
        k_bias = _require_tensor(state_dict, layer_idx, "self_attn.k_proj.bias")
        v_bias = _require_tensor(state_dict, layer_idx, "self_attn.v_proj.bias")

        # The compiler const-eval transposes each projection and concatenates Q, K, V.
        qkv_weight = torch.cat(
            [q_weight.transpose(-2, -1), k_weight.transpose(-2, -1), v_weight.transpose(-2, -1)],
            dim=-1,
        ).to(torch.bfloat16)
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=-1).reshape(1, 1, -1).to(torch.bfloat16)

        o_weight = _require_tensor(state_dict, layer_idx, "self_attn.o_proj.weight")
        o_bias = _require_tensor(state_dict, layer_idx, "self_attn.o_proj.bias")
        router_weight = _require_tensor(state_dict, layer_idx, "mlp.router.weight")
        router_bias = _require_tensor(state_dict, layer_idx, "mlp.router.bias")
        gate_up_weight = _dense_expert_weight(state_dict, layer_idx, "gate_up_proj")
        down_weight = _dense_expert_weight(state_dict, layer_idx, "down_proj")
        gate_up_bias = _require_tensor(state_dict, layer_idx, "mlp.experts.gate_up_proj_bias")
        down_bias = _require_tensor(state_dict, layer_idx, "mlp.experts.down_proj_bias")
        input_norm = _require_tensor(state_dict, layer_idx, "input_layernorm.weight")
        post_attention_norm = _require_tensor(state_dict, layer_idx, "post_attention_layernorm.weight")
        sinks = _require_tensor(state_dict, layer_idx, "self_attn.sinks").to(torch.bfloat16)

        expected_shapes = {
            "qkv_weight": (hidden_size, q_dim + 2 * kv_dim),
            "o_weight": (q_dim, hidden_size),
            "router_weight": (hidden_size, num_experts),
            "gate_up_weight": (num_experts, hidden_size, 2 * intermediate_size),
            "down_weight": (num_experts, intermediate_size, hidden_size),
        }
        actual_shapes = {
            "qkv_weight": tuple(qkv_weight.shape),
            "o_weight": tuple(o_weight.transpose(-2, -1).shape),
            "router_weight": tuple(router_weight.transpose(-2, -1).shape),
            "gate_up_weight": tuple(gate_up_weight.shape),
            "down_weight": tuple(down_weight.shape),
        }
        for name, expected in expected_shapes.items():
            if actual_shapes[name] != expected:
                raise ValueError(f"{name} has shape {actual_shapes[name]}, expected {expected}")

        scale = head_dim**-0.5
        prefill_sinks = (sinks.reshape(1, num_heads, 1, 1) / scale).to(torch.bfloat16)
        decode_sinks = F.pad(sinks.reshape(num_heads, 1), (0, ttnn.TILE_SIZE - 1)) / scale

        rotary = GptOssRotaryEmbedding(hf_config)
        positions = torch.arange(max_cache_len, dtype=torch.long).unsqueeze(0)
        rotary_input = torch.empty(1, 1, max_cache_len, head_dim, dtype=torch.bfloat16)
        cos_half, sin_half = rotary(rotary_input, positions)
        cos = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(1)
        sin = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(1)

        norm_shape = (1, 1, hidden_size // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
        weights = {
            "input_norm": _as_replicated_tensor(
                input_norm.reshape(norm_shape).to(torch.bfloat16),
                mesh_device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            "post_attention_norm": _as_replicated_tensor(
                post_attention_norm.reshape(norm_shape).to(torch.bfloat16),
                mesh_device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            "qkv_weight": _as_replicated_tensor(qkv_weight, mesh_device=mesh_device),
            "qkv_bias": _as_replicated_tensor(qkv_bias, mesh_device=mesh_device),
            "o_weight": _as_replicated_tensor(o_weight.transpose(-2, -1).to(torch.bfloat16), mesh_device=mesh_device),
            "o_bias": _as_replicated_tensor(o_bias.reshape(1, 1, -1).to(torch.bfloat16), mesh_device=mesh_device),
            "prefill_sinks": _as_replicated_tensor(prefill_sinks, mesh_device=mesh_device),
            "decode_sinks": _as_replicated_tensor(decode_sinks.to(torch.bfloat16), mesh_device=mesh_device),
            "router_weight": _as_replicated_tensor(
                router_weight.transpose(-2, -1).to(torch.bfloat16), mesh_device=mesh_device
            ),
            "router_bias": _as_replicated_tensor(
                router_bias.reshape(1, -1).float(),
                mesh_device=mesh_device,
                dtype=ttnn.float32,
            ),
            "gate_up_weight": _as_replicated_tensor(gate_up_weight.to(torch.bfloat16), mesh_device=mesh_device),
            "gate_up_bias": _as_replicated_tensor(
                gate_up_bias.reshape(num_experts, 1, 2 * intermediate_size).to(torch.bfloat16),
                mesh_device=mesh_device,
            ),
            "down_weight": _as_replicated_tensor(down_weight.to(torch.bfloat16), mesh_device=mesh_device),
            "down_bias": _as_replicated_tensor(
                down_bias.reshape(num_experts, 1, hidden_size).to(torch.bfloat16),
                mesh_device=mesh_device,
            ),
        }

        return cls(
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_cache_len=max_cache_len,
            weights=weights,
            cos_cache=_as_replicated_tensor(cos, mesh_device=mesh_device),
            sin_cache=_as_replicated_tensor(sin, mesh_device=mesh_device),
        )

    def create_kv_cache(self):
        shape = (self.batch, self.num_kv_heads, self.max_cache_len, self.head_dim)
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

    def _validate_hidden_states(self, hidden_states, *, expected_seq_len=None):
        shape = tuple(hidden_states.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != self.batch or shape[3] != self.hidden_size:
            raise ValueError(f"hidden_states must have shape [1, {self.batch}, seq, {self.hidden_size}], got {shape}")
        if expected_seq_len is not None and shape[2] != expected_seq_len:
            raise ValueError(f"expected sequence length {expected_seq_len}, got {shape[2]}")
        return shape[2]

    def _prefill_attention(self, hidden_states, key_cache, value_cache, seq_len):
        normalized = ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=self.weights["input_norm"],
            compute_kernel_config=self.compute_kernel_config,
        )
        fused = ttnn.linear(
            normalized,
            self.weights["qkv_weight"],
            bias=self.weights["qkv_bias"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        fused = ttnn.reshape(fused, [self.batch, seq_len, -1])
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.slice(self.cos_cache, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
        sin = ttnn.slice(self.sin_cache, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
        query = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # The emitted graph slices the tile-padded RoPE outputs back to the
        # logical sequence before padding only the cache-write operands.  Keep
        # that logical slice here so the dedicated SDPA sees matching K/V
        # sequence lengths for non-tile-aligned prefills (the captured graph is
        # seq=17).
        query = ttnn.slice(query, [0, 0, 0, 0], [1, self.num_heads, seq_len, self.head_dim])
        key = ttnn.slice(key, [0, 0, 0, 0], [1, self.num_kv_heads, seq_len, self.head_dim])
        ttnn.fill_cache(key_cache, key, batch_idx=0)
        ttnn.fill_cache(value_cache, value, batch_idx=0)
        attended = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            attention_sink=self.weights["prefill_sinks"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        projected = ttnn.linear(
            attended,
            self.weights["o_weight"],
            bias=self.weights["o_bias"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        return ttnn.add(hidden_states, ttnn.reshape(projected, [1, self.batch, seq_len, self.hidden_size]))

    def _decode_attention(
        self,
        hidden_states,
        key_cache,
        value_cache,
        cache_position,
        cache_position_tensor,
    ):
        normalized = ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=self.weights["input_norm"],
            compute_kernel_config=self.compute_kernel_config,
        )
        fused = ttnn.linear(
            normalized,
            self.weights["qkv_weight"],
            bias=self.weights["qkv_bias"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        fused = ttnn.reshape(fused, [1, 1, self.batch, -1])
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        query = ttnn.experimental.rotary_embedding(
            query, self.cos_cache, self.sin_cache, cache_position, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        key = ttnn.experimental.rotary_embedding(
            key, self.cos_cache, self.sin_cache, cache_position, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=cache_position_tensor,
            share_cache=False,
            page_table=None,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=cache_position_tensor,
            share_cache=False,
            page_table=None,
        )
        attended = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            is_causal=True,
            cur_pos_tensor=cache_position_tensor,
            attention_sink=self.weights["decode_sinks"],
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        attended = ttnn.reshape(attended, [self.batch, self.num_heads * self.head_dim])
        projected = ttnn.linear(
            attended,
            self.weights["o_weight"],
            bias=self.weights["o_bias"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        projected = ttnn.reshape(projected, [1, self.batch, 1, self.hidden_size])
        return ttnn.add(hidden_states, projected)

    def _moe_forward(self, hidden_states, seq_len):
        normalized = ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=self.weights["post_attention_norm"],
            compute_kernel_config=self.compute_kernel_config,
        )
        token_count = self.batch * seq_len
        flat = ttnn.reshape(normalized, [token_count, self.hidden_size])
        # The compiler graph promotes the normalized router input and bias to
        # FP32, computes FP32 logits, then casts to BF16 immediately before
        # top-k.  That boundary is important with real expert weights because a
        # changed fourth expert has a much larger effect than ordinary BF16
        # arithmetic noise.
        router_input = ttnn.typecast(flat, ttnn.float32)

        router_logits = ttnn.linear(
            router_input,
            self.weights["router_weight"],
            bias=self.weights["router_bias"],
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        router_logits = ttnn.typecast(router_logits, ttnn.bfloat16)
        top_values, top_indices = ttnn.topk(router_logits, k=self.top_k, dim=-1, sorted=True)
        top_values = ttnn.softmax(top_values, dim=-1, numeric_stable=True)
        routing_weights = ttnn.scatter(ttnn.zeros_like(router_logits), dim=1, index=top_indices, src=top_values)

        expert_input = ttnn.reshape(flat, [1, token_count, self.hidden_size])
        expert_input = ttnn.repeat(expert_input, ttnn.Shape([self.num_experts, 1, 1]))
        gate_up = ttnn.matmul(
            expert_input,
            self.weights["gate_up_weight"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        gate_up = ttnn.add(gate_up, self.weights["gate_up_bias"])
        gate = ttnn.slice(
            gate_up,
            [0, 0, 0],
            [self.num_experts, token_count, 2 * self.intermediate_size],
            [1, 1, 2],
        )
        up = ttnn.slice(
            gate_up,
            [0, 0, 1],
            [self.num_experts, token_count, 2 * self.intermediate_size],
            [1, 1, 2],
        )
        gate = ttnn.clamp(gate, min=None, max=7.0)
        up = ttnn.clamp(up, min=-7.0, max=7.0)
        gate = ttnn.multiply(gate, ttnn.sigmoid(ttnn.multiply(gate, 1.703125)))
        activated = ttnn.multiply(ttnn.add(up, 1.0), gate)
        expert_output = ttnn.matmul(
            activated,
            self.weights["down_weight"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        expert_output = ttnn.add(expert_output, self.weights["down_bias"])
        routing_weights = ttnn.permute(routing_weights, [1, 0])
        routing_weights = ttnn.reshape(routing_weights, [self.num_experts, token_count, 1])
        expert_output = ttnn.multiply(expert_output, routing_weights)
        expert_output = ttnn.sum(expert_output, dim=0)
        expert_output = ttnn.reshape(expert_output, [1, self.batch, seq_len, self.hidden_size])
        return ttnn.add(hidden_states, expert_output)

    def prefill_forward(self, hidden_states, *, key_cache, value_cache):
        seq_len = self._validate_hidden_states(hidden_states)
        if seq_len <= 1:
            raise ValueError("prefill_forward requires seq_len > 1")
        if seq_len > self.max_cache_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_cache_len={self.max_cache_len}")
        hidden_states = self._prefill_attention(hidden_states, key_cache, value_cache, seq_len)
        return self._moe_forward(hidden_states, seq_len)

    def decode_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        cache_position,
        cache_position_tensor,
    ):
        """Decode one token.

        ``cache_position`` and ``cache_position_tensor`` are the host and
        device representations of the same position.  The caller must keep
        them equal: the former selects the RoPE row while the latter selects
        the cache write and decode-SDPA position.
        """
        self._validate_hidden_states(hidden_states, expected_seq_len=1)
        if not 0 <= cache_position < self.max_cache_len:
            raise ValueError(f"cache_position must be in [0, {self.max_cache_len}), got {cache_position}")
        hidden_states = self._decode_attention(
            hidden_states,
            key_cache,
            value_cache,
            cache_position,
            cache_position_tensor,
        )
        return self._moe_forward(hidden_states, 1)

    def forward(self, hidden_states, *, mode, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
