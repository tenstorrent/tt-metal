# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Dense single-device translation of the emitted GPT-OSS decoder graph.

The source EmitPy graphs are flat full-model programs captured on a 1x4 mesh.
This module segments their layer-12 block and replaces every TP4 shard plus
collective with the equivalent dense operation over full Hugging Face weights.
It deliberately retains no mesh partition or collective in either forward.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

import ttnn
from models.common.lightweightmodule import LightweightModule

EMITTED_BATCH = 1
EMITTED_PREFILL_SEQUENCE = 17
EMITTED_CACHE_LENGTH = 128
REPRESENTATIVE_LAYER = 12
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
    raise KeyError(f"Missing decoder weight {suffix!r}; tried {candidates}")


def _expert_tensor(
    state_dict: Mapping[str, torch.Tensor], layer_idx: int, name: str, *, dtype=torch.bfloat16
) -> torch.Tensor:
    """Read a dense expert tensor, dequantizing canonical MXFP4 weights if needed."""

    try:
        return _state_tensor(state_dict, layer_idx, f"mlp.experts.{name}").to(dtype)
    except KeyError:
        if name not in ("gate_up_proj", "down_proj"):
            raise
        blocks = _state_tensor(state_dict, layer_idx, f"mlp.experts.{name}_blocks")
        scales = _state_tensor(state_dict, layer_idx, f"mlp.experts.{name}_scales")
        return convert_moe_packed_tensors(blocks, scales, dtype=dtype)


def _to_device_tensor(tensor: torch.Tensor, mesh_device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class FunctionalDecoder(LightweightModule):
    """Correctness-first GPT-OSS decoder layer translated from EmitPy TTNN."""

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
        num_experts: int,
        experts_per_token: int,
        rms_norm_eps: float,
        sliding_window: int,
        swiglu_limit: float,
        input_norm,
        post_attention_norm,
        qkv_weight,
        qkv_bias,
        output_weight,
        output_bias,
        attention_sinks,
        decode_attention_sinks,
        router_weight,
        router_bias,
        gate_up_weight,
        gate_up_bias,
        down_weight,
        down_bias,
        rotary_cos,
        rotary_sin,
        attention_mask,
        position_indices,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.batch = batch
        self.max_cache_len = max_cache_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.swiglu_limit = swiglu_limit
        self.scale = 1.0 / math.sqrt(head_dim)
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.input_norm = input_norm
        self.post_attention_norm = post_attention_norm
        self.qkv_weight = qkv_weight
        self.qkv_bias = qkv_bias
        self.output_weight = output_weight
        self.output_bias = output_bias
        self.attention_sinks = attention_sinks
        self.decode_attention_sinks = decode_attention_sinks
        self.router_weight = router_weight
        self.router_bias = router_bias
        self.gate_up_weight = gate_up_weight
        self.gate_up_bias = gate_up_bias
        self.down_weight = down_weight
        self.down_bias = down_bias
        self.rotary_cos = rotary_cos
        self.rotary_sin = rotary_sin
        self.attention_mask = attention_mask
        self.position_indices = position_indices

        # Decode head kernels require one small, batch-derived L1 shard.
        # Everything else stays interleaved in DRAM.
        device_grid = mesh_device.compute_with_storage_grid_size()
        batch_grid = ttnn.num_cores_to_corerangeset(batch, device_grid, row_wise=True)
        padded_heads = ((num_heads + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        self.decode_heads_mem_config = ttnn.create_sharded_memory_config(
            shape=(padded_heads, head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
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
        **_kwargs,
    ) -> "FunctionalDecoder":
        """Load dense HF weights and perform all host-side preparation."""

        num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
        if num_devices != 1:
            raise ValueError(f"FunctionalDecoder requires a 1x1 mesh, got {num_devices} devices")
        if batch < 1:
            raise ValueError(f"batch must be positive, got {batch}")
        if max_cache_len < EMITTED_PREFILL_SEQUENCE or max_cache_len > EMITTED_CACHE_LENGTH:
            raise ValueError(
                f"max_cache_len must be in [{EMITTED_PREFILL_SEQUENCE}, {EMITTED_CACHE_LENGTH}], got {max_cache_len}"
            )

        hidden_size = int(_config_value(hf_config, "hidden_size"))
        num_heads = int(_config_value(hf_config, "num_attention_heads"))
        num_kv_heads = int(_config_value(hf_config, "num_key_value_heads"))
        head_dim = int(_config_value(hf_config, "head_dim") or hidden_size // num_heads)
        intermediate_size = int(_config_value(hf_config, "intermediate_size"))
        num_experts = int(_config_value(hf_config, "num_local_experts"))
        experts_per_token = int(_config_value(hf_config, "num_experts_per_tok"))
        rms_norm_eps = float(_config_value(hf_config, "rms_norm_eps"))
        sliding_window = int(_config_value(hf_config, "sliding_window"))
        swiglu_limit = float(_config_value(hf_config, "swiglu_limit"))

        expected_config = {
            "hidden_size": (hidden_size, 2880),
            "num_attention_heads": (num_heads, 64),
            "num_key_value_heads": (num_kv_heads, 8),
            "head_dim": (head_dim, 64),
            "intermediate_size": (intermediate_size, 2880),
            "num_local_experts": (num_experts, 32),
            "num_experts_per_tok": (experts_per_token, 4),
        }
        mismatches = [
            f"{name}={actual} (emit={emitted})"
            for name, (actual, emitted) in expected_config.items()
            if actual != emitted
        ]
        if mismatches:
            raise ValueError("HF config does not match the captured graph: " + ", ".join(mismatches))

        q = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.weight").to(torch.bfloat16)
        k = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.weight").to(torch.bfloat16)
        v = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.weight").to(torch.bfloat16)
        q_bias = _state_tensor(state_dict, layer_idx, "self_attn.q_proj.bias").to(torch.bfloat16)
        k_bias = _state_tensor(state_dict, layer_idx, "self_attn.k_proj.bias").to(torch.bfloat16)
        v_bias = _state_tensor(state_dict, layer_idx, "self_attn.v_proj.bias").to(torch.bfloat16)
        output = _state_tensor(state_dict, layer_idx, "self_attn.o_proj.weight").to(torch.bfloat16)
        output_bias = _state_tensor(state_dict, layer_idx, "self_attn.o_proj.bias").to(torch.bfloat16)
        sinks = _state_tensor(state_dict, layer_idx, "self_attn.sinks").to(torch.bfloat16)
        router = _state_tensor(state_dict, layer_idx, "mlp.router.weight").to(torch.bfloat16)
        router_bias = _state_tensor(state_dict, layer_idx, "mlp.router.bias").to(torch.bfloat16)
        gate_up = _expert_tensor(state_dict, layer_idx, "gate_up_proj")
        gate_up_bias = _state_tensor(state_dict, layer_idx, "mlp.experts.gate_up_proj_bias").to(torch.bfloat16)
        down = _expert_tensor(state_dict, layer_idx, "down_proj")
        down_bias = _state_tensor(state_dict, layer_idx, "mlp.experts.down_proj_bias").to(torch.bfloat16)
        input_norm = _state_tensor(state_dict, layer_idx, "input_layernorm.weight").to(torch.bfloat16)
        post_attention_norm = _state_tensor(state_dict, layer_idx, "post_attention_layernorm.weight").to(torch.bfloat16)

        expected_shapes = {
            "q": (num_heads * head_dim, hidden_size),
            "k": (num_kv_heads * head_dim, hidden_size),
            "v": (num_kv_heads * head_dim, hidden_size),
            "q_bias": (num_heads * head_dim,),
            "k_bias": (num_kv_heads * head_dim,),
            "v_bias": (num_kv_heads * head_dim,),
            "output": (hidden_size, num_heads * head_dim),
            "output_bias": (hidden_size,),
            "sinks": (num_heads,),
            "router": (num_experts, hidden_size),
            "router_bias": (num_experts,),
            "gate_up": (num_experts, hidden_size, 2 * intermediate_size),
            "gate_up_bias": (num_experts, 2 * intermediate_size),
            "down": (num_experts, intermediate_size, hidden_size),
            "down_bias": (num_experts, hidden_size),
            "input_norm": (hidden_size,),
            "post_attention_norm": (hidden_size,),
        }
        tensors = locals()
        for name, expected in expected_shapes.items():
            if tuple(tensors[name].shape) != expected:
                raise ValueError(f"{name} has shape {tuple(tensors[name].shape)}, expected {expected}")

        # The emit's consteval receives V, K, Q but transposes and concatenates
        # them in Q -> K -> V order. Dense weights replace the four column shards.
        qkv = torch.cat((q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)), dim=-1)
        qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=-1)

        rotary = GptOssRotaryEmbedding(hf_config)
        positions = torch.arange(max_cache_len, dtype=torch.long).unsqueeze(0)
        rope_probe = torch.zeros((1, 1, max_cache_len, head_dim), dtype=torch.bfloat16)
        cos, sin = rotary(rope_probe, positions)
        cos = torch.cat((cos, cos), dim=-1).unsqueeze(1).to(torch.bfloat16)
        sin = torch.cat((sin, sin), dim=-1).unsqueeze(1).to(torch.bfloat16)

        query_positions = torch.arange(max_cache_len).view(max_cache_len, 1)
        key_positions = torch.arange(max_cache_len).view(1, max_cache_len)
        causal_mask = torch.zeros((1, 1, max_cache_len, max_cache_len), dtype=torch.bfloat16)
        invalid = key_positions > query_positions
        if sliding_window < max_cache_len:
            invalid = invalid | (key_positions <= query_positions - sliding_window)
        causal_mask.masked_fill_(invalid.view(1, 1, max_cache_len, max_cache_len), torch.finfo(torch.bfloat16).min)

        # TT SDPA multiplies both QK and sink inputs by scale. The manual
        # prefill path uses raw sinks; decode therefore stores sink / scale.
        prefill_sinks = sinks.view(1, num_heads, 1, 1).expand(batch, num_heads, max_cache_len, 1).contiguous()
        decode_sinks = torch.nn.functional.pad(sinks.view(num_heads, 1), (0, ttnn.TILE_SIZE - 1)) / (head_dim**-0.5)

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
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            rms_norm_eps=rms_norm_eps,
            sliding_window=sliding_window,
            swiglu_limit=swiglu_limit,
            input_norm=_to_device_tensor(input_norm, mesh_device),
            post_attention_norm=_to_device_tensor(post_attention_norm, mesh_device),
            qkv_weight=_to_device_tensor(qkv, mesh_device),
            qkv_bias=_to_device_tensor(qkv_bias, mesh_device),
            output_weight=_to_device_tensor(output.transpose(0, 1), mesh_device),
            output_bias=_to_device_tensor(output_bias, mesh_device),
            attention_sinks=_to_device_tensor(prefill_sinks, mesh_device),
            decode_attention_sinks=_to_device_tensor(decode_sinks, mesh_device),
            router_weight=_to_device_tensor(router.transpose(0, 1), mesh_device),
            router_bias=_to_device_tensor(router_bias, mesh_device),
            gate_up_weight=_to_device_tensor(gate_up, mesh_device),
            gate_up_bias=_to_device_tensor(gate_up_bias.unsqueeze(1), mesh_device),
            down_weight=_to_device_tensor(down, mesh_device),
            down_bias=_to_device_tensor(down_bias.unsqueeze(1), mesh_device),
            rotary_cos=_to_device_tensor(cos, mesh_device),
            rotary_sin=_to_device_tensor(sin, mesh_device),
            attention_mask=_to_device_tensor(causal_mask, mesh_device),
            position_indices=_to_device_tensor(
                torch.arange(max_cache_len, dtype=torch.int32),
                mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
            ),
        )

    def _validate_hidden_states(self, hidden_states, expected_seq_len: int | None = None) -> int:
        shape = tuple(hidden_states.shape)
        expected_prefix = (1, self.batch)
        if len(shape) != 4 or shape[:2] != expected_prefix or shape[3] != self.hidden_size:
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

    def _moe_forward(self, hidden_states, seq_len: int):
        tokens = self.batch * seq_len
        token_states = ttnn.reshape(hidden_states, [tokens, self.hidden_size])

        # The TP4 emit repeats each token across eight local experts. Dense
        # translation repeats across all 32 and uses the full expert weights.
        expert_input = ttnn.reshape(token_states, [1, tokens, self.hidden_size])
        expert_input = ttnn.repeat(
            expert_input,
            ttnn.Shape([self.num_experts, 1, 1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate_up = ttnn.matmul(
            expert_input,
            self.gate_up_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        gate_up = ttnn.add(gate_up, self.gate_up_bias, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        up = ttnn.slice(
            gate_up,
            [0, 0, 1],
            [self.num_experts, tokens, 2 * self.intermediate_size],
            [1, 1, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.clamp(up, -self.swiglu_limit, self.swiglu_limit, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.add(up, 1.0, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        gate = ttnn.slice(
            gate_up,
            [0, 0, 0],
            [self.num_experts, tokens, 2 * self.intermediate_size],
            [1, 1, 2],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.clamp(gate, float("-inf"), self.swiglu_limit, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sigmoid = ttnn.sigmoid(
            ttnn.multiply(gate, 1.703125, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gated = ttnn.multiply(gate, sigmoid, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        down_input = ttnn.multiply(up, gated, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        expert_output = ttnn.matmul(
            down_input,
            self.down_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        expert_output = ttnn.add(
            expert_output,
            self.down_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # The emit promotes the normalized token to FP32 for routing, performs
        # the router linear in FP32, then returns to BF16 before top-k.
        router_input = ttnn.typecast(token_states, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        router_logits = ttnn.linear(
            router_input,
            self.router_weight,
            bias=self.router_bias,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        router_logits = ttnn.typecast(router_logits, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        top_values, top_indices = ttnn.topk(
            router_logits,
            self.experts_per_token,
            1,
            True,
            True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        top_weights = ttnn.softmax(top_values, 1, memory_config=ttnn.DRAM_MEMORY_CONFIG, numeric_stable=True)
        routing = ttnn.scatter(
            input=ttnn.zeros_like(router_logits),
            dim=1,
            index=top_indices,
            src=top_weights,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        routing = ttnn.permute(routing, (1, 0), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routing = ttnn.reshape(routing, [self.num_experts, tokens, 1])
        weighted = ttnn.multiply(
            expert_output,
            routing,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        reduced = ttnn.sum(weighted, [0], False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(reduced, [1, self.batch, seq_len, self.hidden_size])

    def prefill_forward(self, hidden_states, key_cache, value_cache):
        """Run the translated g0 ``fill_cache`` path from position zero."""

        seq_len = self._validate_hidden_states(hidden_states)
        if seq_len <= 1:
            raise ValueError("prefill_forward requires seq_len > 1; use decode_forward for one token")
        self._validate_caches(key_cache, value_cache)

        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.input_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        fused_qkv = ttnn.linear(
            normed,
            self.qkv_weight,
            bias=self.qkv_bias,
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

        cos = ttnn.slice(self.rotary_cos, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        sin = ttnn.slice(self.rotary_sin, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim], [1, 1, 1, 1])
        query = ttnn.experimental.rotary_embedding(query, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.experimental.rotary_embedding(key, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
        # ``fill_cache`` takes one input batch row per call. Select each row
        # on device and target its matching cache row so batch remains a true
        # construction parameter without introducing host fallback.
        for user_id in range(self.batch):
            user_key = ttnn.slice(
                key,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            user_value = ttnn.slice(
                value,
                [user_id, 0, 0, 0],
                [user_id + 1, self.num_kv_heads, seq_len, self.head_dim],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.fill_cache(key_cache, user_key, user_id)
            ttnn.fill_cache(value_cache, user_value, user_id)

        repeated_key = ttnn.repeat_interleave(key_cache, self.num_heads // self.num_kv_heads, 1)
        transposed_key = ttnn.permute(repeated_key, (0, 1, 3, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores = ttnn.matmul(query, transposed_key, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores = ttnn.multiply(scores, self.scale, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mask = ttnn.slice(
            self.attention_mask,
            [0, 0, 0, 0],
            [1, 1, seq_len, self.max_cache_len],
            [1, 1, 1, 1],
        )
        scores = ttnn.add(scores, mask, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sinks = ttnn.slice(
            self.attention_sinks,
            [0, 0, 0, 0],
            [self.batch, self.num_heads, seq_len, 1],
            [1, 1, 1, 1],
        )
        probabilities = ttnn.softmax(
            ttnn.concat([scores, sinks], 3, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            numeric_stable=True,
        )
        probabilities = ttnn.slice(
            probabilities,
            [0, 0, 0, 0],
            [self.batch, self.num_heads, seq_len, self.max_cache_len],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        repeated_value = ttnn.repeat_interleave(value_cache, self.num_heads // self.num_kv_heads, 1)
        attention = ttnn.matmul(
            probabilities,
            repeated_value,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.transformer.concatenate_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.reshape(attention, [1, self.batch, seq_len, self.num_heads * self.head_dim])
        attention = ttnn.linear(
            attention,
            self.output_weight,
            bias=self.output_bias,
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
            compute_kernel_config=self.compute_kernel_config,
        )
        hidden_states = self._moe_forward(hidden_states, seq_len)
        return ttnn.add(
            residual,
            hidden_states,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def decode_forward(self, hidden_states, key_cache, value_cache, *, current_pos: int):
        """Run the translated g1 paged-cache/decode-SDPA path for one token."""

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
            compute_kernel_config=self.compute_kernel_config,
        )
        fused_qkv = ttnn.linear(
            normed,
            self.qkv_weight,
            bias=self.qkv_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fused_qkv = ttnn.reshape(
            fused_qkv,
            [1, 1, self.batch, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim],
        )
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=self.decode_heads_mem_config,
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
            memory_config=self.decode_heads_mem_config,
        )
        key = ttnn.experimental.rotary_embedding(
            key,
            cos,
            sin,
            0,
            memory_config=self.decode_heads_mem_config,
        )

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

        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        decode_mask = ttnn.slice(
            self.attention_mask,
            [0, 0, current_pos, 0],
            [1, 1, current_pos + 1, self.max_cache_len],
            [1, 1, 1, 1],
        )
        decode_mask = ttnn.repeat(
            decode_mask,
            ttnn.Shape([1, 1, self.num_heads, 1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            is_causal=False,
            attn_mask=decode_mask,
            cur_pos_tensor=None,
            attention_sink=self.decode_attention_sinks,
            scale=self.scale,
            sliding_window_size=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.to_memory_config(attention, self.decode_heads_mem_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(attention, num_heads=self.num_heads)
        attention = ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.slice(
            attention,
            [0, 0, 0, 0],
            [1, 1, self.batch, self.num_heads * self.head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.permute(attention, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.linear(
            attention,
            self.output_weight,
            bias=self.output_bias,
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
            compute_kernel_config=self.compute_kernel_config,
        )
        hidden_states = self._moe_forward(hidden_states, 1)
        return ttnn.add(
            residual,
            hidden_states,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
