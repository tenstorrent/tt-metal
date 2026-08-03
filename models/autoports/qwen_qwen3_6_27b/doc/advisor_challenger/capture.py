"""Qwen3.6 hooks for advisor-challenger's fixed capture template."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import _state as full_state
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import _state as linear_state
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, _to_device

_template_path = Path(__file__).parents[5] / ".agents/skills/advisor-challenger/scripts/capture_template.py"
_spec = importlib.util.spec_from_file_location("advisor_challenger_capture_template", _template_path)
template = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(template)


def config():
    return AutoConfig.from_pretrained(MODEL_ID).text_config


def synthetic_state_dict(cfg):
    return (linear_state if template.LAYER_KIND == "linear_attention" else full_state)(cfg)


def decode(hidden):
    batch = template.BATCH
    page_table = _to_device(
        torch.arange(batch, dtype=torch.int32).reshape(batch, 1),
        mesh_device=template._DECODER.mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    positions = _to_device(
        torch.zeros(batch, dtype=torch.uint32),
        mesh_device=template._DECODER.mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    return template._DECODER.decode_forward(
        hidden_states=hidden,
        page_table=page_table,
        current_positions=positions,
    )


template._config = config
template._synthetic_state_dict = synthetic_state_dict
template.decode = decode


class _CaptureDecoder(template.OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state, **kwargs):
        kwargs["max_context"] = template.MAX_CONTEXT
        kwargs["page_size"] = 64
        return super().from_state_dict(state, **kwargs)

    def _rms_norm_decode_sharded(self, hidden_states, name):
        memory_config = self._decode_residual_memory_config()
        hidden_states = ttnn.to_memory_config(hidden_states, memory_config)
        return ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=self.weights[name],
            memory_config=memory_config,
            program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[8, 1],
                subblock_w=4,
                block_h=1,
                block_w=self.hidden_size // 8 // 32,
                inplace=False,
            ),
        )

    def decode_forward(self, *, hidden_states, page_table, current_positions):
        memory_config = self._decode_residual_memory_config()
        residual = ttnn.to_memory_config(hidden_states, memory_config)
        hidden_states = self._rms_norm_decode_sharded(residual, "input_norm")
        hidden_states = self._token_mixer_decode(hidden_states, page_table, current_positions)
        hidden_states = ttnn.to_memory_config(hidden_states, memory_config)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=memory_config)
        residual = hidden_states
        hidden_states = self._rms_norm_decode_sharded(hidden_states, "post_attention_norm")
        hidden_states = self._mlp(hidden_states)
        return ttnn.add(residual, hidden_states, memory_config=memory_config)

    def _optimized_decode_linear(self, activation, weight, **kwargs):
        weight_name = next(name for name, value in self.weights.items() if value is weight)
        program_config = self.decode_program_configs.get(weight_name)
        if program_config is not None:
            activation = ttnn.to_memory_config(activation, self.decode_input_memory_configs[weight_name])
            kwargs["memory_config"] = self.decode_output_memory_configs[weight_name]
            kwargs["program_config"] = program_config
        kwargs["compute_kernel_config"] = self.compute_kernel_config
        kwargs["dtype"] = ttnn.bfloat16
        output = ttnn.linear(activation, weight, **kwargs)
        if weight_name == "packed_linear_inputs":
            output = ttnn.to_memory_config(output, ttnn.L1_MEMORY_CONFIG)
        return output

    @staticmethod
    def _optimized_decode_concat(tensors, *args, **kwargs):
        tensors = [ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG) for tensor in tensors]
        kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
        return ttnn.concat(tensors, *args, **kwargs)

    def _full_attention_decode(self, hidden_states, page_table, current_positions):
        cache_positions = ttnn.typecast(current_positions, ttnn.int32)
        projected = self._optimized_decode_linear(
            hidden_states, self.weights["packed_qkv"], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        total_width = 2 * q_width + 2 * kv_width
        q = ttnn.slice(projected, (0, 0, 0, 0), (1, 1, self.batch, q_width))
        gate = ttnn.slice(projected, (0, 0, 0, q_width), (1, 1, self.batch, 2 * q_width))
        k = ttnn.slice(projected, (0, 0, 0, 2 * q_width), (1, 1, self.batch, 2 * q_width + kv_width))
        v = ttnn.slice(
            projected,
            (0, 0, 0, 2 * q_width + kv_width),
            (1, 1, self.batch, total_width),
        )
        fused_qkv = self._optimized_decode_concat([q, k, v], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=self.decode_attention_memory_config,
        )
        q = self._per_head_norm(q, "q_norm")
        k = self._per_head_norm(k, "k_norm")
        q = self._partial_rope_decode(q, current_positions)
        k = self._partial_rope_decode(k, current_positions)
        ttnn.experimental.paged_update_cache(
            self.caches["key"], k, update_idxs_tensor=cache_positions, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            self.caches["value"], v, update_idxs_tensor=cache_positions, page_table=page_table
        )
        attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.caches["key"],
            self.caches["value"],
            cur_pos_tensor=cache_positions,
            page_table_tensor=page_table,
            scale=self.head_dim**-0.5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.to_memory_config(attention, self.decode_attention_memory_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(attention, num_heads=self.num_heads)
        attention = ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.multiply(attention, ttnn.sigmoid(gate))
        attention = self._optimized_decode_linear(
            attention, self.weights["o_proj"], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        attention = ttnn.slice(attention, (0, 0, 0, 0), (1, 1, self.batch, self.hidden_size))
        return ttnn.reshape(attention, (1, 1, self.batch, self.hidden_size))

    @staticmethod
    def _rotate_half(tensor):
        width = tensor.shape[-1]
        half = width // 2
        first = ttnn.slice(tensor, (0, 0, 0, 0), (tensor.shape[0], tensor.shape[1], tensor.shape[2], half))
        second = ttnn.slice(
            tensor, (0, 0, 0, half), (tensor.shape[0], tensor.shape[1], tensor.shape[2], width)
        )
        return ttnn.concat([ttnn.neg(second), first], dim=-1)

    def _partial_rope_decode(self, tensor, current_positions):
        rotary_dim = int(self.head_dim * float(self.hf_config.partial_rotary_factor))
        shape = tensor.shape
        rotary = ttnn.slice(tensor, (0, 0, 0, 0), (shape[0], shape[1], shape[2], rotary_dim))
        passthrough = ttnn.slice(
            tensor, (0, 0, 0, rotary_dim), (shape[0], shape[1], shape[2], shape[3])
        )
        cos = ttnn.embedding(current_positions, self.rope["cos"], layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(current_positions, self.rope["sin"], layout=ttnn.TILE_LAYOUT)
        cos = ttnn.transpose(ttnn.unsqueeze_to_4D(cos), 1, 2)
        sin = ttnn.transpose(ttnn.unsqueeze_to_4D(sin), 1, 2)
        cos = ttnn.slice(cos, (0, 0, 0, 0), (1, self.batch, 1, rotary_dim))
        sin = ttnn.slice(sin, (0, 0, 0, 0), (1, self.batch, 1, rotary_dim))
        cos = ttnn.repeat(cos, ttnn.Shape([1, 1, shape[2], 1]))
        sin = ttnn.repeat(sin, ttnn.Shape([1, 1, shape[2], 1]))
        rotary = ttnn.add(ttnn.multiply(rotary, cos), ttnn.multiply(self._rotate_half(rotary), sin))
        return ttnn.to_memory_config(
            ttnn.concat([rotary, passthrough], dim=-1), self.decode_attention_memory_config
        )

    def _linear_attention_decode(self, hidden_states):
        key_heads = int(self.hf_config.linear_num_key_heads)
        value_heads = int(self.hf_config.linear_num_value_heads)
        key_dim = int(self.hf_config.linear_key_head_dim)
        value_dim = int(self.hf_config.linear_value_head_dim)
        key_width = key_heads * key_dim
        value_width = value_heads * value_dim
        projected = self._optimized_decode_linear(
            hidden_states, self.weights["packed_linear_inputs"], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        conv_width = 2 * key_width + value_width
        total_width = conv_width + value_width + 2 * value_heads
        mixed = ttnn.slice(projected, (0, 0, 0, 0), (1, 1, self.batch, conv_width))
        z = ttnn.slice(
            projected, (0, 0, 0, conv_width), (1, 1, self.batch, conv_width + value_width)
        )
        beta = ttnn.slice(
            projected,
            (0, 0, 0, conv_width + value_width),
            (1, 1, self.batch, conv_width + value_width + value_heads),
        )
        decay = ttnn.slice(
            projected,
            (0, 0, 0, conv_width + value_width + value_heads),
            (1, 1, self.batch, total_width),
        )
        mixed = ttnn.permute(mixed, (0, 2, 3, 1))
        conv_shape = self.caches["conv"].shape
        conv_tail = ttnn.slice(
            self.caches["conv"],
            (0, 0, 0, 1),
            (conv_shape[0], conv_shape[1], conv_shape[2], conv_shape[3]),
        )
        next_conv_state = ttnn.concat([conv_tail, mixed], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mixed = ttnn.sum(ttnn.multiply(next_conv_state, self.weights["conv"]), dim=-1, keepdim=True)
        mixed = ttnn.silu(mixed)
        ttnn.copy(next_conv_state, self.caches["conv"])
        mixed = ttnn.permute(mixed, (0, 3, 1, 2))
        query = ttnn.slice(mixed, (0, 0, 0, 0), (1, 1, self.batch, key_width))
        key = ttnn.slice(mixed, (0, 0, 0, key_width), (1, 1, self.batch, 2 * key_width))
        value = ttnn.slice(mixed, (0, 0, 0, 2 * key_width), (1, 1, self.batch, conv_width))
        query = ttnn.permute(ttnn.reshape(query, (self.batch, 1, key_heads, key_dim)), (0, 2, 1, 3))
        key = ttnn.permute(ttnn.reshape(key, (self.batch, 1, key_heads, key_dim)), (0, 2, 1, 3))
        value = ttnn.permute(ttnn.reshape(value, (self.batch, 1, value_heads, value_dim)), (0, 2, 1, 3))
        repeats = value_heads // key_heads
        query = self._l2_norm(ttnn.repeat_interleave(query, repeats, dim=1))
        key = self._l2_norm(ttnn.repeat_interleave(key, repeats, dim=1))
        query = ttnn.multiply(query, key_dim**-0.5)
        beta = ttnn.reshape(ttnn.sigmoid(beta), (self.batch, value_heads, 1, 1))
        decay = ttnn.exp(
            ttnn.reshape(
                ttnn.multiply(self.weights["a"], ttnn.softplus(ttnn.add(decay, self.weights["dt_bias"]))),
                (self.batch, value_heads, 1, 1),
            )
        )
        recurrent = ttnn.multiply(self.caches["recurrent"], decay)
        memory_value = ttnn.matmul(key, recurrent)
        delta = ttnn.multiply(ttnn.subtract(value, memory_value), beta)
        recurrent = ttnn.add(recurrent, ttnn.matmul(ttnn.transpose(key, -2, -1), delta))
        output = ttnn.matmul(query, recurrent)
        ttnn.copy(recurrent, self.caches["recurrent"])
        output = ttnn.rms_norm(
            output,
            epsilon=self.eps,
            weight=self.weights["gated_norm"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        z = ttnn.reshape(z, (self.batch, value_heads, 1, value_dim))
        output = ttnn.multiply(output, ttnn.silu(z))
        output = ttnn.reshape(ttnn.permute(output, (2, 0, 1, 3)), (1, 1, self.batch, value_width))
        return self._optimized_decode_linear(
            output, self.weights["out_proj"], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )


template.OptimizedDecoder = _CaptureDecoder


def make_inputs(device):
    decoder = template._build(device)
    hidden = _to_device(
        torch.zeros(1, 1, template.BATCH, template._CONFIG.hidden_size, dtype=torch.bfloat16),
        mesh_device=device,
    )
    template._record_traced_dtypes(template.os.environ.get("CHALLENGER_OUT_DIR", "."))
    return (hidden,)


if __name__ == "__main__":
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        (hidden,) = make_inputs(device)
        decode(hidden)
        ttnn.synchronize_device(device)
        print(f"capture target builds: kind={template.LAYER_KIND} idx={template.LAYER_IDX} batch={template.BATCH}")
    finally:
        ttnn.close_mesh_device(device)
