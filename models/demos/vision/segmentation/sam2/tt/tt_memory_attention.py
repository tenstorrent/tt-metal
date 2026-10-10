# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI U.S. Corp., for the TTNN port. Reference code © Meta Platforms, Inc. (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
import math

import torch

import ttnn
from models.demos.vision.segmentation.sam2.tt.tt_hiera import _height_shard_cfg, _hs_mem

ATTENTION_FFN_HIDDEN_DTYPE = ttnn.bfloat8_b
ATTENTION_BANK_DTYPE = ttnn.bfloat8_b
ATTENTION_BF16_FFN_LAYERS = frozenset({2})


def _compute_axial_cis():
    positions = torch.arange(4096, dtype=torch.float32)
    x_positions = positions % 64
    y_positions = torch.div(positions, 64, rounding_mode="floor")
    frequencies = 1.0 / (10000.0 ** (torch.arange(0, 256, 4).float() / 256))
    x_angles = torch.outer(x_positions, frequencies)
    y_angles = torch.outer(y_positions, frequencies)
    return torch.cat(
        [
            torch.polar(torch.ones_like(x_angles), x_angles),
            torch.polar(torch.ones_like(y_angles), y_angles),
        ],
        dim=-1,
    )


class TtRoPE:
    def __init__(self, device):
        self.device = device
        self.head_dim = 256
        self._freqs = _compute_axial_cis()
        rotate_matrix = torch.kron(torch.eye(16), torch.tensor([[0.0, 1.0], [-1.0, 0.0]]))
        self.trans_mat = ttnn.from_torch(
            rotate_matrix.reshape(1, 1, 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._tables = None

    def tables(self):
        if self._tables is None:
            self._tables = tuple(
                ttnn.from_torch(
                    torch.repeat_interleave(values, 2, dim=-1).reshape(1, 1, 4096, self.head_dim),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for values in (self._freqs.real, self._freqs.imag)
            )
        return self._tables

    def apply(self, x, cos, sin):
        return ttnn.experimental.rotary_embedding_llama(x, cos, sin, self.trans_mat, is_decode_mode=False)


def _ln(x, ns, eps=1e-5, compute_kernel_config=None):
    if x.memory_config().is_sharded():
        return ttnn.moreh_layer_norm(
            x,
            1,
            eps,
            ns.weight,
            ns.bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )[0]
    return ttnn.layer_norm(
        x,
        weight=ns.weight,
        bias=ns.bias,
        epsilon=eps,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )


class TtRoPEAttention:
    def __init__(self, params, device, rope: TtRoPE):
        self.p = params
        self.rope = rope
        self.head_dim = rope.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv_split_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.cross_sdpa_query_heads = 8
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self.cross_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=64,
            k_chunk_size=512,
            exp_approx_mode=False,
        )

    def self_attention(self, x, *, output_memory_config):
        batch, tokens, _ = x.shape
        qkv_input = x
        if qkv_input.layout != ttnn.TILE_LAYOUT:
            qkv_input = ttnn.to_layout(
                qkv_input,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        q, k, v = ttnn.experimental.minimal_matmul_split(
            qkv_input,
            self.p.qkv_proj.weight,
            chunks=3,
            dim=-1,
            bias_tensor=self.p.qkv_proj.bias,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.qkv_split_compute_kernel_config,
        )
        if qkv_input is not x:
            ttnn.deallocate(qkv_input)
        q = ttnn.reshape(q, (batch, 1, tokens, self.head_dim))
        k = ttnn.reshape(k, (batch, 1, tokens, self.head_dim))
        v = ttnn.reshape(v, (batch, 1, tokens, self.head_dim))
        cos, sin = self.rope.tables()
        unrotated_q = q
        q = self.rope.apply(unrotated_q, cos, sin)
        ttnn.deallocate(unrotated_q)
        unrotated_k = k
        k = self.rope.apply(unrotated_k, cos, sin)
        ttnn.deallocate(unrotated_k)
        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            memory_config=output_memory_config,
            program_config=self.sdpa_program_config,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        return ttnn.reshape(out, (batch, tokens, self.head_dim))

    def cross_attention(self, q, k, v, *, output_memory_config):
        batch = int(q.shape[0])
        query_tokens = int(q.shape[2])
        cos, sin = self.rope.tables()
        q = self.rope.apply(q, cos, sin)
        q = ttnn.reshape(
            q,
            (batch, self.cross_sdpa_query_heads, query_tokens // self.cross_sdpa_query_heads, self.head_dim),
        )
        out = ttnn.transformer.flash_mla_prefill(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            memory_config=output_memory_config,
            program_config=self.cross_sdpa_program_config,
        )
        ttnn.deallocate(q)
        return ttnn.reshape(out, (batch, query_tokens, int(v.shape[-1])))


class TtMemoryAttentionLayer:
    def __init__(
        self,
        params,
        device,
        rope: TtRoPE,
        layer_index: int,
        tokens: int,
    ):
        self.p = params
        self.tokens = tokens
        self.channels = int(params.linear2.weight.shape[-1])
        self.hidden = int(params.linear1.weight.shape[-1])
        self.cross_value_dim = int(params.cross_attn_image.latent_out_proj.weight.shape[-2])
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.layer_norm_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.self_attn = TtRoPEAttention(params.self_attn, device, rope)
        self.cross_attn = TtRoPEAttention(params.cross_attn_image, device, rope)
        self.ffn_hidden_dtype = (
            ttnn.bfloat16 if layer_index in ATTENTION_BF16_FFN_LAYERS else ATTENTION_FFN_HIDDEN_DTYPE
        )
        self.projection_program, self.grid = _height_shard_cfg(tokens, self.channels, self.channels)
        self.cross_projection_program, _ = _height_shard_cfg(tokens, self.cross_value_dim, self.channels)
        self.first_ffn_program, _ = _height_shard_cfg(
            tokens,
            self.channels,
            self.hidden,
            fused_activation=ttnn.UnaryOpType.RELU,
        )
        self.second_ffn_program, _ = _height_shard_cfg(tokens, self.hidden, self.channels)
        self.state_memory = _hs_mem(tokens, self.channels, self.grid)
        self.cross_output_memory = _hs_mem(tokens, self.cross_value_dim, self.grid)
        self.hidden_memory = _hs_mem(tokens, self.hidden, self.grid)

    def _normalize_state(self, state, parameters):
        return ttnn.moreh_layer_norm(
            state,
            1,
            1e-5,
            parameters.weight,
            parameters.bias,
            compute_kernel_config=self.layer_norm_compute_config,
        )[0]

    def _to_height_sharded(self, x, channels):
        flat = ttnn.reshape(x, (1, self.tokens, channels))
        if flat.memory_config().is_sharded():
            return flat
        if flat.layout != ttnn.TILE_LAYOUT:
            flat = ttnn.to_layout(
                flat,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        memory_config = self.state_memory if channels == self.channels else self.cross_output_memory
        return ttnn.to_memory_config(flat, memory_config)

    def _project_residual(self, state, branch, branch_channels, projection, program_config):
        if branch.memory_config().is_sharded():
            sharded = ttnn.reshape(branch, (1, self.tokens, branch_channels))
        else:
            sharded = self._to_height_sharded(branch, branch_channels)
            ttnn.deallocate(branch)
        projected = ttnn.linear(
            sharded,
            projection.weight,
            bias=projection.bias,
            program_config=program_config,
            memory_config=self.state_memory,
            compute_kernel_config=self.compute_config,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(sharded)
        state = ttnn.add_(state, projected)
        ttnn.deallocate(projected)
        return state

    def __call__(self, tgt, pre_projected_kv):
        batch, tokens, channels = tgt.shape
        normalized = _ln(tgt, self.p.norm1, compute_kernel_config=self.layer_norm_compute_config)
        self_output = self.self_attn.self_attention(
            normalized,
            output_memory_config=self.state_memory,
        )
        ttnn.deallocate(normalized)
        state = self._to_height_sharded(tgt, channels)
        if not tgt.memory_config().is_sharded():
            ttnn.deallocate(tgt)
        state = self._project_residual(
            state,
            self_output,
            channels,
            self.p.self_attn.out_proj,
            self.projection_program,
        )
        normalized = self._normalize_state(state, self.p.norm2)
        (query,) = ttnn.experimental.minimal_matmul_split(
            normalized,
            self.p.cross_attn_image.q_proj.weight,
            chunks=1,
            dim=-1,
            bias_tensor=self.p.cross_attn_image.q_proj.bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        ttnn.deallocate(normalized)
        query_dram = ttnn.reshape(query, (batch, 1, tokens, channels))
        projected_k, projected_v = pre_projected_kv
        cross_output = self.cross_attn.cross_attention(
            query_dram,
            projected_k,
            projected_v,
            output_memory_config=self.cross_output_memory,
        )
        ttnn.deallocate(query_dram)
        state = self._project_residual(
            state,
            cross_output,
            self.cross_value_dim,
            self.p.cross_attn_image.latent_out_proj,
            self.cross_projection_program,
        )
        normalized = self._normalize_state(state, self.p.norm3)
        hidden_state = ttnn.linear(
            normalized,
            self.p.linear1.weight,
            bias=self.p.linear1.bias,
            program_config=self.first_ffn_program,
            memory_config=self.hidden_memory,
            compute_kernel_config=self.compute_config,
            dtype=self.ffn_hidden_dtype,
        )
        ttnn.deallocate(normalized)
        output = ttnn.linear(
            hidden_state,
            self.p.linear2.weight,
            bias=self.p.linear2.bias,
            program_config=self.second_ffn_program,
            memory_config=self.state_memory,
            compute_kernel_config=self.compute_config,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(hidden_state)
        state = ttnn.add_(state, output)
        ttnn.deallocate(output)
        return ttnn.reshape(state, (batch, tokens, channels))


class TtMemoryAttention:
    def __init__(self, params, device):
        self.p = params
        self.rope = TtRoPE(device)
        tokens = 4096
        self.layers = [
            TtMemoryAttentionLayer(layer, device, self.rope, layer_index, tokens)
            for layer_index, layer in enumerate(params.layers)
        ]
        self.layer_norm_compute_config = self.layers[0].layer_norm_compute_config
        self._scaled_curr_pos = None

    def __call__(self, curr, curr_pos, pre_projected_kv_list):
        if self._scaled_curr_pos is None:
            self._scaled_curr_pos = ttnn.multiply(curr_pos, 0.1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.add(curr, self._scaled_curr_pos, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for i, layer in enumerate(self.layers):
            output = layer(output, pre_projected_kv_list[i])
        normalized = _ln(output, self.p.norm, compute_kernel_config=self.layer_norm_compute_config)
        ttnn.deallocate(output)
        return normalized
