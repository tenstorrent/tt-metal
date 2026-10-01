# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiniCPM4 full-sequence component port; AR KV decode is a later milestone.

Derived from OpenBMB/VoxCPM (Apache-2.0), pinned in config.py.
Attention masks cover the physical key padding, including tiny local patches.
No learned forward computation is performed on the host.
"""

import math

import torch
import ttnn

from ..config import validate_minicpm_config
from .ops import TtLinear, TtRMSNorm, compute_config, upload


def rope_constants(config, length):
    """Match upstream's max-position cache policy, not request-length scaling."""
    dim = validate_minicpm_config(config)
    rope = config['rope_scaling']
    original = rope['original_max_position_embeddings']
    maximum = config['max_position_embeddings']
    factor_key = 'long_factor' if maximum > original else 'short_factor'
    factors = torch.tensor(rope[factor_key], dtype=torch.float32)
    inverse = 1.0 / (config['rope_theta'] ** (torch.arange(0, dim, 2).float() / dim))
    frequencies = torch.outer(torch.arange(length).float(), 1.0 / factors) * inverse
    emb = torch.cat((frequencies, frequencies), dim=-1)
    scale = math.sqrt(1.0 + math.log(maximum / original) / math.log(original))
    return emb.cos() * scale, emb.sin() * scale


def rotate_half(hidden):
    shape = list(hidden.shape)
    half = shape[-1] // 2
    first_end = shape.copy()
    first_end[-1] = half
    second_start = [0] * len(shape)
    second_start[-1] = half
    first = ttnn.slice(hidden, [0] * len(shape), first_end)
    second = ttnn.slice(hidden, second_start, shape)
    return ttnn.concat([ttnn.multiply(second, -1.0), first], dim=-1)


class TtAttention:
    def __init__(self, config, state, prefix, device, dtype):
        self.config = config
        self.device, self.dtype = device, dtype
        self.head_dim = validate_minicpm_config(config)
        self.heads = config['num_attention_heads']
        self.kv_heads = config['num_key_value_heads']
        self.query = TtLinear(state, prefix + '.q_proj', device, dtype)
        self.key = TtLinear(state, prefix + '.k_proj', device, dtype)
        self.value = TtLinear(state, prefix + '.v_proj', device, dtype)
        self.output = TtLinear(state, prefix + '.o_proj', device, dtype)
        self.compute = compute_config(device)
        self.constants = {}

    def _constants(self, length, causal, padded_length):
        signature = (length, causal, padded_length)
        if signature not in self.constants:
            # Mask is a host-created structural constant, independent of activations.
            mask = torch.zeros(1, 1, padded_length, padded_length, dtype=torch.float32)
            mask[..., length:] = float('-inf')
            if causal:
                for position in range(length):
                    mask[..., position, position + 1:] = float('-inf')
            # Padded query rows still have a valid key, avoiding all -inf softmax.
            mask_tt = upload(mask, self.device, self.dtype)
            rope_tt = None
            if not self.config.get('no_rope', False):
                cos, sin = rope_constants(self.config, length)
                rope_tt = tuple(upload(x.reshape(1, 1, length, self.head_dim),
                                       self.device, ttnn.float32) for x in (cos, sin))
            self.constants[signature] = mask_tt, rope_tt
        return self.constants[signature]

    def _heads(self, hidden, heads):
        batch, length, _ = hidden.shape
        return ttnn.permute(ttnn.reshape(hidden, (batch, length, heads, self.head_dim)), (0, 2, 1, 3))

    def __call__(self, hidden, is_causal):
        batch, length, _ = hidden.shape
        query = self._heads(self.query(hidden), self.heads)
        key = self._heads(self.key(hidden), self.kv_heads)
        value = self._heads(self.value(hidden), self.kv_heads)
        padded_length = ((length + 31) // 32) * 32
        mask, rope = self._constants(length, is_causal, padded_length)
        if rope is not None:
            cos, sin = rope
            def apply_rope(x):
                # Upstream rotates in FP32 then casts back to model storage.
                x = ttnn.typecast(x, ttnn.float32)
                return ttnn.typecast(ttnn.add(ttnn.multiply(x, cos),
                    ttnn.multiply(rotate_half(x), sin)), self.dtype)
            query, key = apply_rope(query), apply_rope(key)
        if self.heads != self.kv_heads:
            repeats = self.heads // self.kv_heads
            key = ttnn.repeat_interleave(key, repeats=repeats, dim=1)
            value = ttnn.repeat_interleave(value, repeats=repeats, dim=1)
        # Materialize logical sequence padding before softmax. A logical S-wide
        # zero pad would otherwise count as a real key in short local attention.
        padding = [(0, 0), (0, 0), (0, padded_length - length), (0, 0)]
        if padded_length != length:
            query = ttnn.pad(query, padding, 0.0)
            key = ttnn.pad(key, padding, 0.0)
            value = ttnn.pad(value, padding, 0.0)
        scores = ttnn.matmul(query, ttnn.transpose(key, -1, -2), compute_kernel_config=self.compute)
        scores = ttnn.add(ttnn.multiply(scores, self.head_dim ** -0.5), mask)
        probabilities = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute)
        context = ttnn.matmul(probabilities, value, compute_kernel_config=self.compute)
        context = ttnn.slice(context, (0, 0, 0, 0), (batch, self.heads, length, self.head_dim))
        context = ttnn.reshape(ttnn.permute(context, (0, 2, 1, 3)),
                               (batch, length, self.heads * self.head_dim))
        return self.output(context)


class TtDecoderLayer:
    def __init__(self, config, state, prefix, device, dtype):
        self.attention = TtAttention(config, state, prefix + '.self_attn', device, dtype)
        self.input_norm = TtRMSNorm(state, prefix + '.input_layernorm', config['rms_norm_eps'], device, dtype)
        self.post_norm = TtRMSNorm(state, prefix + '.post_attention_layernorm', config['rms_norm_eps'], device, dtype)
        self.gate = TtLinear(state, prefix + '.mlp.gate_proj', device, dtype)
        self.up = TtLinear(state, prefix + '.mlp.up_proj', device, dtype)
        self.down = TtLinear(state, prefix + '.mlp.down_proj', device, dtype)
        self.residual_scale = (config['scale_depth'] / math.sqrt(config['num_hidden_layers'])
                               if config.get('use_mup', True) else 1.0)

    def __call__(self, hidden, is_causal):
        branch = self.attention(self.input_norm(hidden), is_causal)
        hidden = ttnn.add(hidden, ttnn.multiply(branch, self.residual_scale))
        normalized = self.post_norm(hidden)
        branch = self.down(ttnn.multiply(ttnn.silu(self.gate(normalized)), self.up(normalized)))
        return ttnn.add(hidden, ttnn.multiply(branch, self.residual_scale))


class TtMiniCPMModel:
    def __init__(self, config, state, prefix, device, dtype):
        validate_minicpm_config(config)
        self.config = config
        self.layers = [TtDecoderLayer(config, state, f'{prefix}.layers.{i}', device, dtype)
                       for i in range(config['num_hidden_layers'])]
        self.norm = TtRMSNorm(state, prefix + '.norm', config['rms_norm_eps'], device, dtype)

    def __call__(self, hidden, is_causal=True):
        if len(hidden.shape) != 3 or hidden.shape[-1] != self.config['hidden_size']:
            raise ValueError('MiniCPM requires [batch,sequence,hidden_size] embeddings')
        if not 0 < hidden.shape[1] <= self.config['max_position_embeddings']:
            raise ValueError('Sequence length exceeds the configured RoPE cache')
        for layer in self.layers:
            hidden = layer(hidden, is_causal)
        return self.norm(hidden)
