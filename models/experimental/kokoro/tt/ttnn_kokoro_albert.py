# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full Kokoro PL-BERT `AlbertModel` forward on TTNN (12× shared layer, device weights)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

import ttnn
from models.experimental.kokoro.tt.common import default_compute_kernel_config, dram_tile_config
from models.experimental.functional_common.attention_mask_functions import get_extended_attention_mask


def _gelu_new_approx(x: ttnn.Tensor, *, memory_config) -> ttnn.Tensor:
    """HF `gelu_new` (tanh approximation), same as `models/tt_dit/layers/linear.gelu_tanh`."""
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    inner = ttnn.add(x, ttnn.multiply(ttnn.pow(x, 3), 0.044715), memory_config=memory_config)
    tanh_br = ttnn.tanh(ttnn.multiply(inner, sqrt_2_over_pi, memory_config=memory_config), memory_config=memory_config)
    one = ttnn.full(
        tanh_br.shape,
        fill_value=1.0,
        dtype=x.dtype,
        layout=x.layout,
        device=x.device(),
        memory_config=memory_config,
    )
    bracket = ttnn.add(one, tanh_br, memory_config=memory_config)
    out = ttnn.multiply(x, bracket, memory_config=memory_config)
    return ttnn.multiply(out, 0.5, memory_config=memory_config)


def _ensure_bsh(x: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    """
    Normalize activations to rank-3 [B, S, H].

    Some TTNN ops return rank-4 with a singleton dim, e.g. [B, 1, S, H].
    Albert attention math expects [B, S, H].
    """
    shape = list(x.shape)
    if len(shape) == 3:
        return x
    if len(shape) == 4 and shape[1] == 1:
        return ttnn.reshape(x, [shape[0], shape[2], shape[3]], memory_config=memory_config)
    raise ValueError(f"Unexpected hidden_states rank/shape for Albert: rank={len(shape)} shape={shape}")


def _albert_attention(
    config,
    hidden_states: ttnn.Tensor,
    attention_mask: ttnn.Tensor | None,
    *,
    parameters: SimpleNamespace,
    mesh_device: ttnn.MeshDevice,
    memory_config: ttnn.MemoryConfig,
    compute_kernel_config,
) -> ttnn.Tensor:
    fallback_reshape = ttnn.get_fallback_function(ttnn.reshape)
    num_heads = config.num_attention_heads
    hidden_states = _ensure_bsh(hidden_states, memory_config=memory_config)
    batch_size, sequence_size, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    q = ttnn.linear(
        hidden_states,
        parameters.query.weight,
        bias=parameters.query.bias,
        transpose_b=True,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    q = ttnn.to_layout(q, layout=ttnn.ROW_MAJOR_LAYOUT)
    q = fallback_reshape(q, (batch_size, sequence_size, num_heads, head_size))
    q = ttnn.to_layout(q, layout=ttnn.TILE_LAYOUT)
    q = ttnn.permute(q, (0, 2, 1, 3))

    k = ttnn.linear(
        hidden_states,
        parameters.key.weight,
        bias=parameters.key.bias,
        transpose_b=True,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    k = ttnn.to_layout(k, layout=ttnn.ROW_MAJOR_LAYOUT)
    k = fallback_reshape(k, (batch_size, sequence_size, num_heads, head_size))
    k = ttnn.to_layout(k, layout=ttnn.TILE_LAYOUT)
    k = ttnn.permute(k, (0, 2, 3, 1))

    v = ttnn.linear(
        hidden_states,
        parameters.value.weight,
        bias=parameters.value.bias,
        transpose_b=True,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    v = ttnn.to_layout(v, layout=ttnn.ROW_MAJOR_LAYOUT)
    v = fallback_reshape(v, (batch_size, sequence_size, num_heads, head_size))
    v = ttnn.to_layout(v, layout=ttnn.TILE_LAYOUT)
    v = ttnn.permute(v, (0, 2, 1, 3))

    attention_scores = q @ k
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    attention_scores = attention_scores * (1.0 / (head_size**0.5))
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = ttnn.softmax(attention_scores, dim=-1)
    ttnn.deallocate(attention_scores)

    context_layer = attention_probs @ v
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(v)

    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    context_layer = ttnn.to_layout(context_layer, ttnn.ROW_MAJOR_LAYOUT)
    context_layer = fallback_reshape(context_layer, (batch_size, sequence_size, hidden_size))
    context_layer = ttnn.to_layout(context_layer, ttnn.TILE_LAYOUT)

    self_output = ttnn.linear(
        context_layer,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        transpose_b=True,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(context_layer)

    attention_output = ttnn.layer_norm(
        hidden_states + self_output,
        weight=parameters.layernorm.weight,
        bias=parameters.layernorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(self_output)
    return attention_output


def _albert_layer(
    config,
    hidden_states: ttnn.Tensor,
    attention_mask: ttnn.Tensor | None,
    *,
    parameters: SimpleNamespace,
    mesh_device: ttnn.MeshDevice,
    memory_config: ttnn.MemoryConfig,
    compute_kernel_config,
) -> ttnn.Tensor:
    attention_output = _albert_attention(
        config,
        hidden_states,
        attention_mask,
        parameters=parameters.attention,
        mesh_device=mesh_device,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )

    ffn_mid = ttnn.linear(
        attention_output,
        parameters.ffn_weight,
        bias=parameters.ffn_bias,
        transpose_b=True,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    ffn_mid = _gelu_new_approx(ffn_mid, memory_config=memory_config)

    ffn_out = ttnn.linear(
        ffn_mid,
        parameters.ffn_out_weight,
        bias=parameters.ffn_out_bias,
        transpose_b=True,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(ffn_mid)

    out = ttnn.layer_norm(
        attention_output + ffn_out,
        weight=parameters.full_layernorm.weight,
        bias=parameters.full_layernorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(attention_output)
    ttnn.deallocate(ffn_out)
    return out


class TtKokoroAlbert:
    """Runs Kokoro `AlbertModel` encoder on device (matches HF forward numerics at BF16)."""

    def __init__(self, mesh_device: ttnn.MeshDevice, config, parameters: SimpleNamespace):
        self.mesh_device = mesh_device
        self.config = config
        self.parameters = parameters
        self.compute_kernel_config = default_compute_kernel_config(mesh_device)
        self.memory_config = dram_tile_config()

    def __call__(
        self,
        input_ids: torch.LongTensor,
        attention_mask_1_for_valid: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Args:
            input_ids: [B, S] on CPU
            attention_mask_1_for_valid: [B, S] float/bool, 1 = attendable token (HF convention)

        Returns:
            last_hidden_state on device: ttnn.Tensor [B, S, hidden_size]
        """
        cfg = self.config
        device = self.mesh_device
        batch_size, seq_len = input_ids.shape

        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

        tt_input = ttnn.from_torch(
            input_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=self.memory_config,
        )
        tt_pos = ttnn.from_torch(
            position_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=self.memory_config,
        )
        tt_tok = ttnn.from_torch(
            token_type_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=self.memory_config,
        )

        p = self.parameters.embeddings
        word = ttnn.embedding(tt_input, p.word_embeddings_weight, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(tt_input)
        pos = ttnn.embedding(tt_pos, p.position_embeddings_weight, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(tt_pos)
        tok = ttnn.embedding(tt_tok, p.token_type_embeddings_weight, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(tt_tok)

        embeddings = word + pos
        ttnn.deallocate(word)
        ttnn.deallocate(pos)
        embeddings = embeddings + tok
        ttnn.deallocate(tok)

        embeddings = ttnn.layer_norm(
            embeddings,
            weight=p.layernorm.weight,
            bias=p.layernorm.bias,
            epsilon=cfg.layer_norm_eps,
            memory_config=self.memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        hidden = ttnn.linear(
            embeddings,
            self.parameters.embedding_hidden_mapping_weight,
            bias=self.parameters.embedding_hidden_mapping_bias,
            transpose_b=True,
            memory_config=self.memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(embeddings)

        ext = get_extended_attention_mask(attention_mask_1_for_valid.float(), input_ids.shape, torch.float32)
        ext = torch.clamp(ext, min=-100000.0)
        ext = ext.expand((batch_size, -1, -1, -1))
        attention_mask = ttnn.from_torch(
            ext,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=self.memory_config,
        )

        layer_params = self.parameters.layer
        for _ in range(cfg.num_hidden_layers):
            hidden = _albert_layer(
                cfg,
                hidden,
                attention_mask,
                parameters=layer_params,
                mesh_device=device,
                memory_config=self.memory_config,
                compute_kernel_config=self.compute_kernel_config,
            )

        ttnn.deallocate(attention_mask)
        return hidden
