# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Upload Kokoro PL-BERT (HuggingFace `AlbertModel`) weights to TTNN tensors."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn

import ttnn


def _to_ttnn_weight(w: torch.Tensor, mesh_device: ttnn.MeshDevice, *, dtype) -> ttnn.Tensor:
    return ttnn.from_torch(
        w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _to_ttnn_bias_1d(b: torch.Tensor, mesh_device: ttnn.MeshDevice, *, dtype) -> ttnn.Tensor:
    return ttnn.from_torch(
        b.reshape(1, 1, 1, -1),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def preprocess_kokoro_albert_for_ttnn(
    albert: nn.Module,
    mesh_device: ttnn.MeshDevice,
    *,
    weights_dtype: Optional[Any] = None,
) -> SimpleNamespace:
    if weights_dtype is None:
        weights_dtype = ttnn.bfloat16

    emb = albert.embeddings
    enc = albert.encoder
    layer0 = enc.albert_layer_groups[0].albert_layers[0]

    def ln_params(mod: nn.LayerNorm) -> SimpleNamespace:
        return SimpleNamespace(
            weight=_to_ttnn_weight(mod.weight.data, mesh_device, dtype=weights_dtype),
            bias=_to_ttnn_bias_1d(mod.bias.data, mesh_device, dtype=weights_dtype),
        )

    attn = layer0.attention
    embeddings = SimpleNamespace(
        word_embeddings_weight=_to_ttnn_weight(emb.word_embeddings.weight.data, mesh_device, dtype=weights_dtype),
        position_embeddings_weight=_to_ttnn_weight(
            emb.position_embeddings.weight.data, mesh_device, dtype=weights_dtype
        ),
        token_type_embeddings_weight=_to_ttnn_weight(
            emb.token_type_embeddings.weight.data, mesh_device, dtype=weights_dtype
        ),
        layernorm=ln_params(emb.LayerNorm),
    )

    mapping = enc.embedding_hidden_mapping_in

    attention = SimpleNamespace(
        query=SimpleNamespace(
            weight=_to_ttnn_weight(attn.query.weight.data, mesh_device, dtype=weights_dtype),
            bias=_to_ttnn_bias_1d(attn.query.bias.data, mesh_device, dtype=weights_dtype),
        ),
        key=SimpleNamespace(
            weight=_to_ttnn_weight(attn.key.weight.data, mesh_device, dtype=weights_dtype),
            bias=_to_ttnn_bias_1d(attn.key.bias.data, mesh_device, dtype=weights_dtype),
        ),
        value=SimpleNamespace(
            weight=_to_ttnn_weight(attn.value.weight.data, mesh_device, dtype=weights_dtype),
            bias=_to_ttnn_bias_1d(attn.value.bias.data, mesh_device, dtype=weights_dtype),
        ),
        dense=SimpleNamespace(
            weight=_to_ttnn_weight(attn.dense.weight.data, mesh_device, dtype=weights_dtype),
            bias=_to_ttnn_bias_1d(attn.dense.bias.data, mesh_device, dtype=weights_dtype),
        ),
        layernorm=ln_params(attn.LayerNorm),
    )

    layer = SimpleNamespace(
        attention=attention,
        ffn_weight=_to_ttnn_weight(layer0.ffn.weight.data, mesh_device, dtype=weights_dtype),
        ffn_bias=_to_ttnn_bias_1d(layer0.ffn.bias.data, mesh_device, dtype=weights_dtype),
        ffn_out_weight=_to_ttnn_weight(layer0.ffn_output.weight.data, mesh_device, dtype=weights_dtype),
        ffn_out_bias=_to_ttnn_bias_1d(layer0.ffn_output.bias.data, mesh_device, dtype=weights_dtype),
        full_layernorm=ln_params(layer0.full_layer_layer_norm),
    )

    return SimpleNamespace(
        embeddings=embeddings,
        embedding_hidden_mapping_weight=_to_ttnn_weight(mapping.weight.data, mesh_device, dtype=weights_dtype),
        embedding_hidden_mapping_bias=_to_ttnn_bias_1d(mapping.bias.data, mesh_device, dtype=weights_dtype),
        layer=layer,
    )
