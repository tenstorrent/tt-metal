# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest
from models.demos.deepseek_v4_flash.ttnn_decoder_layer import TtDecoderLayer

DEFAULT_TINY_MODEL_LAYER = 2


@dataclass(frozen=True)
class ModelEmbeddingHeadWeights:
    embed_weight: torch.Tensor
    head_weight: torch.Tensor


class TtDeepSeekV4FlashTinyModel(LightweightModule):
    """Tiny batch-1 model-level scaffold for DeepSeek V4 Flash prefill.

    This stepping stone runs host ``input_ids`` shaped ``[1, tokens]`` through a
    host embedding lookup, one or more compressed ``TtDecoderLayer`` instances
    (default layer 2 for the synthetic tiny checkpoint), and a TTNN LM head on
    the first decoder layer's primary 1x1 submesh.

    The public output is a host torch tensor shaped ``[1, tokens, vocab]``. This
    is intentionally not the full model: it does not add hyperconnections, all
    layers, generation/cache semantics, or tensor caches.
    """

    def __init__(
        self,
        *,
        decoder_layers: Iterable[TtDecoderLayer],
        weights: ModelEmbeddingHeadWeights,
        hidden_size: int,
        vocab_size: int,
        layer_ids: Iterable[int],
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        validate_model_embedding_head_weights(weights, hidden_size=hidden_size, vocab_size=vocab_size)
        decoder_layers = tuple(decoder_layers)
        layer_ids = normalize_tiny_model_layer_ids(layer_ids=layer_ids)
        validate_model_decoder_layers(decoder_layers, hidden_size=hidden_size, layer_ids=layer_ids)

        self.decoder_layers = decoder_layers
        self.decoder_layer = decoder_layers[0]
        self.primary_submesh = decoder_layers[0].primary_submesh
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.layer_ids = layer_ids
        self.layer = int(layer_ids[-1])
        self.dtype = dtype
        self.memory_config = memory_config
        self.embed_weight = weights.embed_weight.contiguous().to(torch.bfloat16)
        self.lm_head_weight = _to_tt_lm_head_weight(
            weights.head_weight,
            device=self.primary_submesh,
            dtype=dtype,
            memory_config=memory_config,
        )

    @classmethod
    def from_preprocessed(
        cls,
        preprocessed_path: str | Path,
        *,
        mesh_device,
        layer: int = DEFAULT_TINY_MODEL_LAYER,
        layer_ids: Iterable[int] | None = None,
        primary_submesh_coord: tuple[int, int] | None = None,
        replicas_per_expert: int = 1,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> "TtDeepSeekV4FlashTinyModel":
        preprocessed_path = Path(preprocessed_path)
        manifest = load_tt_manifest(preprocessed_path)
        config = manifest["config"]
        normalized_layer_ids = normalize_tiny_model_layer_ids(
            layer=layer,
            layer_ids=layer_ids,
            num_hidden_layers=int(config["num_hidden_layers"]),
        )
        decoder_layers = []
        primary_submesh = None
        for layer_id in normalized_layer_ids:
            decoder_layer = TtDecoderLayer.from_preprocessed(
                preprocessed_path,
                mesh_device=mesh_device,
                layer=layer_id,
                primary_submesh_coord=primary_submesh_coord,
                primary_submesh=primary_submesh,
                replicas_per_expert=replicas_per_expert,
                dtype=dtype,
                memory_config=memory_config,
            )
            if primary_submesh is None:
                primary_submesh = decoder_layer.primary_submesh
            decoder_layers.append(decoder_layer)
        return cls(
            decoder_layers=tuple(decoder_layers),
            weights=load_model_embedding_head_weights(preprocessed_path, manifest=manifest),
            hidden_size=int(config["hidden_size"]),
            vocab_size=int(config["vocab_size"]),
            layer_ids=normalized_layer_ids,
            dtype=dtype,
            memory_config=memory_config,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        validate_model_input_ids(input_ids, vocab_size=self.vocab_size)
        hidden_states = embed_input_ids_host(input_ids, self.embed_weight).unsqueeze(1)
        for decoder_layer in self.decoder_layers:
            hidden_states = decoder_layer(hidden_states, input_ids=input_ids)
        tt_hidden = ttnn.from_torch(
            hidden_states.contiguous(),
            device=self.primary_submesh,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )
        tt_logits = ttnn.linear(tt_hidden, self.lm_head_weight, memory_config=self.memory_config)
        logits = ttnn.to_torch(tt_logits)
        expected_shape = (1, 1, input_ids.shape[1], self.vocab_size)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(f"LM head returned shape {tuple(logits.shape)}, expected {expected_shape}")
        return logits[:, 0].contiguous()


def normalize_tiny_model_layer_ids(
    *,
    layer: int | None = None,
    layer_ids: Iterable[int] | None = None,
    num_hidden_layers: int | None = None,
) -> tuple[int, ...]:
    if layer_ids is None:
        if layer is None:
            raise ValueError("layer must be provided when layer_ids is not set")
        normalized = (_normalize_layer_id(layer),)
    else:
        if layer is not None and layer != DEFAULT_TINY_MODEL_LAYER:
            raise ValueError("Pass either layer or layer_ids, not both")
        if isinstance(layer_ids, (str, bytes)):
            raise TypeError(f"layer_ids must be an iterable of integers, got {type(layer_ids).__name__}")
        normalized = tuple(_normalize_layer_id(layer_id) for layer_id in layer_ids)
        if not normalized:
            raise ValueError("layer_ids must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"layer_ids must not contain duplicates, got {normalized}")

    if num_hidden_layers is not None:
        if num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}")
        for layer_id in normalized:
            if layer_id >= num_hidden_layers:
                raise ValueError(f"layer id must be in [0, {num_hidden_layers}), got {layer_id}")
    return normalized


def load_model_embedding_head_weights(
    preprocessed_path: str | Path,
    *,
    manifest: dict | None = None,
) -> ModelEmbeddingHeadWeights:
    preprocessed_path = Path(preprocessed_path)
    if manifest is None:
        manifest = load_tt_manifest(preprocessed_path)

    keys = {"embed_weight": "embed.weight", "head_weight": "head.weight"}
    loaded: dict[str, torch.Tensor] = {}
    for artifact in manifest["artifacts"]["non_expert_safetensors"]:
        with safe_open(preprocessed_path / artifact, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name, key in keys.items():
                if name not in loaded and key in available:
                    loaded[name] = handle.get_tensor(key).contiguous()
        if len(loaded) == len(keys):
            break

    missing = sorted(key for name, key in keys.items() if name not in loaded)
    if missing:
        raise KeyError(f"Missing model embedding/head weights: {missing}")
    return ModelEmbeddingHeadWeights(embed_weight=loaded["embed_weight"], head_weight=loaded["head_weight"])


def embed_input_ids_host(input_ids: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
    if embed_weight.ndim != 2:
        raise ValueError(f"embed_weight must have shape [vocab, hidden], got {tuple(embed_weight.shape)}")
    validate_model_input_ids(input_ids, vocab_size=int(embed_weight.shape[0]))
    return F.embedding(input_ids.to(torch.long), embed_weight.contiguous()).contiguous()


def validate_model_embedding_head_weights(
    weights: ModelEmbeddingHeadWeights,
    *,
    hidden_size: int,
    vocab_size: int,
) -> None:
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    _expect_shape(weights.embed_weight, (vocab_size, hidden_size), "embed_weight")
    _expect_shape(weights.head_weight, (vocab_size, hidden_size), "head_weight")


def validate_model_decoder_layers(
    decoder_layers: tuple[TtDecoderLayer, ...],
    *,
    hidden_size: int,
    layer_ids: tuple[int, ...],
) -> None:
    if len(decoder_layers) == 0:
        raise ValueError("decoder_layers must be non-empty")
    if len(decoder_layers) != len(layer_ids):
        raise ValueError(f"decoder_layers length {len(decoder_layers)} does not match layer_ids {layer_ids}")
    for index, decoder_layer in enumerate(decoder_layers):
        if int(decoder_layer.hidden_size) != int(hidden_size):
            raise ValueError(
                f"decoder layer hidden size {int(decoder_layer.hidden_size)} does not match model hidden size "
                f"{int(hidden_size)}"
            )
        if int(decoder_layer.layer) != int(layer_ids[index]):
            raise ValueError(f"decoder layer id {int(decoder_layer.layer)} does not match layer_ids[{index}]")


def validate_model_input_ids(input_ids: torch.Tensor, *, vocab_size: int | None = None) -> None:
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids).__name__}")
    shape = tuple(input_ids.shape)
    if len(shape) != 2 or shape[0] != 1:
        raise ValueError(f"input_ids must have shape [1, tokens], got {shape}")
    if shape[1] == 0:
        raise ValueError("input_ids must contain at least one token")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"input_ids dtype must be int32 or int64, got {input_ids.dtype}")
    if vocab_size is not None:
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        min_id = int(input_ids.min().item())
        max_id = int(input_ids.max().item())
        if min_id < 0 or max_id >= vocab_size:
            raise ValueError(f"input_ids values must be in [0, {vocab_size}), got min={min_id}, max={max_id}")


def _normalize_layer_id(layer_id: int) -> int:
    if not isinstance(layer_id, int) or isinstance(layer_id, bool):
        raise TypeError(f"layer ids must be integers, got {layer_id!r}")
    if layer_id < 0:
        raise ValueError(f"layer ids must be non-negative, got {layer_id}")
    return int(layer_id)


def _to_tt_lm_head_weight(
    head_weight: torch.Tensor,
    *,
    device,
    dtype,
    memory_config,
):
    if head_weight.ndim != 2:
        raise ValueError(f"head_weight must have shape [vocab, hidden], got {tuple(head_weight.shape)}")
    torch_weight = head_weight.transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    return ttnn.from_torch(
        torch_weight,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def _expect_shape(tensor: torch.Tensor, expected_shape: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Expected {name} shape {expected_shape}, got {tuple(tensor.shape)}")
