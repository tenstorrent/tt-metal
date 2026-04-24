# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest
from models.demos.deepseek_v4_flash.ttnn_moe_block import TtMoEFeedForwardBlock
from models.demos.deepseek_v4_flash.ttnn_prefill_attention_block import TtPrefillAttentionBlock


@dataclass(frozen=True)
class DecoderLayerNormWeights:
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor


class TtDecoderLayer(LightweightModule):
    """Small DeepSeek V4 Flash decoder-layer stepping stone.

    This scaffold wires the current TTNN prefill attention block and T3K MoE FFN
    block through explicit RMSNorm and residual boundaries:
    ``x -> attn_norm -> attention -> residual -> ffn_norm -> MoE -> residual``.

    Hyperconnection pre/post mixing is intentionally not implemented in this
    slice. The current contract is batch-1 hidden states shaped
    ``[1, 1, tokens, hidden]`` on the tiny converted checkpoint. RMSNorm runs on
    the primary 1x1 submesh; residual combines and the final output are host
    torch tensors because the attention and MoE stepping stones already use
    host-side fallbacks at their current ABI boundaries. No tensor caches are
    added.
    """

    def __init__(
        self,
        *,
        attention: TtPrefillAttentionBlock,
        ffn: TtMoEFeedForwardBlock,
        norm_weights: DecoderLayerNormWeights,
        primary_submesh,
        hidden_size: int,
        norm_eps: float = 1e-6,
        layer: int = 0,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        validate_decoder_layer_config(
            attention=attention,
            ffn=ffn,
            norm_weights=norm_weights,
            hidden_size=hidden_size,
        )
        self.attention = attention
        self.ffn = ffn
        self.primary_submesh = primary_submesh
        self.hidden_size = int(hidden_size)
        self.norm_eps = float(norm_eps)
        self.layer = int(layer)
        self.dtype = dtype
        self.memory_config = memory_config
        self.attn_norm = _to_tt_norm_weight(
            norm_weights.attn_norm,
            device=primary_submesh,
            dtype=dtype,
            memory_config=memory_config,
        )
        self.ffn_norm = _to_tt_norm_weight(
            norm_weights.ffn_norm,
            device=primary_submesh,
            dtype=dtype,
            memory_config=memory_config,
        )

    @classmethod
    def from_preprocessed(
        cls,
        preprocessed_path: str | Path,
        *,
        mesh_device,
        layer: int = 0,
        primary_submesh_coord: tuple[int, int] | None = None,
        primary_submesh=None,
        replicas_per_expert: int = 1,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> "TtDecoderLayer":
        preprocessed_path = Path(preprocessed_path)
        manifest = load_tt_manifest(preprocessed_path)
        config = manifest["config"]
        _validate_layer_index(config, layer)
        compress_ratio = int(config["compress_ratios"][layer])
        if compress_ratio <= 0:
            raise ValueError(
                f"Layer {layer} has compress_ratio={compress_ratio}; the decoder-layer scaffold requires "
                "a compressed prefill-attention layer"
            )

        ffn = TtMoEFeedForwardBlock.from_preprocessed(
            preprocessed_path,
            mesh_device=mesh_device,
            layer=layer,
            primary_submesh_coord=primary_submesh_coord,
            primary_submesh=primary_submesh,
            replicas_per_expert=replicas_per_expert,
            dtype=dtype,
            memory_config=memory_config,
        )
        attention = TtPrefillAttentionBlock.from_preprocessed(
            preprocessed_path,
            device=ffn.primary_submesh,
            layer=layer,
            dtype=dtype,
            memory_config=memory_config,
        )
        return cls(
            attention=attention,
            ffn=ffn,
            norm_weights=load_decoder_layer_norm_weights(preprocessed_path, manifest=manifest, layer=layer),
            primary_submesh=ffn.primary_submesh,
            hidden_size=int(config["hidden_size"]),
            norm_eps=float(config["rms_norm_eps"]),
            layer=layer,
            dtype=dtype,
            memory_config=memory_config,
        )

    def forward(
        self,
        hidden_states,
        *,
        input_ids: torch.Tensor | None = None,
        topk_idxs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_decoder_layer_input(
            hidden_states,
            input_ids=input_ids,
            topk_idxs=topk_idxs,
            hidden_size=self.hidden_size,
            compress_ratio=self.attention.compressor.compress_ratio,
        )
        torch_hidden_states, tt_hidden_states = self._materialize_hidden_states(hidden_states)

        tt_attn_input = ttnn.rms_norm(
            tt_hidden_states,
            weight=self.attn_norm,
            epsilon=self.norm_eps,
            memory_config=self.memory_config,
        )
        attention_output = ttnn.to_torch(self.attention(tt_attn_input, topk_idxs=topk_idxs)).contiguous()
        hidden_after_attention = (torch_hidden_states.float() + attention_output.float()).to(torch_hidden_states.dtype)

        tt_ffn_residual = ttnn.from_torch(
            hidden_after_attention,
            device=self.primary_submesh,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )
        tt_ffn_input = ttnn.rms_norm(
            tt_ffn_residual,
            weight=self.ffn_norm,
            epsilon=self.norm_eps,
            memory_config=self.memory_config,
        )
        ffn_output = self.ffn(tt_ffn_input, input_ids=input_ids)
        return (hidden_after_attention.float() + ffn_output.float()).to(hidden_after_attention.dtype)

    def _materialize_hidden_states(self, hidden_states) -> tuple[torch.Tensor, object]:
        if isinstance(hidden_states, torch.Tensor):
            torch_hidden_states = hidden_states.contiguous()
        else:
            torch_hidden_states = ttnn.to_torch(hidden_states).contiguous()
        tt_hidden_states = ttnn.from_torch(
            torch_hidden_states,
            device=self.primary_submesh,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )
        return torch_hidden_states, tt_hidden_states


def load_decoder_layer_norm_weights(
    preprocessed_path: str | Path,
    *,
    manifest: dict | None = None,
    layer: int = 0,
) -> DecoderLayerNormWeights:
    preprocessed_path = Path(preprocessed_path)
    if manifest is None:
        manifest = load_tt_manifest(preprocessed_path)

    keys = {
        "attn_norm": f"layers.{layer}.attn_norm.weight",
        "ffn_norm": f"layers.{layer}.ffn_norm.weight",
    }
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
        raise KeyError(f"Missing decoder layer norm weights for layer {layer}: {missing}")
    return DecoderLayerNormWeights(attn_norm=loaded["attn_norm"], ffn_norm=loaded["ffn_norm"])


def validate_decoder_layer_config(
    *,
    attention: TtPrefillAttentionBlock,
    ffn: TtMoEFeedForwardBlock,
    norm_weights: DecoderLayerNormWeights,
    hidden_size: int,
) -> None:
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if attention.attention_projection.hidden_size != hidden_size:
        raise ValueError(
            f"attention hidden size {attention.attention_projection.hidden_size} does not match {hidden_size}"
        )
    if ffn.hidden_size != hidden_size:
        raise ValueError(f"ffn hidden size {ffn.hidden_size} does not match {hidden_size}")
    validate_decoder_layer_norm_weights(norm_weights, hidden_size=hidden_size)


def validate_decoder_layer_norm_weights(norm_weights: DecoderLayerNormWeights, *, hidden_size: int) -> None:
    _expect_shape(norm_weights.attn_norm, (hidden_size,), "attn_norm")
    _expect_shape(norm_weights.ffn_norm, (hidden_size,), "ffn_norm")


def validate_decoder_layer_input(
    hidden_states,
    *,
    input_ids: torch.Tensor | None = None,
    topk_idxs: torch.Tensor | None = None,
    hidden_size: int,
    compress_ratio: int,
) -> None:
    shape = tuple(hidden_states.shape)
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 1:
        raise ValueError(f"hidden_states must have shape [1, 1, tokens, hidden], got {shape}")
    _, _, tokens, width = shape
    if width != hidden_size:
        raise ValueError(f"hidden_states hidden dim must be {hidden_size}, got {width}")
    if tokens < compress_ratio:
        raise ValueError(f"hidden_states tokens must be >= compress_ratio {compress_ratio}, got {tokens}")
    if input_ids is not None:
        if tuple(input_ids.shape) != (1, tokens):
            raise ValueError(f"input_ids must have shape {(1, tokens)}, got {tuple(input_ids.shape)}")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"input_ids dtype must be int32 or int64, got {input_ids.dtype}")
    if topk_idxs is not None:
        if topk_idxs.ndim != 3:
            raise ValueError(f"topk_idxs must have shape [1, tokens, topk], got {tuple(topk_idxs.shape)}")
        if tuple(topk_idxs.shape[:2]) != (1, tokens):
            raise ValueError(f"topk_idxs batch/tokens must be {(1, tokens)}, got {tuple(topk_idxs.shape[:2])}")
        if topk_idxs.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"topk_idxs dtype must be int32 or int64, got {topk_idxs.dtype}")


def _validate_layer_index(config: dict, layer: int) -> None:
    num_layers = int(config["num_hidden_layers"])
    if layer < 0 or layer >= num_layers:
        raise ValueError(f"layer must be in [0, {num_layers}), got {layer}")


def _expect_shape(tensor: torch.Tensor, expected_shape: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Expected {name} shape {expected_shape}, got {tuple(tensor.shape)}")


def _to_tt_norm_weight(
    weight: torch.Tensor,
    *,
    device,
    dtype,
    memory_config,
):
    return ttnn.from_torch(
        weight.contiguous().to(torch.bfloat16),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )
