# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 decoder layer (both kinds), prefill only.

``h = x + mixer(input_norm(x)); out = h + mlp(post_norm(h))`` where the mixer is gated full
attention (layer index 3 mod 4) or Gated DeltaNet (the rest).

Prefill accepts any logical length 1..max_seq_len. It runs bounded physical chunks of
``Optimizations.prefill_chunk`` tokens (page/SDPA aligned); attention K/V and DeltaNet
state carry across chunks on device. The forward contains only TTNN device ops.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.pplx_decider_v1_27b.tt.attention import PplxGatedAttention
from models.demos.pplx_decider_v1_27b.tt.gated_deltanet import PplxGatedDeltaNet
from models.demos.pplx_decider_v1_27b.tt.mlp import PplxMLP
from models.demos.pplx_decider_v1_27b.tt.model_config import PplxDeciderArgs
from models.demos.pplx_decider_v1_27b.tt.norm import PplxRMSNorm
from models.demos.pplx_decider_v1_27b.tt.optimizations import Optimizations
from models.demos.pplx_decider_v1_27b.tt.rope import PplxRotary
from models.demos.pplx_decider_v1_27b.tt.weight_adapter import DecoderLayerWeights, build_decoder_layer_weights


@dataclass
class DecoderLayerConfig:
    weights: DecoderLayerWeights
    args: PplxDeciderArgs
    layer_idx: int
    optimizations: Optimizations


class PplxDecoderLayer(LightweightModule):
    def __init__(self, weights: DecoderLayerWeights, args: PplxDeciderArgs, layer_idx: int, optimizations):
        super().__init__()
        self._build(DecoderLayerConfig(weights=weights, args=args, layer_idx=layer_idx, optimizations=optimizations))

    @classmethod
    def from_config(cls, config: DecoderLayerConfig) -> "PplxDecoderLayer":
        instance = object.__new__(cls)
        LightweightModule.__init__(instance)
        instance._build(config)
        return instance

    @classmethod
    def from_state_dict(
        cls, state_dict, *, args: PplxDeciderArgs, layer_idx: int, optimizations: Optimizations
    ) -> "PplxDecoderLayer":
        """``state_dict``: layer-local HF tensors (keys as in ``Qwen3_5DecoderLayer.state_dict()``)."""
        weights = build_decoder_layer_weights(state_dict, args, layer_idx, optimizations.policy)
        return cls.from_config(
            DecoderLayerConfig(weights=weights, args=args, layer_idx=layer_idx, optimizations=optimizations)
        )

    def _build(self, config: DecoderLayerConfig) -> None:
        self.config = config
        w, a, opts = config.weights, config.args, config.optimizations
        self.kind = w.kind
        if self.kind != a.layer_kind(config.layer_idx):
            raise ValueError(f"Layer {config.layer_idx} weights are {self.kind}, config says otherwise")
        self.input_norm = PplxRMSNorm(w.input_norm.weight, a.rms_norm_eps, opts.norm)
        self.post_norm = PplxRMSNorm(w.post_norm.weight, a.rms_norm_eps, opts.norm)
        self.mlp = PplxMLP(w.mlp, opts.linear, mesh_device=opts.mesh_device)
        if self.kind == "full_attention":
            self.mixer = PplxGatedAttention(w.attention, a, opts)
        elif self.kind == "linear_attention":
            self.mixer = PplxGatedDeltaNet(w.delta, a, opts)
        else:
            raise ValueError(f"Unknown layer kind {self.kind}")

    def _chunks(self, length: int):
        chunk = self.config.optimizations.prefill_chunk
        if length < 1 or length > self.config.args.max_seq_len:
            raise ValueError(f"Prefill length {length} outside 1..{self.config.args.max_seq_len}")
        return [(start, min(chunk, length - start)) for start in range(0, length, chunk)]

    def mixer_prefill(self, x: ttnn.Tensor, rotary: PplxRotary | None = None) -> ttnn.Tensor:
        """Token mixer alone over a whole request (x already input-normed). For module tests."""
        outputs, state = [], None
        for start, count in self._chunks(x.shape[1]):
            chunk = x[:, start : start + count, :] if count != x.shape[1] else x
            if self.kind == "full_attention":
                cos, sin = rotary(start, count)
                outputs.append(self.mixer(chunk, start_pos=start, cos=cos, sin=sin))
            else:
                out, state = self.mixer(chunk, state)
                outputs.append(out)
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1)

    def forward(self, x: ttnn.Tensor, rotary: PplxRotary | None = None) -> ttnn.Tensor:
        """Fresh-request prefill. x: [1, S, 5120] BF16 TILE -> [1, S, 5120]."""
        if self.kind == "full_attention" and rotary is None:
            raise ValueError("Full-attention layers need the shared rotary tables")
        outputs, state = [], None
        for start, count in self._chunks(x.shape[1]):
            chunk = x[:, start : start + count, :] if count != x.shape[1] else x
            normed = self.input_norm(chunk)
            if self.kind == "full_attention":
                cos, sin = rotary(start, count)
                mixed = self.mixer(normed, start_pos=start, cos=cos, sin=sin)
            else:
                mixed, state = self.mixer(normed, state)
            ttnn.deallocate(normed)
            h = ttnn.add(chunk, mixed)
            ttnn.deallocate(mixed)
            post = self.post_norm(h)
            mlp_out = self.mlp(post)
            ttnn.deallocate(post)
            out = ttnn.add(h, mlp_out)
            ttnn.deallocate(h)
            ttnn.deallocate(mlp_out)
            outputs.append(out)
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1)
