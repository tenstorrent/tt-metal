# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TtQwen36Model — full text decoder (embedding + N decoder layers + final norm + LM head).

Currently single mesh-device with replicated weights. Supports configurable layer subset
for memory-constrained testing.
"""
from __future__ import annotations

import torch.nn.functional as F

import ttnn
from models.demos.qwen3_6_27b.tt.decoder import TtDecoderLayer, _make_norm_weight, _rms_norm_device, _t2t


class TtQwen36Model:
    """Hybrid Qwen3.6 text decoder."""

    def __init__(self, device, weights, hf_config, num_layers=None):
        """
        Args:
            device: ttnn device or mesh
            weights: dict of all loaded HF tensors (text path)
            hf_config: Qwen3NextConfig instance
            num_layers: if None, all layers; else first N layers (for memory-constrained testing)
        """
        self.device = device
        self.hf_cfg = hf_config
        self.num_layers = num_layers or hf_config.num_hidden_layers
        self.layer_types = hf_config.layer_types[: self.num_layers]
        self.eps = hf_config.rms_norm_eps
        self.vocab = hf_config.vocab_size
        self.H = hf_config.hidden_size

        # Embedding (host-side lookup; converts to BF16 on output).
        # TODO: move to ttnn.embedding on device for performance.
        self.embed_weight = weights["model.language_model.embed_tokens.weight"]  # [vocab, H]

        # Final norm — on device with zero-centered (1+w) pre-adjustment
        self.final_norm_w = _make_norm_weight(weights["model.language_model.norm.weight"], device, zero_centered=True)

        # LM head — push transposed weight to device DRAM.
        lm = weights["lm_head.weight"].float()  # [vocab, H]
        self.lm_head_w = _t2t(lm.T, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG)
        # Keep host copy too for fallback; will remove once stable
        self.lm_head_weight = weights["lm_head.weight"]

        # Decoder layers
        self.layers = [
            TtDecoderLayer(device, weights, i, self.layer_types[i], hf_config) for i in range(self.num_layers)
        ]

    def forward_hidden(self, input_ids, cos=None, sin=None, attention_mask=None):
        """Returns final hidden state [B, T, H] as a TTNN tensor.

        Args:
            input_ids: torch [B, T] long
            cos, sin: torch [B, T, rotary_dim=64] for full_attention layers
            attention_mask: torch [1, 1, T, T] additive
        """
        # Embedding lookup on host (simplest path)
        h_host = F.embedding(input_ids, self.embed_weight.float())  # [B, T, H]
        h_tt = _t2t(h_host, self.device, dtype=ttnn.bfloat16)

        for layer in self.layers:
            h_tt = layer(h_tt, cos=cos, sin=sin, attention_mask=attention_mask)

        # Final norm on device
        h_tt = _rms_norm_device(h_tt, self.final_norm_w, self.eps)
        return h_tt

    def __call__(self, input_ids, cos=None, sin=None, attention_mask=None):
        """Full forward: returns logits [B, T, vocab] as torch tensor."""
        h_tt = self.forward_hidden(input_ids, cos, sin, attention_mask)
        # LM head on device
        logits_tt = ttnn.linear(h_tt, self.lm_head_w, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        return ttnn.to_torch(logits_tt).float()
