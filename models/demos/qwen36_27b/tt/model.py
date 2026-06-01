# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full Qwen3.6-27B model in TT-NN for single P150a."""

import math

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen36_27b.tt.decoder import TtHybridDecoderLayer
from models.demos.qwen36_27b.tt.deltanet import TtDeltaNetState
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig


class TtQwen36Model(LightweightModule):
    def __init__(self, device, state_dict, config: Qwen36ModelConfig, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device
        self.config = config
        self.dtype = dtype
        self.num_layers = config.num_hidden_layers

        embed_w = state_dict["model.embed_tokens.weight"]
        self.embedding_weight = embed_w  # keep on CPU for lookup

        self.layers = []
        for i in range(self.num_layers):
            layer = TtHybridDecoderLayer(device, state_dict, i, config, dtype=dtype)
            self.layers.append(layer)

        TILE = 32
        norm_w = state_dict["model.norm.weight"]
        dim = norm_w.shape[0]
        torch_norm_w = (norm_w + 1.0).unsqueeze(0).view(1, 1, dim).reshape(1, 1, dim // TILE, TILE)
        self.final_norm_weight = ttnn.from_torch(
            torch_norm_w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
        )

        lm_head_w = state_dict["lm_head.weight"].T.contiguous()
        self.lm_head_w = ttnn.from_torch(
            lm_head_w.unsqueeze(0).unsqueeze(0),
            dtype=config.weights_dtype, layout=ttnn.TILE_LAYOUT, device=device,
        )

        self._build_rope_cache(config)

    def _build_rope_cache(self, config):
        """Precompute RoPE cos/sin for the rotary dimension."""
        dim = config.rotary_dim
        max_seq = config.max_seq_len
        theta = config.rope_theta
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, freqs)
        self.cos_cache = freqs.cos().reshape(1, 1, max_seq, dim // 2).repeat(1, 1, 1, 2)
        self.sin_cache = freqs.sin().reshape(1, 1, max_seq, dim // 2).repeat(1, 1, 1, 2)

    def get_rope(self, position_ids):
        """Get cos/sin for given position IDs. Always returns 4D [1, 1, S, D]."""
        if isinstance(position_ids, int):
            cos = self.cos_cache[:, :, position_ids:position_ids+1, :]
            sin = self.sin_cache[:, :, position_ids:position_ids+1, :]
        else:
            cos = self.cos_cache[:, :, position_ids, :]
            sin = self.sin_cache[:, :, position_ids, :]
        return cos, sin

    def embed(self, token_ids):
        """CPU embedding lookup → device tensor [1, 1, S, H]."""
        embeddings = self.embedding_weight[token_ids]  # [B, S, H]
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)  # [1, S, H]
        return ttnn.from_torch(
            embeddings.unsqueeze(0),  # [1, B, S, H] = [1, 1, S, H]
            dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device,
        )

    def rms_norm(self, x, weight, eps=1e-6):
        return ttnn.rms_norm(x, epsilon=eps, weight=weight)

    def forward(self, token_ids, position_ids, deltanet_state, kv_caches=None, mode="decode"):
        """
        Args:
            token_ids: [B, S] tensor of token IDs (CPU)
            position_ids: int or tensor of position IDs
            deltanet_state: TtDeltaNetState
            kv_caches: dict mapping attention layer_idx → (k_cache, v_cache)
            mode: "decode" or "prefill"

        Returns:
            logits: [1, 1, 1, vocab_padded] on device (last token only for prefill)
            new_kv_caches: updated KV caches
        """
        hidden_states = self.embed(token_ids)
        cos, sin = self.get_rope(position_ids)

        new_kv_caches = {} if kv_caches is None else dict(kv_caches)

        for i, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            kv_cache = new_kv_caches.get(i) if layer_type == "full_attention" else None

            hidden_states, new_kv = layer(
                hidden_states,
                deltanet_state=deltanet_state,
                cos=cos, sin=sin,
                kv_cache=kv_cache,
                mode=mode,
            )

            if new_kv is not None:
                new_kv_caches[i] = new_kv

        if mode == "prefill" and hidden_states.shape[2] > 1:
            last_hidden = ttnn.to_torch(hidden_states)[:, :, -1:, :]
            hidden_states = ttnn.from_torch(
                last_hidden, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
            )

        hidden_states = self.rms_norm(hidden_states, self.final_norm_weight)
        logits = ttnn.linear(hidden_states, self.lm_head_w)

        return logits, new_kv_caches

    def create_deltanet_state(self):
        return TtDeltaNetState(
            self.num_layers, self.config.layer_types, self.device, self.config
        )
