# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT port of the SeamlessM4Tv2 text decoder + lm_head (Phase 4).

NLLB/BART-style pre-norm decoder, batch=1:
  h = embed_tokens(ids) * sqrt(hidden) + sinusoidal_positions      (host)
  per layer:
    h += self_attn(self_attn_ln(h), causal_mask)                    (device)
    h += cross_attn(cross_attn_ln(h), encoder_hidden_states)        (device)
    h += ffn(ffn_ln(h))                                             (device, relu)
  h = layer_norm(h)
  logits = lm_head(h)                                               (tied to embeddings)

Phase 4 validates correctness with a full (non-cached) forward over the decoder
sequence and a causal mask, matching HF `use_cache=False`. KV-cache decode is a
Phase 5/6 optimization. Token embedding + sinusoidal positions are computed on
host (cheap, exact) and uploaded.
"""

from __future__ import annotations

import math

import torch

import ttnn


def _to_tt(t, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _sinusoidal_weights(num_positions, dim, padding_idx):
    """Replicates SeamlessM4Tv2SinusoidalPositionalEmbedding.get_embedding (offset=2)."""
    num_embeddings = num_positions + 2
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, dtype=torch.float32) * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb


def _position_ids(input_ids, padding_idx, past_len=0):
    """create_position_ids_from_input_ids: cumulative non-pad index + padding_idx."""
    mask = input_ids.ne(padding_idx).int()
    incremental = (torch.cumsum(mask, dim=1) + past_len) * mask
    return incremental.long() + padding_idx


class _DecAttention:
    """BART-style MHA. Self-attn (kv from hidden, causal) or cross-attn (kv from encoder)."""

    def __init__(self, weights, prefix, num_heads, head_dim, device, dtype=ttnn.bfloat16):
        self.H = num_heads
        self.d = head_dim
        self.scaling = head_dim**-0.5
        self.device = device
        self.qw = _to_tt(weights[prefix + "q_proj.weight"].t().contiguous(), device, dtype)
        self.qb = _to_tt(weights[prefix + "q_proj.bias"], device, dtype)
        self.kw = _to_tt(weights[prefix + "k_proj.weight"].t().contiguous(), device, dtype)
        self.kb = _to_tt(weights[prefix + "k_proj.bias"], device, dtype)
        self.vw = _to_tt(weights[prefix + "v_proj.weight"].t().contiguous(), device, dtype)
        self.vb = _to_tt(weights[prefix + "v_proj.bias"], device, dtype)
        self.ow = _to_tt(weights[prefix + "out_proj.weight"].t().contiguous(), device, dtype)
        self.ob = _to_tt(weights[prefix + "out_proj.bias"], device, dtype)

    def _heads(self, x, seq):  # (1,seq,C) -> (1,H,seq,d)
        x = ttnn.reshape(x, [1, seq, self.H, self.d])
        return ttnn.transpose(x, 1, 2)

    def __call__(self, hidden, kv_source, mask=None):
        q_seq = hidden.shape[1]
        kv_seq = kv_source.shape[1]
        q = ttnn.linear(hidden, self.qw, bias=self.qb)
        q = ttnn.multiply(q, self.scaling)
        k = ttnn.linear(kv_source, self.kw, bias=self.kb)
        v = ttnn.linear(kv_source, self.vw, bias=self.vb)

        q = self._heads(q, q_seq)  # (1,H,q,d)
        k = self._heads(k, kv_seq)  # (1,H,kv,d)
        v = self._heads(v, kv_seq)  # (1,H,kv,d)

        k_t = ttnn.transpose(k, -2, -1)  # (1,H,d,kv)
        scores = ttnn.matmul(q, k_t)  # (1,H,q,kv)
        if mask is not None:
            scores = ttnn.add(scores, mask)
        attn = ttnn.softmax(scores, dim=-1)
        ctx = ttnn.matmul(attn, v)  # (1,H,q,d)
        ctx = ttnn.transpose(ctx, 1, 2)  # (1,q,H,d)
        ctx = ttnn.reshape(ctx, [1, q_seq, self.H * self.d])
        return ttnn.linear(ctx, self.ow, bias=self.ob)


class _DecoderLayer:
    def __init__(self, weights, prefix, config, device, dtype=ttnn.bfloat16):
        eps = config.layer_norm_eps
        H = config.decoder_attention_heads
        d = config.hidden_size // H

        def ln(name):
            return (_to_tt(weights[prefix + name + ".weight"], device, dtype),
                    _to_tt(weights[prefix + name + ".bias"], device, dtype))

        self.eps = eps
        self.sa_ln = ln("self_attn_layer_norm")
        self.ca_ln = ln("cross_attention_layer_norm")
        self.ffn_ln = ln("ffn_layer_norm")
        self.self_attn = _DecAttention(weights, prefix + "self_attn.", H, d, device, dtype)
        self.cross_attn = _DecAttention(weights, prefix + "cross_attention.", H, d, device, dtype)
        self.fc1_w = _to_tt(weights[prefix + "ffn.fc1.weight"].t().contiguous(), device, dtype)
        self.fc1_b = _to_tt(weights[prefix + "ffn.fc1.bias"], device, dtype)
        self.fc2_w = _to_tt(weights[prefix + "ffn.fc2.weight"].t().contiguous(), device, dtype)
        self.fc2_b = _to_tt(weights[prefix + "ffn.fc2.bias"], device, dtype)

    def _norm(self, x, ln):
        return ttnn.layer_norm(x, weight=ln[0], bias=ln[1], epsilon=self.eps)

    def __call__(self, h, causal_mask, enc):
        residual = h
        x = self._norm(h, self.sa_ln)
        x = self.self_attn(x, x, mask=causal_mask)
        h = ttnn.add(residual, x)

        residual = h
        x = self._norm(h, self.ca_ln)
        x = self.cross_attn(x, enc, mask=None)
        h = ttnn.add(residual, x)

        residual = h
        x = self._norm(h, self.ffn_ln)
        x = ttnn.linear(x, self.fc1_w, bias=self.fc1_b)
        x = ttnn.relu(x)
        x = ttnn.linear(x, self.fc2_w, bias=self.fc2_b)
        h = ttnn.add(residual, x)
        return h


class TtTextDecoder:
    def __init__(self, state_dict, config, device, dtype=ttnn.bfloat16):
        """state_dict: HF `text_decoder.` slice (prefix stripped). lm_head tied to embed_tokens."""
        self.device = device
        self.dtype = dtype
        self.config = config
        self.pad = config.pad_token_id
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        self.embed_weight = state_dict["embed_tokens.weight"].float()  # (vocab, hidden), host
        self.sin_weights = _sinusoidal_weights(config.max_position_embeddings, config.hidden_size, self.pad)

        self.layers = [
            _DecoderLayer(state_dict, f"layers.{i}.", config, device, dtype)
            for i in range(config.decoder_layers)
        ]
        self.final_ln = (_to_tt(state_dict["layer_norm.weight"], device, dtype),
                         _to_tt(state_dict["layer_norm.bias"], device, dtype))
        self.eps = config.layer_norm_eps
        # lm_head tied to embeddings: weight (vocab, hidden) -> linear weight (hidden, vocab)
        self.lm_head_w = _to_tt(self.embed_weight.t().contiguous(), device, dtype)

    def _embed_host(self, input_ids):
        emb = torch.nn.functional.embedding(input_ids, self.embed_weight) * self.embed_scale
        pos_ids = _position_ids(input_ids, self.pad)
        pos = self.sin_weights[pos_ids.view(-1)].view(*input_ids.shape, -1)
        return emb + pos  # (1, seq, hidden)

    def _causal_mask(self, seq):
        m = torch.full((seq, seq), float("-inf"))
        m = torch.triu(m, diagonal=1)
        return m.view(1, 1, seq, seq)

    def __call__(self, input_ids, encoder_hidden_states, return_hidden=False):
        """input_ids: torch LongTensor (1, seq). encoder_hidden_states: ttnn (1, enc_seq, hidden).

        Returns logits (1, seq, vocab); if return_hidden, returns (logits, pre-lm_head hidden).
        """
        seq = input_ids.shape[1]
        h = _to_tt(self._embed_host(input_ids), self.device, self.dtype)
        causal = _to_tt(self._causal_mask(seq), self.device, self.dtype)
        for layer in self.layers:
            h = layer(h, causal, encoder_hidden_states)
        h = ttnn.layer_norm(h, weight=self.final_ln[0], bias=self.final_ln[1], epsilon=self.eps)
        logits = ttnn.linear(h, self.lm_head_w)  # (1, seq, vocab)
        if return_hidden:
            return logits, h
        return logits
