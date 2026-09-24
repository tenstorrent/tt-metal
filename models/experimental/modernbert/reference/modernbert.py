# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Torch reference implementation of ModernBERT.

Written from scratch rather than wrapping HuggingFace so that a TTNN mismatch has
a single possible cause: this reference is verified against HF first, then the
TTNN modules are verified against this.

Module and parameter names mirror HF exactly, so
`load_state_dict(hf_model.state_dict(), strict=True)` doubles as the weight-map
check -- it raises on any missing, unexpected or misshaped key.

The `ablate` arguments deliberately break one architectural detail each. They
exist for the negative-control tests, which assert that a broken variant fails;
they are inert by default.

Matches transformers 5.10.2.
"""

import torch
import torch.nn as nn

from models.experimental.modernbert.common import FULL_ATTENTION, SLIDING_ATTENTION


def rotate_half(x):
    """HF (non-interleaved) convention: split in half, negate the second, concat."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """cos/sin arrive as (B, S, head_dim) and broadcast over heads at dim 1.

    Computed in fp32 then cast back, matching HF.
    """
    orig_dtype = q.dtype
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class ModernBertRotaryEmbedding(nn.Module):
    """One rotary cache per layer type. ModernBERT uses two different thetas:
    full_attention=160000.0, sliding_attention=10000.0 (measured)."""

    def __init__(self, head_dim, theta):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, dtype):
        # (B, head_dim/2, 1) @ (B, 1, S) -> (B, head_dim/2, S) -> (B, S, head_dim/2)
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class ModernBertEmbeddings(nn.Module):
    """Token embeddings + LayerNorm. No positional embeddings - RoPE carries position."""

    def __init__(self, config):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def forward(self, input_ids):
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertMLP(nn.Module):
    """GeGLU. Wi projects to 2 * intermediate_size and is chunked into (input, gate).

    NOTE the activation lands on the FIRST half. This is inverted relative to the
    common SwiGLU convention where the gate is first and receives the activation.
    Getting it backwards yields plausible-looking but wrong output.
    """

    def __init__(self, config, ablate_swap_gate=False):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias)
        self.act = nn.GELU()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.ablate_swap_gate = ablate_swap_gate

    def forward(self, hidden_states):
        a, b = self.Wi(hidden_states).chunk(2, dim=-1)
        if self.ablate_swap_gate:  # negative control NC1
            return self.Wo(self.act(b) * a)
        return self.Wo(self.act(a) * b)


class ModernBertAttention(nn.Module):
    """Bidirectional attention with a fused QKV projection.

    Wqkv output (2304) reshapes to (B, S, 3, n_heads, head_dim), so the 3 is the
    OUTER dim: rows 0:768 are Q, 768:1536 are K, 1536:2304 are V.
    """

    def __init__(self, config, ablate_qkv_permute=False):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.Wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.ablate_qkv_permute = ablate_qkv_permute

    def forward(self, hidden_states, position_embeddings, attention_mask):
        B, S, _ = hidden_states.shape
        qkv = self.Wqkv(hidden_states).view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=-3)
        if self.ablate_qkv_permute:  # negative control NC4
            q, k, v = k, q, v
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = (q @ k.transpose(2, 3)) * (self.head_dim**-0.5)
        if attention_mask is not None:
            attn = attn + attention_mask
        # softmax in fp32 then cast back, matching HF
        attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
        return self.Wo(out)


class ModernBertEncoderLayer(nn.Module):
    """Pre-norm residual block.

    Layer 0 has NO attention norm - HF uses nn.Identity() there because the
    embedding LayerNorm already normalised the input. Consequently there is no
    `layers.0.attn_norm.weight` in the state dict (134 tensors total, not 135).
    """

    def __init__(self, config, layer_idx, ablate=None):
        super().__init__()
        ablate = ablate or {}
        force_norm_l0 = ablate.get("norm_at_layer0", False)  # negative control NC3
        if layer_idx == 0 and not force_norm_l0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = ModernBertAttention(config, ablate_qkv_permute=ablate.get("qkv_permute", False))
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config, ablate_swap_gate=ablate.get("swap_gate", False))
        self.attention_type = config.layer_types[layer_idx]

    def forward(self, hidden_states, position_embeddings, attention_mask):
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states), position_embeddings, attention_mask)
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


def build_masks(config, attention_mask, seq_len, dtype, half_window):
    """Additive attention masks, one per layer type.

    full_attention    : padding only (None when nothing is padded)
    sliding_attention : symmetric band |i - j| <= half_window, measured as +/-64
                        for local_attention=128, i.e. a total width of 129.
    """
    device = attention_mask.device
    neg = torch.finfo(dtype).min

    pad = None
    if not torch.all(attention_mask == 1):
        pad = torch.zeros(attention_mask.shape, dtype=dtype, device=device)
        pad = pad.masked_fill(attention_mask == 0, neg)
        pad = pad[:, None, None, :]  # (B,1,1,S)

    idx = torch.arange(seq_len, device=device)
    band = (idx[None, :] - idx[:, None]).abs() <= half_window
    sliding = torch.zeros(seq_len, seq_len, dtype=dtype, device=device).masked_fill(~band, neg)
    sliding = sliding[None, None]  # (1,1,S,S)
    if pad is not None:
        sliding = sliding + pad

    return {FULL_ATTENTION: pad, SLIDING_ATTENTION: sliding}


class ModernBertModel(nn.Module):
    """Full encoder. Produces last_hidden_state."""

    def __init__(self, config, ablate=None):
        super().__init__()
        self.config = config
        ablate = ablate or {}
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ModernBertEncoderLayer(config, i, ablate=ablate) for i in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

        head_dim = config.hidden_size // config.num_attention_heads
        thetas = {lt: config.rope_parameters[lt]["rope_theta"] for lt in set(config.layer_types)}
        if ablate.get("single_theta", False):  # negative control NC5
            thetas = {lt: config.rope_parameters[FULL_ATTENTION]["rope_theta"] for lt in thetas}
        # not an nn.ModuleDict on purpose: these hold no learnable parameters, and
        # registering them would add keys that break strict state_dict loading.
        self._rotary = {lt: ModernBertRotaryEmbedding(head_dim, th) for lt, th in thetas.items()}

        self.half_window = ablate.get("half_window_override", config.local_attention // 2)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        B, S = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        hidden = self.embeddings(input_ids)
        masks = build_masks(self.config, attention_mask, S, hidden.dtype, self.half_window)
        pos_emb = {lt: rot(position_ids, hidden.dtype) for lt, rot in self._rotary.items()}

        all_hidden = [hidden] if output_hidden_states else None
        for layer in self.layers:
            hidden = layer(hidden, pos_emb[layer.attention_type], masks[layer.attention_type])
            if output_hidden_states:
                all_hidden.append(hidden)

        hidden = self.final_norm(hidden)
        if output_hidden_states:
            # HF exposes 23 entries for 22 layers: (embeddings, L0..L20, final_norm(L21)).
            # The last raw layer output is REPLACED by its normalised form, so
            # hidden_states[-1] is last_hidden_state. Mirror that exactly, otherwise
            # index alignment against HF silently compares the wrong pair.
            all_hidden[-1] = hidden
            return hidden, all_hidden
        return hidden


class ModernBertPredictionHead(nn.Module):
    """MLM prediction head: norm(gelu(dense(x))).

    dense and norm are both bias-free (classifier_bias=False, norm_bias=False).
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.classifier_bias)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def forward(self, hidden_states):
        return self.norm(self.act(self.dense(hidden_states)))


class ModernBertForMaskedLM(nn.Module):
    """Encoder plus MLM head: logits = decoder(head(last_hidden_state)).

    The decoder is the ONLY layer in the entire model carrying a bias
    (decoder_bias=True). The paper states this "might help alleviate weight
    tying's negative effects" - tie_word_embeddings is True, so decoder.weight is
    tied to embeddings.tok_embeddings.weight.

    Module names mirror HF exactly so that
    load_state_dict(hf_mlm.state_dict(), strict=True) is the mapping proof.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.model(input_ids, attention_mask)
        return self.decoder(self.head(hidden))
