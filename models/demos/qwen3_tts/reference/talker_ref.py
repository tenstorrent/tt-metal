# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone PyTorch reference for the Qwen3-TTS Talker (Qwen3-1.7B decoder).
No HuggingFace / transformers dependency — loads weights from safetensors directly.
Used for PCC comparison against the TT implementation.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings. x: [B, H, S, D]"""
    d_half = x.shape[-1] // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    cos = cos.unsqueeze(1)  # [B, 1, S, D/2]
    sin = sin.unsqueeze(1)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


def build_rope_cache(seq_len, head_dim, theta=1000000.0, dtype=torch.float32):
    """Precompute RoPE cos/sin for positions 0..seq_len-1."""
    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=dtype) / half_dim))
    positions = torch.arange(seq_len, dtype=dtype)
    angles = positions[:, None] * freqs[None, :]
    cos = angles.cos()
    sin = angles.sin()
    return cos, sin


class TalkerAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, cos, sin, mask=None, kv_cache=None, start_pos=0):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_cache[:, :, start_pos : start_pos + S] = k.to(k_cache.dtype)
            v_cache[:, :, start_pos : start_pos + S] = v.to(v_cache.dtype)
            k = k_cache[:, :, : start_pos + S].to(q.dtype)
            v = v_cache[:, :, : start_pos + S].to(q.dtype)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class TalkerMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TalkerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim, hidden_dim, norm_eps=1e-6):
        super().__init__()
        self.attention = TalkerAttention(dim, n_heads, n_kv_heads, head_dim)
        self.feed_forward = TalkerMLP(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(self, x, cos, sin, mask=None, kv_cache=None, start_pos=0):
        h = x + self.attention(self.attention_norm(x), cos, sin, mask, kv_cache, start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class TalkerReference(nn.Module):
    """
    Standalone PyTorch Talker reference model.
    Matches the HF Qwen3-TTS Talker architecture exactly.
    """

    def __init__(
        self,
        dim=2048,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        hidden_dim=6144,
        codec_vocab_size=3072,
        text_vocab_size=151936,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        max_seq_len=4096,
        spk_enc_dim=2048,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.codec_vocab_size = codec_vocab_size
        self.text_vocab_size = text_vocab_size
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len

        self.codec_embedding = nn.Embedding(codec_vocab_size, dim)
        self.text_embedding = nn.Embedding(text_vocab_size, dim)
        self.text_projection_fc1 = nn.Linear(dim, dim, bias=True)
        self.text_projection_fc2 = nn.Linear(dim, dim, bias=True)

        self.layers = nn.ModuleList(
            [TalkerBlock(dim, n_heads, n_kv_heads, head_dim, hidden_dim, norm_eps) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(dim, norm_eps)
        self.codec_head = nn.Linear(dim, codec_vocab_size, bias=False)

        cos, sin = build_rope_cache(max_seq_len, head_dim, rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def _causal_mask(self, seq_len, dtype, device):
        mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward_prefill(self, text_token_ids, speaker_emb=None, start_pos=0):
        """Run the Talker in prefill mode with text tokens.

        Args:
            text_token_ids: [B, S] text token IDs
            speaker_emb: optional [B, dim] speaker embedding (same dim as hidden)
            start_pos: starting position for RoPE

        Returns:
            logits: [B, S, codec_vocab_size]
            hidden: [B, S, dim] final hidden state (before LM head)
        """
        B, S = text_token_ids.shape
        x = self.text_embedding(text_token_ids)

        # Text projection MLP: Linear → SiLU → Linear
        x = self.text_projection_fc2(F.silu(self.text_projection_fc1(x)))

        if speaker_emb is not None:
            x = x + speaker_emb.unsqueeze(1)

        cos = self.rope_cos[start_pos : start_pos + S].unsqueeze(0).expand(B, -1, -1)
        sin = self.rope_sin[start_pos : start_pos + S].unsqueeze(0).expand(B, -1, -1)

        mask = self._causal_mask(S, x.dtype, x.device)

        for layer in self.layers:
            x = layer(x, cos, sin, mask=mask)

        hidden = self.norm(x)
        logits = self.codec_head(hidden)
        return logits, hidden

    def forward_decode(self, codec_token_ids, kv_caches, start_pos):
        """Run the Talker in decode mode with a single codec token per position.

        Args:
            codec_token_ids: [B, 1] codec token IDs
            kv_caches: list of (k_cache, v_cache) for each layer
            start_pos: current position in the sequence

        Returns:
            logits: [B, 1, codec_vocab_size]
        """
        B = codec_token_ids.shape[0]
        x = self.codec_embedding(codec_token_ids)

        cos = self.rope_cos[start_pos : start_pos + 1].unsqueeze(0).expand(B, -1, -1)
        sin = self.rope_sin[start_pos : start_pos + 1].unsqueeze(0).expand(B, -1, -1)

        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, kv_cache=kv_caches[i], start_pos=start_pos)

        hidden = self.norm(x)
        logits = self.codec_head(hidden)
        return logits

    def init_kv_caches(self, batch_size, max_seq_len, device, dtype=torch.bfloat16):
        """Allocate empty KV caches for decode mode."""
        caches = []
        for _ in range(self.n_layers):
            k = torch.zeros(batch_size, self.n_kv_heads, max_seq_len, self.head_dim, device=device, dtype=dtype)
            v = torch.zeros(batch_size, self.n_kv_heads, max_seq_len, self.head_dim, device=device, dtype=dtype)
            caches.append((k, v))
        return caches

    @classmethod
    def from_hf_state_dict(cls, state_dict, **kwargs):
        """Create from a HuggingFace state dict (talker.* keys).

        HF key structure:
            talker.model.codec_embedding.weight    → codec_embedding.weight
            talker.model.text_embedding.weight      → text_embedding.weight
            talker.model.layers.N.self_attn.*        → layers.N.attention.*
            talker.model.layers.N.mlp.*              → layers.N.feed_forward.*
            talker.model.norm.weight                 → norm.weight
            talker.codec_head.weight                 → codec_head.weight
            talker.text_projection.linear_fc1.*      → text_projection_fc1.*
            talker.text_projection.linear_fc2.*      → text_projection_fc2.*
        """
        model = cls(**kwargs)

        key_map = {}
        for k in state_dict:
            if not k.startswith("talker."):
                continue
            new_k = k[len("talker."):]
            # Strip "model." prefix from inner model keys
            if new_k.startswith("model."):
                new_k = new_k[len("model."):]
            # Attention / MLP key remapping
            new_k = new_k.replace("self_attn.q_proj", "attention.q_proj")
            new_k = new_k.replace("self_attn.k_proj", "attention.k_proj")
            new_k = new_k.replace("self_attn.v_proj", "attention.v_proj")
            new_k = new_k.replace("self_attn.o_proj", "attention.o_proj")
            new_k = new_k.replace("self_attn.q_norm", "attention.q_norm")
            new_k = new_k.replace("self_attn.k_norm", "attention.k_norm")
            new_k = new_k.replace("mlp.gate_proj", "feed_forward.gate_proj")
            new_k = new_k.replace("mlp.up_proj", "feed_forward.up_proj")
            new_k = new_k.replace("mlp.down_proj", "feed_forward.down_proj")
            new_k = new_k.replace("input_layernorm", "attention_norm")
            new_k = new_k.replace("post_attention_layernorm", "ffn_norm")
            # Text projection: flatten nested key
            new_k = new_k.replace("text_projection.linear_fc1", "text_projection_fc1")
            new_k = new_k.replace("text_projection.linear_fc2", "text_projection_fc2")
            key_map[new_k] = state_dict[k]

        missing, unexpected = model.load_state_dict(key_map, strict=False)
        if missing:
            import warnings

            warnings.warn(f"Missing keys: {missing}")
        return model

    @classmethod
    def from_safetensors(cls, checkpoint_dir, **kwargs):
        """Load from local safetensors files, extracting only talker.* keys."""
        from pathlib import Path

        from safetensors import safe_open

        checkpoint_dir = Path(checkpoint_dir)
        st_files = sorted(checkpoint_dir.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No safetensors files in {checkpoint_dir}")

        state_dict = {}
        for f in st_files:
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    if key.startswith("talker."):
                        state_dict[key] = sf.get_tensor(key)

        return cls.from_hf_state_dict(state_dict, **kwargs)

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        """Load from HuggingFace Hub, downloading if needed."""
        from pathlib import Path

        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        local_dir = Path(model_id)
        if local_dir.is_dir():
            return cls.from_safetensors(model_id, **kwargs)

        path = hf_hub_download(model_id, "model.safetensors")
        state_dict = {}
        with safe_open(path, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if key.startswith("talker."):
                    state_dict[key] = sf.get_tensor(key)

        return cls.from_hf_state_dict(state_dict, **kwargs)


def generate_cb0(
    model,
    text_token_ids,
    speaker_emb=None,
    max_new_tokens=256,
    temperature=0.9,
    top_k=50,
    top_p=1.0,
    device="cpu",
):
    """Full autoregressive CB0 generation: prefill + decode loop.

    Args:
        model: TalkerReference instance
        text_token_ids: [B, S] text token IDs
        speaker_emb: optional [B, dim] speaker embedding
        max_new_tokens: max codec tokens to generate
        temperature: sampling temperature
        top_k: top-k for sampling
        top_p: nucleus sampling threshold
        device: torch device

    Returns:
        generated_tokens: [B, num_tokens] codec token IDs
    """
    model.eval()
    model.to(device)
    text_token_ids = text_token_ids.to(device)

    B, S = text_token_ids.shape

    # Prefill
    logits, hidden = model.forward_prefill(text_token_ids, speaker_emb=speaker_emb)
    last_logits = logits[:, -1:, :]  # [B, 1, vocab]

    # Sample first token
    next_token = _sample(last_logits, temperature, top_k, top_p)  # [B]
    generated = [next_token.unsqueeze(-1)]  # list of [B, 1]

    # Init KV caches and fill with prefill K/V
    max_seq = S + max_new_tokens
    kv_caches = model.init_kv_caches(B, max_seq, device, dtype=hidden.dtype)

    # Re-run prefill to populate KV caches
    x = model.text_embedding(text_token_ids)
    x = model.text_projection_fc2(F.silu(model.text_projection_fc1(x)))
    if speaker_emb is not None:
        x = x + speaker_emb.unsqueeze(1)

    cos = model.rope_cos[:S].unsqueeze(0).expand(B, -1, -1).to(device)
    sin = model.rope_sin[:S].unsqueeze(0).expand(B, -1, -1).to(device)
    mask = model._causal_mask(S, x.dtype, device)

    for i, layer in enumerate(model.layers):
        x = layer(x, cos, sin, mask=mask, kv_cache=kv_caches[i], start_pos=0)

    # Decode loop
    for step in range(max_new_tokens - 1):
        pos = S + step
        decode_logits = model.forward_decode(next_token.unsqueeze(1), kv_caches, start_pos=pos)
        next_token = _sample(decode_logits, temperature, top_k, top_p)
        generated.append(next_token.unsqueeze(-1))

        if (next_token == model.codec_vocab_size - 1).all():
            break

    return torch.cat(generated, dim=1)


def _sample(logits, temperature, top_k, top_p):
    """Sample from logits [B, 1, vocab]. Returns [B] token IDs."""
    logits = logits[:, -1, :]
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=False).unsqueeze(0) if logits.dim() == 1 else torch.argmax(
            logits, dim=-1
        )
    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        vals, _ = torch.topk(logits, top_k)
        logits[logits < vals[:, [-1]]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
