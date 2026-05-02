"""
Reference PyTorch implementation of Voxtral-4B-TTS-2603.

Three components:
  1. text_decoder_forward      — 26-layer GQA transformer (Ministral-3B)
  2. acoustic_transformer_step — flow-matching transformer (3 layers, per ODE step)
  3. codec_decoder_forward     — 4-stage causal conv + ALiBi transformer

Inference flow:
  text_decoder_prefill(voice_tokens + text_tokens) → h [S, 3072]
  semantic_decode(h) → semantic_token_ids [N_frames]
  ode_solve(semantic_h) → acoustic_codes [N_frames, 36]
  codec_decoder_forward(semantic_codes, acoustic_codes) → waveform [samples]

Each function accepts capture_intermediates=False.
When True, returns (output, caps) where caps is a dict of named intermediate tensors.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

# ─────────────────────────── helpers ───────────────────────────


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    norm = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (x.float() * norm).to(x.dtype) * weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device) -> tuple:
    pos = torch.arange(seq_len, device=device).float()
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def build_alibi_bias(
    seq_len: int,
    n_heads: int,
    window_size: int,
    device,
) -> torch.Tensor:
    """ALiBi + causal sliding-window mask [1, n_heads, seq_len, seq_len]."""
    slopes = 2.0 ** (-8.0 * torch.arange(1, n_heads + 1, device=device).float() / n_heads)

    # positions
    i = torch.arange(seq_len, device=device).view(-1, 1)
    j = torch.arange(seq_len, device=device).view(1, -1)
    dist = (j - i).float()  # negative for attending backward

    # ALiBi: add slope * relative position (linear decay for positions to the left)
    alibi = slopes.view(n_heads, 1, 1) * dist.unsqueeze(0)  # [n_heads, S, S]

    # Causal + sliding window mask
    causal_mask = j > i  # future positions: -inf
    window_mask = (i - j) > window_size  # too-far-left positions: -inf
    mask = causal_mask | window_mask
    alibi = alibi.masked_fill(mask.unsqueeze(0), float("-inf"))

    return alibi.unsqueeze(0)  # [1, n_heads, S, S]


def causal_conv1d(
    x: torch.Tensor,  # [B, L, C] (channels-last)
    weight: torch.Tensor,  # [out, in, k]
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
) -> torch.Tensor:
    """Causal conv1d: left-pad by (kernel_size-1) for causal property."""
    k = weight.shape[-1]
    # to channels-first [B, C, L]
    x_ncl = x.permute(0, 2, 1)
    # causal left-pad
    x_padded = F.pad(x_ncl, (k - 1, 0))
    out = F.conv1d(x_padded, weight, bias, stride=stride, padding=0)
    # back to [B, L', C]
    return out.permute(0, 2, 1)


def causal_conv_transpose1d(
    x: torch.Tensor,  # [B, L, C]
    weight: torch.Tensor,  # [in, out, k]
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
) -> torch.Tensor:
    """
    Causal ConvTranspose1d for upsampling.
    Remove first (kernel_size - stride) samples to make it causal.
    """
    x_ncl = x.permute(0, 2, 1)
    k = weight.shape[-1]
    out = F.conv_transpose1d(x_ncl, weight, bias, stride=stride, padding=0)
    # trim non-causal tail introduced by transposed conv
    trim = k - stride
    if trim > 0:
        out = out[:, :, trim:]
    return out.permute(0, 2, 1)


def time_sinusoidal_embedding(t: torch.Tensor, dim: int = 3072) -> torch.Tensor:
    """Sinusoidal timestep embedding for flow-matching transformer."""
    half = dim // 2
    inv_freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    emb = t.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    return torch.cat([emb.cos(), emb.sin()], dim=-1)


# ─────────────────────────── text decoder blocks ───────────────────────────


def text_attention(
    x: torch.Tensor,  # [B, S, D]
    sd: dict,
    layer_idx: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_heads: int = 32,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    kv_cache: Optional[dict] = None,
    capture_intermediates: bool = False,
) -> tuple:
    caps = {}
    pfx = f"layers.{layer_idx}.attention"

    B, S, D = x.shape
    wq = sd[f"{pfx}.wq.weight"]  # [n_heads*head_dim, D]
    wk = sd[f"{pfx}.wk.weight"]  # [n_kv_heads*head_dim, D]
    wv = sd[f"{pfx}.wv.weight"]
    wo = sd[f"{pfx}.wo.weight"]  # [D, n_heads*head_dim]

    # Infer dims from weight shapes if not matching defaults
    n_heads_actual = wq.shape[0] // head_dim
    n_kv_actual = wk.shape[0] // head_dim

    q = F.linear(x, wq).view(B, S, n_heads_actual, head_dim).transpose(1, 2)
    k = F.linear(x, wk).view(B, S, n_kv_actual, head_dim).transpose(1, 2)
    v = F.linear(x, wv).view(B, S, n_kv_actual, head_dim).transpose(1, 2)
    n_heads = n_heads_actual
    n_kv_heads = n_kv_actual

    if capture_intermediates:
        caps["q"] = q
        caps["k"] = k
        caps["v"] = v

    q, k = apply_rope(q, k, cos[:S], sin[:S])

    if kv_cache is not None:
        k = torch.cat([kv_cache["k"], k], dim=2)
        v = torch.cat([kv_cache["v"], v], dim=2)
        kv_cache["k"] = k
        kv_cache["v"] = v

    # GQA: repeat k,v to match n_heads
    repeat = n_heads // n_kv_heads
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)

    scale = head_dim**-0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale

    # causal mask
    kv_len = k.shape[2]
    causal = torch.zeros(S, kv_len, device=x.device)
    causal = causal.masked_fill(
        torch.arange(S, device=x.device).view(-1, 1) < torch.arange(kv_len, device=x.device).view(1, -1) - (kv_len - S),
        float("-inf"),
    )
    attn = attn + causal.unsqueeze(0).unsqueeze(0)

    attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
    out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, n_heads * head_dim)

    if capture_intermediates:
        caps["attn_out_pre_proj"] = out

    out = F.linear(out, wo)
    if capture_intermediates:
        caps["attn_out"] = out

    return out, caps


def text_mlp(
    x: torch.Tensor,
    sd: dict,
    layer_idx: int,
    capture_intermediates: bool = False,
) -> tuple:
    caps = {}
    pfx = f"layers.{layer_idx}.feed_forward"

    gate = F.linear(x, sd[f"{pfx}.w1.weight"])
    up = F.linear(x, sd[f"{pfx}.w3.weight"])
    hidden = F.silu(gate) * up

    if capture_intermediates:
        caps["gate"] = gate
        caps["up"] = up
        caps["hidden"] = hidden

    out = F.linear(hidden, sd[f"{pfx}.w2.weight"])

    if capture_intermediates:
        caps["mlp_out"] = out

    return out, caps


def text_decoder_layer(
    x: torch.Tensor,
    sd: dict,
    layer_idx: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_cache: Optional[dict] = None,
    capture_intermediates: bool = False,
) -> tuple:
    caps = {}
    pfx = f"layers.{layer_idx}"

    normed = rms_norm(x, sd[f"{pfx}.attention_norm.weight"])
    attn_out, attn_caps = text_attention(
        normed, sd, layer_idx, cos, sin, kv_cache=kv_cache, capture_intermediates=capture_intermediates
    )
    x = x + attn_out

    normed2 = rms_norm(x, sd[f"{pfx}.ffn_norm.weight"])
    mlp_out, mlp_caps = text_mlp(normed2, sd, layer_idx, capture_intermediates)
    x = x + mlp_out

    if capture_intermediates:
        caps.update({f"attn_{k}": v for k, v in attn_caps.items()})
        caps.update({f"mlp_{k}": v for k, v in mlp_caps.items()})
        caps["layer_out"] = x

    return x, caps


def text_decoder_forward(
    input_ids: Optional[torch.Tensor],  # [B, S] or None if inputs_embeds given
    sd: dict,  # full state dict
    n_layers: int = 26,
    rope_theta: float = 1e6,
    head_dim: int = 128,
    kv_caches: Optional[list] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    capture_intermediates: bool = False,
) -> tuple:
    """
    Full text decoder forward pass.

    Returns: (hidden_states [B, S, 3072], caps)
    """
    caps = {}

    if inputs_embeds is None:
        x = F.embedding(input_ids, sd["mm_audio_embeddings.tok_embeddings.weight"])
    else:
        x = inputs_embeds

    B, S, D = x.shape
    cos, sin = build_rope_cache(S, head_dim, rope_theta, x.device)

    if capture_intermediates:
        caps["embed"] = x

    for layer_idx in range(n_layers):
        kv = kv_caches[layer_idx] if kv_caches else None
        x, layer_caps = text_decoder_layer(
            x,
            sd,
            layer_idx,
            cos,
            sin,
            kv_cache=kv,
            capture_intermediates=capture_intermediates,
        )
        if capture_intermediates:
            caps[f"layer_{layer_idx}"] = layer_caps

    x = rms_norm(x, sd["norm.weight"])

    if capture_intermediates:
        caps["final_norm"] = x

    return x, caps


# ─────────────────────────── acoustic transformer ───────────────────────────


def acoustic_transformer_step(
    h: torch.Tensor,  # [B, N_frames, 3072] — text decoder hidden states
    x_t: torch.Tensor,  # [B, N_frames, 36] — current acoustic embedding
    t: float,  # ODE timestep in [0, 1]
    sd: dict,  # acoustic transformer state dict (stripped prefix)
    n_layers: int = 3,
    capture_intermediates: bool = False,
) -> tuple:
    """
    One step of the flow-matching acoustic transformer.
    Returns: (velocity [B, N_frames, 36], semantic_logits [B, N_frames, 8320], caps)

    Inputs are combined via three separate projections before the transformer:
      h_proj:  LLM hidden state → D
      t_proj:  sinusoidal timestep → D
      x_proj:  36-dim acoustic noise → D
    All three are summed to form the transformer input.
    """
    caps = {}
    B, N, D = h.shape
    dtype = sd["llm_projection.weight"].dtype
    h = h.to(dtype)
    x_t = x_t.to(dtype)

    # Project LLM hidden state
    h_proj = F.linear(h, sd["llm_projection.weight"])

    # Sinusoidal timestep embedding
    t_tensor = torch.full((B,), t, device=h.device, dtype=h.dtype)
    t_emb = time_sinusoidal_embedding(t_tensor, dim=D)  # [B, D]
    t_emb = t_emb.to(dtype)
    t_proj = F.linear(t_emb.unsqueeze(1).expand(-1, N, -1), sd["time_projection.weight"])

    # Project acoustic noise
    x_proj = F.linear(x_t, sd["input_projection.weight"])

    combined = h_proj + t_proj + x_proj

    if capture_intermediates:
        caps["h_proj"] = h_proj
        caps["t_proj"] = t_proj
        caps["x_proj"] = x_proj
        caps["combined"] = combined

    # 3-layer transformer (bidirectional — no causal mask for flow-matching)
    hidden = combined
    cos, sin = build_rope_cache(N, 128, 10000.0, h.device)

    for layer_idx in range(n_layers):
        pfx = f"layers.{layer_idx}"

        normed = rms_norm(hidden, sd[f"{pfx}.attention_norm.weight"])

        # GQA attention (no causal mask — bidirectional)
        wq = sd[f"{pfx}.attention.wq.weight"]
        wk = sd[f"{pfx}.attention.wk.weight"]
        wv = sd[f"{pfx}.attention.wv.weight"]
        wo = sd[f"{pfx}.attention.wo.weight"]
        n_heads, n_kv, head_dim = 32, 8, 128

        q = F.linear(normed, wq).view(B, N, n_heads, head_dim).transpose(1, 2)
        k = F.linear(normed, wk).view(B, N, n_kv, head_dim).transpose(1, 2)
        v = F.linear(normed, wv).view(B, N, n_kv, head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos[:N], sin[:N])

        repeat = n_heads // n_kv
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        scale = head_dim**-0.5
        attn = F.softmax((torch.matmul(q, k.transpose(-2, -1)) * scale).float(), dim=-1).to(hidden.dtype)
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, n_heads * head_dim)
        attn_out = F.linear(attn_out, wo)
        hidden = hidden + attn_out

        normed2 = rms_norm(hidden, sd[f"{pfx}.ffn_norm.weight"])
        gate = F.linear(normed2, sd[f"{pfx}.feed_forward.w1.weight"])
        up = F.linear(normed2, sd[f"{pfx}.feed_forward.w3.weight"])
        mlp_out = F.linear(F.silu(gate) * up, sd[f"{pfx}.feed_forward.w2.weight"])
        hidden = hidden + mlp_out

        if capture_intermediates:
            caps[f"layer_{layer_idx}_out"] = hidden

    hidden = rms_norm(hidden, sd["norm.weight"])

    # Predict semantic logits and acoustic velocity
    semantic_logits = F.linear(hidden, sd["semantic_codebook_output.weight"])  # [B, N, 8320]
    velocity = F.linear(hidden, sd["acoustic_codebook_output.weight"])  # [B, N, 36]

    if capture_intermediates:
        caps["hidden_normed"] = hidden
        caps["semantic_logits"] = semantic_logits
        caps["velocity"] = velocity

    return velocity, semantic_logits, caps


def ode_solve(
    h: torch.Tensor,  # [B, N_frames, 3072] from text decoder
    sd_acoustic: dict,  # acoustic transformer state (stripped prefix)
    n_steps: int = 8,
    cfg_alpha: float = 1.2,
    null_h: Optional[torch.Tensor] = None,
    capture_intermediates: bool = False,
) -> tuple:
    """
    Euler ODE solver for flow-matching acoustic token generation.

    Args:
        h: conditioning hidden states from text decoder (or null for CFG)
        sd_acoustic: acoustic transformer weights
        n_steps: Euler steps (default 8)
        cfg_alpha: classifier-free guidance scale (1.2)
        null_h: unconditioned hidden states for CFG (zeros if None)

    Returns: (acoustic_codes [B, N_frames, 36], final_x [B, N_frames, 36], caps)
    """
    caps = {}
    B, N, D = h.shape
    device = h.device

    if null_h is None:
        null_h = torch.zeros_like(h)

    x_t = torch.randn(B, N, 36, device=device, dtype=h.dtype)
    dt = 1.0 / n_steps

    for step in range(n_steps):
        t = step * dt

        # Conditioned pass
        v_cond, _, _ = acoustic_transformer_step(h, x_t, t, sd_acoustic)
        # Unconditioned pass (CFG)
        v_uncond, _, _ = acoustic_transformer_step(null_h, x_t, t, sd_acoustic)

        v_guided = cfg_alpha * v_cond + (1.0 - cfg_alpha) * v_uncond
        x_t = x_t + v_guided * dt

        if capture_intermediates:
            caps[f"step_{step}_x"] = x_t.clone()

    # FSQ quantization: 21 levels in [-1, 1] with step 0.1.
    # Continuous x_t ∈ approx [-1, 1] → code = round(x * 10 + 10), clamped to [0, 20].
    # x_t.round() alone was wrong: maps [-1, 1] → {-1, 0, 1} → all codes near 0.
    acoustic_codes = (x_t * 10 + 10).round().long().clamp(0, 20)

    if capture_intermediates:
        caps["final_x_continuous"] = x_t
        caps["acoustic_codes"] = acoustic_codes

    return acoustic_codes, x_t, caps


# ─────────────────────────── codec decoder ───────────────────────────


def codec_attention(
    x: torch.Tensor,  # [B, L, C]
    sd: dict,  # codec decoder state dict
    block_idx: int,
    layer_idx: int,
    window_size: int,
    n_heads: int = 8,
    head_dim: int = 128,
    capture_intermediates: bool = False,
) -> tuple:
    caps = {}
    B, L, C = x.shape
    pfx = f"decoder_blocks.{block_idx}.layers.{layer_idx}.attention"

    wq = sd[f"{pfx}.wq.weight"]  # [C, C]
    x = x.to(wq.dtype)
    wk = sd[f"{pfx}.wk.weight"]
    wv = sd[f"{pfx}.wv.weight"]
    wo = sd[f"{pfx}.wo.weight"]

    q = F.linear(x, wq).view(B, L, n_heads, head_dim).transpose(1, 2)
    k = F.linear(x, wk).view(B, L, n_heads, head_dim).transpose(1, 2)
    v = F.linear(x, wv).view(B, L, n_heads, head_dim).transpose(1, 2)

    # QK-norm: reshape to [B*L, C], apply rms_norm with weight [C], reshape back
    q_norm_w = sd[f"{pfx}.q_norm.weight"]  # [C=1024]
    k_norm_w = sd[f"{pfx}.k_norm.weight"]
    eps = 1e-6

    q_flat = q.permute(0, 2, 1, 3).contiguous().view(B * L, n_heads * head_dim)
    q_flat = rms_norm(q_flat, q_norm_w, eps=eps)
    q = q_flat.view(B, L, n_heads, head_dim).permute(0, 2, 1, 3)

    k_flat = k.permute(0, 2, 1, 3).contiguous().view(B * L, n_heads * head_dim)
    k_flat = rms_norm(k_flat, k_norm_w, eps=eps)
    k = k_flat.view(B, L, n_heads, head_dim).permute(0, 2, 1, 3)

    scale = head_dim**-0.5
    attn_bias = build_alibi_bias(L, n_heads, window_size, x.device)

    attn = torch.matmul(q, k.transpose(-2, -1)) * scale + attn_bias
    attn = F.softmax(attn.float(), dim=-1).to(x.dtype)

    out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, C)
    out = F.linear(out, wo)

    if capture_intermediates:
        caps["attn_out"] = out

    return out, caps


def codec_mlp(
    x: torch.Tensor,
    sd: dict,
    block_idx: int,
    layer_idx: int,
    capture_intermediates: bool = False,
) -> tuple:
    caps = {}
    pfx = f"decoder_blocks.{block_idx}.layers.{layer_idx}.feed_forward"

    gate = F.linear(x, sd[f"{pfx}.w1.weight"])
    up = F.linear(x, sd[f"{pfx}.w3.weight"])
    out = F.linear(F.silu(gate) * up, sd[f"{pfx}.w2.weight"])

    if capture_intermediates:
        caps["mlp_out"] = out
    return out, caps


def codec_transformer_block(
    x: torch.Tensor,  # [B, L, C=1024]
    sd: dict,
    block_idx: int,  # odd index in decoder_blocks (1, 3, 5, 7)
    layer_idx: int,  # 0 or 1 within the block
    window_size: int,
    capture_intermediates: bool = False,
) -> tuple:
    """Single transformer layer inside a codec decoder block."""
    caps = {}
    pfx = f"decoder_blocks.{block_idx}.layers.{layer_idx}"

    # Pre-norm + attention
    normed = rms_norm(x, sd[f"{pfx}.attention_norm.weight"], eps=0.01)
    attn_out, attn_caps = codec_attention(
        normed, sd, block_idx, layer_idx, window_size, capture_intermediates=capture_intermediates
    )
    # LayerScale
    attn_scale = sd[f"{pfx}.attention_scale"]  # [C]
    x = x + attn_out * attn_scale

    # Pre-norm + MLP
    normed2 = rms_norm(x, sd[f"{pfx}.ffn_norm.weight"], eps=0.01)
    mlp_out, mlp_caps = codec_mlp(normed2, sd, block_idx, layer_idx, capture_intermediates)
    ffn_scale = sd[f"{pfx}.ffn_scale"]  # [C]
    x = x + mlp_out * ffn_scale

    if capture_intermediates:
        caps.update({f"attn_{k}": v for k, v in attn_caps.items()})
        caps.update({f"mlp_{k}": v for k, v in mlp_caps.items()})
        caps["layer_out"] = x

    return x, caps


def codec_decoder_forward(
    semantic_codes: torch.Tensor,  # [B, N_frames]  int64, values 0..8191
    acoustic_codes: torch.Tensor,  # [B, N_frames, 36]  int64, values 0..20
    sd_codec: dict,  # codec decoder state dict (stripped prefix)
    capture_intermediates: bool = False,
) -> tuple:
    """
    Decode semantic + acoustic tokens to waveform patches.

    Audio tokenizer decoder block layout (8 blocks):
      Block 0: initial conv [292→1024, k=3, stride=1]  (weight_norm fused)
      Block 1: 2× transformer layers (window=2)
      Block 2: conv_transpose [1024→1024, k=4, stride=2]
      Block 3: 2× transformer layers (window=4)
      Block 4: conv_transpose [1024→1024, k=4, stride=2]
      Block 5: 2× transformer layers (window=8)
      Block 6: conv_transpose [1024→1024, k=4, stride=2]
      Block 7: 2× transformer layers (window=16)
      output_proj: conv [1024→240, k=7, stride=1]

    Output: waveform patches [B, N_frames*8, 240] → reshape to [B, N_frames * 1920]
    (at 12.5 Hz × 8 upsampling = 100 Hz × 240 samples = 24kHz)
    """
    caps = {}
    B, N = semantic_codes.shape

    # ── Dequantize inputs ──
    # Semantic: VQ lookup [8192→256], from audio_codebook_embeddings table
    # The combined embedding table is [9088, 3072] for INPUT to the text backbone.
    # For the CODEC decoder, semantic is re-embedded into 256-dim via VQ codebook vectors.
    # Acoustic: FSQ dequantize — integer code → float value in [-1, 1]
    #   code k in 0..20 → float = (k - 10) / 10.0  (21 levels centered at 0)
    semantic_float = sd_codec["quantizer.semantic_codebook.embedding_sum"] / sd_codec[
        "quantizer.semantic_codebook.cluster_usage"
    ].clamp(min=1e-8).unsqueeze(-1)
    # semantic_float: [8192, 256]
    sem_emb = semantic_float[semantic_codes]  # [B, N, 256]

    # Acoustic FSQ dequantize to [-1, 1] range
    acoustic_float = (acoustic_codes.float() - 10.0) / 10.0  # [B, N, 36]

    # Concatenate: [B, N, 292]
    codec_input = torch.cat([sem_emb, acoustic_float], dim=-1)

    if capture_intermediates:
        caps["codec_input"] = codec_input

    # ── Block 0: initial projection conv (kernel=3, stride=1) ──
    w0 = sd_codec["decoder_blocks.0.conv.weight"]  # [1024, 292, 3] (weight_norm fused)
    codec_input = codec_input.to(w0.dtype)
    x = causal_conv1d(codec_input, w0)  # [B, N, 1024]

    if capture_intermediates:
        caps["block0_out"] = x

    # ── Blocks 1-7: alternating transformer + conv_transpose ──
    # window sizes in decoder (mirrors encoder in reverse):
    # encoder windows: 16→8→4→2 (downsampling)
    # decoder windows: 2→4→8→16 (upsampling)
    window_sizes = {1: 2, 3: 4, 5: 8, 7: 16}

    for block_idx in range(1, 8):
        if block_idx % 2 == 1:
            # Transformer block (2 layers)
            win = window_sizes[block_idx]
            for layer_idx in range(2):
                x, layer_caps = codec_transformer_block(
                    x,
                    sd_codec,
                    block_idx,
                    layer_idx,
                    win,
                    capture_intermediates=capture_intermediates,
                )
                if capture_intermediates:
                    caps[f"block{block_idx}_layer{layer_idx}"] = layer_caps
        else:
            # ConvTranspose for 2× upsampling
            w_ct = sd_codec[f"decoder_blocks.{block_idx}.conv.weight"]  # [1024, 1024, 4]
            x = causal_conv_transpose1d(x, w_ct, stride=2)
            if capture_intermediates:
                caps[f"block{block_idx}_out"] = x

    # ── Output projection: conv [1024→240, k=7] ──
    w_out = sd_codec["output_proj.conv.weight"]  # [240, 1024, 7]
    x = causal_conv1d(x, w_out)  # [B, N*8, 240]

    if capture_intermediates:
        caps["output_proj"] = x

    # Reshape to waveform: [B, N*8*240] = [B, N*1920]
    waveform = x.contiguous().view(B, -1)

    return waveform, caps


# ─────────────────────────── full TTS inference ───────────────────────────


def embed_audio_tokens(
    semantic_ids: torch.Tensor,  # [B, N_frames] int64
    acoustic_ids: torch.Tensor,  # [B, N_frames, 36] int64, values 0..20
    emb_table: torch.Tensor,  # [9088, 3072] unified audio embedding table
) -> torch.Tensor:
    """
    Compute input embeddings for audio frames fed into the text backbone.

    The embedding table [9088, 3072] is indexed as:
      - Semantic frames: entry = semantic_code (0..8191)
      - Acoustic codebook k, level v: entry = 8192 + k * 25 + v (approx scheme)
        (exact scheme TBD from model behavior; 9088 = 8192 + 36*25 - 4 approximately)

    All 37 per-frame embeddings are summed → single [D] vector per frame.
    """
    B, N = semantic_ids.shape

    # Semantic embedding
    sem_emb = F.embedding(semantic_ids, emb_table)  # [B, N, 3072]

    # Acoustic embeddings — sum across 36 codebooks
    # ID scheme: 8192 + codebook_k * (9088-8192)//36 + level_v
    # Using approximate mapping for now; will verify against reference
    acoustic_base = 8192
    n_acoustic = emb_table.shape[0] - acoustic_base  # 896
    codes_per_book = n_acoustic // 36  # ~24 (actual will differ if 25)

    acoustic_ids_clamped = acoustic_ids.clamp(0, codes_per_book - 1)
    flat_ids = (
        acoustic_base
        + torch.arange(36, device=acoustic_ids.device).view(1, 1, 36) * codes_per_book
        + acoustic_ids_clamped
    )  # [B, N, 36]

    aco_emb = F.embedding(flat_ids.view(-1), emb_table).view(B, N, 36, 3072).sum(dim=2)

    return sem_emb + aco_emb  # [B, N, 3072]


def tts_generate(
    text_ids: torch.Tensor,  # [1, T] text token IDs
    voice_emb: torch.Tensor,  # [1, V_frames, 3072] pre-computed voice embeddings
    sd_text: dict,  # text decoder state dict
    sd_acoustic: dict,  # acoustic transformer state dict
    sd_codec: dict,  # codec decoder state dict
    max_audio_frames: int = 1000,
    cfg_alpha: float = 1.2,
    temperature: float = 1.0,
    capture_intermediates: bool = False,
) -> tuple:
    """
    End-to-end TTS generation.

    1. Prefill text backbone with [voice_tokens + text_tokens]
    2. Decode semantic tokens autoregressively
    3. ODE solve for acoustic tokens
    4. Decode to waveform
    """
    caps = {}

    # Embed text tokens
    tok_emb = sd_text["mm_audio_embeddings.tok_embeddings.weight"]
    text_emb = F.embedding(text_ids, tok_emb)  # [1, T, 3072]

    # Concatenate: voice embeddings (pre-computed) + text
    prefill_input = torch.cat([voice_emb, text_emb], dim=1)  # [1, V+T, 3072]

    # Prefill run to get hidden states h
    h, text_caps = text_decoder_forward(
        input_ids=None,
        sd=sd_text,
        inputs_embeds=prefill_input,
        capture_intermediates=capture_intermediates,
    )  # h: [1, V+T, 3072]

    # Extract only the text-position hidden states for acoustic conditioning
    V = voice_emb.shape[1]
    h_text = h[:, V:, :]  # [1, T, 3072]

    # ODE solve: flow matching on the text positions
    # (simplified; real impl would decode semantic tokens autoregressively
    #  and then condition flow-matching on those positions)
    acoustic_codes, x_continuous, ode_caps = ode_solve(
        h_text[:, -max_audio_frames:, :],
        sd_acoustic,
        cfg_alpha=cfg_alpha,
        capture_intermediates=capture_intermediates,
    )

    # Semantic token prediction from acoustic transformer
    _, sem_logits, at_caps = acoustic_transformer_step(
        h_text[:, -max_audio_frames:, :],
        x_continuous,
        t=1.0,
        sd=sd_acoustic,
        capture_intermediates=capture_intermediates,
    )
    if temperature > 0:
        semantic_codes = sem_logits[:, :, :8192].argmax(dim=-1)  # [1, N]
    else:
        semantic_codes = sem_logits[:, :, :8192].argmax(dim=-1)

    # Codec decode
    waveform, codec_caps = codec_decoder_forward(
        semantic_codes,
        acoustic_codes,
        sd_codec,
        capture_intermediates=capture_intermediates,
    )

    if capture_intermediates:
        caps["text_decoder"] = text_caps
        caps["ode"] = ode_caps
        caps["acoustic_transformer"] = at_caps
        caps["codec"] = codec_caps

    return waveform, semantic_codes, acoustic_codes, caps
