# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Pure-PyTorch reference (functional) implementations for the MiniMax-M3 model.

These mirror the HuggingFace `transformers` (>=5.12.0) MiniMaxM3VL modeling
submodules and are used to generate golden tensors for the TTNN bring-up.
No ttnn imports here.
"""

import torch


def rms_norm_forward(hidden: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Gemma-style RMSNorm matching MiniMaxM3VLRMSNorm.

    Normalizes in fp32 and scales by ``(1.0 + weight)`` (the Gemma ``+1`` on the
    gamma), with the multiply by weight performed before casting back to the
    input dtype, i.e. ``(x_norm * (1 + w)).to(input_dtype)``.

    Args:
        hidden: Input tensor of shape [..., hidden_size].
        weight: Gamma weight of shape [hidden_size].
        eps: Variance epsilon.

    Returns:
        Tensor with the same dtype/shape as ``hidden``.
    """
    x = hidden.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x = x * (1.0 + weight.float())
    return x.type_as(hidden)


def embedding_forward(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Text token embedding matching MiniMaxM3VLTextModel.embed_tokens.

    A plain ``nn.Embedding`` row lookup. MiniMax-M3 applies NO scaling
    (no sqrt(hidden_size) multiplier): the HF forward sets
    ``inputs_embeds = self.embed_tokens(input_ids)`` and feeds it directly
    into the decoder stack. ``tie_word_embeddings`` is False.

    The embedding ``padding_idx`` only affects gradient/initialization, not
    the forward output values, so it is irrelevant for inference golden gen.

    Args:
        input_ids: Long tensor of token ids, shape [..., seq_len].
        weight: Embedding matrix of shape [vocab_size, hidden_size].

    Returns:
        Tensor of shape [..., seq_len, hidden_size] with the weight dtype.
    """
    return torch.nn.functional.embedding(input_ids.long(), weight)


def vision_layernorm_forward(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """CLIP-style vision LayerNorm matching the vision tower's ``nn.LayerNorm``.

    True LayerNorm over the last dim with mean subtraction AND a bias term
    (unlike the text RMSNorm): ``(x - mean) / sqrt(var + eps) * weight + bias``.
    Computed in fp32 for numerical stability, then cast back to the input dtype.

    Args:
        x: Input tensor of shape [..., hidden_size].
        weight: Gamma scale of shape [hidden_size].
        bias: Beta shift of shape [hidden_size].
        eps: Variance epsilon (vision_config.layer_norm_eps, default 1e-5).

    Returns:
        Tensor with the same dtype/shape as ``x``.
    """
    orig_dtype = x.dtype
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = xf.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (xf - mean) / torch.sqrt(var + eps)
    out = x_norm * weight.float() + bias.float()
    return out.type_as(x).to(orig_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Half-split rotation matching HF MiniMaxM3VL ``rotate_half``.

    This is the GPT-NeoX / Llama style ``rotate_half`` (NOT the interleaved
    GPT-J style): the last dim is split into two contiguous halves and rotated
    as ``cat(-x2, x1)``::

        x1 = x[..., : d/2]
        x2 = x[..., d/2 :]
        return cat(-x2, x1)
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cos_sin(
    seq_len: int,
    rotary_dim: int = 64,
    theta: float = 5e6,
    position_ids: torch.Tensor = None,
    dtype: torch.dtype = torch.float32,
    device=None,
):
    """Build the partial-RoPE cos/sin tables matching MiniMaxM3VLRotaryEmbedding.

    Mirrors the HF ``compute_default_rope_parameters`` + ``forward`` for
    ``rope_type == "default"`` (``attention_scaling == 1.0``):

      inv_freq = 1 / theta^(arange(0, rotary_dim, 2) / rotary_dim)   # len rotary_dim/2
      freqs    = position_ids[:, :, None] * inv_freq[None, None, :]  # [B, S, rotary_dim/2]
      emb      = cat(freqs, freqs, dim=-1)                           # [B, S, rotary_dim]
      cos, sin = emb.cos(), emb.sin()

    Note ``rotary_dim`` is the PARTIAL rotary dim (head_dim * partial_rotary_factor
    = 128 * 0.5 = 64), so the produced cos/sin have last dim == 64, and only the
    first 64 dims of each head get rotated downstream (see ``rope_forward``).

    Args:
        seq_len: Sequence length (used to build default position_ids 0..seq_len-1).
        rotary_dim: The partial rotary dimension (64 for MiniMax-M3 text).
        theta: RoPE base / theta (5e6 for MiniMax-M3 text).
        position_ids: Optional explicit positions, shape [B, S]; defaults to
            a single batch arange(seq_len).
        dtype: Output dtype for cos/sin.
        device: Output device.

    Returns:
        ``(cos, sin)`` each of shape [B, seq_len, rotary_dim].
    """
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_ids = position_ids.to(device=device)

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / rotary_dim)
    )
    # Force fp32 accumulation as HF does.
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [B, S, rotary_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, rotary_dim]
    cos = emb.cos()
    sin = emb.sin()
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rope_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
):
    """Apply PARTIAL rotary position embedding matching HF ``apply_rotary_pos_emb``.

    PARTIAL rope: ``rotary_dim = cos.shape[-1]`` (== 64 for MiniMax-M3 text).
    The first ``rotary_dim`` channels of each head are rotated with the
    half-split ``rotate_half`` convention; the remaining ``head_dim - rotary_dim``
    (== 64) channels PASS THROUGH unchanged, then both parts are concatenated
    back to full ``head_dim``::

        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        q_out   = cat(q_embed, q_pass)

    ``cos``/``sin`` are unsqueezed at ``unsqueeze_dim`` (default 1) so a
    [B, S, rotary_dim] table broadcasts against [B, H, S, head_dim] q/k.

    Args:
        q: Query tensor [B, num_heads, S, head_dim].
        k: Key tensor   [B, num_kv_heads, S, head_dim].
        cos: Cosine table [B, S, rotary_dim].
        sin: Sine table   [B, S, rotary_dim].
        unsqueeze_dim: Dim to unsqueeze cos/sin on (1 for [B,H,S,D] layout).

    Returns:
        ``(q_embed, k_embed)`` rotated, same shapes/dtypes as inputs.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed
