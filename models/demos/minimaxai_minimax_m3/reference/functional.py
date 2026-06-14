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


def build_vision_rope_3d(grid_thw, head_dim: int = 80, theta: float = 1e4, spatial_merge_size: int = 2):
    """Build the CLIP vision 3D-RoPE cos/sin tables matching MiniMaxM3VL3DRotaryEmbedding.

    3D RoPE: every visual token has a ``(t, h, w)`` grid position and the per-head
    rotary dims are split *evenly across the three axes*. With ``rope_dims =
    2 * (head_dim // 2)`` total rotary dims, each axis gets::

        axis_dim = 2 * ((rope_dims // 3) // 2)   # rounded down to a multiple of 2

    so each axis owns ``axis_dim`` channels (``axis_dim // 2`` frequencies). For
    ``head_dim = 80``: ``rope_dims = 80``, ``axis_dim = 26`` (13 freqs/axis),
    ``rot_dim = 3 * axis_dim = 78`` rotated channels, and the final
    ``head_dim - 3 * axis_dim = 2`` channels are NEVER rotated (pass through).

    The single ``inv_freq`` table (length ``axis_dim // 2``) is SHARED across axes;
    each axis multiplies it by *its own* coordinate. The three resulting frequency
    bands are concatenated in ``T | H | W`` order, then duplicated as
    ``cat([freqs, freqs])`` (GPT-NeoX half-rotation pairing). So the cos/sin layout is::

        [ T(13) | H(13) | W(13) | T(13) | H(13) | W(13) ]   -> rot_dim = 78

    Coordinate generation handles ``spatial_merge_size`` (m): within each frame the
    H/W indices are reordered into ``m x m`` spatial-merge blocks via
    ``reshape(h//m, m, w//m, m).permute(0,2,1,3).flatten()`` so that the 2x2 merged
    patches are contiguous; the temporal index ``t`` is ``repeat_interleave(h*w)``.

    Args:
        grid_thw: Long tensor of shape [num_images, 3] with rows ``(t, h, w)``.
        head_dim: Per-head dimension (80 for the M3 vision tower).
        theta: RoPE base (10000.0).
        spatial_merge_size: Spatial merge factor m (2 for M3).

    Returns:
        ``(cos, sin)`` each of shape [total_tokens, rot_dim] where
        ``rot_dim = 3 * axis_dim``, dtype float32.
    """
    import torch as _torch

    if not isinstance(grid_thw, _torch.Tensor):
        grid_thw = _torch.tensor(grid_thw, dtype=_torch.long)

    rope_dims = 2 * (head_dim // 2)
    axis_dim = 2 * ((rope_dims // 3) // 2)
    m = spatial_merge_size

    coords = []
    for t, h, w in grid_thw.tolist():
        hi = _torch.arange(h).unsqueeze(1).expand(-1, w)
        hi = hi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        wi = _torch.arange(w).unsqueeze(0).expand(h, -1)
        wi = wi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        ti = _torch.arange(t).repeat_interleave(h * w)
        coords.append(_torch.stack([ti, hi.repeat(t), wi.repeat(t)], dim=-1))
    coords = _torch.cat(coords).to(dtype=_torch.float32)

    inv_freq = 1.0 / (theta ** (_torch.arange(0, axis_dim, 2, dtype=_torch.float32) / axis_dim))
    freqs = _torch.cat([coords[:, i : i + 1] * inv_freq for i in range(3)], dim=-1)
    emb = _torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def vision_rope_3d_forward(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply CLIP vision 3D RoPE matching HF ``apply_rotary_pos_emb_vision``.

    Only the first ``rot_dim = cos.shape[-1]`` head channels carry the 3D rotation
    (using the half-split ``rotate_half`` convention); the tail channels pass
    through untouched, then both parts are concatenated back to full ``head_dim``::

        q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
        q_rot = q_rot * cos + rotate_half(q_rot) * sin
        q_out = cat(q_rot, q_pass)

    cos/sin are [total_tokens, rot_dim]; they are unsqueezed to broadcast against q/k.
    This implementation accepts q/k in ``[B, num_heads, S, head_dim]`` layout (the
    bring-up golden layout) and broadcasts cos/sin as ``[1, 1, S, rot_dim]``. The HF
    reference uses ``[B, S, num_heads, head_dim]`` with cos/sin ``[1, S, 1, rot_dim]``;
    the per-token math is identical.

    Args:
        q: Query tensor [B, num_heads, S, head_dim].
        k: Key tensor   [B, num_heads, S, head_dim].
        cos: Cosine table [S, rot_dim].
        sin: Sine table   [S, rot_dim].

    Returns:
        ``(q_embed, k_embed)`` rotated, same shapes/dtypes as inputs.
    """
    rot_dim = cos.shape[-1]
    # [S, rot_dim] -> [1, 1, S, rot_dim] to broadcast against [B, H, S, D]
    cos_b = cos[None, None, :, :].to(q.dtype)
    sin_b = sin[None, None, :, :].to(q.dtype)

    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]

    q_rot = q_rot * cos_b + _rotate_half(q_rot) * sin_b
    k_rot = k_rot * cos_b + _rotate_half(k_rot) * sin_b

    q_embed = torch.cat([q_rot, q_pass], dim=-1)
    k_embed = torch.cat([k_rot, k_pass], dim=-1)
    return q_embed, k_embed


def qk_norm_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
):
    """Per-head QK-norm matching MiniMaxM3VLAttention's q_norm/k_norm.

    M3 attention uses ``use_qk_norm=True`` with ``qk_norm_type="per_head"``: it
    applies a Gemma-style RMSNorm (``MiniMaxM3VLRMSNorm``) over the per-head
    ``head_dim`` (128) of the query and key states. In HF the norm is applied to
    the ``[B, S, num_heads, head_dim]`` view BEFORE ``transpose(1, 2)`` and
    BEFORE RoPE:

        query_states = self.q_norm(self.q_proj(h).view(*shape, -1, head_dim)).transpose(1, 2)
        key_states   = self.k_norm(self.k_proj(h).view(*shape, -1, head_dim)).transpose(1, 2)
        ...
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    Because RMSNorm only reduces over the last dim (``head_dim``), it is
    equivalent whether q/k are laid out as ``[B, S, H, D]`` or
    ``[B, H, S, D]``. This implementation accepts the post-transpose layout
    used elsewhere in the bring-up:

        q: [..., num_attention_heads (64), S, head_dim (128)]
        k: [..., num_key_value_heads (4), S, head_dim (128)]

    The weight is shaped ``[head_dim]`` (==128) and is broadcast identically
    across every head. The Gemma ``+1`` on gamma applies (delegated to
    :func:`rms_norm_forward`). This is APPLIED PRE-ROPE.

    Args:
        q: Query tensor, head_dim as last axis.
        k: Key tensor, head_dim as last axis.
        q_weight: q_norm gamma, shape [head_dim].
        k_weight: k_norm gamma, shape [head_dim].
        eps: RMSNorm variance epsilon (config.rms_norm_eps, default 1e-6).

    Returns:
        ``(q_normed, k_normed)`` with the same shapes/dtypes as the inputs.
    """
    q_normed = rms_norm_forward(q, q_weight, eps=eps)
    k_normed = rms_norm_forward(k, k_weight, eps=eps)
    return q_normed, k_normed


def patch_embedding_forward(
    pixel_values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    in_channels: int = 3,
    temporal_patch_size: int = 2,
    patch_size: int = 14,
) -> torch.Tensor:
    """CLIP/Qwen2.5-VL-style patch embedding matching MiniMaxM3VLVisionEmbeddings.

    The vision patchify is a **Conv3d** (NOT a Conv2d), identical to
    ``Qwen2_5_VisionPatchEmbed``. The checkpoint stores the weight under
    ``vision_tower.vision_model.embeddings.patch_embedding.weight`` (renamed to
    the inherited ``proj`` at load time), with shape
    ``[embed_dim, in_channels, temporal_patch_size, patch_size, patch_size]``
    (e.g. ``[1280, 3, 2, 14, 14]``). There is **NO bias** (Conv3d ``bias=False``).

    HF forward::

        hidden_states = pixel_values.view(
            -1, in_channels, temporal_patch_size, patch_size, patch_size
        )
        hidden_states = self.proj(hidden_states).view(-1, embed_dim)

    Because the Conv3d kernel size equals its stride and exactly covers each
    flattened patch, this is mathematically a linear projection of each
    ``[C * tps * p * p]``-flattened patch onto ``embed_dim``. Position
    embeddings / 3D-RoPE are applied SEPARATELY downstream and are intentionally
    kept OUT of this block.

    Args:
        pixel_values: Pre-patchified pixel tensor. Either already shaped
            ``[num_patches, C * tps * p * p]`` (Qwen-VL processor output) or any
            tensor whose total numel is a multiple of ``C*tps*p*p``; it is
            ``view``-reshaped to ``[-1, C, tps, p, p]``.
        weight: Conv3d weight, shape
            ``[embed_dim, in_channels, temporal_patch_size, patch_size, patch_size]``.
        bias: Optional bias of shape ``[embed_dim]`` (None for MiniMax-M3).
        in_channels: Number of input channels (3).
        temporal_patch_size: Temporal patch size (2).
        patch_size: Spatial patch size (14).

    Returns:
        Tensor of shape ``[num_patches, embed_dim]`` with the weight dtype.
    """
    target_dtype = weight.dtype
    embed_dim = weight.shape[0]
    hidden_states = pixel_values.view(-1, in_channels, temporal_patch_size, patch_size, patch_size)
    hidden_states = torch.nn.functional.conv3d(
        hidden_states.to(dtype=target_dtype),
        weight,
        bias=bias,
        stride=(temporal_patch_size, patch_size, patch_size),
    )
    return hidden_states.view(-1, embed_dim)


def _repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA key/value head expansion matching HF ``repeat_kv``.

    Expands [B, num_kv_heads, S, head_dim] to [B, num_kv_heads*n_rep, S, head_dim]
    via interleaved expand+reshape (NOT a tile), so kv head ``i`` is repeated into
    output heads ``i*n_rep : (i+1)*n_rep``.
    """
    b, n_kv, s, d = hidden.shape
    if n_rep == 1:
        return hidden
    hidden = hidden[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return hidden.reshape(b, n_kv * n_rep, s, d)


def gqa_attention_forward(
    x: torch.Tensor,
    weights_dict: dict,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_attention_heads: int = 64,
    num_key_value_heads: int = 4,
    head_dim: int = 128,
    eps: float = 1e-6,
    attention_mask: torch.Tensor = None,
):
    """Dense-layer full GQA attention matching MiniMaxM3VLAttention.forward (layers 0-2).

    Exact HF pipeline order (full_attention layers, indexer is None):

      1. q = q_proj(x).view(B, S, num_heads, head_dim)     # [hidden -> 64*128=8192]
         k = k_proj(x).view(B, S, num_kv_heads, head_dim)  # [hidden -> 4*128=512]
         v = v_proj(x).view(B, S, num_kv_heads, head_dim)  # [hidden -> 4*128=512]
      2. PER-HEAD Gemma QK-norm on the LAST dim (head_dim) BEFORE transpose:
            q = q_norm(q); k = k_norm(k)   # rms_norm_forward over head_dim, (1+w) scale
         (v is NOT normed)
      3. transpose(1,2) -> [B, H, S, head_dim]
      4. PARTIAL rope (rotary_dim 64) on q,k via rope_forward (qk_norm BEFORE rope)
      5. repeat_kv(k/v, n_rep=16) : 4 kv heads -> 64
      6. scaled_dot_product, scale = head_dim**-0.5 = 1/sqrt(128), causal
      7. attn.transpose(1,2).reshape(B, S, 8192) -> o_proj -> [hidden]

    Args:
        x: hidden states [B, S, hidden_size].
        weights_dict: dict with keys q_proj,k_proj,v_proj,o_proj,q_norm,k_norm
            (each the ``.weight`` tensor; proj are nn.Linear weights [out,in]).
        cos, sin: partial-rope tables [B, S, rotary_dim] (rotary_dim 64), from
            build_rope_cos_sin(theta=5e6, rotary_dim=64).
        num_attention_heads, num_key_value_heads, head_dim, eps: config.
        attention_mask: optional additive mask [B,1,S,S]; if None a causal mask
            is built internally (matching is_causal=True / sdpa causal).

    Returns:
        Attention block output [B, S, hidden_size] (post o_proj), x's dtype.
    """
    B, S, _ = x.shape
    n_rep = num_attention_heads // num_key_value_heads

    q = torch.nn.functional.linear(x, weights_dict["q_proj"])
    k = torch.nn.functional.linear(x, weights_dict["k_proj"])
    v = torch.nn.functional.linear(x, weights_dict["v_proj"])

    q = q.view(B, S, num_attention_heads, head_dim)
    k = k.view(B, S, num_key_value_heads, head_dim)
    v = v.view(B, S, num_key_value_heads, head_dim)

    # Per-head Gemma RMSNorm over head_dim, applied BEFORE transpose & rope.
    q = rms_norm_forward(q, weights_dict["q_norm"], eps=eps)
    k = rms_norm_forward(k, weights_dict["k_norm"], eps=eps)

    q = q.transpose(1, 2)  # [B, H, S, head_dim]
    k = k.transpose(1, 2)  # [B, n_kv, S, head_dim]
    v = v.transpose(1, 2)  # [B, n_kv, S, head_dim]

    # Partial rope (rotary_dim = cos.shape[-1] = 64), qk_norm done first above.
    q, k = rope_forward(q, k, cos, sin, unsqueeze_dim=1)

    # GQA: expand kv heads 4 -> 64.
    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)

    scaling = head_dim**-0.5
    attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling

    if attention_mask is None:
        causal = torch.full((S, S), float("-inf"), dtype=attn_weights.dtype, device=x.device)
        causal = torch.triu(causal, diagonal=1)
        attn_weights = attn_weights + causal[None, None, :, :]
    else:
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)  # [B, H, S, head_dim]

    attn_output = attn_output.transpose(1, 2).reshape(B, S, -1).contiguous()  # [B, S, 8192]
    out = torch.nn.functional.linear(attn_output, weights_dict["o_proj"])  # [B, S, hidden]
    return out
