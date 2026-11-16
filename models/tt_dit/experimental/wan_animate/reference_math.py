# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Torch-only math for Wan2.2-Animate on TT-DiT.

Nothing here imports ttnn. These functions are shared by:

* the TT modules (weight preparation in ``_prepare_torch_state``),
* the pipeline (host-side conditioning: pose channels, attention masks, segment bookkeeping),
* the CPU tests (``experimental/tests/test_wan_animate_cpu.py``), which check every
  reformulation below against the Diffusers reference ``WanAnimateTransformer3DModel``.

Each reformulation exists to turn an Animate-specific op into something the existing
TT-DiT building blocks already do well (one patch-embed matmul, plain SDPA with an
additive mask, plain matmuls), without changing the math.
"""

from __future__ import annotations

import re

import torch

# The Diffusers checkpoint concatenates the CLIP image context in front of the text context
# and splits it off again by `encoder_hidden_states.shape[1] - 512`. 257 = 1 CLS + 16x16 patches
# of CLIP ViT-H/14 at 224px.
IMAGE_CONTEXT_TOKENS = 257
# Number of tokens per face-motion frame after the face encoder: 4 heads + 1 learned padding token.
FACE_TOKENS_PER_FRAME = 5
# Key-length granule for the masked cross-attentions. The existing Wan SDPA program config
# uses k_chunk_size=256, so keys padded to a multiple of 256 reuse it unchanged.
KEY_PAD_MULTIPLE = 256

NEG_INF = float("-inf")


def round_up(x: int, m: int) -> int:
    return -(-x // m) * m


# ---------------------------------------------------------------------------------------------
# 1. Pose latent patch embedding, fused into the main patch embedding.
# ---------------------------------------------------------------------------------------------
#
# Reference (Diffusers / official):
#     h = patch_embedding(cat[noise(16), y(20)])                # all T+1 latent frames
#     h[:, :, 1:] += pose_patch_embedding(pose_latents(16))     # frames 1..T only
#
# Both are Conv3d with kernel == stride == patch_size, i.e. a per-patch linear map. Appending the
# pose latents (zero on frame 0) and one indicator channel (0 on frame 0, 1 on frames 1..T) to the
# input turns the sum into ONE linear map: the pose bias rides on the indicator channel. The TT
# WanPatchEmbed then runs a single matmul with K = 4 * (36 + 16 + 1) = 212 instead of 144.

POSE_EXTRA_CHANNELS = 17  # 16 pose latent channels + 1 indicator


def fuse_pose_patch_embedding(
    patch_weight: torch.Tensor,
    patch_bias: torch.Tensor,
    pose_weight: torch.Tensor,
    pose_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate Conv3d weights [dim, C, pt, ph, pw] along C: [patch | pose | indicator]."""
    dim, _, pt, ph, pw = patch_weight.shape
    assert pose_weight.shape[0] == dim and pose_weight.shape[2:] == patch_weight.shape[2:]
    indicator = torch.zeros(dim, 1, pt, ph, pw, dtype=patch_weight.dtype)
    indicator[:, 0, 0, 0, 0] = pose_bias.to(patch_weight.dtype)
    return torch.cat([patch_weight, pose_weight.to(patch_weight.dtype), indicator], dim=1), patch_bias


def pose_conditioning(pose_latents: torch.Tensor) -> torch.Tensor:
    """[B, 16, T, H, W] pose latents -> [B, 17, T + 1, H, W] extra input channels (see above)."""
    b, c, t, h, w = pose_latents.shape
    out = torch.zeros(b, c + 1, t + 1, h, w, dtype=pose_latents.dtype)
    out[:, :c, 1:] = pose_latents
    out[:, c, 1:] = 1.0
    return out


# ---------------------------------------------------------------------------------------------
# 2. Face encoder causal Conv1d as matmuls.
# ---------------------------------------------------------------------------------------------
#
# Reference: x -> F.pad(x, (k - 1, 0), mode="replicate") -> Conv1d(k=3, stride s).
# Equivalent: y = sum_k (P_k @ X) @ W_k^T + b, with X [T_in, C_in] and P_k a 0/1 [T_out, T_in]
# matrix that does the replicate padding, the shift and the stride. Everything stays a matmul,
# which the device does in tile layout without any strided slicing. Rows/cols beyond the
# logical lengths are zero, so tile padding never leaks into real outputs.


def conv1d_out_len(t_in: int, kernel: int = 3, stride: int = 1) -> int:
    return (t_in + (kernel - 1) - kernel) // stride + 1


def causal_conv_selectors(
    t_in: int, *, kernel: int = 3, stride: int = 1, rows: int | None = None, cols: int | None = None
) -> torch.Tensor:
    """[kernel, rows, cols] selection matrices; rows >= t_out and cols >= t_in pad with zeros."""
    t_out = conv1d_out_len(t_in, kernel, stride)
    rows = rows or t_out
    cols = cols or t_in
    p = torch.zeros(kernel, rows, cols)
    for k in range(kernel):
        for t in range(t_out):
            src = t * stride + k - (kernel - 1)
            p[k, t, max(src, 0)] = 1.0
    return p


def conv1d_weight_as_linear(weight: torch.Tensor) -> torch.Tensor:
    """Conv1d weight [C_out, C_in, k] -> per-tap linear weights [k, C_out, C_in] (torch Linear layout)."""
    return weight.permute(2, 0, 1).contiguous()


def face_encoder_matmul_reference(face_encoder: torch.nn.Module, motion_vec: torch.Tensor) -> torch.Tensor:
    """Torch emulation of the device face encoder (selection-matmul form).

    motion_vec: [B, T, C_motion] -> [B, T', heads + 1, dim], identical to
    ``diffusers.WanAnimateFaceEncoder.forward``. Used by the CPU test to prove the reformulation.
    """
    fe = face_encoder
    b, t, _ = motion_vec.shape
    eps = fe.norm1.eps
    heads = fe.num_heads

    def conv(x, conv_mod, stride):
        sel = causal_conv_selectors(x.shape[-2], stride=stride)
        w = conv1d_weight_as_linear(conv_mod.weight)
        y = sum((sel[k] @ x) @ w[k].T for k in range(sel.shape[0]))
        return y + conv_mod.bias

    def ln_silu(x):
        return torch.nn.functional.silu(torch.nn.functional.layer_norm(x, x.shape[-1:], eps=eps))

    outs = []
    for bi in range(b):
        x = conv(motion_vec[bi], fe.conv1_local, 1)  # [T, heads * hidden]
        per_head = []
        for n in x.chunk(heads, dim=-1):
            h = ln_silu(n)
            h = ln_silu(conv(h, fe.conv2, 2))
            h = ln_silu(conv(h, fe.conv3, 2))
            per_head.append(fe.out_proj(h))
        outs.append(torch.stack(per_head, dim=1))  # [T', heads, dim]
    x = torch.stack(outs)
    pad = fe.padding_tokens.expand(b, x.shape[1], 1, -1)
    return torch.cat([x, pad], dim=2)


# ---------------------------------------------------------------------------------------------
# 3. Face adapter (temporally aligned cross-attention) as one masked SDPA.
# ---------------------------------------------------------------------------------------------
#
# Reference: the query sequence (F latent frames x HW tokens, frame-major) is split per latent
# frame; frame f attends only to the 5 motion tokens of frame f (frame 0, the reference, gets a
# zero motion vector). Equivalent: flatten all motion tokens to one key sequence of 5F (padded to
# 256) and give SDPA an additive mask that is block-diagonal in (frame of query, frame of key).
# The mask is per query row, so it can be sharded along the sequence-parallel axis together with
# the queries, which a per-frame batched attention could not do across frame boundaries.


def face_motion_tokens(motion_vec: torch.Tensor, *, key_len: int | None = None) -> torch.Tensor:
    """[B, F, 5, D] (zero frame already prepended) -> [1, B, K, D] with K padded to 256."""
    b, f, n, d = motion_vec.shape
    flat = motion_vec.reshape(b, f * n, d)
    key_len = key_len or round_up(f * n, KEY_PAD_MULTIPLE)
    out = torch.zeros(b, key_len, d, dtype=motion_vec.dtype)
    out[:, : f * n] = flat
    return out.unsqueeze(0)


def face_attention_mask(
    *, num_frames: int, tokens_per_frame: int, padded_seq_len: int, key_len: int | None = None
) -> torch.Tensor:
    """Additive mask [1, 1, padded_seq_len, key_len]: 0 where query frame == key frame, -inf else.

    Rows beyond the logical sequence (sequence-parallel padding) attend to frame 0's keys so that
    no row is fully masked (a fully masked softmax row is NaN).
    """
    n = FACE_TOKENS_PER_FRAME
    key_len = key_len or round_up(num_frames * n, KEY_PAD_MULTIPLE)
    q_frame = torch.arange(padded_seq_len) // tokens_per_frame
    q_frame = torch.where(q_frame < num_frames, q_frame, torch.zeros_like(q_frame))
    k_frame = torch.arange(key_len) // n
    k_frame = torch.where(torch.arange(key_len) < num_frames * n, k_frame, torch.full_like(k_frame, -1))
    allowed = q_frame[:, None] == k_frame[None, :]
    mask = torch.zeros(padded_seq_len, key_len)
    mask.masked_fill_(~allowed, NEG_INF)
    return mask[None, None]


def image_attention_mask(*, padded_seq_len: int, num_tokens: int = IMAGE_CONTEXT_TOKENS) -> torch.Tensor:
    """Additive mask [1, 1, padded_seq_len, round_up(num_tokens, 256)] hiding the key padding."""
    key_len = round_up(num_tokens, KEY_PAD_MULTIPLE)
    mask = torch.zeros(padded_seq_len, key_len)
    mask[:, num_tokens:] = NEG_INF
    return mask[None, None]


def pad_image_context(image_context: torch.Tensor) -> torch.Tensor:
    """[B, 257, D] -> [1, B, 512, D] zero padded (the padding is masked out)."""
    b, t, d = image_context.shape
    out = torch.zeros(b, round_up(t, KEY_PAD_MULTIPLE), d, dtype=image_context.dtype)
    out[:, :t] = image_context
    return out.unsqueeze(0)


def face_adapter_masked_reference(
    adapter: torch.nn.Module,
    hidden_states: torch.Tensor,
    motion_vec: torch.Tensor,
    *,
    tokens_per_frame: int,
) -> torch.Tensor:
    """Torch emulation of the device face adapter (single masked SDPA). Returns the adapter output
    (to be added to hidden_states), like ``diffusers.WanAnimateFaceBlockCrossAttention``."""
    b, s, d = hidden_states.shape
    f = motion_vec.shape[1]
    heads = adapter.heads
    x = adapter.pre_norm_q(hidden_states)
    kv = face_motion_tokens(adapter.pre_norm_kv(motion_vec))[0]
    q = adapter.norm_q(adapter.to_q(x).unflatten(-1, (heads, -1))).transpose(1, 2)
    k = adapter.norm_k(adapter.to_k(kv).unflatten(-1, (heads, -1))).transpose(1, 2)
    v = adapter.to_v(kv).unflatten(-1, (heads, -1)).transpose(1, 2)
    mask = face_attention_mask(num_frames=f, tokens_per_frame=tokens_per_frame, padded_seq_len=s, key_len=k.shape[2])
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return adapter.to_out(out.transpose(1, 2).flatten(2))


# ---------------------------------------------------------------------------------------------
# 4. Segment bookkeeping (identical to Diffusers WanAnimatePipeline / official WanAnimate).
# ---------------------------------------------------------------------------------------------


def num_segments_and_padding(num_frames: int, segment_len: int = 77, prev_frames: int = 1) -> tuple[int, int]:
    """(num_segments, num_padding_frames) for a driving video of ``num_frames`` frames."""
    eff = segment_len - prev_frames
    last = (num_frames - prev_frames) % eff
    pad = 0 if last == 0 else eff - last
    return (num_frames + pad) // eff, pad


def reflect_pad_frames(frames: list, target: int) -> list:
    """Diffusers ``pad_video_frames``: [1,2,3,4,5] -> [1,2,3,4,5,4,3,2,1,2,...]."""
    if len(frames) == 1:
        return [frames[0]] * target
    idx, flip, out = 0, False, []
    while len(out) < target:
        out.append(frames[idx])
        idx = idx - 1 if flip else idx + 1
        if idx == 0 or idx == len(frames) - 1:
            flip = not flip
    return out


def i2v_mask(latent_t: int, latent_h: int, latent_w: int, mask_len: int, mask_pixels: torch.Tensor | None = None):
    """Diffusers ``get_i2v_mask`` for batch 1: [1, 4, latent_t, latent_h, latent_w]."""
    if mask_pixels is None:
        m = torch.zeros(1, 1, (latent_t - 1) * 4 + 1, latent_h, latent_w)
    else:
        m = mask_pixels.clone()
    m[:, :, :mask_len] = 1
    m = torch.cat([torch.repeat_interleave(m[:, :, 0:1], 4, dim=2), m[:, :, 1:]], dim=2)
    return m.view(1, -1, 4, latent_h, latent_w).transpose(1, 2)


def segment_conditioning(
    *,
    encode,
    y_ref: torch.Tensor,
    pose_segment: torch.Tensor,
    prev_frames: torch.Tensor | None,
    start: int,
    mode: str,
    prev_len: int,
    background_segment: torch.Tensor | None = None,
    mask_segment: torch.Tensor | None = None,
) -> torch.Tensor:
    """The per-segment model conditioning, [1, 20 + 17, T_lat + 1, h, w].

    ``encode`` maps pixels [1, 3, T, H, W] in [-1, 1] to *normalized* latents (argmax mode), like the
    TT ``WanEncoder`` with the latent mean/std folded in. ``y_ref`` is [mask(4) | ref latent(16)] for
    the reference image. ``prev_frames`` are the previous segment's decoded frames (None on the first
    segment). Mirrors Diffusers ``prepare_pose_latents`` + ``prepare_prev_segment_cond_latents``.
    """
    _, _, length, height, width = pose_segment.shape
    latent_t = (length - 1) // 4 + 1
    h, w = height // 8, width // 8
    pose_latent = encode(pose_segment)
    if start > 0:
        prev = prev_frames[:, :, -prev_len:]
        if prev.shape[-2:] != (height, width):
            prev = torch.nn.functional.interpolate(
                prev[0].transpose(0, 1), size=(height, width), mode="bicubic"
            ).transpose(0, 1)[None]
    elif mode == "replace":
        prev = background_segment[:, :, :prev_len]
    else:
        prev = torch.zeros(1, 3, prev_len, height, width)
    if mode == "replace":
        rest = background_segment[:, :, prev_len:]
    else:
        rest = torch.zeros(1, 3, length - prev_len, height, width)
    cond_latent = encode(torch.cat([prev, rest], dim=2))
    if mode == "replace":
        keep = 1 - mask_segment  # [1, 1, T, H, W]; 1 = keep the background
        keep = torch.nn.functional.interpolate(keep[0].transpose(0, 1), size=(h, w), mode="nearest")
        keep = keep.transpose(0, 1)[None]
    else:
        keep = None
    msk = i2v_mask(latent_t, h, w, prev_len if start > 0 else 0, keep)
    y = torch.cat([y_ref, torch.cat([msk, cond_latent], dim=1)], dim=2)
    return torch.cat([y, pose_conditioning(pose_latent)], dim=1)


# ---------------------------------------------------------------------------------------------
# 5. Relighting LoRA (replace mode): official-repo PEFT keys -> Diffusers keys, merged on host.
# ---------------------------------------------------------------------------------------------
#
# Wan-AI/Wan2.2-Animate-14B/relighting_lora/adapter_model.safetensors holds 960 fp32 tensors:
# base_model.model.blocks.{i}.{self_attn,cross_attn}.{q,k,v,o[,k_img,v_img]}.lora_{A,B}.weight and
# ...ffn.{0,2}.lora_{A,B}.weight, rank 128, alpha 128 (scale 1.0). Face modules are not targeted.

_LORA_MODULE_MAP = {
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.k_img": "attn2.add_k_proj",
    "cross_attn.v_img": "attn2.add_v_proj",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
}
_LORA_KEY_RE = re.compile(r"^(?:base_model\.model\.)?blocks\.(\d+)\.(.+)\.lora_([AB])\.weight$")


def map_relighting_lora(lora_state: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """{diffusers_linear_prefix: (A [r, in], B [out, r])}. Raises on any key it does not understand."""
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in lora_state.items():
        m = _LORA_KEY_RE.match(key)
        if m is None or m.group(2) not in _LORA_MODULE_MAP:
            msg = f"unrecognized relighting LoRA key {key!r}"
            raise ValueError(msg)
        target = f"blocks.{m.group(1)}.{_LORA_MODULE_MAP[m.group(2)]}"
        pairs.setdefault(target, {})[m.group(3)] = value
    return {t: (ab["A"], ab["B"]) for t, ab in pairs.items()}


def merge_relighting_lora(state: dict[str, torch.Tensor], lora_state: dict[str, torch.Tensor], scale: float = 1.0):
    """W += scale * B @ A for each targeted Linear, in place on a Diffusers-keyed state dict."""
    merged = 0
    for target, (a, b) in map_relighting_lora(lora_state).items():
        key = f"{target}.weight"
        if key not in state:
            msg = f"relighting LoRA targets {key!r}, which is not in the transformer state dict"
            raise KeyError(msg)
        w = state[key]
        state[key] = (w.float() + scale * (b.float() @ a.float())).to(w.dtype)
        merged += 1
    return merged
