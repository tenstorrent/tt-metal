# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch reference for Hunyuan 2D Rotary Position Embeddings.
# Extracted verbatim from:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#     _to_tuple          lines  345-351
#     get_meshgrid_nd    lines  354-400
#     build_2d_rope      lines  403-484
#     build_batch_2d_rope lines 487-516
#     rotate_half        lines  519-523
#     apply_rotary_pos_emb lines 526-555
#
# Used as the golden reference for TT-Metal numeric validation.

from typing import List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2, device="cpu"):
    """
    Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1,
            start is start, args[0] is stop, step is 1; If len(args) == 2, start is
            start, args[0] is stop, args[1] is num. For n-dim, start/stop/num should
            be int or n-tuple.
    Returns:
        grid: [dim, H, W]
    """
    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
        num_int = [int(x) for x in num]
        assert (torch.tensor(num) == torch.tensor(num_int)).all()
        num = num_int
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim)  # Left-Top    e.g. 12, 0
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom e.g. 20, 32
        num = _to_tuple(args[1], dim=dim)  # Target Size  e.g. 32, 124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32, device=device)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [H, W]
    grid = torch.stack(grid, dim=0)  # [dim, H, W]
    return grid


# ---------------------------------------------------------------------------
# Core 2D RoPE builders
# ---------------------------------------------------------------------------


def build_2d_rope(
    seq_len: int,
    n_elem: int,
    image_infos: Optional[List[Tuple[slice, Tuple[int, int]]]] = None,
    device: Optional[torch.device] = None,
    base: int = 10000,
    base_rescale_factor: float = 1.0,
    return_all_pos: bool = False,
):
    """
    Build 2-D Rotary Position Embeddings for a single sample.

    Reference: https://kexue.fm/archives/10352

    Text tokens receive sequential 1-D position IDs on both x and y axes.
    Image tokens receive 2-D grid coordinates centered so that:
        beta_y = L + (w*h - h) / 2
        beta_x = L + (w*h - w) / 2
    where L is the token offset of the image in the sequence.

    Args:
        seq_len:   Total sequence length (text + image tokens).
        n_elem:    Head dimension (must be divisible by 4).
        image_infos: List of (slice, (h, w)) describing each image segment.
                   slice.start = L (start token index of image in sequence).
                   h, w = image height/width in tokens (after VAE patch).
        device:    Torch device for coordinate tensors.
        base:      RoPE theta base frequency.
        base_rescale_factor: Optional frequency rescaling.
        return_all_pos: If True, also return the raw (y, x) position tensor.

    Returns:
        cos: [seq_len, n_elem]
        sin: [seq_len, n_elem]
        (all_pos: [seq_len, 1, 2] — only when return_all_pos=True)
    """
    assert n_elem % 4 == 0, f"n_elem must be divisible by 4, but got {n_elem}."

    # Build per-dimension theta frequencies
    if base_rescale_factor != 1.0:
        base *= base_rescale_factor ** (n_elem / (n_elem - 2))
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    theta = theta.reshape(1, n_elem // 4, 2)  # [1, half_d/2, 2]

    if image_infos is None:
        image_infos = []

    image_infos_list = [image_infos]
    sample_seq_lens = [seq_len]

    x_sections: List[torch.Tensor] = []
    y_sections: List[torch.Tensor] = []

    for sample_id, sample_image_infos in enumerate(image_infos_list):
        last_pos = 0
        for sec_slice, (h, w) in sample_image_infos:
            L = sec_slice.start
            # Text tokens before this image
            if last_pos < L:
                y_sections.append(torch.arange(last_pos, L, device=device))
                x_sections.append(torch.arange(last_pos, L, device=device))
            elif h is None:
                # Interleaved special tokens: use sequential positions
                y_sections.append(torch.arange(sec_slice.start, sec_slice.stop, device=device))
                x_sections.append(torch.arange(sec_slice.start, sec_slice.stop, device=device))
                continue
            # Image tokens: 2-D grid coordinates
            beta_y = L + (w * h - h) / 2
            beta_x = L + (w * h - w) / 2
            grid = get_meshgrid_nd((beta_y, beta_x), (beta_y + h, beta_x + w), device=device)  # [2, h, w]
            grid = grid.reshape(2, -1)  # [2, h*w]  (row=y, col=x)
            y_sections.append(grid[0])
            x_sections.append(grid[1])
            last_pos = L + w * h
        # Remaining text tokens
        y_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id], device=device))
        x_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id], device=device))

    x_pos = torch.cat(x_sections).long()
    y_pos = torch.cat(y_sections).long()
    # Trim to seq_len (handles overlap from interleaved data)
    x_pos = x_pos[:seq_len]
    y_pos = y_pos[:seq_len]
    all_pos = torch.stack((y_pos, x_pos), dim=1).unsqueeze(1).to(device)  # [seq_len, 1, 2]

    # Compute cos/sin: idx_theta shape [seq_len, n_elem]
    idx_theta = (all_pos * theta).reshape(all_pos.shape[0], n_elem // 2).repeat(1, 2)
    cos = torch.cos(idx_theta)
    sin = torch.sin(idx_theta)

    if return_all_pos:
        return cos, sin, all_pos
    return cos, sin


def build_batch_2d_rope(
    seq_len: int,
    n_elem: int,
    image_infos: Optional[List[List[Tuple[slice, Tuple[int, int]]]]] = None,
    device: Optional[torch.device] = None,
    base: int = 10000,
    base_rescale_factor: float = 1.0,
    return_all_pos: bool = False,
):
    """
    Build batched 2-D RoPE tensors — one call per batch item, then stack.

    Args:
        seq_len:     Sequence length (same for all items in batch).
        n_elem:      Head dimension.
        image_infos: List[List[Tuple[slice, (h, w)]]] — outer list = batch.
                     Pass None to treat the whole sequence as text.
        device:      Torch device.

    Returns:
        cos: [batch, seq_len, n_elem]
        sin: [batch, seq_len, n_elem]
    """
    cos_list, sin_list, all_pos_list = [], [], []
    if image_infos is None:
        image_infos = [None]
    for image_info in image_infos:
        res = build_2d_rope(
            seq_len,
            n_elem,
            image_infos=image_info,
            device=device,
            base=base,
            base_rescale_factor=base_rescale_factor,
            return_all_pos=return_all_pos,
        )
        if return_all_pos:
            cos, sin, all_pos = res
        else:
            cos, sin = res
            all_pos = None
        cos_list.append(cos)
        sin_list.append(sin)
        all_pos_list.append(all_pos)

    stacked_cos = torch.stack(cos_list, dim=0)  # [batch, seq_len, n_elem]
    stacked_sin = torch.stack(sin_list, dim=0)

    if return_all_pos:
        return stacked_cos, stacked_sin, all_pos_list
    return stacked_cos, stacked_sin


# ---------------------------------------------------------------------------
# RoPE application
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids=None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding to Q and K.

    Args:
        q:   [batch, num_heads, seq_len, head_dim]
        k:   [batch, num_kv_heads, seq_len, head_dim]
        cos: [batch, seq_len, head_dim]  (from build_batch_2d_rope)
        sin: [batch, seq_len, head_dim]
        position_ids: optional index tensor to select rows from cos/sin
        unsqueeze_dim: dim along which to unsqueeze cos/sin for broadcasting.
                       Use 1 when q/k are [B, heads, S, D].

    Returns:
        q_embed, k_embed — same shapes as inputs
    """
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]

    cos = cos.unsqueeze(unsqueeze_dim)  # [B, 1, S, D]
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Quick numeric smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, S, H, num_heads, head_dim = 1, 256, 4096, 32, 128
    n_elem = head_dim  # RoPE dim = head_dim

    # Text-only (no image regions)
    cos, sin = build_batch_2d_rope(seq_len=S, n_elem=n_elem, image_infos=None)
    print(f"cos shape: {cos.shape}  sin shape: {sin.shape}")  # [1, 256, 128]

    q = torch.randn(B, num_heads, S, head_dim)
    k = torch.randn(B, num_heads, S, head_dim)
    q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"q_rot shape: {q_r.shape}  k_rot shape: {k_r.shape}")

    # With an image region: tokens 10..10+16*16 are a 16×16 image patch
    h, w = 16, 16
    img_slice = slice(10, 10 + h * w)
    image_infos = [[(img_slice, (h, w))]]
    cos2, sin2 = build_batch_2d_rope(seq_len=S, n_elem=n_elem, image_infos=image_infos)
    print(f"cos2 (with image) shape: {cos2.shape}")
    print("smoke-test passed")
