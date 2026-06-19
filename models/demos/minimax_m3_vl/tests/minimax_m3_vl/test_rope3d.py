# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Test the 3D vision RoPE cos/sin precompute against the HF golden.

The rotary cos/sin are a host-side, position-only precompute (no device
involvement), so this test runs on CPU. It reproduces HF's
`MiniMaxM3VL3DRotaryEmbedding` from `image_grid_thw` and checks it matches
the golden `rope.cos`/`rope.sin` (captured from the HF rotary module).

It also self-checks `apply_rope_ref` against an inline HF-style application.

Run:
    pytest models/demos/minimax_m3_vl/tests/minimax_m3_vl/test_rope3d.py -q
"""
from __future__ import annotations

import pytest
import torch
from loguru import logger

from models.demos.minimax_m3_vl.tt.rope import apply_rope_ref, rope_cos_sin


@torch.no_grad()
@pytest.mark.parametrize("grid_tag", ["224x224", "448x448"])
def test_m3vl_rope_cos_sin(model_args, goldens, grid_tag):
    """Host cos/sin precompute matches the HF golden (near bit-exact, fp32 trig)."""
    g = goldens(grid_tag)
    grid = g["image_grid_thw"].to(torch.int64)  # (num_images, 3)
    cos_ref = g["rope.cos"].float()  # (L, 78)
    sin_ref = g["rope.sin"].float()

    cos, sin = rope_cos_sin(
        grid,
        head_dim=model_args.head_dim,
        theta=model_args.rope_theta,
        spatial_merge_size=model_args.spatial_merge_size,
    )
    assert cos.shape == cos_ref.shape, f"cos shape {tuple(cos.shape)} != {tuple(cos_ref.shape)}"

    cos_err = (cos - cos_ref).abs().max().item()
    sin_err = (sin - sin_ref).abs().max().item()
    logger.info(
        f"[rope {grid_tag}] L={cos.shape[0]} rot_dim={cos.shape[1]} max|dcos|={cos_err:.2e} max|dsin|={sin_err:.2e}"
    )
    assert cos_err < 1e-4 and sin_err < 1e-4, f"rope cos/sin mismatch: dcos={cos_err}, dsin={sin_err}"


@torch.no_grad()
def test_apply_rope_ref_matches_hf():
    """apply_rope_ref reproduces the HF apply_rotary_pos_emb_vision math."""
    torch.manual_seed(0)
    L, H, hd, rot = 12, 4, 80, 78
    q = torch.randn(1, L, H, hd)
    k = torch.randn(1, L, H, hd)
    cos = torch.randn(L, rot)
    sin = torch.randn(L, rot)

    def hf_rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    c, s = cos[None, :, None, :], sin[None, :, None, :]
    qr, qp = q[..., :rot], q[..., rot:]
    kr, kp = k[..., :rot], k[..., rot:]
    q_hf = torch.cat([qr * c + hf_rotate_half(qr) * s, qp], dim=-1)
    k_hf = torch.cat([kr * c + hf_rotate_half(kr) * s, kp], dim=-1)

    q_ours, k_ours = apply_rope_ref(q, k, cos, sin)
    assert torch.allclose(q_ours, q_hf, atol=1e-6), (q_ours - q_hf).abs().max()
    assert torch.allclose(k_ours, k_hf, atol=1e-6), (k_ours - k_hf).abs().max()
