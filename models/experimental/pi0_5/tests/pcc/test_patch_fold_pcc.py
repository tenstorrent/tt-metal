# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone PCC test: ttnn.fold + linear vs torch reference for SigLIP patch embed.

Mirrors the SmolVLA pattern (models/experimental/smolvla/tt/ttnn_optimized_vit_smolvla.py)
for our specific shape (B=3, C=3, H=224, W=224, patch=14):
  1. Pre-permute on host BCHW → BHWC (cheap)
  2. Pad C 3 → 4 (power-of-2 align)
  3. Pre-reshape (B, H, W, C') → (B, H, W/patch, C'*patch) — metadata only
  4. ttnn.fold(stride_h=patch, stride_w=1) → (B, num_patches, kH*kW*C')
  5. to_layout TILE
  6. linear → (B, num_patches, hidden)

If PCC passes, this is a strong candidate to replace the current
permute+reshape+permute+reshape+linear chain (~0.92 ms / inference).
"""

import os
from pathlib import Path

import pytest
import torch
import ttnn

from safetensors import safe_open
from models.common.utility_functions import comp_pcc

_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_libero_upstream"
CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", str(_DEFAULT_CHECKPOINT_DIR)))

PCC_THRESHOLD = 0.999


pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _load_patch_weights():
    sf = safe_open(str(CHECKPOINT_DIR / "model.safetensors"), framework="pt")
    keys = list(sf.keys())
    w_key = next(k for k in keys if "vision_tower.vision_model.embeddings.patch_embedding.weight" in k)
    b_key = next(k for k in keys if "vision_tower.vision_model.embeddings.patch_embedding.bias" in k)
    return sf.get_tensor(w_key).to(torch.float32), sf.get_tensor(b_key).to(torch.float32)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_patch_fold_pcc(device):
    """ttnn.fold + linear matches torch.nn.functional.conv2d output (within bf16 tolerance)."""
    torch.manual_seed(0)

    w_torch, b_torch = _load_patch_weights()  # w: (1152, 3, 14, 14), b: (1152,)
    out_channels, in_channels, kh, kw = w_torch.shape
    assert (in_channels, kh, kw) == (3, 14, 14)
    patch_size = kh  # 14

    B, C, H, W = 3, 3, 224, 224
    x_torch = torch.randn(B, C, H, W, dtype=torch.float32) * 0.5

    # === Torch reference ===
    out_torch_ref = torch.nn.functional.conv2d(x_torch, w_torch, bias=b_torch, stride=(kh, kw))
    out_torch_ref = out_torch_ref.flatten(2).transpose(1, 2)  # (B, 256, 1152)

    # === ttnn fold-based path ===
    # 1. Pre-permute BCHW → BHWC + pad C 3 → 4 on host
    C_padded = 4
    x_nhwc = x_torch.permute(0, 2, 3, 1).contiguous()  # (B, H, W, 3)
    x_nhwc = torch.nn.functional.pad(x_nhwc, (0, C_padded - C), "constant", 0)  # (B, H, W, 4)

    # 2. Pre-reshape: (B, H, W, C') → (B, H, W/patch, C'*patch)
    #    This is metadata-only because the inner dims (C'*patch) are still contiguous.
    x_nhwc = x_nhwc.reshape(B, H, W // patch_size, C_padded * patch_size)  # (B, 224, 16, 56)

    x_ttnn = ttnn.from_torch(
        x_nhwc.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # 3. fold(stride_h=patch_size, stride_w=1)
    #    Output: (B, num_patches, kH*kW*C') = (B, 256, 14*14*4=784)
    x_folded = ttnn.fold(x_ttnn, patch_size, 1)
    print(f"\nfold output shape: {x_folded.shape}")

    # 4. to_layout TILE for matmul
    x_folded = ttnn.to_layout(x_folded, ttnn.TILE_LAYOUT)

    # 5. Weight prep (one-time at init in production):
    #    Conv weight (1152, 3, 14, 14) → linear weight (14*14*4=784, 1152)
    w_padded = torch.nn.functional.pad(w_torch, (0, 0, 0, 0, 0, C_padded - C))  # (1152, 4, 14, 14)
    w_reshaped = w_padded.permute(2, 3, 1, 0).contiguous()  # (14, 14, 4, 1152)
    w_reshaped = w_reshaped.reshape(-1, out_channels)  # (784, 1152)
    w_ttnn = ttnn.from_torch(
        w_reshaped.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b_ttnn = ttnn.from_torch(
        b_torch.reshape(1, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # 6. Linear: (B, 256, 784) × (784, 1152) → (B, 256, 1152)
    out_ttnn = ttnn.linear(
        x_folded,
        w_ttnn,
        bias=b_ttnn,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    out_ttnn = ttnn.to_torch(out_ttnn).float()
    # Output shape from linear is (B, 256, 1152) or (1, B*256, 1152) depending on internals
    if out_ttnn.dim() == 3:
        # Could be (B, 256, 1152) or (1, B*256, 1152)
        if out_ttnn.shape[0] == 1 and out_ttnn.shape[1] == B * 256:
            out_ttnn = out_ttnn.reshape(B, 256, out_channels)
    out_ttnn = out_ttnn.reshape(B, 256, out_channels)

    print(f"Reference shape: {out_torch_ref.shape}, range: [{out_torch_ref.min():.4f}, {out_torch_ref.max():.4f}]")
    print(f"ttnn fold+linear shape: {out_ttnn.shape}, range: [{out_ttnn.min():.4f}, {out_ttnn.max():.4f}]")

    ok, pcc = comp_pcc(out_torch_ref, out_ttnn, pcc=PCC_THRESHOLD)
    print(f"\nPCC: {pcc}")
    print(f"Max abs diff: {(out_torch_ref - out_ttnn).abs().max():.4f}")
    print(f"Mean abs diff: {(out_torch_ref - out_ttnn).abs().mean():.4f}")
    assert ok, f"PCC {pcc} below threshold {PCC_THRESHOLD}"
