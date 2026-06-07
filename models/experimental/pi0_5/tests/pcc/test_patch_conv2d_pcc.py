# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone PCC test: ttnn.conv2d vs the existing linear-unfold patch embed path.

Validates that ttnn.conv2d(kernel=14, stride=14) on the SigLIP patch shape
(B=3, C=3, H=224, W=224) matches the linear-unfold output within bf16 tolerance
(PCC >= 0.999).

If this passes, the conv2d path is a safe drop-in for PatchEmbeddingTTNN.forward.
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
    """Load SigLIP patch_embedding.weight and .bias from the safetensors checkpoint."""
    sf = safe_open(str(CHECKPOINT_DIR / "model.safetensors"), framework="pt")
    keys = list(sf.keys())
    w_key = next(k for k in keys if "vision_tower.vision_model.embeddings.patch_embedding.weight" in k)
    b_key = next(k for k in keys if "vision_tower.vision_model.embeddings.patch_embedding.bias" in k)
    return sf.get_tensor(w_key).to(torch.float32), sf.get_tensor(b_key).to(torch.float32)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_patch_conv2d_pcc(device):
    """Compare conv2d-based patch extraction to torch reference."""
    torch.manual_seed(0)

    # Real SigLIP patch weights
    w_torch, b_torch = _load_patch_weights()  # w: (1152, 3, 14, 14), b: (1152,)
    out_channels, in_channels, kh, kw = w_torch.shape
    assert (in_channels, kh, kw) == (3, 14, 14), f"unexpected weight shape {w_torch.shape}"

    # Synthetic input: 3 cameras, (3, 3, 224, 224) bf16
    B, C, H, W = 3, 3, 224, 224
    x_torch = torch.randn(B, C, H, W, dtype=torch.float32) * 0.5

    # === Torch reference (the "truth" we compare against) ===
    out_torch_ref = torch.nn.functional.conv2d(x_torch, w_torch, bias=b_torch, stride=(kh, kw))  # (B, 1152, 16, 16)
    # Reshape to (B, num_patches, hidden) like SigLIP downstream expects
    out_torch_ref = out_torch_ref.flatten(2).transpose(1, 2)  # (B, 256, 1152)

    # === ttnn.conv2d path ===
    # conv2d expects input as (1, 1, N*H*W, C) row-major
    x_nhwc_flat = x_torch.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16)  # (B, H, W, C)
    x_nhwc_flat = x_nhwc_flat.reshape(1, 1, B * H * W, C)
    x_ttnn = ttnn.from_torch(
        x_nhwc_flat,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Weight in PyTorch convention; ttnn.conv2d will internally re-arrange.
    w_ttnn = ttnn.from_torch(w_torch.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_ttnn = ttnn.from_torch(
        b_torch.reshape(1, 1, 1, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    out_ttnn = ttnn.conv2d(
        input_tensor=x_ttnn,
        weight_tensor=w_ttnn,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=b_ttnn,
        kernel_size=(kh, kw),
        stride=(kh, kw),
        padding=(0, 0),
        dilation=(1, 1),
        batch_size=B,
        input_height=H,
        input_width=W,
        compute_config=compute_config,
        groups=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        return_output_dim=False,
        return_weights_and_bias=False,
        dtype=ttnn.bfloat16,
    )

    # Output is (1, 1, B * (H//kh) * (W//kw), out_channels)
    out_ttnn = ttnn.to_torch(out_ttnn).float()
    P_h = H // kh
    P_w = W // kw
    num_patches = P_h * P_w
    # Reshape to match torch reference (B, num_patches, hidden)
    if out_ttnn.dim() == 4:
        out_ttnn = out_ttnn.reshape(B, num_patches, out_channels)
    elif out_ttnn.dim() == 3:
        out_ttnn = out_ttnn.reshape(B, num_patches, out_channels)

    print(f"\nReference shape: {out_torch_ref.shape}")
    print(f"ttnn.conv2d  out shape: {out_ttnn.shape}")
    print(f"Reference range: [{out_torch_ref.min():.4f}, {out_torch_ref.max():.4f}]")
    print(f"ttnn output  range: [{out_ttnn.min():.4f}, {out_ttnn.max():.4f}]")

    ok, pcc = comp_pcc(out_torch_ref, out_ttnn, pcc=PCC_THRESHOLD)
    print(f"\nPCC: {pcc}")
    print(f"Max abs diff: {(out_torch_ref - out_ttnn).abs().max():.4f}")
    print(f"Mean abs diff: {(out_torch_ref - out_ttnn).abs().mean():.4f}")
    assert ok, f"PCC {pcc} below threshold {PCC_THRESHOLD}"
