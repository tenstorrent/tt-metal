# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Layer-by-layer PCC debugging for SFace model.
Identifies where accuracy drops between PyTorch and TTNN.
"""

import torch
import numpy as np
import ttnn

from models.experimental.sface.common import get_sface_onnx_path, SFACE_L1_SMALL_SIZE
from models.experimental.sface.reference.sface_model import load_sface_from_onnx


def compute_pcc(a, b):
    """Compute Pearson Correlation Coefficient."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.corrcoef(a_flat, b_flat)[0, 1]


def compute_mse(a, b):
    """Compute Mean Squared Error."""
    return np.mean((a.flatten() - b.flatten()) ** 2)


def compute_max_diff(a, b):
    """Compute maximum absolute difference."""
    return np.max(np.abs(a.flatten() - b.flatten()))


def debug_conv_bn_prelu(
    torch_conv_bn_prelu, ttnn_conv_bn_prelu, x_torch_nchw, x_ttnn, device, layer_name, batch_size, h, w
):
    """Debug a single ConvBNPReLU block."""
    print(f"\n{'='*60}")
    print(f"Layer: {layer_name}")
    print(f"{'='*60}")

    # PyTorch forward
    with torch.no_grad():
        y_torch = torch_conv_bn_prelu(x_torch_nchw)

    # TTNN forward
    y_ttnn, new_h, new_w = ttnn_conv_bn_prelu(x_ttnn, batch_size, h, w)

    # Convert TTNN to numpy for comparison
    y_ttnn_torch = ttnn.to_torch(y_ttnn)

    # TTNN output is [B, 1, H*W, C], convert to [B, C, H, W] for comparison
    y_ttnn_nchw = y_ttnn_torch.reshape(batch_size, new_h, new_w, -1).permute(0, 3, 1, 2)

    y_torch_np = y_torch.float().numpy()
    y_ttnn_np = y_ttnn_nchw.float().numpy()

    pcc = compute_pcc(y_torch_np, y_ttnn_np)
    mse = compute_mse(y_torch_np, y_ttnn_np)
    max_diff = compute_max_diff(y_torch_np, y_ttnn_np)

    print(f"Output shape: PyTorch {y_torch.shape}, TTNN {y_ttnn_nchw.shape}")
    print(f"PCC: {pcc:.6f}")
    print(f"MSE: {mse:.8f}")
    print(f"Max Diff: {max_diff:.6f}")
    print(f"PyTorch range: [{y_torch_np.min():.4f}, {y_torch_np.max():.4f}]")
    print(f"TTNN range: [{y_ttnn_np.min():.4f}, {y_ttnn_np.max():.4f}]")

    # Sample values
    print(f"PyTorch sample (first 5): {y_torch_np.flatten()[:5]}")
    print(f"TTNN sample (first 5): {y_ttnn_np.flatten()[:5]}")

    # Return for next layer
    return y_torch, y_ttnn, new_h, new_w, pcc


def debug_depthwise_separable(torch_block, ttnn_block, x_torch_nchw, x_ttnn, device, layer_name, batch_size, h, w):
    """Debug a DepthwiseSeparable block."""
    print(f"\n{'='*60}")
    print(f"Block: {layer_name}")
    print(f"{'='*60}")

    # PyTorch forward
    with torch.no_grad():
        y_torch = torch_block(x_torch_nchw)

    # TTNN forward
    y_ttnn, new_h, new_w = ttnn_block(x_ttnn, batch_size, h, w)

    # Convert TTNN to numpy for comparison
    y_ttnn_torch = ttnn.to_torch(y_ttnn)

    # TTNN output is [B, 1, H*W, C], convert to [B, C, H, W] for comparison
    y_ttnn_nchw = y_ttnn_torch.reshape(batch_size, new_h, new_w, -1).permute(0, 3, 1, 2)

    y_torch_np = y_torch.float().numpy()
    y_ttnn_np = y_ttnn_nchw.float().numpy()

    pcc = compute_pcc(y_torch_np, y_ttnn_np)
    mse = compute_mse(y_torch_np, y_ttnn_np)
    max_diff = compute_max_diff(y_torch_np, y_ttnn_np)

    print(f"Output shape: PyTorch {y_torch.shape}, TTNN {y_ttnn_nchw.shape}")
    print(f"PCC: {pcc:.6f}")
    print(f"MSE: {mse:.8f}")
    print(f"Max Diff: {max_diff:.6f}")
    print(f"PyTorch range: [{y_torch_np.min():.4f}, {y_torch_np.max():.4f}]")
    print(f"TTNN range: [{y_ttnn_np.min():.4f}, {y_ttnn_np.max():.4f}]")

    return y_torch, y_ttnn, new_h, new_w, pcc


def main():
    print("=" * 70)
    print("SFace Layer-by-Layer PCC Debug")
    print("=" * 70)

    # Load models
    onnx_path = get_sface_onnx_path()
    torch_model = load_sface_from_onnx(onnx_path)
    torch_model.eval()

    # Open device
    device = ttnn.open_device(device_id=0, l1_small_size=SFACE_L1_SMALL_SIZE)

    # Create TTNN model
    from models.experimental.sface.tt.ttnn_sface import create_sface_model

    ttnn_model = create_sface_model(device, torch_model)

    # Create test input
    batch_size = 1
    torch.manual_seed(42)
    x_torch_nchw = torch.randn(batch_size, 3, 112, 112, dtype=torch.float32)
    x_torch_nhwc = x_torch_nchw.permute(0, 2, 3, 1).contiguous()

    # Create TTNN input
    x_ttnn = ttnn.from_torch(
        x_torch_nhwc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    h, w = 112, 112
    pcc_results = []

    # ============ Layer 1: conv1 ============
    y_torch, y_ttnn, h, w, pcc = debug_conv_bn_prelu(
        torch_model.conv1, ttnn_model.conv1, x_torch_nchw, x_ttnn, device, "conv1", batch_size, 112, 112
    )
    pcc_results.append(("conv1", pcc))
    x_torch_nchw = y_torch
    x_ttnn = y_ttnn

    # ============ Blocks 2-14 ============
    blocks = [
        ("block2", torch_model.block2, ttnn_model.block2),
        ("block3", torch_model.block3, ttnn_model.block3),
        ("block4", torch_model.block4, ttnn_model.block4),
        ("block5", torch_model.block5, ttnn_model.block5),
        ("block6", torch_model.block6, ttnn_model.block6),
        ("block7", torch_model.block7, ttnn_model.block7),
        ("block8", torch_model.block8, ttnn_model.block8),
        ("block9", torch_model.block9, ttnn_model.block9),
        ("block10", torch_model.block10, ttnn_model.block10),
        ("block11", torch_model.block11, ttnn_model.block11),
        ("block12", torch_model.block12, ttnn_model.block12),
        ("block13", torch_model.block13, ttnn_model.block13),
        ("block14", torch_model.block14, ttnn_model.block14),
    ]

    for name, torch_block, ttnn_block in blocks:
        y_torch, y_ttnn, h, w, pcc = debug_depthwise_separable(
            torch_block, ttnn_block, x_torch_nchw, x_ttnn, device, name, batch_size, h, w
        )
        pcc_results.append((name, pcc))
        x_torch_nchw = y_torch
        x_ttnn = y_ttnn

    # ============ Head ============
    print(f"\n{'='*60}")
    print("Head: BN1 + FC + BN2 + L2Norm")
    print(f"{'='*60}")

    # PyTorch head
    with torch.no_grad():
        x_torch_head = torch_model.bn1(x_torch_nchw)
        x_torch_head = x_torch_head.flatten(1)
        x_torch_head = torch_model.fc(x_torch_head)
        x_torch_head = torch_model.bn2(x_torch_head)
        y_torch_final = torch.nn.functional.normalize(x_torch_head, p=2, dim=1)

    # TTNN head (already in model)
    # The TTNN model returns the final output after head
    # We need to run full model to get head output
    x_ttnn_input = ttnn.from_torch(
        torch.randn(batch_size, 3, 112, 112, dtype=torch.float32).permute(0, 2, 3, 1).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Actually, let's just compare full model output
    torch.manual_seed(42)
    x_full = torch.randn(batch_size, 3, 112, 112, dtype=torch.float32)

    with torch.no_grad():
        y_torch_full = torch_model(x_full)

    x_full_nhwc = x_full.permute(0, 2, 3, 1).contiguous()
    x_full_ttnn = ttnn.from_torch(x_full_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y_ttnn_full = ttnn_model(x_full_ttnn)
    y_ttnn_full_np = ttnn.to_torch(y_ttnn_full).float().numpy()
    y_torch_full_np = y_torch_full.numpy()

    final_pcc = compute_pcc(y_torch_full_np, y_ttnn_full_np)
    pcc_results.append(("Final Output", final_pcc))

    print(f"Final Output PCC: {final_pcc:.6f}")
    print(f"PyTorch output: {y_torch_full_np[0, :10]}")
    print(f"TTNN output: {y_ttnn_full_np[0, :10]}")

    # ============ Summary ============
    print(f"\n{'='*70}")
    print("PCC SUMMARY (Layer by Layer)")
    print(f"{'='*70}")
    print(f"{'Layer':<20} {'PCC':>12}")
    print("-" * 35)
    for name, pcc in pcc_results:
        status = "✅" if pcc > 0.99 else "⚠️" if pcc > 0.95 else "❌"
        print(f"{name:<20} {pcc:>12.6f} {status}")

    # Find where PCC drops
    print(f"\n{'='*70}")
    print("ANALYSIS: Where does PCC drop?")
    print(f"{'='*70}")
    prev_pcc = 1.0
    for name, pcc in pcc_results:
        drop = prev_pcc - pcc
        if drop > 0.01:
            print(f"⚠️  Significant drop at {name}: {prev_pcc:.4f} → {pcc:.4f} (drop: {drop:.4f})")
        prev_pcc = pcc

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
