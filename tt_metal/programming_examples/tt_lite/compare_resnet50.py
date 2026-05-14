# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Compare ResNet50 C++ replay output with Python TTNN reference.

Usage:
    python compare_resnet50.py <ref_dir> <replay_output.bin>
"""

import sys
import struct
import torch
import numpy as np


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ref_dir> <replay_output.bin>")
        return 1

    ref_dir = sys.argv[1]
    replay_output_path = sys.argv[2]

    # Load reference outputs
    pytorch_ref = torch.load(f"{ref_dir}/pytorch_reference_output.pt", weights_only=True)
    ttnn_trace_ref = torch.load(f"{ref_dir}/ttnn_trace_output.pt", weights_only=True)

    # Load C++ replay output (raw bfloat16 bytes from DRAM)
    with open(replay_output_path, "rb") as f:
        raw_bytes = f.read()

    # Parse as bfloat16: each value is 2 bytes
    num_values = len(raw_bytes) // 2
    # Convert raw bytes to int16 then reinterpret as bfloat16
    int16_array = np.frombuffer(raw_bytes, dtype=np.int16)
    replay_tensor = torch.from_numpy(int16_array.copy()).view(torch.bfloat16).float()

    print(f"=== ResNet50 Output Comparison ===")
    print(f"PyTorch reference shape: {pytorch_ref.shape}")
    print(f"TTNN trace reference shape: {ttnn_trace_ref.shape}")
    print(f"C++ replay raw values: {num_values}")

    # Try to reshape replay output to match reference
    ref_numel = pytorch_ref.numel()
    if num_values >= ref_numel:
        replay_output = replay_tensor[:ref_numel].reshape(pytorch_ref.shape)
    else:
        print(f"WARNING: replay output ({num_values}) smaller than reference ({ref_numel})")
        replay_output = replay_tensor

    # PCC comparisons
    pt_flat = pytorch_ref.flatten()
    ttnn_flat = ttnn_trace_ref.flatten()
    replay_flat = replay_output.flatten()[:pt_flat.shape[0]]

    pcc_ttnn_vs_pt = torch.corrcoef(torch.stack([pt_flat, ttnn_flat]))[0, 1].item()
    pcc_replay_vs_pt = torch.corrcoef(torch.stack([pt_flat, replay_flat]))[0, 1].item()
    pcc_replay_vs_ttnn = torch.corrcoef(torch.stack([ttnn_flat, replay_flat]))[0, 1].item()

    print(f"\n--- Pearson Correlation Coefficient ---")
    print(f"  TTNN trace vs PyTorch:   {pcc_ttnn_vs_pt:.6f}")
    print(f"  C++ replay vs PyTorch:   {pcc_replay_vs_pt:.6f}")
    print(f"  C++ replay vs TTNN trace: {pcc_replay_vs_ttnn:.6f}")

    # Top-5 predictions comparison (sample 0)
    pt_probs = torch.softmax(pytorch_ref[0], dim=-1)
    ttnn_probs = torch.softmax(ttnn_trace_ref[0], dim=-1)
    replay_probs = torch.softmax(replay_output[0], dim=-1)

    pt_top5 = torch.topk(pt_probs, 5)
    ttnn_top5 = torch.topk(ttnn_probs, 5)
    replay_top5 = torch.topk(replay_probs, 5)

    # Load ImageNet class names if available
    class_names = None
    try:
        from torchvision.models import ResNet50_Weights
        class_names = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    except Exception:
        pass

    def fmt_class(idx):
        if class_names and idx < len(class_names):
            return f"{idx:4d} ({class_names[idx]})"
        return f"{idx:4d}"

    print(f"\n--- Top-5 Predictions (sample 0) ---")
    print(f"{'Rank':<6} {'PyTorch reference':<40} {'TTNN trace':<40} {'C++ replay':<40}")
    print("-" * 126)
    for i in range(5):
        pt_str = f"{fmt_class(pt_top5.indices[i].item()):<30} {pt_top5.values[i].item():.4f}"
        tt_str = f"{fmt_class(ttnn_top5.indices[i].item()):<30} {ttnn_top5.values[i].item():.4f}"
        rp_str = f"{fmt_class(replay_top5.indices[i].item()):<30} {replay_top5.values[i].item():.4f}"
        print(f"  #{i+1:<3} {pt_str:<40} {tt_str:<40} {rp_str:<40}")

    # Top-1 agreement
    pt_top1 = pytorch_ref.argmax(dim=-1)
    ttnn_top1 = ttnn_trace_ref.argmax(dim=-1)
    replay_top1 = replay_output.argmax(dim=-1)
    batch_size = pytorch_ref.shape[0]

    ttnn_agree = (pt_top1 == ttnn_top1).sum().item()
    replay_agree = (pt_top1 == replay_top1).sum().item()
    ttnn_replay_agree = (ttnn_top1 == replay_top1).sum().item()

    print(f"\n--- Top-1 Agreement (batch={batch_size}) ---")
    print(f"  TTNN trace vs PyTorch:    {ttnn_agree}/{batch_size} ({100*ttnn_agree/batch_size:.1f}%)")
    print(f"  C++ replay vs PyTorch:    {replay_agree}/{batch_size} ({100*replay_agree/batch_size:.1f}%)")
    print(f"  C++ replay vs TTNN trace: {ttnn_replay_agree}/{batch_size} ({100*ttnn_replay_agree/batch_size:.1f}%)")

    # Summary
    print(f"\n--- Summary ---")
    if pcc_replay_vs_ttnn > 0.999:
        print(f"  PASS: C++ replay output matches TTNN trace output (PCC={pcc_replay_vs_ttnn:.6f} > 0.999)")
    elif pcc_replay_vs_ttnn > 0.99:
        print(f"  CLOSE: C++ replay output is close to TTNN trace (PCC={pcc_replay_vs_ttnn:.6f})")
    else:
        print(f"  FAIL: C++ replay output diverges from TTNN trace (PCC={pcc_replay_vs_ttnn:.6f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
