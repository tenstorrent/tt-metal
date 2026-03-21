# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
analyze rt-detr model structure and identify ops needed for ttnn
uses huggingface so no need to deal with weights manually
"""

from collections import defaultdict

import torch
from transformers import RTDetrForObjectDetection


def analyze_model():
    print("loading rt-detr from huggingface (may take a moment first run)...")
    model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd", ignore_mismatched_sizes=True)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\ntotal parameters: {total_params:,}\n")

    # count module types
    counts = defaultdict(int)
    for _, m in model.named_modules():
        counts[m.__class__.__name__] += 1

    print("module type counts:")
    for mtype, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {mtype:40s}: {n}")

    # collect ops we need in ttnn
    ops = set()
    type_map = {
        "Conv2d": "Conv2d",
        "BatchNorm": "BatchNorm2d",
        "MaxPool": "MaxPool2d",
        "AvgPool": "AvgPool2d",
        "ReLU": "ReLU",
        "GELU": "GELU",
        "SiLU": "SiLU",
        "Swish": "SiLU",
        "Attention": "MultiheadAttention",
        "MultiheadAttention": "MultiheadAttention",
        "LayerNorm": "LayerNorm",
        "Linear": "Linear",
    }
    for _, m in model.named_modules():
        for key, op in type_map.items():
            if key in m.__class__.__name__:
                ops.add(op)

    print("\nops needed for ttnn:")
    for op in sorted(ops):
        print(f"  {op}")

    # quick forward pass to check shapes
    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model(pixel_values=dummy)

    print(f"\nforward pass ok")
    if hasattr(out, "logits"):
        print(f"  logits:    {out.logits.shape}")
    if hasattr(out, "pred_boxes"):
        print(f"  pred_boxes:{out.pred_boxes.shape}")

    with open("reference/rt_detr_analysis.txt", "w") as f:
        f.write(f"total params: {total_params:,}\n\nops needed:\n")
        for op in sorted(ops):
            f.write(f"  {op}\n")

    print("\nanalysis saved to reference/rt_detr_analysis.txt")


if __name__ == "__main__":
    analyze_model()
