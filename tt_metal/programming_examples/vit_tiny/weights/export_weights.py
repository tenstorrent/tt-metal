#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Export ViT Tiny weights from timm to binary files for C++ inference.

Usage:
    python export_weights.py --output-dir ./vit_tiny_weights

Weight format: raw float32 binary, row-major.
The C++ loader handles padding and tilization.
"""

import argparse
import os
import numpy as np

def export_weights(output_dir: str):
    import timm
    import torch

    os.makedirs(output_dir, exist_ok=True)

    model = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
    model.eval()
    state = model.state_dict()

    def save(name: str, tensor: torch.Tensor):
        arr = tensor.detach().cpu().float().numpy()
        path = os.path.join(output_dir, name + ".bin")
        arr.tofile(path)
        print(f"  {name}: {arr.shape} -> {path}")

    # Patch embedding: conv2d weight [192, 3, 16, 16] -> reshape to [768, 192]^T
    # timm stores as conv2d: out_channels=192, in_channels=3, kH=16, kW=16
    # For matmul: input is [N, 768], weight should be [768, 192]
    # conv weight: [192, 3, 16, 16] -> reshape to [192, 768] -> transpose to [768, 192]
    conv_w = state["patch_embed.proj.weight"]  # [192, 3, 16, 16]
    conv_w_2d = conv_w.reshape(192, -1).T  # [768, 192]
    save("patch_embed.proj.weight", conv_w_2d)
    save("patch_embed.proj.bias", state["patch_embed.proj.bias"])  # [192]

    # CLS token [1, 1, 192] -> [1, 192]
    save("cls_token", state["cls_token"].squeeze(0))

    # Position embedding [1, 197, 192] -> pad to [224, 192]
    pos = state["pos_embed"].squeeze(0)  # [197, 192]
    pos_padded = torch.zeros(224, 192)
    pos_padded[:197, :] = pos
    save("pos_embed", pos_padded)

    # Transformer blocks
    for i in range(12):
        prefix = f"blocks.{i}."

        # LayerNorm 1
        save(f"{prefix}norm1.weight", state[f"blocks.{i}.norm1.weight"])
        save(f"{prefix}norm1.bias", state[f"blocks.{i}.norm1.bias"])

        # Attention QKV: timm stores as [576, 192] (out_features, in_features)
        # For matmul: input [N, 192] x weight [192, 576] -> transpose
        qkv_w = state[f"blocks.{i}.attn.qkv.weight"]  # [576, 192]
        save(f"{prefix}attn.qkv.weight", qkv_w.T)  # [192, 576]
        save(f"{prefix}attn.qkv.bias", state[f"blocks.{i}.attn.qkv.bias"])  # [576]

        # Attention output projection
        proj_w = state[f"blocks.{i}.attn.proj.weight"]  # [192, 192]
        save(f"{prefix}attn.proj.weight", proj_w.T)  # [192, 192]
        save(f"{prefix}attn.proj.bias", state[f"blocks.{i}.attn.proj.bias"])

        # LayerNorm 2
        save(f"{prefix}norm2.weight", state[f"blocks.{i}.norm2.weight"])
        save(f"{prefix}norm2.bias", state[f"blocks.{i}.norm2.bias"])

        # MLP FC1: [768, 192] -> transpose to [192, 768]
        fc1_w = state[f"blocks.{i}.mlp.fc1.weight"]  # [768, 192]
        save(f"{prefix}mlp.fc1.weight", fc1_w.T)  # [192, 768]
        save(f"{prefix}mlp.fc1.bias", state[f"blocks.{i}.mlp.fc1.bias"])

        # MLP FC2: [192, 768] -> transpose to [768, 192]
        fc2_w = state[f"blocks.{i}.mlp.fc2.weight"]  # [192, 768]
        save(f"{prefix}mlp.fc2.weight", fc2_w.T)  # [768, 192]
        save(f"{prefix}mlp.fc2.bias", state[f"blocks.{i}.mlp.fc2.bias"])

    # Final LayerNorm
    save("norm.weight", state["norm.weight"])
    save("norm.bias", state["norm.bias"])

    # Classification head: [1000, 192] -> transpose to [192, 1000]
    head_w = state["head.weight"]  # [1000, 192]
    save("head.weight", head_w.T)  # [192, 1000]
    save("head.bias", state["head.bias"])

    print(f"\nAll weights exported to {output_dir}")
    print(f"Total files: {len(os.listdir(output_dir))}")


def export_test_image(output_dir: str, image_path: str = None):
    """Export a preprocessed test image as float32 binary [3, 224, 224]."""
    import torch
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if image_path:
        img = Image.open(image_path).convert("RGB")
    else:
        # Create a synthetic test image
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

    tensor = transform(img).numpy()  # [3, 224, 224]
    path = os.path.join(output_dir, "test_image.bin")
    tensor.astype(np.float32).tofile(path)
    print(f"Test image exported: {tensor.shape} -> {path}")

    # Also run reference inference
    import timm
    model = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(tensor).unsqueeze(0))
    top5 = torch.topk(logits, 5)
    print(f"Reference top-5 classes: {top5.indices[0].tolist()}")
    print(f"Reference top-5 scores: {top5.values[0].tolist()}")

    # Save reference logits
    logits.numpy().astype(np.float32).tofile(os.path.join(output_dir, "reference_logits.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./vit_tiny_weights")
    parser.add_argument("--test-image", default=None, help="Path to test image")
    parser.add_argument("--export-image", action="store_true", help="Also export a test image")
    args = parser.parse_args()

    export_weights(args.output_dir)
    if args.export_image:
        export_test_image(args.output_dir, args.test_image)
