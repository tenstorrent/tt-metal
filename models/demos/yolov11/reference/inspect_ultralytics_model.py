#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Inspect Ultralytics YOLO11n-pose model structure
"""

import torch
from ultralytics import YOLO


def inspect_model():
    print("=" * 70)
    print("Inspecting Ultralytics YOLO11n-pose Model Structure")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = YOLO("yolo11n-pose.pt")
    pt_model = model.model

    # Print model architecture
    print("\n" + "=" * 70)
    print("Model Architecture:")
    print("=" * 70)
    print(pt_model)

    # Print state dict keys and shapes
    print("\n" + "=" * 70)
    print("Model State Dict (Weights):")
    print("=" * 70)

    state_dict = pt_model.state_dict()

    # Group by module
    from collections import defaultdict

    by_module = defaultdict(list)

    for name, param in state_dict.items():
        # Extract module name (before first digit or 'model')
        parts = name.split(".")
        if len(parts) > 1:
            module = parts[1] if parts[0] == "model" else parts[0]
        else:
            module = "root"
        by_module[module].append((name, param.shape))

    for module, params in sorted(by_module.items()):
        print(f"\n{module}:")
        for name, shape in params[:5]:  # Show first 5
            print(f"  {name}: {shape}")
        if len(params) > 5:
            print(f"  ... and {len(params) - 5} more")

    print(f"\nTotal parameters: {len(state_dict)}")

    # Inspect the detection/pose head specifically
    print("\n" + "=" * 70)
    print("Pose Head Structure (layer 23):")
    print("=" * 70)

    # The pose head is typically the last layer
    if hasattr(pt_model, "model") and len(pt_model.model) > 0:
        pose_head = pt_model.model[-1]
        print(f"\nPose head type: {type(pose_head)}")
        print(f"\nPose head structure:")
        print(pose_head)

        # Check attributes
        print("\nPose head attributes:")
        for attr in dir(pose_head):
            if not attr.startswith("_"):
                try:
                    val = getattr(pose_head, attr)
                    if not callable(val):
                        print(f"  {attr}: {type(val).__name__}")
                except:
                    pass

    # Test forward pass
    print("\n" + "=" * 70)
    print("Testing Forward Pass:")
    print("=" * 70)

    dummy_input = torch.randn(1, 3, 640, 640)
    print(f"\nInput shape: {dummy_input.shape}")

    with torch.no_grad():
        outputs = pt_model(dummy_input)

    if isinstance(outputs, (list, tuple)):
        print(f"\nNumber of outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            if isinstance(out, torch.Tensor):
                print(f"  Output {i}: {out.shape}")
            else:
                print(f"  Output {i}: {type(out)}")
    else:
        print(f"\nOutput shape: {outputs.shape}")

    # Save detailed architecture to file
    output_file = "ultralytics_model_structure.txt"
    with open(output_file, "w") as f:
        f.write("Ultralytics YOLO11n-pose Model Structure\n")
        f.write("=" * 70 + "\n\n")
        f.write(str(pt_model))
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("State Dict:\n")
        f.write("=" * 70 + "\n")
        for name, param in state_dict.items():
            f.write(f"{name}: {param.shape}\n")

    print(f"\n✓ Detailed structure saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    inspect_model()
