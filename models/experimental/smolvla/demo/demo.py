# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
SmolVLA Demo - Vision-Language-Action Model on TT Hardware

This demo shows how to run SmolVLA for robotic action prediction.
The model takes an image and instruction, and predicts robot actions.

Usage:
    python models/experimental/smolvla/demo/demo.py
"""

import os
import time
import numpy as np
import torch
from PIL import Image

import ttnn
from models.experimental.smolvla.tt.smol_vla import SmolVLAForActionPrediction


def load_lerobot_images():
    """Load real LeRobot sample images."""
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(demo_dir, "images")

    images = []
    for i in range(1, 4):
        path = os.path.join(images_dir, f"lerobot_sample_{i}.png")
        if os.path.exists(path):
            img = Image.open(path)
            images.append(img)
            print(f"      Loaded: {path} ({img.size})")

    if not images:
        # Fallback: create synthetic image
        print("      No LeRobot images found, using synthetic image")
        img_array = np.zeros((480, 640, 3), dtype=np.uint8)
        img_array[:, :, 0] = 180  # Red channel
        img_array[:, :, 1] = 160  # Green channel
        img_array[:, :, 2] = 140  # Blue channel
        images = [Image.fromarray(img_array)]

    return images


def run_demo():
    """Run SmolVLA demo with various instructions."""
    print("=" * 60)
    print("SmolVLA Demo - Vision-Language-Action on TT Hardware")
    print("=" * 60)

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Open TT device
    print("\n[1/4] Opening TT device...")
    device = ttnn.open_device(device_id=0)

    try:
        # Load model
        print("[2/4] Loading SmolVLA model (this may take a minute)...")
        model = SmolVLAForActionPrediction.from_pretrained("lerobot/smolvla_base", ttnn_device=device)
        model.processor.image_processor.do_image_splitting = False
        model.eval()
        print("      Model loaded: lerobot/smolvla_base")

        # Load real LeRobot images
        print("[3/4] Loading LeRobot images...")
        images = load_lerobot_images()
        image = images[0]  # Use first image for demo
        print(f"      Using image: {image.size}, mode={image.mode}")

        # Test instructions
        instructions = [
            "pick up the red block",
            "move to the blue block",
            "push the red block to the right",
            "lift up",
            "rotate clockwise",
        ]

        print("[4/4] Running inference...")
        print("\n" + "-" * 60)

        # Robot state (6-DOF: x, y, z, roll, pitch, yaw)
        robot_state = torch.zeros(6).float()

        # Warmup
        print("Warmup run...", end=" ", flush=True)
        with torch.no_grad():
            _ = model.predict_action(
                images=[image], robot_state=robot_state, instruction=instructions[0], num_inference_steps=5
            )
        print("done")
        print("-" * 60)

        # Run each instruction
        for instruction in instructions:
            print(f'\nInstruction: "{instruction}"')

            start = time.time()
            with torch.no_grad():
                action = model.predict_action(
                    images=[image],
                    robot_state=robot_state,
                    instruction=instruction,
                    num_inference_steps=10,  # Default: 10 steps for ~200ms latency
                )
            elapsed = time.time() - start

            # Convert to numpy if needed
            if hasattr(action, "cpu"):
                action = action.cpu().numpy()
            action = np.asarray(action).flatten()

            # Print action output
            fps = 1.0 / elapsed if elapsed > 0 else 0
            print(f"  Action (6 DoF): [{', '.join([f'{a:+.3f}' for a in action[:6]])}]")
            print(f"  Latency: {elapsed*1000:.1f}ms | FPS: {fps:.1f}")

            # Interpret the action
            dof_names = ["X (left/right)", "Y (forward/back)", "Z (up/down)", "Roll", "Pitch", "Yaw (rotate)"]
            max_idx = np.argmax(np.abs(action[:6]))
            print(f"  Dominant motion: {dof_names[max_idx]} = {action[max_idx]:+.3f}")

        print("\n" + "=" * 60)
        print("Demo complete!")
        print("=" * 60)

    finally:
        # Clean up
        print("\nClosing TT device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    run_demo()
