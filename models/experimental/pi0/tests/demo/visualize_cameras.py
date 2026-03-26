#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Visualize camera views from the PI0 PyBullet simulation.

Captures RGB and depth images from both cameras and saves them
to ./camera_debug/ for inspection. Useful for verifying the cube
is visible and the robot is in frame.

Usage:
    python models/experimental/pi0/tests/demo/visualize_cameras.py
    python models/experimental/pi0/tests/demo/visualize_cameras.py --image-size 224
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("TT_METAL_HOME", os.path.join(os.path.dirname(__file__), *[".."] * 4)))

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("PyBullet required: pip install pybullet")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Visualize PI0 sim camera views")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "camera_debug")
    os.makedirs(out_dir, exist_ok=True)

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    cube = p.loadURDF("cube_small.urdf", [0.5, 0.0, 0.025])

    cameras = {
        "front": {"eye": [1.0, 0.0, 0.5], "target": [0.3, 0, 0.3]},
        "side": {"eye": [0.3, 1.0, 0.5], "target": [0.3, 0, 0.3]},
        "overview": {"eye": [1.2, 0.8, 0.8], "target": [0.3, 0, 0.3]},
    }

    sz = args.image_size
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=5.0)

    print(f"Capturing {sz}x{sz} images from {len(cameras)} cameras...")

    for name, cam in cameras.items():
        view = p.computeViewMatrix(
            cameraEyePosition=cam["eye"],
            cameraTargetPosition=cam["target"],
            cameraUpVector=[0, 0, 1],
        )
        _, _, rgba, depth, _ = p.getCameraImage(
            width=sz, height=sz,
            viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb = np.array(rgba[:, :, :3], dtype=np.uint8)
        depth_arr = np.array(depth, dtype=np.float32)
        depth_norm = ((depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min() + 1e-8) * 255).astype(np.uint8)

        try:
            from PIL import Image
            Image.fromarray(rgb).save(os.path.join(out_dir, f"{name}_rgb.png"))
            Image.fromarray(depth_norm).save(os.path.join(out_dir, f"{name}_depth.png"))
        except ImportError:
            np.save(os.path.join(out_dir, f"{name}_rgb.npy"), rgb)
            np.save(os.path.join(out_dir, f"{name}_depth.npy"), depth_norm)

        print(f"  {name}: saved RGB and depth to {out_dir}/")

    p.disconnect()
    print(f"\nAll camera views saved to: {os.path.abspath(out_dir)}")
    print("Check that the cube is visible in the RGB images.")


if __name__ == "__main__":
    main()
