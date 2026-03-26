# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Data-parallel SmolVLA inference across multiple Tenstorrent devices.

Uses MeshDevice submeshes to run independent SmolVLA model replicas,
one per chip, each handling its own simulation environment.
"""

import time
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

import ttnn

from models.experimental.smolvla.tt.smol_vla import SmolVLAForActionPrediction


class DataParallelSmolVLA:
    """
    Manages N SmolVLA replicas, one per Tenstorrent chip.

    SmolVLA uses PIL images and its own HF processor, so the
    interface is simpler than PI0 (no manual TTNN tensor prep).
    """

    def __init__(
        self,
        num_devices: int,
        repo_id: str = "lerobot/smolvla_base",
        mesh_device: Optional[ttnn.Device] = None,
        l1_small_size: int = 81920,
    ):
        self.num_devices = num_devices
        self.repo_id = repo_id

        if mesh_device is not None:
            self.mesh_device = mesh_device
            self._owns_mesh = False
        else:
            self.mesh_device = ttnn.open_mesh_device(
                ttnn.MeshShape(1, num_devices),
                l1_small_size=l1_small_size,
            )
            self._owns_mesh = True

        self.submeshes = self.mesh_device.create_submeshes(ttnn.MeshShape(1, 1))
        actual = len(self.submeshes)
        if actual < num_devices:
            print(f"[DataParallelSmolVLA] Requested {num_devices} but got {actual} submeshes")
            self.num_devices = actual

        print(f"[DataParallelSmolVLA] Loading {self.num_devices} SmolVLA replicas...")
        self.models: List[SmolVLAForActionPrediction] = []
        for i, sub in enumerate(self.submeshes[:self.num_devices]):
            device = sub.get_devices()[0] if hasattr(sub, "get_devices") else sub
            model = SmolVLAForActionPrediction.from_pretrained(
                repo_id=repo_id, ttnn_device=device,
            )
            model.processor.image_processor.do_image_splitting = False
            model.eval()
            self.models.append(model)
            print(f"  SmolVLA replica {i} loaded")

    def infer_all(
        self,
        all_images: List[List[Image.Image]],
        instructions: List[str],
        num_inference_steps: int = 10,
        action_dim: int = 6,
    ) -> List[np.ndarray]:
        """
        Run inference on all replicas.

        Args:
            all_images: Per-environment list of PIL images.
            instructions: Per-environment task instruction strings.
            num_inference_steps: Flow matching steps.
            action_dim: Output action dimensionality.

        Returns:
            List of action arrays, one per replica.
        """
        results = []
        for i, model in enumerate(self.models):
            imgs = all_images[i] if i < len(all_images) else all_images[-1]
            inst = instructions[i] if i < len(instructions) else instructions[-1]
            with torch.no_grad():
                actions = model.sample_actions(
                    images=imgs,
                    instruction=inst,
                    num_inference_steps=num_inference_steps,
                    action_dim=action_dim,
                )
            results.append(actions)
        return results

    def close(self):
        if self._owns_mesh:
            ttnn.close_mesh_device(self.mesh_device)
