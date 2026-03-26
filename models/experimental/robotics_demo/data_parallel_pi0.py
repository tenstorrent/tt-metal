# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Data-parallel PI0 inference across multiple Tenstorrent devices.

Uses MeshDevice submeshes to run independent PI0 model replicas,
one per chip, each handling its own simulation environment.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

import ttnn

from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN
from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader


def create_pi0_config() -> PI0ModelConfig:
    config = PI0ModelConfig(
        action_dim=32, action_horizon=50, state_dim=32,
        paligemma_variant="gemma_2b", action_expert_variant="gemma_300m", pi05=False,
    )
    config.siglip_config = SigLIPConfig(
        hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
        num_attention_heads=16, image_size=224, patch_size=14,
    )
    return config


class DataParallelPI0:
    """
    Manages N PI0 replicas, one per Tenstorrent chip.

    Each replica is a full PI0ModelTTNN on its own submesh device.
    Observations are dispatched round-robin; actions are returned per-replica.
    """

    def __init__(
        self,
        num_devices: int,
        checkpoint_path: str,
        mesh_device: Optional[ttnn.Device] = None,
        l1_small_size: int = 24576,
    ):
        self.num_devices = num_devices
        self.checkpoint_path = checkpoint_path

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
            print(f"[DataParallelPI0] Requested {num_devices} devices but got {actual} submeshes")
            self.num_devices = actual

        self.config = create_pi0_config()
        self.weight_loader = PI0WeightLoader(checkpoint_path)

        print(f"[DataParallelPI0] Loading {self.num_devices} PI0 replicas...")
        self.models: List[PI0ModelTTNN] = []
        for i, sub in enumerate(self.submeshes[:self.num_devices]):
            device = sub.get_devices()[0] if hasattr(sub, "get_devices") else sub
            model = PI0ModelTTNN(self.config, self.weight_loader, device, fresh_noise_per_call=True)
            self.models.append(model)
            print(f"  Replica {i} loaded on submesh")

    def infer_all(
        self,
        observations: List[Dict],
    ) -> List:
        """
        Run inference on all replicas in sequence.

        Args:
            observations: List of dicts per environment, each containing:
                - 'images_ttnn': list of TTNN image tensors
                - 'img_masks': list of torch bool tensors
                - 'lang_tokens_ttnn': TTNN token tensor
                - 'lang_masks_ttnn': TTNN mask tensor
                - 'state_ttnn': TTNN state tensor

        Returns:
            List of action tensors (one per replica).
        """
        results = []
        for i, (model, obs) in enumerate(zip(self.models, observations)):
            with torch.no_grad():
                actions = model.sample_actions(
                    images=obs["images_ttnn"],
                    img_masks=obs["img_masks"],
                    lang_tokens=obs["lang_tokens_ttnn"],
                    lang_masks=obs["lang_masks_ttnn"],
                    state=obs["state_ttnn"],
                )
            results.append(actions)
        return results

    def preprocess_for_device(
        self,
        device_idx: int,
        images: List[torch.Tensor],
        state: torch.Tensor,
        tokens: torch.Tensor,
        masks: torch.Tensor,
    ) -> Dict:
        """Convert raw torch tensors to TTNN tensors on the correct device."""
        device = self._get_device(device_idx)

        images_ttnn = [
            ttnn.from_torch(img, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for img in images
        ]
        img_masks = [torch.ones(1, dtype=torch.bool) for _ in images]

        lang_tokens_ttnn = ttnn.from_torch(tokens, dtype=ttnn.uint32,
                                           layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        lang_masks_ttnn = ttnn.from_torch(masks.float(), dtype=ttnn.bfloat16,
                                          layout=ttnn.TILE_LAYOUT, device=device)
        state_ttnn = ttnn.from_torch(state, dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=device)

        return {
            "images_ttnn": images_ttnn,
            "img_masks": img_masks,
            "lang_tokens_ttnn": lang_tokens_ttnn,
            "lang_masks_ttnn": lang_masks_ttnn,
            "state_ttnn": state_ttnn,
        }

    def _get_device(self, idx: int):
        sub = self.submeshes[idx]
        return sub.get_devices()[0] if hasattr(sub, "get_devices") else sub

    def close(self):
        if self._owns_mesh:
            ttnn.close_mesh_device(self.mesh_device)
