# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ModelConfig for DiffusionDrive TTNN bring-up.

All architecture constants confirmed in Stage 0 (2026-04-13) by reading
navsim/agents/diffusiondrive/transfuser_config.py and transfuser_model_v2.py
from hustvl/DiffusionDrive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class ModelConfig:
    """Confirmed architecture constants for DiffusionDrive on NavSim."""

    # ------------------------------------------------------------------
    # Backbone (TransFuser: ResNet-34 image + ResNet-34 LiDAR)
    # ------------------------------------------------------------------
    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"

    # Camera input resolution (stitched 3-camera front view)
    camera_height: int = 256
    camera_width: int = 1024

    # LiDAR BEV input
    lidar_resolution_height: int = 256
    lidar_resolution_width: int = 256
    lidar_seq_len: int = 1  # 1 height-split channel
    lidar_max_x: float = 32.0  # BEV grid extent (metres)
    lidar_max_y: float = 32.0

    # GPT cross-modal fusion blocks
    img_vert_anchors: int = 8
    img_horz_anchors: int = 32
    lidar_vert_anchors: int = 8
    lidar_horz_anchors: int = 8
    n_layer: int = 2  # GPT layers per scale
    n_head: int = 4  # GPT attention heads
    block_exp: int = 4  # MLP expansion
    embd_pdrop: float = 0.0  # 0.0 at eval time
    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    gpt_linear_layer_init_mean: float = 0.0
    gpt_linear_layer_init_std: float = 0.02
    gpt_layer_norm_init_weight: float = 1.0

    # 3-level top-down FPN on LiDAR branch
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2
    bev_num_classes: int = 7

    # ------------------------------------------------------------------
    # Perception TransformerDecoder
    # ------------------------------------------------------------------
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0  # 0.0 at eval time
    num_bounding_boxes: int = 30

    # ------------------------------------------------------------------
    # Planner / diffusion head
    # ------------------------------------------------------------------
    num_poses: int = 8  # T = 8 waypoints (4 s at 0.5 s steps)
    ego_fut_mode: int = 20  # K = 20 anchor trajectory modes

    # DDIM noise schedule
    ddim_num_train_timesteps: int = 1000
    ddim_beta_schedule: str = "scaled_linear"
    ddim_trunc_timestep: int = 8  # noise added at t=8 at inference
    ddim_step_num: int = 2  # denoising steps
    ddim_roll_timesteps: tuple = (10, 0)  # np.arange(0,2)*10 reversed

    # ------------------------------------------------------------------
    # Assets (set at load time)
    # ------------------------------------------------------------------
    plan_anchor_path: Optional[str] = str(_DEFAULT_DATA_DIR / "kmeans_navsim_traj_20.npy")
    checkpoint_path: Optional[str] = str(_DEFAULT_DATA_DIR / "diffusiondrive_navsim.pth")

    # Use a learned LiDAR latent (replaces real LiDAR input — useful for unit tests)
    latent: bool = False

    # ------------------------------------------------------------------
    # TTNN execution options
    # ------------------------------------------------------------------
    # Set TTNN_WEIGHT_BF8=1 to store linear weights as bfloat8_b (opt-in, ~5-7% gain)
    use_bf8_weights: bool = field(default_factory=lambda: os.environ.get("TTNN_WEIGHT_BF8", "0") == "1")

    @property
    def bev_h(self) -> int:
        """Height of deepest LiDAR BEV feature (8 for 256-px input, dsf=4 then ÷8 from ResNet stride)."""
        return self.lidar_resolution_height // 32  # ResNet-34 stride-32 → 8

    @property
    def bev_w(self) -> int:
        return self.lidar_resolution_width // 32

    @property
    def num_bev_tokens(self) -> int:
        """Number of BEV key-val tokens = 8×8 + 1 status = 65."""
        return self.lidar_vert_anchors * self.lidar_horz_anchors + 1

    @property
    def num_queries(self) -> int:
        """Total perception query slots = 1 ego + num_bounding_boxes."""
        return 1 + self.num_bounding_boxes
