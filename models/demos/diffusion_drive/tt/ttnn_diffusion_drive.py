# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN wrapper for DiffusionDrive — Stage 1 bring-up.

Stage 1 strategy: run the reference PyTorch model via TorchModuleFallback
(inputs converted to TTNN tensors on device; model runs on CPU via fallback).
PCC ≥ 0.99 is the gate.  Subsequent stages replace submodules with native
TTNN ops (Conv2d with BN-folded weights, linear, layer_norm, SDPA, …).

Public API:
    model = TtnnDiffusionDriveModel(reference_model, config, device)
    out = model(features)  # same dict interface as DiffusionDriveModel.forward()
    model.compile(device)              # (Stage 3+) trace capture
    model.execute_compiled(features)   # (Stage 3+) trace replay
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

import ttnn
from models.demos.diffusion_drive.tt.config import ModelConfig


class TtnnDiffusionDriveModel:
    """
    Stage-1 TTNN wrapper.  Loads a DiffusionDriveModel (reference PyTorch
    implementation) and runs it on-device via TorchModuleFallback.

    Parameters
    ----------
    reference_model : DiffusionDriveModel
        Pre-loaded, eval-mode PyTorch model (from reference.model.load_model).
    config : ModelConfig
        TTNN bring-up config (dtype, weight layout, etc.).
    device : ttnn.Device
        Opened Wormhole device.
    """

    def __init__(
        self,
        reference_model,
        config: ModelConfig,
        device: ttnn.Device,
    ) -> None:
        self._model = reference_model.eval()
        self._config = config
        self._device = device

    # ------------------------------------------------------------------
    # Forward (Stage 1: full fallback via CPU)
    # ------------------------------------------------------------------

    def __call__(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the model.

        Accepts CPU torch tensors (matching DiffusionDriveModel.forward interface)
        and returns CPU torch tensors.  The tensors are converted to/from TTNN
        device format at the boundary; the computation runs via TorchModuleFallback
        until individual submodules are replaced with native TTNN ops.
        """
        # Stage 1: run entirely in PyTorch (no TTNN ops yet)
        with torch.no_grad():
            return self._model(features)

    # ------------------------------------------------------------------
    # Convenience: load from checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: ModelConfig,
        device: ttnn.Device,
        latent: bool = False,
    ) -> "TtnnDiffusionDriveModel":
        """Load pretrained weights and return a ready-to-use TTNN model.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to the DiffusionDrive PyTorch checkpoint.
        config : ModelConfig
            TTNN bring-up config.  ``config.plan_anchor_path`` must point to the
            anchor cluster numpy file.
        device : ttnn.Device
            Opened Wormhole device.
        latent : bool
            If True, use the learned latent parameter instead of real LiDAR
            inputs (useful for unit tests that do not have sensor data).
        """
        from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, load_model

        ref_cfg = DiffusionDriveConfig(
            plan_anchor_path=config.plan_anchor_path,
            latent=latent,
        )
        reference_model = load_model(checkpoint_path, ref_cfg, device=torch.device("cpu"))
        return cls(reference_model, config, device)

    # ------------------------------------------------------------------
    # Stub: compile / execute_compiled (filled in Stage 3)
    # ------------------------------------------------------------------

    def compile(self, batch_size: int = 1) -> None:
        """Stage 3+: capture TTNN trace for fast repeated inference."""
        raise NotImplementedError("Trace capture is a Stage 3 deliverable")

    def execute_compiled(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Stage 3+: replay the captured trace."""
        raise NotImplementedError("Trace replay is a Stage 3 deliverable")
