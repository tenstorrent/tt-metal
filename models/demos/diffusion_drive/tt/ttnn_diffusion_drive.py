# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN wrapper for DiffusionDrive.

Stage 1: run the reference PyTorch model entirely on CPU (TorchModuleFallback).
Stage 2: replace ResNet-34 BasicBlock stages with native TTNN conv2d ops via
         TtnnTransfuserBackbone; GPT fusion, FPN, perception decoder, and
         trajectory head remain in PyTorch.

Public API:
    model = TtnnDiffusionDriveModel(reference_model, config, device)
    out = model(features)              # Stage-1 forward (full PyTorch)
    model.build_stage2(device)         # install TTNN backbone in-place
    out = model(features)              # Stage-2 forward (TTNN backbone)
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
    # Stage 2: install TTNN backbone
    # ------------------------------------------------------------------

    def build_stage2(self, device: ttnn.Device) -> "TtnnDiffusionDriveModel":
        """Replace the PyTorch backbone with the TTNN Stage-2 implementation.

        Swaps ``DiffusionDriveModel._backbone`` for a ``_TtnnBackboneAdapter``
        wrapping a ``TtnnTransfuserBackbone``.  All downstream modules (GPT
        fusion, FPN, perception decoder, trajectory head) remain in PyTorch.

        After this call, ``__call__`` automatically runs the TTNN backbone.
        Returns self for chaining.
        """
        from models.demos.diffusion_drive.tt.ttnn_backbone import TtnnTransfuserBackbone, _TtnnBackboneAdapter

        ttnn_bb = TtnnTransfuserBackbone(self._model._backbone, device)
        self._model._backbone = _TtnnBackboneAdapter(ttnn_bb)
        return self

    # ------------------------------------------------------------------
    # Forward (Stage 1 / Stage 2)
    # ------------------------------------------------------------------

    def __call__(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the model.

        Stage 1 (before build_stage2): runs entirely on CPU via PyTorch.
        Stage 2+ (after build_stage2): ResNet-34 BasicBlock stages run on
        the TTNN device; everything else stays on CPU.
        """
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
