# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN wrapper for DiffusionDrive.

Stage 1: run the reference PyTorch model entirely on CPU (TorchModuleFallback).
Stage 2: replace ResNet-34 BasicBlock stages with native TTNN conv2d ops via
         TtnnTransfuserBackbone; GPT fusion, FPN, perception decoder, and
         trajectory head remain in PyTorch.
Stage 3: additionally replace the 3-level FPN (c5_conv + up_conv5 + up_conv4)
         with native TTNN conv2d ops via TtnnFPN; bilinear upsampling stays in
         PyTorch (ttnn.upsample does not support bilinear at these scales).

Public API:
    model = TtnnDiffusionDriveModel(reference_model, config, device)
    out = model(features)              # Stage-1 forward (full PyTorch)
    model.build_stage2(device)         # install TTNN backbone in-place
    out = model(features)              # Stage-2 forward (TTNN backbone)
    model.build_stage3(device)         # install TTNN FPN in-place (requires Stage 2)
    out = model(features)              # Stage-3 forward (TTNN backbone + TTNN FPN)
    model.build_stage3_6(device)       # stems+fusion → backbone goes device-native
                                       #   (consolidated path auto-enabled here)
    model.build_stage5(device)         # explicit/redundant: force consolidated path
    model.compile(device)              # (Stage 4+) trace capture
    model.execute_compiled(features)   # (Stage 4+) trace replay

Once build_stage3 (FPN) and build_stage3_6 (stems+fusion) have both run the
TransFuser backbone runs as one device-native graph by default (no per-stage host
round-trips); set DD_CONSOLIDATE=0 to fall back to the staged path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

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
        # Stage 4: consolidated on-device perception forward (set by build_stage4)
        self._perception = None
        # Stage 7: True once compile() has captured the backbone-loop trace.
        self._compiled = False

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
    # Stage 3: install TTNN FPN (requires Stage 2 first)
    # ------------------------------------------------------------------

    def build_stage3(self, device: ttnn.Device) -> "TtnnDiffusionDriveModel":
        """Replace the PyTorch FPN with the TTNN Stage-3 implementation.

        Installs ``TtnnFPN`` into the ``TtnnTransfuserBackbone`` so that the
        three FPN conv2d layers (c5_conv, up_conv5, up_conv4) run on-device.
        Bilinear upsampling steps remain in PyTorch.

        Must be called after ``build_stage2``.  Returns self for chaining.
        """
        if not hasattr(self._model._backbone, "_ttnn"):
            raise RuntimeError("build_stage3 requires build_stage2 to be called first")

        from models.demos.diffusion_drive.tt.ttnn_fpn import TtnnFPN

        ref_backbone = self._model._backbone._ttnn._ref
        ttnn_fpn = TtnnFPN(ref_backbone, device)
        self._model._backbone._ttnn._ttnn_fpn = ttnn_fpn
        # FPN was the last consolidation prerequisite if stems+fusion (3_6) already
        # ran; flip the backbone to the device-native path (no-op until 3_6 lands).
        self._model._backbone._ttnn._maybe_enable_consolidated()
        return self

    # ------------------------------------------------------------------
    # Stage 3.6: install TTNN ResNet stems + GPT cross-modal fusion
    # ------------------------------------------------------------------

    def build_stage3_6(self, device: ttnn.Device) -> "TtnnDiffusionDriveModel":
        """Replace the ResNet-34 stems and the 4× GPT cross-modal fusion with TTNN.

        After this call the TransFuser backbone runs **entirely** on-device:
        stems (conv1+bn1+relu+maxpool), all BasicBlocks (Stage 2), the FPN
        (Stage 3) and the cross-modal fusion (avg_pool2d + 1×1 channel projections
        + GPT transformer + bilinear upsample + residual) all execute via TTNN ops.

        Valid only at the production input resolution (camera 256×1024, LiDAR
        256×256), where the adaptive-pool and bilinear-upsample ratios are integer
        — the fusion asserts this and raises otherwise.  Requires ``build_stage2``.
        Chainable.  Returns self.
        """
        if not hasattr(self._model._backbone, "_ttnn"):
            raise RuntimeError("build_stage3_6 requires build_stage2 to be called first")

        from models.demos.diffusion_drive.tt.ttnn_gpt_fusion import TtnnFuseFeatures

        ttnn_bb = self._model._backbone._ttnn
        ttnn_bb.install_stems(device)
        ttnn_bb.install_fusion(TtnnFuseFeatures(ttnn_bb._ref, device))
        # Stems+fusion were the last consolidation prerequisites if the FPN (3) is
        # installed; flip the backbone to the device-native path by default here
        # (no-op if build_stage3 hasn't run, or if DD_CONSOLIDATE=0).
        ttnn_bb._maybe_enable_consolidated()
        return self

    # ------------------------------------------------------------------
    # Stage 3.4: install TTNN perception head (drop-in submodule swaps)
    # ------------------------------------------------------------------

    def build_stage3_4(self, device: ttnn.Device) -> "TtnnDiffusionDriveModel":
        """Replace the perception-head ops with TTNN drop-ins.

        Swaps four submodules of ``DiffusionDriveModel`` in place:
          ``_bev_downscale`` (1×1 conv), ``_status_encoding`` (Linear),
          ``bev_proj`` (Linear+ReLU+LN), and ``_tf_decoder`` (3-layer
          TransformerDecoder).  Each TTNN drop-in keeps the original call
          signature, so ``DiffusionDriveModel.forward`` is untouched.

        Chainable; typically called after ``build_stage3``.  Returns self.
        """
        from models.demos.diffusion_drive.tt.ttnn_perception import (
            TtnnBevProj,
            TtnnConv1x1,
            TtnnLinear,
            TtnnTransformerDecoder,
        )

        m = self._model
        m._bev_downscale = TtnnConv1x1(m._bev_downscale, device)
        m._status_encoding = TtnnLinear(m._status_encoding, device)
        m.bev_proj = TtnnBevProj(m.bev_proj, device)
        m._tf_decoder = TtnnTransformerDecoder(m._tf_decoder, device)
        return self

    # ------------------------------------------------------------------
    # Stage 3.5: install TTNN trajectory-head denoiser (drop-in submodule swaps)
    # ------------------------------------------------------------------

    def build_stage3_5(self, device: ttnn.Device) -> "TtnnDiffusionDriveModel":
        """Replace the TrajectoryHead denoiser's weight-bearing modules with TTNN
        drop-ins (plan_anchor_encoder, time_mlp, and per decoder layer: grid-sample
        cross-attention, the two MHAs, FFN, norms, FiLM modulation, task heads).

        The DDIM ``scheduler.step``, sinusoidal embed and norm/denorm glue stay on
        host (Stage-3.7 consolidation target).  Chainable.  Returns self.
        """
        from models.demos.diffusion_drive.tt.ttnn_trajectory import install_ttnn_trajectory_head

        install_ttnn_trajectory_head(self._model._trajectory_head, device)
        return self

    # ------------------------------------------------------------------
    # Stage 3.7: install TTNN agent head (last weight-bearing fallback)
    # ------------------------------------------------------------------

    def build_stage3_7(self, device: ttnn.Device) -> "TtnnDiffusionDriveModel":
        """Replace the AgentHead's two MLPs with TTNN drop-ins.

        After this call every weight-bearing op in the model runs on TTNN.  The
        agent head does not feed the trajectory/scores; the only host residue
        left is non-weight scalar/control-flow glue (DDIM ``scheduler.step``,
        ``gen_sineembed`` trig, norm/denorm, ``argmax``/``gather`` best-mode
        select, per-index tanh scaling) — see ``tt/ttnn_trajectory.py`` for the
        rationale on what cannot become a tensor op.
        Chainable.  Returns self.
        """
        from models.demos.diffusion_drive.tt.ttnn_trajectory import install_ttnn_agent_head

        install_ttnn_agent_head(self._model._agent_head, device)
        return self

    # ------------------------------------------------------------------
    # Stage 4: single-graph consolidation — perception path
    # ------------------------------------------------------------------

    def build_stage4(self, device: ttnn.Device) -> "TtnnDiffusionDriveModel":
        """Single-graph consolidation — collapse the per-drop-in host round-trips.

        Two consolidations:
          1. **Perception** (``DiffusionDriveModel.forward`` lines 955–984) into one
             on-device graph, running the ``concat_cross_bev`` bilinear interpolate
             (``reference/model.py:974`` — the last non-glue PyTorch compute op) on
             device via ``ttnn.upsample``.  ``__call__`` then routes through
             ``_forward_ttnn``.
          2. **Diffusion decoder** — replace the Stage-3.5 ``CustomTransformerDecoder``
             with ``TtnnDiffDecoder``, which keeps ``traj_feature`` on device across
             each layer (the ~9 per-drop-in round-trips collapse to one layer
             boundary).

        Requires ``build_stage3_4`` (perception drop-ins) and ``build_stage3_5``
        (trajectory drop-ins) — their prepared weights are reused.  Valid only at
        production resolution (bev 8×8 → 64×64).  Chainable.  Returns self.
        """
        from models.demos.diffusion_drive.tt.ttnn_consolidated import TtnnPerceptionForward
        from models.demos.diffusion_drive.tt.ttnn_perception import TtnnConv1x1
        from models.demos.diffusion_drive.tt.ttnn_trajectory import TtnnSequentialMLP, install_ttnn_diff_decoder

        m = self._model
        if not isinstance(m._bev_downscale, TtnnConv1x1):
            raise RuntimeError("build_stage4 requires build_stage3_4 to be called first")

        # 1. perception path
        self._perception = TtnnPerceptionForward(m, device)

        # 2. diffusion decoder (requires the Stage-3.5 per-layer drop-ins)
        th = m._trajectory_head
        if not isinstance(th.diff_decoder.layers[0].ffn, TtnnSequentialMLP):
            raise RuntimeError("build_stage4 requires build_stage3_5 to be called first")
        install_ttnn_diff_decoder(th, device)
        return self

    # ------------------------------------------------------------------
    # Stage 5: device-native backbone (consolidated stage→fusion chaining)
    # ------------------------------------------------------------------

    def build_stage5(self, device: ttnn.Device) -> "TtnnDiffusionDriveModel":
        """Route the TransFuser backbone through the device-native consolidated path.

        Keeps the image/LiDAR feature maps on-device across all four ResNet-34
        stages and the 4× GPT fusion, removing the 8 per-stage host round-trips of
        the staged path (the prerequisite for whole-model trace capture).

        This is already enabled **by default** once ``build_stage3`` (FPN) and
        ``build_stage3_6`` (stems + fusion) have both run — the production server
        and full-stack tests pick it up automatically — so calling it explicitly is
        usually redundant. It is kept as a named, raising entry point (and for
        forcing the path even when ``DD_CONSOLIDATE=0`` would otherwise opt out).
        Requires ``build_stage2``/``3``/``3_6``.  Chainable.  Returns self.
        """
        if not hasattr(self._model._backbone, "_ttnn"):
            raise RuntimeError("build_stage5 requires build_stage2/3/3_6 to be called first")
        self._model._backbone._ttnn.enable_consolidated()
        return self

    # ------------------------------------------------------------------
    # Forward (Stage 1 / Stage 2 / Stage 3 / Stage 4)
    # ------------------------------------------------------------------

    def __call__(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the model.

        Stage 1 (before build_stage2): runs entirely on CPU via PyTorch.
        Stage 2+ (after build_stage2): ResNet-34 BasicBlock stages run on
        the TTNN device; everything else stays on CPU.
        Stage 4 (after build_stage4): the perception path runs as one on-device
        graph via ``_forward_ttnn``.
        """
        with torch.no_grad():
            if self._perception is not None:
                return self._forward_ttnn(features)
            return self._model(features)

    def _forward_ttnn(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Consolidated forward: backbone → on-device perception → trajectory/agent heads.

        Mirrors ``DiffusionDriveModel.forward`` but replaces lines 955–984 (the
        perception pre-decoder block, incl. the ``concat_cross_bev`` interpolate)
        with the single-graph ``TtnnPerceptionForward``.  The trajectory and agent
        heads are the existing Stage-3.5/3.7 drop-ins, called as in the reference.
        """
        m = self._model
        camera = features["camera_feature"]
        lidar = features["lidar_feature"]
        status = features["status_feature"]

        bev_upscale, bev_feature, _ = m._backbone(camera, lidar)
        bev_spatial_shape = bev_upscale.shape[2:]

        traj_query, agents_query, cross_bev_feature, status_enc = self._perception(bev_upscale, bev_feature, status)

        output = m._trajectory_head(traj_query, agents_query, cross_bev_feature, bev_spatial_shape, status_enc)
        agents = m._agent_head(agents_query)
        output.update(agents)
        return output

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
    # Stage 7: compile / execute_compiled (backbone-loop trace)
    # ------------------------------------------------------------------

    @staticmethod
    def _dummy_features(batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Production-resolution random features for trace capture (see DD-4 sizes)."""
        return {
            "camera_feature": torch.randn(batch_size, 3, 256, 1024),
            "lidar_feature": torch.randn(batch_size, 1, 256, 256),
            "status_feature": torch.randn(batch_size, 8),
        }

    def compile(self, features: Optional[Dict[str, torch.Tensor]] = None, batch_size: int = 1) -> None:
        """Capture a TTNN trace of the consolidated backbone loop for fast replay.

        The ``[stage → fusion] × 4`` backbone loop is the bulk of the per-forward
        host op dispatch; capturing it collapses those dispatches into one
        ``execute_trace`` (measured ~1.76× on the backbone alone). The stems, FPN
        tail, perception and heads are not yet trace-legal and still run per-op,
        so ``execute_compiled`` replays only the loop and runs the rest eagerly —
        the output is identical to ``__call__`` (PCC ≈ 1.0).

        Requires ``build_stage4`` (consolidated perception) and the consolidated
        backbone (``build_stage3``/``build_stage3_6``; ``DD_CONSOLIDATE`` not 0).
        ``features`` is a representative sample used to size/shape the fixed
        trace inputs; if omitted a production-resolution dummy is synthesised
        (camera 256×1024, LiDAR 256×256).
        """
        if self._perception is None:
            raise RuntimeError("compile() requires build_stage4 (consolidated perception path)")
        if not hasattr(self._model._backbone, "_ttnn"):
            raise RuntimeError("compile() requires build_stage2..build_stage4 first")
        bb = self._model._backbone._ttnn
        if not bb._consolidated:
            raise RuntimeError("compile() requires the consolidated backbone (do not set DD_CONSOLIDATE=0)")
        if features is None:
            features = self._dummy_features(batch_size)
        with torch.no_grad():
            # Warm up the FULL eager forward TWICE before capture so every kernel
            # (stem, FPN, perception, heads — and all variants) is JIT-compiled and
            # program-cached first. Compiling a NEW device program while a trace is
            # active hangs the device; execute_compiled's eager tail still allocates
            # buffers per forward (only a warning), but must never trigger a JIT.
            self._forward_ttnn(features)
            self._forward_ttnn(features)
            bb.capture_backbone_trace(features["camera_feature"], features["lidar_feature"])
            # Stage 8: also capture the perception forward as a second trace. Feed it
            # a real sample from the (now captured) backbone trace so its fixed-address
            # inputs are sized correctly. Its own double warm-up JITs any remaining
            # kernel before begin_trace_capture.
            bev_upscale, bev_feature, _ = bb.run_backbone_trace(features["camera_feature"], features["lidar_feature"])
            self._perception.capture_trace(bev_upscale, bev_feature, features["status_feature"])
        self._compiled = True

    def execute_compiled(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Replay the captured backbone trace, then run FPN/perception/heads eagerly.

        Mirrors ``_forward_ttnn`` exactly but replaces the eager backbone call
        with ``run_backbone_trace`` (trace replay). Call ``compile()`` first."""
        if not self._compiled:
            raise RuntimeError("call compile() before execute_compiled()")
        with torch.no_grad():
            m = self._model
            camera = features["camera_feature"]
            lidar = features["lidar_feature"]
            status = features["status_feature"]

            bev_upscale, bev_feature, _ = m._backbone._ttnn.run_backbone_trace(camera, lidar)
            bev_spatial_shape = bev_upscale.shape[2:]

            traj_query, agents_query, cross_bev_feature, status_enc = self._perception.run_trace(
                bev_upscale, bev_feature, status
            )

            output = m._trajectory_head(traj_query, agents_query, cross_bev_feature, bev_spatial_shape, status_enc)
            agents = m._agent_head(agents_query)
            output.update(agents)
            return output

    def release_compiled(self) -> None:
        """Release the captured backbone + perception traces (frees the trace region)."""
        if self._compiled and hasattr(self._model._backbone, "_ttnn"):
            self._model._backbone._ttnn.release_backbone_trace()
            if self._perception is not None:
                self._perception.release_trace()
        self._compiled = False
