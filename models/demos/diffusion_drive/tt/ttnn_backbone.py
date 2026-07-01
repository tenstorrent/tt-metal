# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN TransFuser backbone — Stage 2.

Replaces the 32 ResNet-34 BasicBlock stages (16 per image/LiDAR encoder ×
4 backbone stages each) with native TTNN conv2d using BN-folded weights.

What runs in TTNN:
    All timm BasicBlock layers in image_encoder.layer{1-4} and
    lidar_encoder.layer{1-4} (conv3×3 + BN-fold + ReLU + optional 1×1).

What stays in PyTorch (TorchModuleFallback):
    • Stem: conv1 + bn1 + act1 + MaxPool2d
    • AdaptiveAvgPool2d (for GPT anchor tokens)
    • 1×1 channel-projection Conv2d (lidar↔image)
    • GPT self-attention fusion blocks
    • F.interpolate (bilinear)
    • 3-level top-down FPN (Conv2d + bilinear Upsample) — unless TtnnFPN
      is installed via build_stage3, in which case the 3 FPN conv layers
      run in TTNN and only the bilinear upsamples stay in PyTorch

Public API::
    ttnn_bb = TtnnTransfuserBackbone(ref_backbone, device)
    bev_upscale, bev_feature, _ = ttnn_bb(image, lidar)

The return signature matches TransfuserBackbone.forward() so that
_TtnnBackboneAdapter can be assigned as a drop-in for the nn.Module.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

import ttnn
from models.demos.diffusion_drive.tt.common import fold_bn
from models.demos.diffusion_drive.tt.ttnn_resnet34 import (
    _RELU_ACT,
    TtnnBasicBlock,
    _ttnn_conv2d,
    prep_conv_weights,
    prepare_resnet34_stage_params,
)

# ---------------------------------------------------------------------------
# Tensor-format helpers
# ---------------------------------------------------------------------------


def _to_ttnn_tile(x: torch.Tensor, B: int, H: int, W: int, C: int, device: ttnn.Device) -> ttnn.Tensor:
    """Convert (B, C, H, W) float32 PyTorch → (1,1,B*H*W,C) bfloat16 TILE on device."""
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    x_flat = x_nhwc.reshape(1, 1, B * H * W, C).to(torch.bfloat16)
    return ttnn.from_torch(x_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _to_host_tile(x: torch.Tensor, B: int, H: int, W: int, C: int) -> ttnn.Tensor:
    """Like ``_to_ttnn_tile`` but stays on host (no ``device=``).

    Used to refill a fixed-address pre-allocated device input via
    ``ttnn.copy_host_to_device_tensor`` for trace replay — the device tensor's
    address was bound at capture time, so we cannot allocate a fresh one.
    """
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    x_flat = x_nhwc.reshape(1, 1, B * H * W, C).to(torch.bfloat16)
    return ttnn.from_torch(x_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def _from_ttnn_tile(x_ttnn: ttnn.Tensor, B: int, H: int, W: int, C: int) -> torch.Tensor:
    """Convert (1,1,B*H*W,C) TTNN → (B, C, H, W) float32 PyTorch."""
    if x_ttnn.is_sharded():
        x_ttnn = ttnn.sharded_to_interleaved(x_ttnn, ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.to_torch(x_ttnn)  # (1, 1, B*H*W, C)
    out = out.reshape(B, H, W, C)
    return out.permute(0, 3, 1, 2).float()  # (B, C, H, W)


def _pool_out_dim(size: int, kernel: int, stride: int, padding: int) -> int:
    """Output dim of a (max/avg) pool with dilation=1 (floor mode)."""
    return (size + 2 * padding - kernel) // stride + 1


# ---------------------------------------------------------------------------
# TTNN ResNet-34 stem (conv1 + bn1 + act1 + maxpool)
# ---------------------------------------------------------------------------


class TtnnStem:
    """TTNN drop-in for a timm ResNet-34 stem: 7×7 s2 conv (BN-folded) + ReLU + 3×3 s2 maxpool.

    torch (B, Cin, H, W) → torch (B, 64, H//4, W//4).  The conv, ReLU and maxpool
    all run on-device; only the input/output torch<->ttnn boundary conversion is
    on host (consistent with the per-stage round-trips in this staged backbone).
    """

    def __init__(self, conv1: nn.Conv2d, bn1: nn.BatchNorm2d, maxpool: nn.MaxPool2d, device: ttnn.Device) -> None:
        self._device = device
        w, b = fold_bn(conv1, bn1)  # fp32 fold → bfloat16
        self._w, self._b = prep_conv_weights(w.to(torch.bfloat16), b.to(torch.bfloat16))
        self._cin = conv1.in_channels
        self._cout = conv1.out_channels
        self._k = int(conv1.kernel_size[0])
        self._s = int(conv1.stride[0])
        self._p = int(conv1.padding[0])
        self._mp_k = int(maxpool.kernel_size if isinstance(maxpool.kernel_size, int) else maxpool.kernel_size[0])
        self._mp_s = int(maxpool.stride if isinstance(maxpool.stride, int) else maxpool.stride[0])
        self._mp_p = int(maxpool.padding if isinstance(maxpool.padding, int) else maxpool.padding[0])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        xt = _to_ttnn_tile(x, B, H, W, C, self._device)
        # Cache the device-prepared stem weights (trace-safe; no per-forward H2D upload).
        out, Ho, Wo, self._w, self._b = _ttnn_conv2d(
            self._device, xt, self._w, self._b, B, H, W, C, self._cout, self._k, self._s, self._p, activation=_RELU_ACT
        )
        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.max_pool2d(
            out,
            batch_size=B,
            input_h=Ho,
            input_w=Wo,
            channels=self._cout,
            kernel_size=[self._mp_k, self._mp_k],
            stride=[self._mp_s, self._mp_s],
            padding=[self._mp_p, self._mp_p],
            dilation=[1, 1],
        )
        Hp = _pool_out_dim(Ho, self._mp_k, self._mp_s, self._mp_p)
        Wp = _pool_out_dim(Wo, self._mp_k, self._mp_s, self._mp_p)
        return _from_ttnn_tile(out, B, Hp, Wp, self._cout)


# ---------------------------------------------------------------------------
# TTNN TransFuser backbone
# ---------------------------------------------------------------------------


class TtnnTransfuserBackbone:
    """
    TTNN Stage-2 wrapper for TransfuserBackbone.

    All ResNet-34 BasicBlock stages run on-device via TtnnBasicBlock
    (TTNN conv2d + BN-folded weights). Everything else remains in PyTorch.

    Parameters
    ----------
    ref : TransfuserBackbone
        Pre-loaded, eval-mode reference backbone.
    device : ttnn.Device
        Opened Wormhole device (must have been opened with l1_small_size ≥ 32768).
    """

    def __init__(self, ref, device: ttnn.Device) -> None:
        self._ref = ref
        self._device = device

        # Pre-fold BN and build TtnnBasicBlock objects ONCE (weights are
        # converted to TTNN host tensors at construction, not per forward).
        # _img_stages[i] / _lidar_stages[i] are lists of TtnnBasicBlock for the
        # i-th ResNet-34 stage (0 = layer1, …, 3 = layer4).
        self._img_stages: List[List[TtnnBasicBlock]] = []
        self._lidar_stages: List[List[TtnnBasicBlock]] = []
        for i in range(4):
            img_layer = getattr(ref.image_encoder, f"layer{i + 1}")
            lidar_layer = getattr(ref.lidar_encoder, f"layer{i + 1}")
            self._img_stages.append(self._build_stage(prepare_resnet34_stage_params(img_layer), device))
            self._lidar_stages.append(self._build_stage(prepare_resnet34_stage_params(lidar_layer), device))

        # Stage 3: optional TTNN FPN (set by build_stage3)
        self._ttnn_fpn = None
        # Stage 3.6: optional TTNN stems + GPT cross-modal fusion (set by build_stage3_6)
        self._img_stem: Optional[TtnnStem] = None
        self._lidar_stem: Optional[TtnnStem] = None
        self._ttnn_fusion = None  # callable(image_features, lidar_features, layer_idx)
        # Stage 5: keep img/lidar feats on-device across all 4 stages + fusion
        # (removes the per-stage host round-trips). Enabled by default once
        # stems+fusion+FPN are all installed (_maybe_enable_consolidated, called
        # from build_stage3/3_6); set DD_CONSOLIDATE=0 to force the staged path.
        self._consolidated = False

        # Stage 7: optional captured trace of the [stage→fusion]×4 device loop.
        # capture_backbone_trace() pre-allocates the stem-output device inputs
        # (fixed addresses) + records the graph; run_backbone_trace() replays it.
        self._bb_trace_id = None
        self._bb_img_in = None  # fixed-address device input (image stem output)
        self._bb_lid_in = None  # fixed-address device input (lidar stem output)
        self._bb_out = None  # persistent device output (deepest LiDAR feature)
        self._bb_ish = None  # (B,H,W,C) of the lifted image input
        self._bb_lsh = None  # (B,H,W,C) of the lifted lidar input
        self._bb_out_shape = None  # (B,H,W,C) of the trace output
        self._bb_in_torch_shapes = None  # ((B,C,H,W)_img, (B,C,H,W)_lid) for host refill

    # ------------------------------------------------------------------
    # Stage 3.6 installers
    # ------------------------------------------------------------------

    def install_stems(self, device: ttnn.Device) -> None:
        """Build TTNN stems (conv1+bn1+relu+maxpool) for both encoders."""
        ref = self._ref
        self._img_stem = TtnnStem(ref.image_encoder.conv1, ref.image_encoder.bn1, ref.image_encoder.maxpool, device)
        self._lidar_stem = TtnnStem(ref.lidar_encoder.conv1, ref.lidar_encoder.bn1, ref.lidar_encoder.maxpool, device)

    def install_fusion(self, fusion) -> None:
        """Install a TTNN GPT cross-modal fusion callable (see ttnn_gpt_fusion.TtnnFuseFeatures)."""
        self._ttnn_fusion = fusion

    def enable_consolidated(self) -> None:
        """Force __call__ through the device-native consolidated path (Stage 5).

        Explicit primitive: raises if the prerequisites are missing and ignores the
        DD_CONSOLIDATE escape hatch (use this to opt in regardless of the env).
        Requires on-device stems (install_stems), TTNN fusion (install_fusion), and
        the TTNN FPN (build_stage3). Keeps img/lidar feats on device across all 4
        stage→fusion seams — only the stem input and the FPN-input boundary remain
        host hops (next increments)."""
        if self._img_stem is None or self._ttnn_fusion is None or self._ttnn_fpn is None:
            raise RuntimeError("enable_consolidated requires stems + fusion + FPN installed (build_stage3/3_6/3_7)")
        self._consolidated = True

    @staticmethod
    def _consolidation_disabled() -> bool:
        """True iff DD_CONSOLIDATE explicitly opts out of the device-native path."""
        return os.environ.get("DD_CONSOLIDATE", "1").strip().lower() in ("0", "false", "no", "off")

    def _maybe_enable_consolidated(self) -> bool:
        """Enable the consolidated path once stems+fusion+FPN are all installed.

        Called at the end of build_stage3 and build_stage3_6 so it fires whichever
        of the two prerequisites lands last (order-independent). Idempotent and
        non-raising — unlike enable_consolidated it silently no-ops when a piece is
        still missing, and it honours the DD_CONSOLIDATE=0 escape hatch so the
        staged path stays reachable for A/B without code edits. Returns whether the
        consolidated path is now active."""
        if (
            not self._consolidated
            and self._img_stem is not None
            and self._ttnn_fusion is not None
            and self._ttnn_fpn is not None
            and not self._consolidation_disabled()
        ):
            self._consolidated = True
        return self._consolidated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_stage(stage_params: List[Tuple[int, dict]], device: ttnn.Device) -> List[TtnnBasicBlock]:
        """Instantiate (and pre-convert weights for) every BasicBlock in a stage once."""
        return [TtnnBasicBlock(params, stride=stride, device=device) for stride, params in stage_params]

    def _run_ttnn_stage(
        self,
        x: torch.Tensor,
        stage_blocks: List[TtnnBasicBlock],
    ) -> torch.Tensor:
        """Convert to TTNN, run all (pre-built) BasicBlocks in one stage, convert back.

        Args:
            x:            (B, C, H, W) float32 PyTorch tensor.
            stage_blocks: list of pre-built TtnnBasicBlock for this stage.
        Returns:
            (B, C_out, H_out, W_out) float32 PyTorch tensor.
        """
        B, C, H, W = x.shape
        x_ttnn = _to_ttnn_tile(x, B, H, W, C, self._device)
        x_ttnn, shape = self._run_ttnn_stage_dev(x_ttnn, (B, H, W, C), stage_blocks)
        B_out, H_out, W_out, C_out = shape
        return _from_ttnn_tile(x_ttnn, B_out, H_out, W_out, C_out)

    @staticmethod
    def _run_ttnn_stage_dev(x_ttnn, shape, stage_blocks):
        """Device core (no host hop): run all pre-built BasicBlocks in one stage.

        Args:
            x_ttnn: (1,1,B*H*W,C) TILE device tensor.
            shape:  (B, H, W, C).
        Returns:
            (x_ttnn, shape) on device, ready for the next stage or fusion.
        """
        for block in stage_blocks:
            x_ttnn, shape = block(x_ttnn, shape)
        return x_ttnn, shape

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        image: torch.Tensor,
        lidar: torch.Tensor,
    ):
        """
        Args:
            image: (B, 3, H_cam, W_cam)   — camera feature map.
            lidar: (B, 1, H_lid, W_lid)   — LiDAR BEV; ignored if config.latent.
        Returns:
            (bev_upscale, bev_feature, None) matching TransfuserBackbone.forward().
              bev_upscale:  (B, 64, H_bev, W_bev)
              bev_feature:  (B, 512, H_bev/8, W_bev/8)
        """
        if self._consolidated:
            return self.forward_consolidated(image, lidar)

        ref = self._ref

        if ref.config.latent:
            lidar = ref.lidar_latent.expand(image.shape[0], -1, -1, -1)

        img_feats: torch.Tensor = image
        lidar_feats: torch.Tensor = lidar

        # ------------------------------------------------------------------
        # Step 1: Stem — conv1 + bn1 + act1 (PyTorch).
        # timm resnet34 features_only has 5 return_layers (start_index = 1):
        #   {'act1': '0', 'layer1': '1', …, 'layer4': '4'}
        # ------------------------------------------------------------------
        si = ref._start_index  # 1 for standard timm resnet34
        stems_on_device = self._img_stem is not None
        if si > 0 and not stems_on_device:
            img_feats = ref.image_encoder.act1(ref.image_encoder.bn1(ref.image_encoder.conv1(img_feats)))
            lidar_feats = ref.lidar_encoder.act1(ref.lidar_encoder.bn1(ref.lidar_encoder.conv1(lidar_feats)))
        elif si > 0:
            # TTNN stem already folds conv1+bn1+act1+maxpool — produces post-maxpool feature.
            img_feats = self._img_stem(img_feats)
            lidar_feats = self._lidar_stem(lidar_feats)

        # ------------------------------------------------------------------
        # Steps 2-5: Four backbone stages, each followed by GPT fusion.
        # For stage 0 (i=0) the timm iterator first runs maxpool (PyTorch),
        # unless the TTNN stem already did it on-device.
        # ------------------------------------------------------------------
        for i in range(4):
            # MaxPool is part of the stage-0 iteration in the reference model.
            if i == 0 and si > 0 and not stems_on_device:
                img_feats = ref.image_encoder.maxpool(img_feats)
                lidar_feats = ref.lidar_encoder.maxpool(lidar_feats)

            # TTNN: run layer{i+1} BasicBlocks for both encoders.
            img_feats = self._run_ttnn_stage(img_feats, self._img_stages[i])
            lidar_feats = self._run_ttnn_stage(lidar_feats, self._lidar_stages[i])

            # GPT cross-modal fusion (avgpool + 1×1 proj + attention + upsample +
            # residual add) — on TTNN if installed (Stage 3.6), else PyTorch.
            if self._ttnn_fusion is not None:
                img_feats, lidar_feats = self._ttnn_fusion(img_feats, lidar_feats, i)
            else:
                img_feats, lidar_feats = ref._fuse_features(img_feats, lidar_feats, i)

        # ------------------------------------------------------------------
        # Step 6: 3-level top-down FPN.
        # Stage 3+: TTNN conv2d (bilinear upsample stays in PyTorch).
        # Stage 2:  pure PyTorch reference.
        # ------------------------------------------------------------------
        if self._ttnn_fpn is not None:
            bev_upscale = self._ttnn_fpn(lidar_feats)
        else:
            bev_upscale = ref._top_down(lidar_feats)
        return bev_upscale, lidar_feats, None

    def forward_consolidated(self, image: torch.Tensor, lidar: torch.Tensor):
        """Device-native backbone (Stage 5): stems → [stage → fusion] × 4 on device → FPN.

        Eliminates the 8 per-stage host round-trips of __call__ (the stage↔fusion
        seams): img/lidar feats stay as on-device (1,1,B*H*W,C) tensors across all
        four stages and fusions. Only two host hops remain — lifting the stem output
        to ttnn, and lowering the deepest LiDAR feature back to torch for the FPN
        tail (FPN-on-device is the next increment). Output matches __call__:
        (bev_upscale, bev_feature, None).
        """
        ref = self._ref
        if ref.config.latent:
            lidar = ref.lidar_latent.expand(image.shape[0], -1, -1, -1)

        # Stems fold conv1+bn1+act1+maxpool on device (returns torch); lift to ttnn once.
        img_feats = self._img_stem(image)
        lidar_feats = self._lidar_stem(lidar)
        Bi, Ci, Hi, Wi = img_feats.shape
        Bl, Cl, Hl, Wl = lidar_feats.shape
        img_t = _to_ttnn_tile(img_feats, Bi, Hi, Wi, Ci, self._device)
        lid_t = _to_ttnn_tile(lidar_feats, Bl, Hl, Wl, Cl, self._device)
        img_shape, lid_shape = (Bi, Hi, Wi, Ci), (Bl, Hl, Wl, Cl)

        lid_t, lid_shape = self._run_loop_dev(img_t, img_shape, lid_t, lid_shape)

        # One host hop: deepest LiDAR feature → torch for the FPN tail.
        B, H, W, C = lid_shape
        lidar_feats = _from_ttnn_tile(lid_t, B, H, W, C)
        bev_upscale = self._ttnn_fpn(lidar_feats)
        return bev_upscale, lidar_feats, None

    def _run_loop_dev(self, img_t, img_shape, lid_t, lid_shape):
        """The device-native ``[stage → fusion] × 4`` loop, ttnn-in/ttnn-out.

        Shared by ``forward_consolidated`` (fresh per-forward inputs, fine to
        consume) and the trace path (inputs are persistent, so the traced fn
        clones them before calling this). Returns ``(lid_t, lid_shape)`` — the
        deepest LiDAR feature that feeds the FPN tail. The final image feature is
        unused downstream, so it is deallocated."""
        for i in range(4):
            img_t, img_shape = self._run_ttnn_stage_dev(img_t, img_shape, self._img_stages[i])
            lid_t, lid_shape = self._run_ttnn_stage_dev(lid_t, lid_shape, self._lidar_stages[i])
            img_t, img_shape, lid_t, lid_shape = self._ttnn_fusion.forward_dev(img_t, img_shape, lid_t, lid_shape, i)
        ttnn.deallocate(img_t)
        return lid_t, lid_shape

    # ------------------------------------------------------------------
    # Stage 7: trace capture / replay of the consolidated backbone loop
    # ------------------------------------------------------------------

    def _stem_to_device_inputs(self, image: torch.Tensor, lidar: torch.Tensor):
        """Run the stems (torch) and return their NHWC-tiled torch outputs + shapes.

        The stems themselves run on-device but return torch (post-maxpool); this
        is the boundary the trace cannot yet cross, so the stem output is what we
        lift into the fixed-address trace input."""
        ref = self._ref
        if ref.config.latent:
            lidar = ref.lidar_latent.expand(image.shape[0], -1, -1, -1)
        img_feats = self._img_stem(image)
        lidar_feats = self._lidar_stem(lidar)
        return img_feats, lidar_feats

    def capture_backbone_trace(self, image: torch.Tensor, lidar: torch.Tensor) -> None:
        """Capture the ``[stage → fusion] × 4`` device loop as a replayable trace.

        Pre-allocates the two stem-output device tensors as fixed-address trace
        inputs, double-warms the loop (so every conv kernel variant is JIT-built
        before capture), then records ``clone(inputs) → loop → deepest-LiDAR``.
        The clone inside the captured region lets ``run_backbone_trace`` refill
        the persistent inputs (a legal write outside capture) before each replay.

        Requires the consolidated path (stems + fusion + FPN installed).
        """
        if self._img_stem is None or self._ttnn_fusion is None or self._ttnn_fpn is None:
            raise RuntimeError("capture_backbone_trace requires stems + fusion + FPN (build_stage3/3_6)")

        img_feats, lidar_feats = self._stem_to_device_inputs(image, lidar)
        Bi, Ci, Hi, Wi = img_feats.shape
        Bl, Cl, Hl, Wl = lidar_feats.shape
        self._bb_img_in = _to_ttnn_tile(img_feats, Bi, Hi, Wi, Ci, self._device)
        self._bb_lid_in = _to_ttnn_tile(lidar_feats, Bl, Hl, Wl, Cl, self._device)
        self._bb_ish, self._bb_lsh = (Bi, Hi, Wi, Ci), (Bl, Hl, Wl, Cl)
        self._bb_in_torch_shapes = ((Bi, Ci, Hi, Wi), (Bl, Cl, Hl, Wl))

        def _traced():
            # clone the persistent inputs INSIDE the captured region so replay
            # reads the refilled fixed-address tensors and the loop mutates copies.
            it = ttnn.clone(self._bb_img_in)
            lt = ttnn.clone(self._bb_lid_in)
            lt, lsh = self._run_loop_dev(it, self._bb_ish, lt, self._bb_lsh)
            self._bb_out_shape = lsh
            return lt

        # Double warm-up populates the program cache for all kernel variants.
        for _ in range(2):
            tmp = _traced()
            ttnn.deallocate(tmp)
        ttnn.synchronize_device(self._device)

        try:
            self._bb_trace_id = ttnn.begin_trace_capture(self._device, cq_id=0)
            self._bb_out = _traced()
            ttnn.end_trace_capture(self._device, self._bb_trace_id, cq_id=0)
        except Exception:
            # Never leave an open trace: a leaked trace_id_ fatals every later op.
            if self._bb_trace_id is not None:
                try:
                    ttnn.release_trace(self._device, self._bb_trace_id)
                except Exception:
                    pass
            self._bb_trace_id = None
            raise

    def run_backbone_trace(self, image: torch.Tensor, lidar: torch.Tensor):
        """Replay the captured backbone loop, then run the FPN tail on host hop.

        Refills the fixed-address trace inputs with this call's stem outputs,
        executes the trace, lowers the deepest LiDAR feature to torch and runs the
        (not-yet-traced) FPN. Output matches ``forward_consolidated``:
        ``(bev_upscale, bev_feature, None)``."""
        if self._bb_trace_id is None:
            raise RuntimeError("run_backbone_trace called before capture_backbone_trace")

        img_feats, lidar_feats = self._stem_to_device_inputs(image, lidar)
        (Bi, Ci, Hi, Wi), (Bl, Cl, Hl, Wl) = self._bb_in_torch_shapes
        # Refill the persistent device inputs (legal H2D write, outside capture).
        ttnn.copy_host_to_device_tensor(_to_host_tile(img_feats, Bi, Hi, Wi, Ci), self._bb_img_in)
        ttnn.copy_host_to_device_tensor(_to_host_tile(lidar_feats, Bl, Hl, Wl, Cl), self._bb_lid_in)

        ttnn.execute_trace(self._device, self._bb_trace_id, cq_id=0, blocking=True)

        B, H, W, C = self._bb_out_shape
        lidar_out = _from_ttnn_tile(self._bb_out, B, H, W, C)
        bev_upscale = self._ttnn_fpn(lidar_out)
        return bev_upscale, lidar_out, None

    def release_backbone_trace(self) -> None:
        """Release the captured trace (frees the trace region)."""
        if self._bb_trace_id is not None:
            try:
                ttnn.release_trace(self._device, self._bb_trace_id)
            finally:
                self._bb_trace_id = None


# ---------------------------------------------------------------------------
# Drop-in nn.Module adapter
# ---------------------------------------------------------------------------


class _TtnnBackboneAdapter(nn.Module):
    """Thin nn.Module wrapper so TtnnTransfuserBackbone can be assigned
    directly to ``DiffusionDriveModel._backbone``."""

    def __init__(self, ttnn_backbone: TtnnTransfuserBackbone) -> None:
        super().__init__()
        self._ttnn = ttnn_backbone
        # Forward config so downstream code (FPN size queries) still works.
        self.config = ttnn_backbone._ref.config

    def forward(self, image: torch.Tensor, lidar: torch.Tensor):
        return self._ttnn(image, lidar)
