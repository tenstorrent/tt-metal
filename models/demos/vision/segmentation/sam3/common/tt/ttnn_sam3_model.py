# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end SAM3 image inference pipeline on ttnn.

This module wires together all SAM3 components for image-level inference:
  Image → ViT Backbone → FPN Neck → (Text Encoder) → Transformer Encoder → Decoder → Segmentation

For the initial implementation, many components run on CPU (PyTorch fallback).
The ViT backbone uses ttnn for linear projections. Other components will be
progressively moved to ttnn as support matures.
"""

import os
import sys
from typing import Dict, List, Optional

import torch
import ttnn

# Ensure sam3 is importable
_venv_path = os.environ.get("SAM3_VENV_PATH", os.path.join(os.path.expanduser("~"), ".tenstorrent-venv/lib/python3.12/site-packages"))
if _venv_path not in sys.path:
    sys.path.insert(0, _venv_path)


def build_sam3_reference_model():
    """Build the SAM3 PyTorch reference model on CPU.

    Returns:
        SAM3 model on CPU in eval mode.
    """
    import unittest.mock as mock

    # Patch CUDA tensor allocations to CPU
    orig = {
        n: getattr(torch, n)
        for n in [
            "zeros", "ones", "arange", "empty", "full",
            "randn", "rand", "tensor", "linspace", "logspace", "eye",
        ]
    }

    def _redirect(fn):
        def wrapper(*args, **kwargs):
            if "device" in kwargs:
                dev = kwargs["device"]
                if dev is not None and "cuda" in str(dev):
                    kwargs["device"] = "cpu"
            return fn(*args, **kwargs)
        return wrapper

    patches = [mock.patch("torch.cuda.is_available", return_value=False)]
    for name, fn in orig.items():
        patches.append(mock.patch(f"torch.{name}", _redirect(fn)))

    for p in patches:
        p.start()

    try:
        from sam3.model_builder import build_sam3_image_model

        bpe_path = os.environ.get("SAM3_BPE_PATH", os.path.join(os.environ.get("TT_METAL_HOME", os.path.expanduser("~/tt-metal")), "python_env/lib/python3.12/site-packages/open_clip/bpe_simple_vocab_16e6.txt.gz"))
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device="cpu",
            eval_mode=True,
            load_from_HF=False,
            checkpoint_path=None,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
    finally:
        for p in patches:
            p.stop()

    return model


class TtSam3ImagePipeline:
    """End-to-end SAM3 image inference pipeline.

    Combines ttnn-accelerated ViT backbone with CPU-based remaining components.
    """

    def __init__(self, sam3_model, device):
        """Initialize the pipeline.

        Args:
            sam3_model: PyTorch SAM3 model (from build_sam3_reference_model or checkpoint).
            device: ttnn device handle.
        """
        from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import (
            preprocess_vit_backbone_weights,
            move_backbone_params_to_device,
        )
        from sam3.model.vitdet import ViT

        self.device = device
        self.sam3_model = sam3_model

        # Extract and preprocess ViT backbone weights for ttnn
        vit_backbone = None
        for _, module in sam3_model.named_modules():
            if isinstance(module, ViT):
                vit_backbone = module
                break

        if vit_backbone is None:
            raise RuntimeError("Could not find ViT backbone in SAM3 model")

        self.vit_backbone_ref = vit_backbone
        self.backbone_params = preprocess_vit_backbone_weights(vit_backbone)
        self.backbone_params = move_backbone_params_to_device(self.backbone_params, device)

        # Keep other components as PyTorch modules (CPU)
        self.neck = sam3_model.backbone.visual if hasattr(sam3_model.backbone, "visual") else None
        self.text_encoder = sam3_model.backbone.text if hasattr(sam3_model.backbone, "text") else None
        self.transformer = sam3_model.transformer
        self.segmentation_head = sam3_model.segmentation_head
        self.geometry_encoder = sam3_model.geometry_encoder
        self.dot_prod_scoring = getattr(sam3_model, "dot_prod_scoring", None)

    def run_vit_backbone(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        """Run the ViT backbone on ttnn device.

        Args:
            pixel_values: (B, 3, 1008, 1008) preprocessed image tensor.

        Returns:
            List of feature tensors [(B, 1024, 72, 72)] in NCHW format.
        """
        from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_vitdet import tt_vit_backbone

        return tt_vit_backbone(pixel_values, self.backbone_params, self.device)

    def run_backbone_and_neck(self, pixel_values: torch.Tensor) -> Dict:
        """Run ViT backbone + FPN neck.

        Uses ttnn for ViT backbone, CPU for neck convolutions.

        Args:
            pixel_values: (B, 3, 1008, 1008) preprocessed image tensor.

        Returns:
            Dict with 'backbone_fpn' (list of feature maps) and 'vision_pos_enc'.
        """
        # Run ViT backbone on ttnn
        vit_features = self.run_vit_backbone(pixel_values)
        vit_feat = vit_features[-1]  # (B, 1024, 72, 72)

        # Run FPN neck on CPU
        from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_neck import (
            tt_fpn_neck,
            preprocess_neck_weights,
        )
        from sam3.model.necks import Sam3DualViTDetNeck

        neck_module = None
        for _, module in self.sam3_model.named_modules():
            if isinstance(module, Sam3DualViTDetNeck):
                neck_module = module
                break

        if neck_module is None:
            raise RuntimeError("Could not find neck in SAM3 model")

        neck_params = preprocess_neck_weights(neck_module)
        return tt_fpn_neck(vit_feat, neck_params, self.device)

    def run_text_encoding(self, text_prompts: List[str]) -> Optional[torch.Tensor]:
        """Run text encoder on CPU.

        Args:
            text_prompts: List of text strings.

        Returns:
            Text features tensor or None if no text encoder.
        """
        if self.text_encoder is None:
            return None

        with torch.no_grad():
            text_features = self.text_encoder(text_prompts)
        return text_features

    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,
        text_prompts: Optional[List[str]] = None,
    ) -> Dict:
        """Run full SAM3 image inference.

        Args:
            pixel_values: (B, 3, 1008, 1008) preprocessed image tensor.
            text_prompts: Optional list of text prompts for open-vocabulary detection.

        Returns:
            Dict with:
                'vit_features': ViT backbone output
                'fpn_features': FPN multi-scale features
                (additional outputs added as more components are integrated)
        """
        results = {}

        # Step 1: ViT backbone (ttnn)
        vit_features = self.run_vit_backbone(pixel_values)
        results["vit_features"] = vit_features

        # Step 2: FPN neck (CPU)
        backbone_output = self.run_backbone_and_neck(pixel_values)
        results["fpn_features"] = backbone_output

        # Step 3: Text encoding (CPU, optional)
        if text_prompts is not None:
            text_features = self.run_text_encoding(text_prompts)
            results["text_features"] = text_features

        # Steps 4-7: Transformer + Segmentation (CPU fallback)
        # TODO: Integrate transformer encoder/decoder and segmentation head
        # For now, return intermediate features for downstream processing

        return results


def preprocess_image(image: torch.Tensor, target_size: int = 1008) -> torch.Tensor:
    """Preprocess image for SAM3 inference.

    Args:
        image: (B, C, H, W) or (C, H, W) tensor in [0, 255] or [0, 1] range.
        target_size: Target image size (SAM3 uses 1008).

    Returns:
        Preprocessed tensor (B, 3, target_size, target_size) normalized to [-1, 1].
    """
    import torch.nn.functional as F

    if image.ndim == 3:
        image = image.unsqueeze(0)

    if image.max() > 1.0:
        image = image.float() / 255.0

    if image.shape[-2:] != (target_size, target_size):
        image = F.interpolate(
            image, size=(target_size, target_size), mode="bilinear", align_corners=False
        )

    # SAM3 normalization: mean=0.5, std=0.5
    image = (image - 0.5) / 0.5
    return image
