# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import PIL.Image
import torch

from .config import DPTLargeConfig, DEFAULT_CONFIG

try:
    from transformers import DPTConfig, DPTForDepthEstimation, DPTImageProcessor
except Exception:  # pragma: no cover - transformers might be missing in dev env
    DPTConfig = None
    DPTForDepthEstimation = None
    DPTImageProcessor = None

LOG = logging.getLogger(__name__)


def _require_transformers():
    if DPTConfig is None or DPTForDepthEstimation is None:
        raise ImportError(
            "transformers is required for the DPT fallback pipeline. "
            "Install with `pip install transformers>=4.38 pillow`."
        )


def default_image_processor(model_name: str):
    if DPTImageProcessor is None:
        return None
    try:
        return DPTImageProcessor.from_pretrained(model_name)
    except Exception:
        # Fall back to manual preprocessing if model assets are unavailable.
        LOG.warning("Could not load DPTImageProcessor for %s; using manual preprocessing.", model_name)
        return None


def manual_preprocess(img: PIL.Image.Image, size: Tuple[int, int]) -> torch.Tensor:
    """Lightweight preprocessing mirroring HF defaults."""
    img = img.convert("RGB").resize(size, resample=PIL.Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    arr = (arr - mean) / std
    # HWC -> CHW
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


@dataclass
class DPTFallbackPipeline:
    """
    CPU reference pipeline using Hugging Face `DPTForDepthEstimation`.

    This is intentionally lightweight so it can run in environments without TT
    hardware and serves as the numerical reference for PCC tests.
    """

    # Use default_factory to avoid sharing a mutable config instance across
    # pipelines and to keep Python 3.11+ dataclass semantics happy.
    config: DPTLargeConfig = field(default_factory=DPTLargeConfig)
    pretrained: bool = True
    device: str = "cpu"
    _model: Optional[torch.nn.Module] = None
    _processor: Optional[object] = None

    def __post_init__(self):
        _require_transformers()

        if self._model is None:
            if self.pretrained:
                self._model = DPTForDepthEstimation.from_pretrained(
                    self.config.model_name,
                    output_hidden_states=True,
                )
            else:
                hf_cfg = DPTConfig(**self.config.to_hf_kwargs())
                # Ensure backbone returns per-layer hidden states for HF neck
                hf_cfg.output_hidden_states = True
                self._model = DPTForDepthEstimation(hf_cfg)
        self._model.to(self.device)
        self._model.eval()

        if self._processor is None:
            self._processor = default_image_processor(self.config.model_name)

    # ------------------------------------------------------------------ utils
    def _prepare(self, image_path: str) -> torch.Tensor:
        img = PIL.Image.open(image_path)
        target_size = (self.config.image_size, self.config.image_size)
        if self._processor is not None:
            proc_out = self._processor(images=img, return_tensors="pt", size=target_size)
            pixel_values = proc_out["pixel_values"]
        else:
            pixel_values = manual_preprocess(img, target_size)
        return pixel_values.to(self.device)

    def _forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # HF-like forward with tiny-config guard for taps
        with torch.no_grad():
            dpt = getattr(self._model, "dpt", None)
            if dpt is not None:
                outputs = dpt(
                    pixel_values=pixel_values,
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states
                cfg = self._model.config
                taps = [feat for idx, feat in enumerate(hidden_states[1:]) if idx in cfg.backbone_out_indices]
                if len(taps) == 0:
                    out = self._model(pixel_values=pixel_values, output_hidden_states=True)
                    depth = out.predicted_depth
                else:
                    need = len(cfg.neck_hidden_sizes)
                    while len(taps) < need:
                        taps.append(taps[-1])
                    # compute patch grid dims as HF does for non-hybrid
                    _, _, H, W = pixel_values.shape
                    patch = getattr(cfg, "patch_size", 16)
                    patch_h = H // patch
                    patch_w = W // patch
                    neck_feats = self._model.neck(taps, patch_h, patch_w)
                    depth = self._model.head(neck_feats)
            else:
                out = self._model(pixel_values=pixel_values, output_hidden_states=True)
                depth = out.predicted_depth
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
        return depth

    @staticmethod
    def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
        # Support shapes [B,1,H,W] or [B,H,W]
        if depth.dim() == 4:
            dims = [1, 2, 3]
        elif depth.dim() == 3:
            dims = [1, 2]
        else:
            dims = list(range(1, depth.dim()))
        min_d = depth.amin(dim=dims, keepdim=True)
        max_d = depth.amax(dim=dims, keepdim=True)
        norm = (depth - min_d) / (max_d - min_d + 1e-8)
        return norm

    # ------------------------------------------------------------------ public
    def run_depth_cpu(self, image_path: str, normalize: bool = True) -> np.ndarray:
        pixel_values = self._prepare(image_path)
        depth = self._forward(pixel_values)
        if normalize:
            depth = self._normalize_depth(depth)
        return depth.cpu().numpy()

    # Small convenience wrapper so evaluation helpers can treat CPU and TT
    # pipelines uniformly.
    def forward(self, image_path: str, normalize: bool = True) -> np.ndarray:
        return self.run_depth_cpu(image_path, normalize=normalize)


def run_depth_cpu(image_path: str, config: DPTLargeConfig = DEFAULT_CONFIG, **kwargs) -> np.ndarray:
    pipeline = DPTFallbackPipeline(config=config, **kwargs)
    return pipeline.run_depth_cpu(image_path)
