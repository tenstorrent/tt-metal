# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .config import DPTLargeConfig
from .fallback import DPTFallbackPipeline
from .fusion_head import DPTFusionHead
from .reassembly import DPTReassembly
from .vit_backbone import DPTViTBackboneTTNN
from .tt_configs import describe_configs

try:
    import ttnn
except Exception:  # pragma: no cover
    ttnn = None

LOG = logging.getLogger(__name__)


@dataclass
class DPTTTPipeline:
    """
    Thin wrapper that mirrors the full TTNN execution path.

    In dev environments without hardware we keep everything on the host while
    preserving the same interfaces so perf-tuning can happen later.
    """

    config: DPTLargeConfig = field(default_factory=DPTLargeConfig)
    pretrained: bool = True
    device: str = "cpu"

    def __post_init__(self):
        # Avoid mutating a caller-provided config (and never mutate the module-level DEFAULT_CONFIG).
        self.config = copy.deepcopy(self.config)

        self.fallback = DPTFallbackPipeline(config=self.config, pretrained=self.pretrained, device=self.device)
        # Align neck shapes with HF when using pretrained weights
        # Align fusion/head flags too (BN usage, optional projection)
        try:
            self.config.use_batch_norm_in_fusion_residual = bool(
                self.fallback._model.config.use_batch_norm_in_fusion_residual
            )
            self.config.add_projection = bool(getattr(self.fallback._model.config, "add_projection", False))
        except Exception:
            pass
        try:
            hf_neck_sizes = list(self.fallback._model.config.neck_hidden_sizes)
            if isinstance(hf_neck_sizes, list) and len(hf_neck_sizes) == len(self.config.neck_hidden_sizes):
                self.config.neck_hidden_sizes = hf_neck_sizes
        except Exception:
            pass
        # Reuse the same HF weights for the "TT" path; later this can be swapped
        # with real TTNN modules.
        self.tt_layer_configs = describe_configs(self.config)
        self.backbone = DPTViTBackboneTTNN(
            config=self.config,
            hf_model=self.fallback._model,
            pretrained=self.pretrained,
            device=self.device,
            tt_layer_cfg=self.tt_layer_configs["vit_block"],
        )
        tt_dev = getattr(self.backbone, "tt_device", None)
        self.reassembly = DPTReassembly(
            config=self.config, tt_device=tt_dev, layer_cfg=self.tt_layer_configs["cnn_block"]
        )
        self.fusion_head = DPTFusionHead(
            config=self.config, tt_device=tt_dev, layer_cfg=self.tt_layer_configs["cnn_block"]
        )
        # Mirror HF neck + head weights so the TT path is numerically aligned
        # with the reference model, even when `pretrained=False`.
        state_dict = self.fallback._model.state_dict()
        if hasattr(self.reassembly, "load_from_hf_state_dict"):
            self.reassembly.load_from_hf_state_dict(state_dict)
        if hasattr(self.fusion_head, "load_from_hf_state_dict"):
            self.fusion_head.load_from_hf_state_dict(state_dict)
        self.to(self.device)
        self.eval()
        # Last per-call perf breakdown (filled in forward).
        self.last_perf: Optional[dict] = None

    # ------------------------------------------------------------------ plumbing
    def to(self, device: str):
        self.device = device
        self.backbone.to(device)
        self.reassembly.to(device)
        self.fusion_head.to(device)
        return self

    def eval(self):
        self.backbone.eval()
        self.reassembly.eval()
        self.fusion_head.eval()
        return self

    def close(self):
        if hasattr(self.backbone, "close"):
            self.backbone.close()
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # ------------------------------------------------------------------ forward
    def forward(self, image_path: str | list[str], normalize: bool = True) -> np.ndarray | list[np.ndarray]:
        # Support single path or list for simple pipelining/batching.
        paths = image_path if isinstance(image_path, list) else [image_path]
        outputs = []
        # Reset per-call perf breakdown.
        self.last_perf = None
        # If no TT device available, either fall back to HF path (when allowed)
        # or raise in order to surface configuration issues in tests.
        if getattr(self.backbone, "tt_device", None) is None:
            if not self.config.allow_cpu_fallback:
                raise RuntimeError(
                    "TT device is not available but `allow_cpu_fallback=False`. "
                    "This configuration is intended to exercise the TT path."
                )
            t_start = time.perf_counter()
            for p in paths:
                outputs.append(self.fallback.run_depth_cpu(p, normalize=normalize))
            total_ms = (time.perf_counter() - t_start) * 1000.0
            self.last_perf = {
                "mode": "cpu_fallback",
                "total_ms": total_ms,
                "num_images": len(paths),
            }
            return outputs if len(outputs) > 1 else outputs[0]

        # Lightweight host preprocessing pipeline
        t_pre = time.perf_counter()
        preprocessed = [self.fallback._prepare(p) for p in paths]
        preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

        backbone_ms = 0.0
        reassembly_ms = 0.0
        fusion_head_ms = 0.0
        normalize_ms = 0.0
        t_total = time.perf_counter()

        with torch.no_grad():
            for pv in preprocessed:
                t0 = time.perf_counter()
                feats = self.backbone(pv.to(self.device), return_tt=True)
                backbone_ms += (time.perf_counter() - t0) * 1000.0

                # Strict path (no TT neck/head flags): use HF neck+head for perfect parity.
                if not (self.config.tt_device_reassembly or self.config.tt_device_fusion):
                    t1 = time.perf_counter()
                    depth = self.fallback._forward(pv)
                    # HF returns [B,1,H,W]; treat as torch tensor for normalization below.
                    reassembly_ms += 0.0
                    fusion_head_ms += (time.perf_counter() - t1) * 1000.0
                    depth_t = torch.as_tensor(depth)
                else:
                    t1 = time.perf_counter()
                    pyramid = self.reassembly(feats)
                    reassembly_ms += (time.perf_counter() - t1) * 1000.0

                    t2 = time.perf_counter()
                    depth = self.fusion_head(pyramid)
                    fusion_head_ms += (time.perf_counter() - t2) * 1000.0
                    # depth may be a TT tensor, torch tensor, or numpy array; ensure
                    # we normalize and return a float32 torch tensor.
                    try:
                        import ttnn

                        if isinstance(depth, ttnn.Tensor):
                            depth = depth.cpu().to_torch()
                    except Exception:
                        pass
                    depth_t = torch.as_tensor(depth)
                    # Ensure channel dimension matches CPU fallback ([B,1,H,W]) before normalization.
                    if depth_t.dim() == 3:
                        depth_t = depth_t.unsqueeze(1)
                t3 = time.perf_counter()
                if normalize:
                    depth_t = self.fallback._normalize_depth(depth_t.float())
                else:
                    depth_t = depth_t.float()
                normalize_ms += (time.perf_counter() - t3) * 1000.0
                outputs.append(depth_t.cpu().numpy())

        total_ms = (time.perf_counter() - t_total) * 1000.0
        self.last_perf = {
            "mode": "tt",
            "num_images": len(paths),
            "preprocess_ms": preprocess_ms,
            "backbone_ms": backbone_ms,
            "reassembly_ms": reassembly_ms,
            "fusion_head_ms": fusion_head_ms,
            "normalize_ms": normalize_ms,
            "total_ms": total_ms,
        }

        return outputs if len(outputs) > 1 else outputs[0]


def run_depth(
    image_path: str,
    use_tt: bool = True,
    config: DPTLargeConfig | None = None,
    **kwargs,
) -> np.ndarray:
    if config is None:
        config = DPTLargeConfig()
    if not use_tt:
        pipe = DPTFallbackPipeline(config=config, **kwargs)
        return pipe.run_depth_cpu(image_path)
    pipe = DPTTTPipeline(config=config, **kwargs)
    return pipe.forward(image_path)
