# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Swin-B backbone wrapper for MaskFormer Swin-B.

For bounty #30876, the Swin backbone runs on CPU using Hugging Face's
`transformers.MaskFormerSwinBackbone`. The TTNN implementation focuses on the
transformer decoder + heads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import warnings

import torch

try:
    from models.common.utility_functions import tt_to_torch_tensor
except ModuleNotFoundError:  # pragma: no cover - optional outside tt-metal repo context
    tt_to_torch_tensor = None

from .ttnn_compat import get_default_dtype
from .weights import extract_backbone_state

DEFAULT_TT_DTYPE = get_default_dtype()


class MaskFormerSwinBackbone:
    """CPU backbone wrapper around `transformers.MaskFormerSwinBackbone`."""

    def __init__(self, *, config_dict: Dict[str, object]) -> None:
        self._torch_device = torch.device("cpu")
        self._config_dict = dict(config_dict)
        self._hf_backbone = self._build_hf_backbone(self._config_dict)

    @classmethod
    def from_huggingface(
        cls,
        weights: Dict[str, object],
        device: Optional[object],  # kept for backwards compatibility with older runner paths
        *,
        config_dict: Optional[Dict[str, object]] = None,
    ) -> "MaskFormerSwinBackbone":
        _ = device  # backbone is CPU-only in this implementation
        backbone = cls(config_dict=dict(config_dict or {}))
        backbone.load_weights(weights)
        return backbone

    def load_weights(self, weights: Dict[str, object]) -> None:
        if self._hf_backbone is None:
            raise RuntimeError("transformers is required to load/run the Swin backbone.")

        state = extract_backbone_state(weights)
        torch_state = {name: self._ensure_torch_tensor(tensor) for name, tensor in state.items()}
        missing, unexpected = self._hf_backbone.load_state_dict(torch_state, strict=False)
        if missing or unexpected:
            warnings.warn(
                f"Backbone weight load mismatch. Missing: {missing[:5]} Unexpected: {unexpected[:5]}",
                RuntimeWarning,
            )

    def forward(self, images: Any) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self._hf_backbone is None:
            raise RuntimeError("transformers is required to run the Swin backbone.")

        torch_images = self._ensure_torch_tensor(images).to(self._torch_device)
        with torch.no_grad():
            outputs = self._hf_backbone(torch_images)

        feature_maps = list(outputs.feature_maps)
        hidden_states = getattr(outputs, "hidden_states", None)
        encoder_hidden = (
            [self._ensure_torch_tensor(t) for t in hidden_states]
            if isinstance(hidden_states, (list, tuple))
            else feature_maps
        )
        return feature_maps, encoder_hidden

    def _build_hf_backbone(self, config_dict: Dict[str, object]):
        try:
            from transformers import MaskFormerSwinBackbone as HFBackbone, MaskFormerSwinConfig as HFConfig
        except ModuleNotFoundError:
            return None

        hf_config = HFConfig(**dict(config_dict))
        backbone = HFBackbone(hf_config)
        backbone.eval()
        return backbone.to(self._torch_device)

    def _ensure_torch_tensor(self, tensor: Any) -> torch.Tensor:
        if isinstance(tensor, torch.Tensor):
            return tensor
        if tt_to_torch_tensor is not None:
            try:
                return tt_to_torch_tensor(tensor)
            except Exception:
                pass
        if hasattr(tensor, "to_torch"):
            return tensor.to_torch()
        if hasattr(tensor, "cpu"):
            return torch.tensor(tensor.cpu().numpy())
        if isinstance(tensor, (list, tuple)):
            return torch.tensor(tensor)
        raise TypeError(f"Unsupported tensor type for conversion to torch: {type(tensor)!r}")
