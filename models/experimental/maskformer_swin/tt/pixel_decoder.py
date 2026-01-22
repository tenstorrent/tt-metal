# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pixel decoder (multi-scale fusion) for MaskFormer.

For bounty #30876 the pixel decoder runs on CPU using Hugging Face's reference
implementation. The final 3×3 mask projection is optionally executed on device
when TTNN `conv2d` is available; otherwise it falls back to the HF projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Any, Dict

import torch
import warnings

try:
    from transformers.models.maskformer.modeling_maskformer import MaskFormerPixelDecoder as HFPixelDecoder
except ModuleNotFoundError:  # pragma: no cover - only used for fallback path
    HFPixelDecoder = None

try:
    from models.common.utility_functions import tt_to_torch_tensor
except ModuleNotFoundError:  # pragma: no cover - optional utility
    tt_to_torch_tensor = None

from .backbone_swin import DEFAULT_TT_DTYPE
from .ttnn_compat import ttnn, require_ttnn, get_default_dtype
from .weights import extract_pixel_decoder_state


@dataclass
class PixelDecoderConfig:
    """Configuration knobs for the pixel decoder."""

    fpn_dim: int = 256
    mask_dim: int = 256
    upsample_mode: str = "bilinear"
    align_corners: bool = False
    input_channels: Tuple[int, int, int, int] = (128, 256, 512, 1024)
    feature_strides: Tuple[int, int, int, int] = (4, 8, 16, 32)


class MaskFormerPixelDecoder:
    """Consumes multi-scale backbone features and produces pixel embeddings."""

    def __init__(
        self,
        config: PixelDecoderConfig,
        device: Optional[object],
        *,
        dtype: Optional[object] = DEFAULT_TT_DTYPE,
    ) -> None:
        if device is not None and ttnn is None:
            require_ttnn("allocate the MaskFormer pixel decoder on a TT device")
        self.config = config
        self.device = device
        self.dtype = dtype

        if HFPixelDecoder:
            in_features = config.input_channels[-1]
            lateral_widths = list(config.input_channels[:-1])
            self._hf_decoder = HFPixelDecoder(
                in_features,
                lateral_widths=lateral_widths,
                feature_size=config.fpn_dim,
                mask_feature_size=config.mask_dim,
            )
        else:
            self._hf_decoder = None
        self._torch_state: Dict[str, Any] = {}
        # TTNN-prepared mask projection conv weights
        self._mask_proj_kernel: Dict[str, Any] = {}

    def forward(
        self,
        features: Iterable[torch.Tensor],
        *,
        long_skip: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Fuse backbone features into pixel-level embeddings. Use TT conv for mask projection when available."""

        if self._hf_decoder is None:
            raise NotImplementedError("TT-NN pixel decoder pending; install transformers for fallback execution.")

        with torch.no_grad():
            result = self._hf_decoder(list(features), output_hidden_states=True)
        fpn_hidden = list(result.hidden_states) if isinstance(result.hidden_states, (list, tuple)) else []

        if self.device is not None and ttnn is not None and self._mask_proj_kernel:
            try:
                last_fpn = fpn_hidden[-1]  # [B, C=fpn_dim, H, W]
                nhwc = last_fpn.detach().contiguous().permute(0, 2, 3, 1)
                mem_cfg = getattr(ttnn, "DRAM_MEMORY_CONFIG", None) or ttnn.DRAM_MEMORY_CONFIG
                tt_in = ttnn.from_torch(
                    nhwc,
                    dtype=self.dtype or get_default_dtype(),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=mem_cfg,
                )
                out_tt = ttnn.conv2d(
                    input_tensor=tt_in,
                    weight_tensor=self._mask_proj_kernel["weight"],
                    bias_tensor=self._mask_proj_kernel.get("bias"),
                    in_channels=int(tt_in.shape[-1]),
                    out_channels=self.config.mask_dim,
                    batch_size=int(tt_in.shape[0]),
                    input_height=int(tt_in.shape[1]),
                    input_width=int(tt_in.shape[2]),
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    dilation=(1, 1),
                    groups=1,
                    device=self.device,
                    memory_config=mem_cfg,
                )
                out_rm = (
                    out_tt
                    if out_tt.get_layout() == ttnn.ROW_MAJOR_LAYOUT
                    else ttnn.to_layout(out_tt, ttnn.ROW_MAJOR_LAYOUT)
                )
                mask_features = self._to_torch(out_rm).permute(0, 3, 1, 2).contiguous()
                return mask_features, fpn_hidden
            except Exception as exc:
                warnings.warn(f"TT mask projection failed; falling back to HF projection: {exc}", RuntimeWarning)

        # Fallback to HF mask projection
        return result.last_hidden_state, fpn_hidden

    def load_weights(self, weights: Dict[str, object]) -> None:
        """Load HuggingFace decoder weights into the fallback model."""

        if self._hf_decoder is None:
            return

        state = extract_pixel_decoder_state(weights)
        torch_state = {name: self._ensure_torch_tensor(tensor) for name, tensor in state.items()}
        missing, unexpected = self._hf_decoder.load_state_dict(torch_state, strict=False)
        if missing or unexpected:
            warnings.warn(
                f"Pixel decoder weight load mismatch. Missing: {missing[:5]} Unexpected: {unexpected[:5]}",
                RuntimeWarning,
            )
        self._torch_state = torch_state
        # Prepare TTNN mask projection kernel if device present
        if self.device is not None and ttnn is not None:
            try:
                mp = getattr(self._hf_decoder, "mask_projection", None)
                if mp is not None and hasattr(mp, "weight"):
                    w = mp.weight.detach().contiguous()
                    b = mp.bias.detach().contiguous() if hasattr(mp, "bias") and mp.bias is not None else None
                    dtype = self.dtype or get_default_dtype()
                    wt = ttnn.from_torch(w, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
                    kernel: Dict[str, Any] = {"weight": wt}
                    if b is not None:
                        bt = ttnn.from_torch(b.view(1, 1, 1, -1), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
                        kernel["bias"] = bt
                    self._mask_proj_kernel = kernel
            except Exception as e:
                warnings.warn(f"Failed to prepare TTNN mask projection kernel: {e}")

    @classmethod
    def from_huggingface(
        cls,
        weights: Dict[str, object],
        *,
        config: Optional[PixelDecoderConfig] = None,
        device: Optional[object] = None,
    ) -> "MaskFormerPixelDecoder":
        config = config or PixelDecoderConfig()
        decoder = cls(config=config, device=device)
        decoder.load_weights(weights)
        return decoder

    def _ensure_torch_tensor(self, tensor: Any) -> torch.Tensor:
        if isinstance(tensor, torch.Tensor):
            return tensor
        if tt_to_torch_tensor is not None:
            try:
                return tt_to_torch_tensor(tensor)
            except Exception:
                pass
        # Move tensor from device to host before converting to torch
        if hasattr(tensor, "to_torch"):
            # Always call from_device for TT tensors - handles device storage properly
            if ttnn is not None:
                tensor = ttnn.from_device(tensor)
            return tensor.to_torch()
        if hasattr(tensor, "cpu"):
            return torch.tensor(tensor.cpu().numpy())
        if isinstance(tensor, (list, tuple)):
            return torch.tensor(tensor)
        raise TypeError(f"Unsupported tensor type for conversion to torch: {type(tensor)!r}")

    def _to_torch(self, tttensor: Any) -> torch.Tensor:
        return self._ensure_torch_tensor(tttensor)
