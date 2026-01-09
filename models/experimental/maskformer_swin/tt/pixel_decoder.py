# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pixel decoder (multi-scale fusion) for MaskFormer.

The pixel decoder refines Swin backbone features into high-resolution pixel
embeddings.  This file will house TT-NN implementations of:

* Lateral convolutions + normalization layers per FPN level.
* Feature fusion and upsample steps using ``ttnn.interpolate`` with parity
  against PyTorch ``torch.nn.functional.interpolate`` (bilinear, align_corners=False).
* Optional fused activation support via the TT-CNN builder utilities once the
  modules are wired up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Any, Dict
import os

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

        # If TTNN mask projection kernel is ready, perform the final 3x3 projection on device.
        if (
            self.device is not None
            and ttnn is not None
            and self._mask_proj_kernel
            and os.environ.get("MASKFORMER_TT_MASK_PROJ") == "1"
        ):
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
            # Minimal DRAM-based conv2d; error if unsupported to avoid CPU fallback.
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

    def _build_mask_conv_params(self, tt_input):
        """Build TT-CNN conv config for final mask projection (3x3, stride=1, pad=1)."""
        try:
            from models.tt_cnn.tt.builder import (
                Conv2dConfiguration,
                L1FullSliceStrategyConfiguration,
                to_conv2d_config,
                to_compute_config,
                to_slice_config,
            )
        except ModuleNotFoundError:
            raise RuntimeError("TT-CNN Conv2d helpers unavailable; install TTNN to run mask projection on device.")

        dtype = self.dtype or get_default_dtype()
        if dtype is None:
            raise RuntimeError("Unable to determine default TT dtype for mask projection.")

        # For stability across small spatial sizes, avoid aggressive slicing.
        slice_strategy = None

        in_ch = int(tt_input.shape[-1]) if len(tt_input.shape) == 4 else self.config.fpn_dim
        out_ch = self.config.mask_dim
        sharding = None
        if "L1FullSliceStrategyConfiguration" in locals() and L1FullSliceStrategyConfiguration is not None:
            try:
                sharding = L1FullSliceStrategyConfiguration()
            except Exception:
                sharding = None

        configuration = Conv2dConfiguration(
            input_height=int(tt_input.shape[1]),
            input_width=int(tt_input.shape[2]),
            in_channels=in_ch,
            out_channels=out_ch,
            batch_size=int(tt_input.shape[0]),
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=1,
            weight=self._mask_proj_kernel.get("weight"),
            bias=self._mask_proj_kernel.get("bias"),
            activation_dtype=dtype,
            weights_dtype=dtype,
            output_dtype=dtype,
            output_layout=ttnn.TILE_LAYOUT,
            sharding_strategy=sharding,
            slice_strategy=slice_strategy,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            deallocate_activation=False,
            reallocate_halo_output=True,
            config_tensors_in_dram=True,
        )

        conv_config = to_conv2d_config(configuration)
        compute_config = to_compute_config(configuration, self.device)
        slice_config = None
        if to_slice_config is not None and slice_strategy is not None:
            derived_slice = to_slice_config(slice_strategy)
            if derived_slice is not None:
                slice_config = derived_slice
        conv_kwargs = {
            "in_channels": configuration.in_channels,
            "out_channels": configuration.out_channels,
            "batch_size": configuration.batch_size,
            "input_height": configuration.input_height,
            "input_width": configuration.input_width,
            "kernel_size": configuration.kernel_size,
            "stride": configuration.stride,
            "padding": configuration.padding,
            "dilation": configuration.dilation,
            "groups": configuration.groups,
            "device": self.device,
            "conv_config": conv_config,
        }
        if slice_config is not None:
            conv_kwargs["slice_config"] = slice_config
        return conv_kwargs, compute_config
