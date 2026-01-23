# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pixel decoder (multi-scale fusion) for MaskFormer.

MaskFormer-Swin-B uses an FPN-style pixel decoder consisting of:
  - A stem 3x3 conv applied to the last backbone stage
  - Three top-down lateral projections (1x1 convs) and 3x3 refinement blocks
  - A final 3x3 mask projection producing the mask feature map (mask_dim=256)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Any, Dict

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .backbone_swin import DEFAULT_TT_DTYPE
from .ttnn_compat import ttnn, require_ttnn, get_default_dtype
from .weights import extract_pixel_decoder_state


@dataclass
class PixelDecoderConfig:
    """Configuration knobs for the pixel decoder."""

    fpn_dim: int = 256
    mask_dim: int = 256
    group_norm_groups: int = 32
    group_norm_epsilon: float = 1e-5
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
        self._tt_weights: Dict[str, Any] = {}

    def forward(
        self,
        features: Iterable[Any],
    ) -> Tuple[Any, List[Any]]:
        """Fuse backbone stage features into a mask feature map (TT tensor, NHWC)."""

        if self.device is None or ttnn is None:
            require_ttnn("run MaskFormer pixel decoder on device")
        if not self._tt_weights:
            raise RuntimeError("Pixel decoder weights are not loaded.")

        feats = list(features)
        if len(feats) != 4:
            raise ValueError(f"Expected 4 backbone feature maps, got {len(feats)}")

        # Ensure NHWC TT tensors
        tt_feats: List[Any] = []
        for feat in feats:
            tt_feats.append(self._to_tt_nhwc(feat))

        # FPN stem on last stage (C4, stride 32)
        x = self._conv2d(
            tt_feats[-1],
            weight=self._tt_weights["fpn_stem_w"],
            bias=self._tt_weights.get("fpn_stem_b"),
            out_channels=self.config.fpn_dim,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        x = self._group_norm(
            x,
            weight=self._tt_weights["fpn_stem_gn_w"],
            bias=self._tt_weights["fpn_stem_gn_b"],
            mask=self._tt_weights["fpn_stem_gn_mask"],
        )

        hidden: List[Any] = [x]

        # Top-down pathway: stage2 -> stage1 -> stage0
        lateral_order = [tt_feats[2], tt_feats[1], tt_feats[0]]
        for i, lateral in enumerate(lateral_order):
            lateral_proj = self._conv2d(
                lateral,
                weight=self._tt_weights[f"fpn_l{i}_proj_w"],
                bias=self._tt_weights.get(f"fpn_l{i}_proj_b"),
                out_channels=self.config.fpn_dim,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
            lateral_proj = self._group_norm(
                lateral_proj,
                weight=self._tt_weights[f"fpn_l{i}_proj_gn_w"],
                bias=self._tt_weights[f"fpn_l{i}_proj_gn_b"],
                mask=self._tt_weights[f"fpn_l{i}_proj_gn_mask"],
            )

            x = ttnn.upsample(x, scale_factor=(2.0, 2.0), memory_config=ttnn.L1_MEMORY_CONFIG)
            x = ttnn.add(x, lateral_proj, memory_config=ttnn.L1_MEMORY_CONFIG, use_legacy=False)

            x = self._conv2d(
                x,
                weight=self._tt_weights[f"fpn_l{i}_block_w"],
                bias=self._tt_weights.get(f"fpn_l{i}_block_b"),
                out_channels=self.config.fpn_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )
            x = self._group_norm(
                x,
                weight=self._tt_weights[f"fpn_l{i}_block_gn_w"],
                bias=self._tt_weights[f"fpn_l{i}_block_gn_b"],
                mask=self._tt_weights[f"fpn_l{i}_block_gn_mask"],
            )
            hidden.append(x)

        # Final mask projection (3x3 conv)
        mask_features = self._conv2d(
            x,
            weight=self._tt_weights["mask_proj_w"],
            bias=self._tt_weights.get("mask_proj_b"),
            out_channels=self.config.mask_dim,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        return mask_features, hidden

    def load_weights(self, weights: Dict[str, object]) -> None:
        """Load pixel decoder weights and prepare TTNN tensors."""

        if self.device is None or ttnn is None:
            require_ttnn("load MaskFormer pixel decoder weights on device")
        if torch is None:
            raise RuntimeError("torch is required to load MaskFormer pixel decoder weights.")

        state = extract_pixel_decoder_state(weights)
        dtype = self.dtype or get_default_dtype()
        mem_cfg = ttnn.L1_MEMORY_CONFIG

        def _to_tt_conv_weight_bias(w_key: str, b_key: Optional[str]):
            w = state[w_key]
            if not isinstance(w, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for {w_key}, got {type(w)!r}")
            wt = ttnn.from_torch(
                w.detach().contiguous(),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=mem_cfg,
            )
            bt = None
            if b_key is not None and b_key in state:
                b = state[b_key]
                if isinstance(b, torch.Tensor):
                    bt = ttnn.from_torch(
                        b.detach().contiguous().view(1, 1, 1, -1),
                        dtype=dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=self.device,
                        memory_config=mem_cfg,
                    )
            return wt, bt

        # GroupNorm params need to be prepared for TTNN group_norm
        def _prep_group_norm(prefix: str):
            w = state[f"{prefix}.weight"]
            b = state[f"{prefix}.bias"]
            if not isinstance(w, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor groupnorm params at {prefix}.*")
            # returns ([tt_w, tt_b], tt_mask)
            (tt_w, tt_b), tt_mask = ttnn.dram_group_norm_params_from_torch(
                [w.detach().contiguous(), b.detach().contiguous()],
                channels_per_device=int(w.numel()),
                groups_per_device=int(self.config.group_norm_groups),
                device=self.device,
                return_mask=True,
                dtype=dtype,
            )
            return {"w": tt_w, "b": tt_b, "mask": tt_mask}

        # Stem: conv3x3 + GN
        stem_w, stem_b = _to_tt_conv_weight_bias("fpn.stem.0.weight", None)
        gn = _prep_group_norm("fpn.stem.1")
        self._tt_weights["fpn_stem_w"] = stem_w
        self._tt_weights["fpn_stem_b"] = stem_b
        self._tt_weights["fpn_stem_gn_w"] = gn["w"]
        self._tt_weights["fpn_stem_gn_b"] = gn["b"]
        self._tt_weights["fpn_stem_gn_mask"] = gn["mask"]

        # Lateral layers: 0..2 each has proj (1x1) + GN and block (3x3) + GN
        for i in range(3):
            proj_w, proj_b = _to_tt_conv_weight_bias(f"fpn.layers.{i}.proj.0.weight", None)
            proj_gn = _prep_group_norm(f"fpn.layers.{i}.proj.1")
            block_w, block_b = _to_tt_conv_weight_bias(f"fpn.layers.{i}.block.0.weight", None)
            block_gn = _prep_group_norm(f"fpn.layers.{i}.block.1")
            self._tt_weights[f"fpn_l{i}_proj_w"] = proj_w
            self._tt_weights[f"fpn_l{i}_proj_b"] = proj_b
            self._tt_weights[f"fpn_l{i}_proj_gn_w"] = proj_gn["w"]
            self._tt_weights[f"fpn_l{i}_proj_gn_b"] = proj_gn["b"]
            self._tt_weights[f"fpn_l{i}_proj_gn_mask"] = proj_gn["mask"]
            self._tt_weights[f"fpn_l{i}_block_w"] = block_w
            self._tt_weights[f"fpn_l{i}_block_b"] = block_b
            self._tt_weights[f"fpn_l{i}_block_gn_w"] = block_gn["w"]
            self._tt_weights[f"fpn_l{i}_block_gn_b"] = block_gn["b"]
            self._tt_weights[f"fpn_l{i}_block_gn_mask"] = block_gn["mask"]

        # Mask projection conv3x3
        mp_w, mp_b = _to_tt_conv_weight_bias("mask_projection.weight", "mask_projection.bias")
        self._tt_weights["mask_proj_w"] = mp_w
        self._tt_weights["mask_proj_b"] = mp_b

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

    def _to_tt_nhwc(self, tensor: Any) -> Any:
        if self.device is None or ttnn is None:
            require_ttnn("convert MaskFormer pixel decoder inputs to TT tensors")
        dtype = self.dtype or get_default_dtype()
        mem_cfg = ttnn.DRAM_MEMORY_CONFIG

        if torch is not None and isinstance(tensor, torch.Tensor):
            nhwc = tensor.detach().contiguous().permute(0, 2, 3, 1)
            return ttnn.from_torch(
                nhwc, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, memory_config=mem_cfg
            )

        tt = tensor
        if getattr(tt, "get_layout", None) is not None and tt.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            tt = ttnn.to_layout(tt, ttnn.ROW_MAJOR_LAYOUT)
        return tt

    def _conv2d(
        self,
        x: Any,
        *,
        weight: Any,
        bias: Optional[Any],
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> Any:
        if self.device is None or ttnn is None:
            require_ttnn("run MaskFormer pixel decoder conv2d ops")
        mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        return ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=int(x.shape[-1]),
            out_channels=int(out_channels),
            batch_size=int(x.shape[0]),
            input_height=int(x.shape[1]),
            input_width=int(x.shape[2]),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=(1, 1),
            groups=1,
            device=self.device,
            memory_config=mem_cfg,
        )

    def _group_norm(self, x: Any, *, weight: Any, bias: Any, mask: Optional[Any]) -> Any:
        if self.device is None or ttnn is None:
            require_ttnn("run MaskFormer pixel decoder group norm")
        # GroupNorm expects (B, 1, HW, C)
        B, H, W, C = x.shape
        x_1 = ttnn.reshape(x, (int(B), 1, int(H) * int(W), int(C)))

        x_1 = ttnn.group_norm(
            x_1,
            num_groups=int(self.config.group_norm_groups),
            epsilon=float(self.config.group_norm_epsilon),
            weight=weight,
            bias=bias,
            input_mask=mask,
            core_grid=getattr(self.device, "core_grid", None) or ttnn.CoreGrid(x=8, y=8),
            inplace=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.reshape(x_1, (int(B), int(H), int(W), int(C)))
        return x
