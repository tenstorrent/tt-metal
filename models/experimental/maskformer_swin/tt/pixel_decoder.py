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
import os
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
        self._prefer_native_group_norm = os.environ.get("MASKFORMER_TT_USE_NATIVE_GROUP_NORM", "0").strip() != "0"
        self._prefer_moreh_group_norm = os.environ.get("MASKFORMER_TT_USE_MOREH_GROUP_NORM", "1").strip() != "0"
        self._debug_group_norm = os.environ.get("MASKFORMER_TT_DEBUG_GROUP_NORM", "0").strip() == "1"

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
            weight_key="fpn_stem_w",
            bias_key="fpn_stem_b",
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
            moreh_weight=self._tt_weights["fpn_stem_gn_w_moreh"],
            moreh_bias=self._tt_weights["fpn_stem_gn_b_moreh"],
            manual_weight=self._tt_weights["fpn_stem_gn_w_manual"],
            manual_bias=self._tt_weights["fpn_stem_gn_b_manual"],
        )
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        hidden: List[Any] = [x]

        # Top-down pathway: stage2 -> stage1 -> stage0
        lateral_order = [tt_feats[2], tt_feats[1], tt_feats[0]]
        for i, lateral in enumerate(lateral_order):
            lateral_proj = self._conv2d(
                lateral,
                weight=self._tt_weights[f"fpn_l{i}_proj_w"],
                bias=self._tt_weights.get(f"fpn_l{i}_proj_b"),
                weight_key=f"fpn_l{i}_proj_w",
                bias_key=f"fpn_l{i}_proj_b",
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
                moreh_weight=self._tt_weights[f"fpn_l{i}_proj_gn_w_moreh"],
                moreh_bias=self._tt_weights[f"fpn_l{i}_proj_gn_b_moreh"],
                manual_weight=self._tt_weights[f"fpn_l{i}_proj_gn_w_manual"],
                manual_bias=self._tt_weights[f"fpn_l{i}_proj_gn_b_manual"],
            )

            if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.upsample(x, scale_factor=(2.0, 2.0), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if x.get_layout() != ttnn.TILE_LAYOUT:
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if lateral_proj.get_layout() != ttnn.TILE_LAYOUT:
                lateral_proj = ttnn.to_layout(lateral_proj, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.add(x, lateral_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_legacy=True)
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            x = self._conv2d(
                x,
                weight=self._tt_weights[f"fpn_l{i}_block_w"],
                bias=self._tt_weights.get(f"fpn_l{i}_block_b"),
                weight_key=f"fpn_l{i}_block_w",
                bias_key=f"fpn_l{i}_block_b",
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
                moreh_weight=self._tt_weights[f"fpn_l{i}_block_gn_w_moreh"],
                moreh_bias=self._tt_weights[f"fpn_l{i}_block_gn_b_moreh"],
                manual_weight=self._tt_weights[f"fpn_l{i}_block_gn_w_manual"],
                manual_bias=self._tt_weights[f"fpn_l{i}_block_gn_b_manual"],
            )
            hidden.append(x)

        # Final mask projection (3x3 conv)
        mask_features = self._conv2d(
            x,
            weight=self._tt_weights["mask_proj_w"],
            bias=self._tt_weights.get("mask_proj_b"),
            weight_key="mask_proj_w",
            bias_key="mask_proj_b",
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
        # Persistent conv weights live in DRAM to reduce init-time L1 pressure.
        mem_cfg = ttnn.DRAM_MEMORY_CONFIG

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
            manual_w = ttnn.from_torch(
                w.detach().contiguous().view(1, 1, 1, -1),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            manual_b = ttnn.from_torch(
                b.detach().contiguous().view(1, 1, 1, -1),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            moreh_w = ttnn.to_layout(manual_w, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            moreh_b = ttnn.to_layout(manual_b, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            gn_w = manual_w
            gn_b = manual_b
            gn_mask = None
            if self._prefer_native_group_norm:
                try:
                    if hasattr(ttnn, "create_group_norm_input_mask") and hasattr(ttnn, "create_group_norm_weight_bias_rm"):
                        # Conservative native GN setup: keep channel sharding width at 1 virtual column.
                        cores_across_channel = 1
                        gn_mask = ttnn.create_group_norm_input_mask(
                            int(w.numel()),
                            int(self.config.group_norm_groups),
                            cores_across_channel,
                            dtype,
                        )
                        gn_mask = ttnn.to_device(gn_mask, self.device)
                        gn_w_rm = ttnn.create_group_norm_weight_bias_rm(
                            w.detach().contiguous(),
                            int(w.numel()),
                            cores_across_channel,
                        )
                        gn_b_rm = ttnn.create_group_norm_weight_bias_rm(
                            b.detach().contiguous(),
                            int(b.numel()),
                            cores_across_channel,
                        )
                        gn_w = ttnn.from_torch(
                            gn_w_rm,
                            dtype=dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=self.device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        gn_b = ttnn.from_torch(
                            gn_b_rm,
                            dtype=dtype,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=self.device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    else:
                        self._prefer_native_group_norm = False
                except Exception:
                    self._prefer_native_group_norm = False
                    gn_w = manual_w
                    gn_b = manual_b
                    gn_mask = None
            return {
                "w": gn_w,
                "b": gn_b,
                "mask": gn_mask,
                "w_moreh": moreh_w,
                "b_moreh": moreh_b,
                "w_manual": manual_w,
                "b_manual": manual_b,
            }

        # Stem: conv3x3 + GN
        stem_w, stem_b = _to_tt_conv_weight_bias("fpn.stem.0.weight", None)
        gn = _prep_group_norm("fpn.stem.1")
        self._tt_weights["fpn_stem_w"] = stem_w
        self._tt_weights["fpn_stem_b"] = stem_b
        self._tt_weights["fpn_stem_gn_w"] = gn["w"]
        self._tt_weights["fpn_stem_gn_b"] = gn["b"]
        self._tt_weights["fpn_stem_gn_mask"] = gn["mask"]
        self._tt_weights["fpn_stem_gn_w_moreh"] = gn["w_moreh"]
        self._tt_weights["fpn_stem_gn_b_moreh"] = gn["b_moreh"]
        self._tt_weights["fpn_stem_gn_w_manual"] = gn["w_manual"]
        self._tt_weights["fpn_stem_gn_b_manual"] = gn["b_manual"]

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
            self._tt_weights[f"fpn_l{i}_proj_gn_w_moreh"] = proj_gn["w_moreh"]
            self._tt_weights[f"fpn_l{i}_proj_gn_b_moreh"] = proj_gn["b_moreh"]
            self._tt_weights[f"fpn_l{i}_proj_gn_w_manual"] = proj_gn["w_manual"]
            self._tt_weights[f"fpn_l{i}_proj_gn_b_manual"] = proj_gn["b_manual"]
            self._tt_weights[f"fpn_l{i}_block_w"] = block_w
            self._tt_weights[f"fpn_l{i}_block_b"] = block_b
            self._tt_weights[f"fpn_l{i}_block_gn_w"] = block_gn["w"]
            self._tt_weights[f"fpn_l{i}_block_gn_b"] = block_gn["b"]
            self._tt_weights[f"fpn_l{i}_block_gn_mask"] = block_gn["mask"]
            self._tt_weights[f"fpn_l{i}_block_gn_w_moreh"] = block_gn["w_moreh"]
            self._tt_weights[f"fpn_l{i}_block_gn_b_moreh"] = block_gn["b_moreh"]
            self._tt_weights[f"fpn_l{i}_block_gn_w_manual"] = block_gn["w_manual"]
            self._tt_weights[f"fpn_l{i}_block_gn_b_manual"] = block_gn["b_manual"]

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
        weight_key: Optional[str],
        bias_key: Optional[str],
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> Any:
        if self.device is None or ttnn is None:
            require_ttnn("run MaskFormer pixel decoder conv2d ops")
        out, [out_h, out_w], [prepared_weight, prepared_bias] = ttnn.conv2d(
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
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        if weight_key is not None:
            self._tt_weights[weight_key] = prepared_weight
        if bias_key is not None:
            self._tt_weights[bias_key] = prepared_bias
        return ttnn.reshape(out, (int(x.shape[0]), out_h, out_w, int(out_channels)))

    def _group_norm(
        self,
        x: Any,
        *,
        weight: Any,
        bias: Any,
        mask: Optional[Any],
        moreh_weight: Optional[Any],
        moreh_bias: Optional[Any],
        manual_weight: Optional[Any],
        manual_bias: Optional[Any],
    ) -> Any:
        if self.device is None or ttnn is None:
            require_ttnn("run MaskFormer pixel decoder group norm")
        B, H, W, C = x.shape
        groups = int(self.config.group_norm_groups)
        if int(C) % groups != 0:
            raise ValueError(f"GroupNorm channels ({int(C)}) must be divisible by groups ({groups}).")

        if self._prefer_native_group_norm and mask is not None:
            try:
                x_1 = ttnn.reshape(x, (int(B), 1, int(H) * int(W), int(C)))
                core_grid = ttnn.CoreGrid(x=1, y=1) if hasattr(ttnn, "CoreGrid") else None
                kwargs = {
                    "num_groups": groups,
                    "epsilon": float(self.config.group_norm_epsilon),
                    "weight": weight,
                    "bias": bias,
                    "input_mask": mask,
                    "inplace": False,
                    "memory_config": ttnn.DRAM_MEMORY_CONFIG,
                }
                if core_grid is not None:
                    kwargs["core_grid"] = core_grid
                x_1 = ttnn.group_norm(x_1, **kwargs)
                return ttnn.reshape(x_1, (int(B), int(H), int(W), int(C)))
            except Exception:
                self._prefer_native_group_norm = False

        if self._prefer_moreh_group_norm and moreh_weight is not None and moreh_bias is not None:
            try:
                x_tile = x
                if hasattr(x_tile, "storage_type") and hasattr(ttnn, "StorageType"):
                    if x_tile.storage_type() != ttnn.StorageType.DEVICE:
                        x_tile = ttnn.to_device(x_tile, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if x_tile.get_layout() != ttnn.TILE_LAYOUT:
                    x_tile = ttnn.to_layout(x_tile, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                w_moreh = moreh_weight
                b_moreh = moreh_bias
                if hasattr(w_moreh, "storage_type") and hasattr(ttnn, "StorageType"):
                    if w_moreh.storage_type() != ttnn.StorageType.DEVICE:
                        w_moreh = ttnn.to_device(w_moreh, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if hasattr(b_moreh, "storage_type") and hasattr(ttnn, "StorageType"):
                    if b_moreh.storage_type() != ttnn.StorageType.DEVICE:
                        b_moreh = ttnn.to_device(b_moreh, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                x_nchw = ttnn.permute(x_tile, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                x_nchw, _, _ = ttnn.operations.moreh.group_norm(
                    x_nchw,
                    groups,
                    float(self.config.group_norm_epsilon),
                    w_moreh,
                    b_moreh,
                    are_required_outputs=(True, False, False),
                    mean=None,
                    rstd=None,
                )
                x_out = ttnn.permute(x_nchw, (0, 2, 3, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if x_out.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                    x_out = ttnn.to_layout(x_out, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                return x_out
            except Exception:
                if self._debug_group_norm:
                    try:
                        x_store = x_tile.storage_type() if hasattr(x_tile, "storage_type") else "n/a"
                        x_layout = x_tile.get_layout() if hasattr(x_tile, "get_layout") else "n/a"
                        w_store = w_moreh.storage_type() if hasattr(w_moreh, "storage_type") else "n/a"
                        w_layout = w_moreh.get_layout() if hasattr(w_moreh, "get_layout") else "n/a"
                        b_store = b_moreh.storage_type() if hasattr(b_moreh, "storage_type") else "n/a"
                        b_layout = b_moreh.get_layout() if hasattr(b_moreh, "get_layout") else "n/a"
                        print(
                            "[maskformer][pixel_decoder] moreh_group_norm fallback "
                            f"x=({x_store},{x_layout}) w=({w_store},{w_layout}) b=({b_store},{b_layout})"
                        )
                    except Exception:
                        pass
                self._prefer_moreh_group_norm = False

        channels_per_group = int(C) // groups
        w = manual_weight if manual_weight is not None else weight
        b = manual_bias if manual_bias is not None else bias
        # Fallback path for runtimes where fused group_norm is unavailable/unstable.
        xg = ttnn.reshape(x, (int(B), int(H), int(W), groups, channels_per_group))
        mean = ttnn.mean(xg, dim=[1, 2, 4], keepdim=True)
        var = ttnn.var(xg, dim=[1, 2, 4], keepdim=True)
        xg = (xg - mean) / ttnn.sqrt(var + float(self.config.group_norm_epsilon))
        x = ttnn.reshape(xg, (int(B), int(H), int(W), int(C)))
        x = x * w
        x = x + b
        return x
