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
        # moreh_group_norm has shown instability (TT_FATAL / segfault) on N300 in 320x320 runs.
        # Default to disabled; enable explicitly to experiment.
        self._prefer_moreh_group_norm = os.environ.get("MASKFORMER_TT_USE_MOREH_GROUP_NORM", "0").strip() != "0"
        # Cache conv2d prepared weights across forwards to avoid repeated host-side preparation.
        # Stable on N300 for this model; set to 0 explicitly for A/B validation.
        self._cache_conv2d_weights = os.environ.get("MASKFORMER_TT_CACHE_PIXEL_DECODER_CONV2D_WEIGHTS", "1").strip() != "0"
        self._debug_group_norm = os.environ.get("MASKFORMER_TT_DEBUG_GROUP_NORM", "0").strip() == "1"
        self._debug_conv2d = os.environ.get("MASKFORMER_TT_DEBUG_CONV2D", "0").strip() == "1"
        self._debug_seen_conv2d_sites: set[str] = set()
        self._debug_seen_group_norm_sites: set[str] = set()
        # Native group_norm preparation depends on sharded memory config + core grid selected for a given (H, W, C).
        # Cache per-site prepared params to amortize host work.
        self._native_gn_params_cache: Dict[Tuple[str, int, int, int, str], Dict[str, Any]] = {}
        self._native_gn_torch_params: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(
        self,
        features: Iterable[Any],
    ) -> Tuple[Any, List[Any]]:
        """Fuse backbone stage features into a mask feature map (TT tensor, NHWC)."""

        if self.device is None or ttnn is None:
            require_ttnn("run MaskFormer pixel decoder on device")
        if not self._tt_weights:
            raise RuntimeError("Pixel decoder weights are not loaded.")

        if self._debug_conv2d:
            self._debug_seen_conv2d_sites.clear()
        if self._debug_group_norm:
            self._debug_seen_group_norm_sites.clear()

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
            name="fpn_stem_gn",
            weight=self._tt_weights["fpn_stem_gn_w"],
            bias=self._tt_weights["fpn_stem_gn_b"],
            mask=self._tt_weights["fpn_stem_gn_mask"],
            moreh_weight=self._tt_weights["fpn_stem_gn_w_moreh"],
            moreh_bias=self._tt_weights["fpn_stem_gn_b_moreh"],
            manual_weight=self._tt_weights["fpn_stem_gn_w_manual"],
            manual_bias=self._tt_weights["fpn_stem_gn_b_manual"],
        )
        # HF reference: Conv2d -> GroupNorm -> ReLU
        if not hasattr(ttnn, "relu"):
            raise RuntimeError("Pixel decoder requires ttnn.relu, but it is unavailable in this runtime.")
        if x.get_layout() != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.relu(x)
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
                name=f"fpn_l{i}_proj_gn",
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
                name=f"fpn_l{i}_block_gn",
                weight=self._tt_weights[f"fpn_l{i}_block_gn_w"],
                bias=self._tt_weights[f"fpn_l{i}_block_gn_b"],
                mask=self._tt_weights[f"fpn_l{i}_block_gn_mask"],
                moreh_weight=self._tt_weights[f"fpn_l{i}_block_gn_w_moreh"],
                moreh_bias=self._tt_weights[f"fpn_l{i}_block_gn_b_moreh"],
                manual_weight=self._tt_weights[f"fpn_l{i}_block_gn_w_manual"],
                manual_bias=self._tt_weights[f"fpn_l{i}_block_gn_b_manual"],
            )
            # HF reference: Conv2d -> GroupNorm -> ReLU
            if x.get_layout() != ttnn.TILE_LAYOUT:
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.relu(x)
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
        self._native_gn_params_cache.clear()
        self._native_gn_torch_params.clear()
        # Keep raw conv weights on host; conv2d will prepare and return device weights on first use.
        # This follows established repo patterns (e.g., Whisper/VAE conv blocks) and avoids conv2d
        # device-weight "pull back to host and reprocess" paths.

        def _to_tt_conv_weight_bias(w_key: str, b_key: Optional[str]):
            w = state[w_key]
            if not isinstance(w, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for {w_key}, got {type(w)!r}")
            wt = ttnn.from_torch(
                w.detach().contiguous(),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            bt = None
            if b_key is not None and b_key in state:
                b = state[b_key]
                if isinstance(b, torch.Tensor):
                    bt = ttnn.from_torch(
                        b.detach().contiguous().view(1, 1, 1, -1),
                        dtype=dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
            return wt, bt

        # GroupNorm params: keep manual/moreh params on device; stash raw torch params for native GN preparation.
        def _prep_group_norm(site: str, prefix: str):
            w = state[f"{prefix}.weight"]
            b = state[f"{prefix}.bias"]
            if not isinstance(w, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor groupnorm params at {prefix}.*")
            self._native_gn_torch_params[site] = (
                w.detach().contiguous().to(torch.float32),
                b.detach().contiguous().to(torch.float32),
            )
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
            return {
                # Keep legacy keys for forward(); native GN will prepare/cached per runtime shape in _group_norm().
                "w": manual_w,
                "b": manual_b,
                "mask": None,
                "w_moreh": moreh_w,
                "b_moreh": moreh_b,
                "w_manual": manual_w,
                "b_manual": manual_b,
            }

        # Stem: conv3x3 + GN
        stem_w, stem_b = _to_tt_conv_weight_bias("fpn.stem.0.weight", None)
        gn = _prep_group_norm("fpn_stem_gn", "fpn.stem.1")
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
            proj_gn = _prep_group_norm(f"fpn_l{i}_proj_gn", f"fpn.layers.{i}.proj.1")
            block_w, block_b = _to_tt_conv_weight_bias(f"fpn.layers.{i}.block.0.weight", None)
            block_gn = _prep_group_norm(f"fpn_l{i}_block_gn", f"fpn.layers.{i}.block.1")
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
        if hasattr(tt, "storage_type") and hasattr(ttnn, "StorageType"):
            if tt.storage_type() != ttnn.StorageType.DEVICE:
                tt = ttnn.to_device(tt, self.device, memory_config=mem_cfg)
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

        def _tensor_info(label: str, tensor: Any) -> str:
            if tensor is None:
                return f"{label}=None"
            info = [label]
            try:
                info.append(f"shape={tuple(int(dim) for dim in tensor.shape)}")
            except Exception:
                pass
            try:
                if getattr(tensor, "get_layout", None) is not None:
                    info.append(f"layout={tensor.get_layout()}")
            except Exception:
                pass
            try:
                if hasattr(tensor, "storage_type") and hasattr(ttnn, "StorageType"):
                    info.append(f"storage={tensor.storage_type()}")
            except Exception:
                pass
            try:
                mem_cfg = tensor.memory_config() if callable(getattr(tensor, "memory_config", None)) else None
                if mem_cfg is not None:
                    info.append(f"mem={mem_cfg}")
            except Exception:
                pass
            return " ".join(info)

        site = str(weight_key) if weight_key is not None else "<unnamed>"

        def _is_device_tensor(tensor: Any) -> bool:
            if tensor is None:
                return True
            try:
                return tensor.storage_type() == ttnn.StorageType.DEVICE
            except Exception:
                return True

        def _is_device_tensor_nonnull(tensor: Any) -> bool:
            if tensor is None:
                return False
            try:
                return tensor.storage_type() == ttnn.StorageType.DEVICE
            except Exception:
                return False

        if hasattr(x, "storage_type") and hasattr(ttnn, "StorageType"):
            if x.storage_type() != ttnn.StorageType.DEVICE:
                if self._debug_conv2d:
                    try:
                        print(
                            "[maskformer][pixel_decoder] conv2d pre to_device "
                            f"key={site} {_tensor_info('x', x)}"
                        )
                    except Exception:
                        pass
                x = ttnn.to_device(x, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if getattr(x, "get_layout", None) is not None and x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        log_enter = self._debug_conv2d and (
            site not in self._debug_seen_conv2d_sites
            or not _is_device_tensor(x)
            or _is_device_tensor_nonnull(weight)
            or _is_device_tensor_nonnull(bias)
        )
        if log_enter:
            try:
                print(
                    "[maskformer][pixel_decoder] conv2d enter "
                    f"key={weight_key} k={kernel_size} s={stride} p={padding} "
                    f"{_tensor_info('x', x)} {_tensor_info('w', weight)} {_tensor_info('b', bias)}"
                )
            except Exception:
                pass
            self._debug_seen_conv2d_sites.add(site)

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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        # Defensive: conv2d should return device tensors, but avoid TT_FATALs downstream if a host tensor slips through.
        if hasattr(out, "storage_type") and hasattr(ttnn, "StorageType"):
            if out.storage_type() != ttnn.StorageType.DEVICE:
                out = ttnn.to_device(out, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if hasattr(prepared_weight, "storage_type") and hasattr(ttnn, "StorageType"):
            if prepared_weight.storage_type() != ttnn.StorageType.DEVICE:
                prepared_weight = ttnn.to_device(prepared_weight, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if prepared_bias is not None and hasattr(prepared_bias, "storage_type") and hasattr(ttnn, "StorageType"):
            if prepared_bias.storage_type() != ttnn.StorageType.DEVICE:
                prepared_bias = ttnn.to_device(prepared_bias, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        log_exit = log_enter or (
            self._debug_conv2d
            and (not _is_device_tensor(out) or not _is_device_tensor(prepared_weight) or not _is_device_tensor(prepared_bias))
        )
        if log_exit:
            try:
                print(
                    "[maskformer][pixel_decoder] conv2d exit "
                    f"key={weight_key} out_hw=({out_h},{out_w}) "
                    f"{_tensor_info('out', out)} {_tensor_info('w_prep', prepared_weight)} {_tensor_info('b_prep', prepared_bias)}"
                )
            except Exception:
                pass

        if self._cache_conv2d_weights:
            if weight_key is not None:
                self._tt_weights[weight_key] = prepared_weight
            if bias_key is not None:
                self._tt_weights[bias_key] = prepared_bias
        return ttnn.reshape(out, (int(x.shape[0]), out_h, out_w, int(out_channels)))

    def _group_norm(
        self,
        x: Any,
        *,
        name: Optional[str] = None,
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

        def _tensor_info(label: str, tensor: Any) -> str:
            if tensor is None:
                return f"{label}=None"
            info = [label]
            try:
                info.append(f"shape={tuple(int(dim) for dim in tensor.shape)}")
            except Exception:
                pass
            try:
                if getattr(tensor, "get_layout", None) is not None:
                    info.append(f"layout={tensor.get_layout()}")
            except Exception:
                pass
            try:
                if hasattr(tensor, "storage_type") and hasattr(ttnn, "StorageType"):
                    info.append(f"storage={tensor.storage_type()}")
            except Exception:
                pass
            try:
                mem_cfg = tensor.memory_config() if callable(getattr(tensor, "memory_config", None)) else None
                if mem_cfg is not None:
                    info.append(f"mem={mem_cfg}")
            except Exception:
                pass
            return " ".join(info)

        site = name or "<unnamed>"
        if hasattr(x, "storage_type") and hasattr(ttnn, "StorageType"):
            if x.storage_type() != ttnn.StorageType.DEVICE:
                if self._debug_group_norm:
                    try:
                        print(
                            "[maskformer][pixel_decoder] group_norm pre to_device "
                            f"name={site} {_tensor_info('x', x)}"
                        )
                    except Exception:
                        pass
                x = ttnn.to_device(x, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        log_enter = self._debug_group_norm and site not in self._debug_seen_group_norm_sites
        if log_enter:
            try:
                print(
                    "[maskformer][pixel_decoder] group_norm enter "
                    f"name={site} groups={groups} eps={float(self.config.group_norm_epsilon)} "
                    f"{_tensor_info('x', x)} {_tensor_info('w', weight)} {_tensor_info('b', bias)}"
                )
            except Exception:
                pass
            self._debug_seen_group_norm_sites.add(site)

        skip_moreh = name is not None and name.endswith("_block_gn")
        input_nhw = int(B) * int(H) * int(W)
        # Native group_norm requires NHW to be divisible by the tile size (32). Some MaskFormer FPN levels
        # (e.g., 10x10 -> 100) don't satisfy this, so keep them on the manual/moreh paths.
        if self._prefer_native_group_norm and site in self._native_gn_torch_params and (input_nhw % 32 == 0):
            try:
                dtype = self.dtype or get_default_dtype()
                dtype_key = str(dtype)
                cache_key = (site, int(H), int(W), int(C), dtype_key)
                cached = self._native_gn_params_cache.get(cache_key)
                if cached is None:
                    if not hasattr(ttnn, "determine_expected_group_norm_sharded_config_and_grid_size"):
                        raise RuntimeError("ttnn.determine_expected_group_norm_sharded_config_and_grid_size unavailable.")
                    if not hasattr(ttnn, "create_group_norm_input_mask") or not hasattr(
                        ttnn, "create_group_norm_weight_bias_rm"
                    ):
                        raise RuntimeError("Native group_norm helper APIs unavailable.")

                    mem_cfg, core_grid = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
                        device=self.device,
                        num_channels=int(C),
                        num_groups=groups,
                        input_nhw=input_nhw,
                        is_height_sharded=False,
                        is_row_major=True,
                    )

                    num_cores_across_channel = 1
                    try:
                        if (
                            hasattr(ttnn, "TensorMemoryLayout")
                            and mem_cfg.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
                        ):
                            num_cores_across_channel = int(core_grid.y)
                        elif (
                            hasattr(ttnn, "TensorMemoryLayout")
                            and mem_cfg.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                        ):
                            num_cores_across_channel = 1
                        else:
                            num_cores_across_channel = int(core_grid.x) * int(core_grid.y)
                    except Exception:
                        num_cores_across_channel = 1

                    w_torch, b_torch = self._native_gn_torch_params[site]
                    w_rm = ttnn.create_group_norm_weight_bias_rm(w_torch, int(C), num_cores_across_channel)
                    b_rm = ttnn.create_group_norm_weight_bias_rm(b_torch, int(C), num_cores_across_channel)
                    gn_mask = ttnn.create_group_norm_input_mask(int(C), groups, num_cores_across_channel, dtype)
                    gn_mask = ttnn.to_device(gn_mask, self.device)

                    gn_w = ttnn.from_torch(
                        w_rm,
                        dtype=dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    gn_b = ttnn.from_torch(
                        b_rm,
                        dtype=dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    cached = {"mem_cfg": mem_cfg, "core_grid": core_grid, "mask": gn_mask, "w": gn_w, "b": gn_b}
                    self._native_gn_params_cache[cache_key] = cached

                x_1 = ttnn.reshape(x, (int(B), 1, int(H) * int(W), int(C)))
                if getattr(x_1, "get_layout", None) is not None and x_1.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                    x_1 = ttnn.to_layout(x_1, ttnn.ROW_MAJOR_LAYOUT)
                if hasattr(ttnn, "get_memory_config"):
                    if ttnn.get_memory_config(x_1) != cached["mem_cfg"]:
                        x_1 = ttnn.to_memory_config(x_1, cached["mem_cfg"])
                if hasattr(ttnn, "reallocate"):
                    x_1 = ttnn.reallocate(x_1)

                x_norm = ttnn.group_norm(
                    x_1,
                    num_groups=groups,
                    input_mask=cached["mask"],
                    weight=cached["w"],
                    bias=cached["b"],
                    epsilon=float(self.config.group_norm_epsilon),
                    memory_config=ttnn.get_memory_config(x_1),
                    core_grid=cached["core_grid"],
                    dtype=dtype,
                )
                if hasattr(ttnn, "to_memory_config"):
                    x_norm = ttnn.to_memory_config(x_norm, ttnn.DRAM_MEMORY_CONFIG)
                x_out = ttnn.reshape(x_norm, (int(B), int(H), int(W), int(C)))
                if hasattr(x_out, "storage_type") and hasattr(ttnn, "StorageType"):
                    if x_out.storage_type() != ttnn.StorageType.DEVICE:
                        x_out = ttnn.to_device(x_out, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if log_enter:
                    try:
                        print(
                            f"[maskformer][pixel_decoder] group_norm exit name={site} impl=native {_tensor_info('out', x_out)}"
                        )
                    except Exception:
                        pass
                return x_out
            except Exception:
                self._prefer_native_group_norm = False

        if self._prefer_moreh_group_norm and not skip_moreh and moreh_weight is not None and moreh_bias is not None:
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
                if hasattr(x_nchw, "storage_type") and hasattr(ttnn, "StorageType"):
                    if x_nchw.storage_type() != ttnn.StorageType.DEVICE:
                        if self._debug_group_norm and log_enter:
                            try:
                                print(
                                    "[maskformer][pixel_decoder] moreh_group_norm permute returned HOST; moving to device "
                                    f"name={site} {_tensor_info('x_nchw', x_nchw)}"
                                )
                            except Exception:
                                pass
                        x_nchw = ttnn.to_device(x_nchw, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if getattr(x_nchw, "get_layout", None) is not None and x_nchw.get_layout() != ttnn.TILE_LAYOUT:
                    x_nchw = ttnn.to_layout(x_nchw, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if self._debug_group_norm and log_enter:
                    try:
                        print(
                            f"[maskformer][pixel_decoder] moreh_group_norm input name={site} {_tensor_info('x_nchw', x_nchw)}"
                        )
                    except Exception:
                        pass
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
                if hasattr(x_out, "storage_type") and hasattr(ttnn, "StorageType"):
                    if x_out.storage_type() != ttnn.StorageType.DEVICE:
                        x_out = ttnn.to_device(x_out, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if log_enter:
                    try:
                        print(
                            f"[maskformer][pixel_decoder] group_norm exit name={site} impl=moreh {_tensor_info('out', x_out)}"
                        )
                    except Exception:
                        pass
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
        xg = ttnn.reshape(x, (int(B), int(H) * int(W), groups, channels_per_group))
        mean = ttnn.mean(xg, dim=[1, 3], keepdim=True)
        # Avoid ttnn.var here: it has shown instability across repeated invocations on N300.
        # Compute variance via E[x^2] - (E[x])^2.
        xg_sq = xg * xg
        mean_sq = ttnn.mean(xg_sq, dim=[1, 3], keepdim=True)
        if hasattr(ttnn, "deallocate"):
            try:
                ttnn.deallocate(xg_sq)
            except Exception:
                pass
        var = mean_sq - (mean * mean)
        if hasattr(ttnn, "deallocate"):
            try:
                ttnn.deallocate(mean_sq)
            except Exception:
                pass
        denom = ttnn.sqrt(var + float(self.config.group_norm_epsilon))
        if hasattr(ttnn, "deallocate"):
            try:
                ttnn.deallocate(var)
            except Exception:
                pass
        xg_norm = (xg - mean) / denom
        if hasattr(ttnn, "deallocate"):
            try:
                ttnn.deallocate(mean)
            except Exception:
                pass
            try:
                ttnn.deallocate(denom)
            except Exception:
                pass
            try:
                ttnn.deallocate(xg)
            except Exception:
                pass
        x = ttnn.reshape(xg_norm, (int(B), int(H), int(W), int(C)))
        if hasattr(ttnn, "deallocate"):
            try:
                ttnn.deallocate(xg_norm)
            except Exception:
                pass
        x = x * w
        x = x + b
        if hasattr(x, "storage_type") and hasattr(ttnn, "StorageType"):
            if x.storage_type() != ttnn.StorageType.DEVICE:
                x = ttnn.to_device(x, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if log_enter:
            try:
                print(
                    f"[maskformer][pixel_decoder] group_norm exit name={site} impl=manual {_tensor_info('out', x)}"
                )
            except Exception:
                pass
        return x
