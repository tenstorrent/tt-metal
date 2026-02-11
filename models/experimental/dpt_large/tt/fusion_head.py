# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DPTLargeConfig
from .tt_configs import TTLayerConfig
from models.common.utility_functions import torch_to_tt_tensor_rm


def _ensure_tt_device_tensor(x, tt_device, ttnn):
    """Ensure TT tensors are materialized in DEVICE storage for TT ops."""
    if tt_device is None:
        return x
    if not isinstance(x, ttnn.Tensor):
        if torch.is_tensor(x):
            return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=tt_device)
        return x
    try:
        if x.storage_type() == ttnn.StorageType.DEVICE:
            return x
    except Exception:
        return x
    try:
        return x.to(tt_device)
    except Exception:
        x_host = x.cpu()
        if hasattr(x_host, "layout") and x_host.layout == ttnn.TILE_LAYOUT:
            x_host = x_host.to(ttnn.ROW_MAJOR_LAYOUT)
        x_torch = x_host.to_torch()
        return ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=tt_device)


def _tt_relu_with_fallback(hidden_state, tt_device, ttnn):
    """Apply ReLU on device if possible, fallback to host otherwise."""
    try:
        # ttnn.relu requires TILE_LAYOUT; convert if needed
        if hasattr(hidden_state, "layout") and hidden_state.layout != ttnn.TILE_LAYOUT:
            hidden_state = ttnn.to_layout(hidden_state, ttnn.TILE_LAYOUT)
        return ttnn.relu(hidden_state)
    except Exception:
        # Fallback to host relu if ttnn.relu is unavailable
        x_host = hidden_state.cpu()
        if hasattr(x_host, "layout") and x_host.layout == ttnn.TILE_LAYOUT:
            x_host = x_host.to(ttnn.ROW_MAJOR_LAYOUT)
        x_torch = x_host.to_torch()
        x_torch = torch.relu(x_torch)
        return ttnn.from_torch(
            x_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=tt_device,
        )


class DPTPreActResidualLayerTT(nn.Module):
    """
    TT-aware variant of Hugging Face's `DPTPreActResidualLayer`.
    """

    def __init__(self, channels: int, use_batch_norm: bool, tt_device=None, memcfg=None):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.tt_device = tt_device
        self.memcfg = memcfg

        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not use_batch_norm,
        )

        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not use_batch_norm,
        )

        if self.use_batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(channels)
            self.batch_norm2 = nn.BatchNorm2d(channels)

        # TT caches
        self._tt_conv1 = None
        self._tt_conv2 = None
        self._tt_bn1 = None
        self._tt_bn2 = None

    def _tt_convolution(self, which: int, x):
        import tt_lib.fallback_ops as fallback_ops  # type: ignore
        import ttnn  # type: ignore

        conv = self.convolution1 if which == 1 else self.convolution2
        cache = self._tt_conv1 if which == 1 else self._tt_conv2

        if cache is None:
            wt = torch_to_tt_tensor_rm(conv.weight.detach(), self.tt_device, put_on_device=True)
            bs = (
                torch_to_tt_tensor_rm(conv.bias.detach(), self.tt_device, put_on_device=True)
                if conv.bias is not None
                else None
            )
            cache = fallback_ops.Conv2d(
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size[0],
                weights=wt,
                biases=bs,
                stride=1,
                padding=1,
            )
            if which == 1:
                self._tt_conv1 = cache
            else:
                self._tt_conv2 = cache
        return cache(x)

    def _tt_batch_norm(self, which: int, x):
        import tt_lib.fallback_ops as fallback_ops  # type: ignore
        import ttnn  # type: ignore

        if not self.use_batch_norm:
            return x

        bnorm = self.batch_norm1 if which == 1 else self.batch_norm2
        cache = self._tt_bn1 if which == 1 else self._tt_bn2

        if cache is None:
            w = torch_to_tt_tensor_rm(bnorm.weight.detach(), self.tt_device, put_on_device=True)
            b = torch_to_tt_tensor_rm(bnorm.bias.detach(), self.tt_device, put_on_device=True)
            rm = torch_to_tt_tensor_rm(bnorm.running_mean.detach(), self.tt_device, put_on_device=True)
            rv = torch_to_tt_tensor_rm(bnorm.running_var.detach(), self.tt_device, put_on_device=True)
            cache = fallback_ops.BatchNorm2d(
                weights=w,
                biases=b,
                running_mean=rm,
                running_var=rv,
                num_batches_tracked=bnorm.num_batches_tracked,
                num_features=bnorm.num_features,
                eps=bnorm.eps,
                momentum=bnorm.momentum,
            )
            if which == 1:
                self._tt_bn1 = cache
            else:
                self._tt_bn2 = cache
        return cache(x)

    def forward(self, hidden_state: torch.Tensor):
        try:
            import tt_lib.fallback_ops as fallback_ops  # type: ignore
            import ttnn  # type: ignore
        except Exception:
            fallback_ops = None
            ttnn = None

        if (
            fallback_ops is not None
            and ttnn is not None
            and self.tt_device is not None
            and isinstance(hidden_state, ttnn.Tensor)
        ):
            residual = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)
            # Device ReLU with fallback to host
            hidden_state = _tt_relu_with_fallback(residual, self.tt_device, ttnn)
            hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)

            hidden_state = self._tt_convolution(1, hidden_state)
            hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)
            hidden_state = self._tt_batch_norm(1, hidden_state)
            hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)

            # Device ReLU with fallback to host
            hidden_state = _tt_relu_with_fallback(hidden_state, self.tt_device, ttnn)
            hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)

            hidden_state = self._tt_convolution(2, hidden_state)
            hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)
            hidden_state = self._tt_batch_norm(2, hidden_state)
            hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)
            # residual add on device
            return ttnn.add(hidden_state, residual)

        residual = hidden_state
        hidden_state = self.activation1(hidden_state)

        hidden_state = self.convolution1(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm1(hidden_state)

        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm2(hidden_state)

        return hidden_state + residual


class DPTFeatureFusionLayerTT(nn.Module):
    """
    TT-aware variant of Hugging Face's `DPTFeatureFusionLayer`.
    """

    def __init__(self, channels: int, use_batch_norm: bool, tt_device=None, memcfg=None, align_corners: bool = True):
        super().__init__()
        self.tt_device = tt_device
        self.memcfg = memcfg
        self.align_corners = align_corners

        self.projection = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.residual_layer1 = DPTPreActResidualLayerTT(channels, use_batch_norm, tt_device=tt_device, memcfg=memcfg)
        self.residual_layer2 = DPTPreActResidualLayerTT(channels, use_batch_norm, tt_device=tt_device, memcfg=memcfg)

        self._tt_proj = None

    def _tt_projection(self, x):
        import tt_lib.fallback_ops as fallback_ops  # type: ignore
        import ttnn  # type: ignore

        if self._tt_proj is None:
            wt = torch_to_tt_tensor_rm(self.projection.weight.detach(), self.tt_device, put_on_device=True)
            bs = torch_to_tt_tensor_rm(self.projection.bias.detach(), self.tt_device, put_on_device=True)
            self._tt_proj = fallback_ops.Conv2d(
                in_channels=self.projection.in_channels,
                out_channels=self.projection.out_channels,
                kernel_size=self.projection.kernel_size[0],
                weights=wt,
                biases=bs,
                stride=1,
                padding=0,
            )
        return self._tt_proj(x)

    def forward(self, hidden_state, residual=None):
        try:
            import tt_lib.fallback_ops as fallback_ops  # type: ignore
            import ttnn  # type: ignore
        except Exception:
            fallback_ops = None
            ttnn = None

        if (
            fallback_ops is not None
            and ttnn is not None
            and isinstance(hidden_state, ttnn.Tensor)
            and self.tt_device is not None
        ):
            hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)
            if residual is not None:
                residual = _ensure_tt_device_tensor(residual, self.tt_device, ttnn)
                # `ttnn.Shape` does not support slicing, so work with a plain
                # Python tuple when comparing spatial dims.
                hs_shape = tuple(hidden_state.shape)
                res_shape = tuple(residual.shape)
                if hs_shape[-2] != res_shape[-2] or hs_shape[-1] != res_shape[-1]:
                    residual = fallback_ops.interpolate(
                        residual,
                        size=(hs_shape[-2], hs_shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    residual = _ensure_tt_device_tensor(residual, self.tt_device, ttnn)
                residual_out = self.residual_layer1(residual)
                residual_out = _ensure_tt_device_tensor(residual_out, self.tt_device, ttnn)
                hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)
                hidden_state = ttnn.add(hidden_state, residual_out)

            hidden_state = self.residual_layer2(hidden_state)
            hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)
            hidden_state = fallback_ops.interpolate(
                hidden_state,
                scale_factor=2,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)
            hidden_state = self._tt_projection(hidden_state)
            hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)
            return hidden_state

        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = F.interpolate(
                    residual,
                    size=(hidden_state.shape[2], hidden_state.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            hidden_state = hidden_state + self.residual_layer1(residual)

        hidden_state = self.residual_layer2(hidden_state)
        hidden_state = F.interpolate(
            hidden_state,
            scale_factor=2,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        hidden_state = self.projection(hidden_state)
        return hidden_state


class DPTFeatureFusionStageTT(nn.Module):
    """
    Feature fusion stage composed of multiple `DPTFeatureFusionLayerTT`s.
    """

    def __init__(self, config: DPTLargeConfig, tt_device=None, memcfg=None):
        super().__init__()
        channels = config.fusion_hidden_size
        use_bn = config.use_batch_norm_in_fusion_residual
        self.layers = nn.ModuleList(
            [
                DPTFeatureFusionLayerTT(
                    channels=channels,
                    use_batch_norm=use_bn,
                    tt_device=tt_device,
                    memcfg=memcfg,
                )
                for _ in range(len(config.neck_hidden_sizes))
            ]
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        # Reverse to start from the last feature map, mirroring HF.
        hidden_states = hidden_states[::-1]

        fused_hidden_states: List[torch.Tensor] = []
        fused_hidden_state = self.layers[0](hidden_states[0])
        fused_hidden_states.append(fused_hidden_state)

        for hidden_state, layer in zip(hidden_states[1:], self.layers[1:]):
            fused_hidden_state = layer(fused_hidden_state, hidden_state)
            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


class DPTDepthEstimationHeadTT(nn.Module):
    """
    TT-aware variant of Hugging Face's `DPTDepthEstimationHead`.
    """

    def __init__(self, config: DPTLargeConfig, tt_device=None, memcfg=None):
        super().__init__()
        self.config = config
        self.tt_device = tt_device
        self.memcfg = memcfg

        self.projection: Optional[nn.Conv2d] = None
        if config.add_projection:
            self.projection = nn.Conv2d(
                256,
                256,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        features = config.fusion_hidden_size
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        # TT caches for head convs / projection.
        self._tt_proj = None
        self._tt_head_convs: Dict[int, object] = {}

    def load_from_hf_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        # Optional projection
        if self.projection is not None:
            if "head.projection.weight" in state_dict:
                self.projection.weight.data.copy_(state_dict["head.projection.weight"])
            if "head.projection.bias" in state_dict:
                self.projection.bias.data.copy_(state_dict["head.projection.bias"])

        # Main conv stack
        conv0 = self.head[0]
        conv1 = self.head[2]
        conv2 = self.head[4]

        conv0.weight.data.copy_(state_dict["head.head.0.weight"])
        conv0.bias.data.copy_(state_dict["head.head.0.bias"])

        conv1.weight.data.copy_(state_dict["head.head.2.weight"])
        conv1.bias.data.copy_(state_dict["head.head.2.bias"])

        conv2.weight.data.copy_(state_dict["head.head.4.weight"])
        conv2.bias.data.copy_(state_dict["head.head.4.bias"])

    def _tt_conv_for(self, index: int, conv: nn.Conv2d):
        import tt_lib.fallback_ops as fallback_ops  # type: ignore

        if index in self._tt_head_convs:
            return self._tt_head_convs[index]

        wt = torch_to_tt_tensor_rm(conv.weight.detach(), self.tt_device, put_on_device=True)
        bs = torch_to_tt_tensor_rm(conv.bias.detach(), self.tt_device, put_on_device=True)
        tt_conv = fallback_ops.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size[0],
            weights=wt,
            biases=bs,
            stride=conv.stride[0],
            padding=conv.padding[0],
        )
        self._tt_head_convs[index] = tt_conv
        return tt_conv

    def _forward_torch(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        hidden_state = hidden_states[self.config.head_in_index]

        if self.projection is not None:
            hidden_state = self.projection(hidden_state)
            hidden_state = F.relu(hidden_state)

        predicted_depth = self.head(hidden_state)
        return predicted_depth

    def _forward_tt(self, hidden_states):
        import tt_lib.fallback_ops as fallback_ops  # type: ignore
        import ttnn  # type: ignore

        hidden_state = hidden_states[self.config.head_in_index]

        if self.projection is not None:
            if self._tt_proj is None:
                wt = torch_to_tt_tensor_rm(self.projection.weight.detach(), self.tt_device, put_on_device=True)
                bs = torch_to_tt_tensor_rm(self.projection.bias.detach(), self.tt_device, put_on_device=True)
                self._tt_proj = fallback_ops.Conv2d(
                    in_channels=self.projection.in_channels,
                    out_channels=self.projection.out_channels,
                    kernel_size=self.projection.kernel_size[0],
                    weights=wt,
                    biases=bs,
                    stride=1,
                    padding=1,
                )
            hidden_state = self._tt_proj(hidden_state)
            hidden_state = _tt_relu_with_fallback(hidden_state, self.tt_device, ttnn)

        # Head[0] conv
        conv0 = self.head[0]
        tt_conv0 = self._tt_conv_for(0, conv0)
        hidden_state = tt_conv0(hidden_state)
        hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)

        # Upsample
        hidden_state = fallback_ops.interpolate(
            hidden_state,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)

        # Conv to 32 channels + device ReLU
        conv1 = self.head[2]
        tt_conv1 = self._tt_conv_for(2, conv1)
        hidden_state = tt_conv1(hidden_state)
        hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)
        try:
            hidden_state = ttnn.relu(hidden_state)
        except Exception:
            # Fallback to host relu if ttnn.relu is unavailable
            x_host = hidden_state.cpu()
            if hasattr(x_host, "layout") and x_host.layout == ttnn.TILE_LAYOUT:
                x_host = x_host.to(ttnn.ROW_MAJOR_LAYOUT)
            x_torch = x_host.to_torch()
            x_torch = torch.relu(x_torch)
            hidden_state = ttnn.from_torch(
                x_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.tt_device,
            )

        # Final 1x1 conv to depth + device ReLU
        conv2 = self.head[4]
        tt_conv2 = self._tt_conv_for(4, conv2)
        hidden_state = tt_conv2(hidden_state)
        hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)
        try:
            hidden_state = ttnn.relu(hidden_state)
        except Exception:
            x_host = hidden_state.cpu()
            if hasattr(x_host, "layout") and x_host.layout == ttnn.TILE_LAYOUT:
                x_host = x_host.to(ttnn.ROW_MAJOR_LAYOUT)
            x_torch = x_host.to_torch()
            x_torch = torch.relu(x_torch)
            hidden_state = ttnn.from_torch(
                x_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.tt_device,
            )

        return hidden_state

    def forward(self, hidden_states: List[torch.Tensor]):
        """
        Dispatch to either the TT or torch implementation depending on the
        type of the incoming feature maps and TT device availability.
        """
        try:
            import tt_lib.fallback_ops as fallback_ops  # type: ignore
            import ttnn  # type: ignore
        except Exception:
            fallback_ops = None
            ttnn = None

        # Optionally convert incoming torch tensors to TT tensors to keep fusion entirely on device
        if (
            fallback_ops is not None
            and ttnn is not None
            and self.tt_device is not None
            and (getattr(self.config, "tt_device_fusion", False) or getattr(self.config, "tt_perf_neck", False))
        ):
            new_states: List[torch.Tensor] = []
            for x in hidden_states:
                if isinstance(x, ttnn.Tensor):
                    new_states.append(_ensure_tt_device_tensor(x, self.tt_device, ttnn))
                else:
                    try:
                        new_states.append(
                            ttnn.from_torch(
                                x,
                                dtype=ttnn.bfloat16,
                                layout=ttnn.ROW_MAJOR_LAYOUT,
                                device=self.tt_device,
                            )
                        )
                    except Exception:
                        new_states.append(x)
            hidden_states = new_states

        use_tt = (
            fallback_ops is not None
            and ttnn is not None
            and self.tt_device is not None
            and len(hidden_states) > self.config.head_in_index
            and isinstance(hidden_states[self.config.head_in_index], ttnn.Tensor)
        )

        if use_tt:
            return self._forward_tt(hidden_states)
        return self._forward_torch(hidden_states)


class DPTFusionHead(nn.Module):
    """
    Fusion + depth head that mirrors Hugging Face's DPT neck fusion stage and
    `DPTDepthEstimationHead`, but with TT device support via fallback ops.
    """

    def __init__(
        self,
        config: DPTLargeConfig | None = None,
        channels: int = 256,
        tt_device=None,
        layer_cfg: TTLayerConfig | None = None,
    ):
        super().__init__()
        self.config = config if config is not None else DPTLargeConfig()
        self.channels = channels
        self.tt_device = tt_device
        self.layer_cfg = layer_cfg
        self.memcfg = layer_cfg.memcfg() if layer_cfg else None

        # Fusion stage (merges multi-scale features to a pyramid of fused maps).
        self.fusion_stage = DPTFeatureFusionStageTT(config=self.config, tt_device=tt_device, memcfg=self.memcfg)

        # Depth head operating on fused feature maps.
        self.depth_head = DPTDepthEstimationHeadTT(config=self.config, tt_device=tt_device, memcfg=self.memcfg)

    # ------------------------------------------------------------------ weight loading
    def load_from_hf_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        # Fusion stage weights
        for i, layer in enumerate(self.fusion_stage.layers):
            base = f"neck.fusion_stage.layers.{i}"
            # projection
            layer.projection.weight.data.copy_(state_dict[f"{base}.projection.weight"])
            layer.projection.bias.data.copy_(state_dict[f"{base}.projection.bias"])

            # residual layers
            for which, res_name in enumerate(["residual_layer1", "residual_layer2"], start=1):
                res = getattr(layer, res_name)
                prefix = f"{base}.{res_name}"

                res.convolution1.weight.data.copy_(state_dict[f"{prefix}.convolution1.weight"])
                bkey = f"{prefix}.convolution1.bias"
                if res.convolution1.bias is not None and bkey in state_dict:
                    res.convolution1.bias.data.copy_(state_dict[bkey])

                res.convolution2.weight.data.copy_(state_dict[f"{prefix}.convolution2.weight"])
                bkey2 = f"{prefix}.convolution2.bias"
                if res.convolution2.bias is not None and bkey2 in state_dict:
                    res.convolution2.bias.data.copy_(state_dict[bkey2])

                if res.use_batch_norm:
                    res.batch_norm1.weight.data.copy_(state_dict[f"{prefix}.batch_norm1.weight"])
                    res.batch_norm1.bias.data.copy_(state_dict[f"{prefix}.batch_norm1.bias"])
                    res.batch_norm1.running_mean.data.copy_(state_dict[f"{prefix}.batch_norm1.running_mean"])
                    res.batch_norm1.running_var.data.copy_(state_dict[f"{prefix}.batch_norm1.running_var"])

                    res.batch_norm2.weight.data.copy_(state_dict[f"{prefix}.batch_norm2.weight"])
                    res.batch_norm2.bias.data.copy_(state_dict[f"{prefix}.batch_norm2.bias"])
                    res.batch_norm2.running_mean.data.copy_(state_dict[f"{prefix}.batch_norm2.running_mean"])
                    res.batch_norm2.running_var.data.copy_(state_dict[f"{prefix}.batch_norm2.running_var"])

        # Depth head weights
        self.depth_head.load_from_hf_state_dict(state_dict)

    # ------------------------------------------------------------------ forward
    def forward(self, pyramid_feats: List[torch.Tensor]):
        """
        Args:
            pyramid_feats: list of feature maps from the neck, each with
                `fusion_hidden_size` channels.
        """
        fused = self.fusion_stage(pyramid_feats)
        depth = self.depth_head(fused)
        return depth
