# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DPTLargeConfig
from .tt_configs import TTLayerConfig
from .tt_cnn_ops import (
    TTConv2dCached,
    ensure_tt_device_tensor,
    tt_canonicalize_nchw_spatial,
    tt_relu,
    tt_resize_to_nchw,
    tt_upsample_nchw,
)


def _ensure_tt_device_tensor(x, tt_device, ttnn):
    """Ensure TT tensors are materialized in DEVICE storage for TT ops."""
    return ensure_tt_device_tensor(x, tt_device)


def _tt_relu_with_fallback(hidden_state, tt_device, ttnn):
    """Apply ReLU entirely on TT device in the TT fast path."""
    return tt_relu(hidden_state)


def _shape4_hw(x) -> Tuple[int, int]:
    shape = tuple(int(v) for v in tuple(x.shape))
    if len(shape) != 4:
        raise RuntimeError(f"Expected rank-4 tensor, got shape={shape}")
    return int(shape[-2]), int(shape[-1])


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

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        # Fold BN into conv for inference: y = BN(Conv(x)) -> Conv'(x)
        w = conv.weight.detach()
        b = conv.bias.detach() if conv.bias is not None else torch.zeros(w.shape[0], dtype=w.dtype, device=w.device)
        gamma = bn.weight.detach()
        beta = bn.bias.detach()
        mean = bn.running_mean.detach()
        var = bn.running_var.detach()
        inv_std = gamma / torch.sqrt(var + bn.eps)
        fused_w = w * inv_std.reshape(-1, 1, 1, 1)
        fused_b = (b - mean) * inv_std + beta
        return fused_w, fused_b

    def _tt_convolution(self, which: int, x):
        conv = self.convolution1 if which == 1 else self.convolution2
        cache = self._tt_conv1 if which == 1 else self._tt_conv2

        if cache is None:
            if self.use_batch_norm:
                bn = self.batch_norm1 if which == 1 else self.batch_norm2
                fused_w, fused_b = self._fuse_conv_bn(conv, bn)
                cache = TTConv2dCached.from_tensors(
                    weight_torch=fused_w,
                    bias_torch=fused_b,
                    stride=(1, 1),
                    padding=(1, 1),
                )
            else:
                cache = TTConv2dCached.from_conv(conv)
            if which == 1:
                self._tt_conv1 = cache
            else:
                self._tt_conv2 = cache
        return cache(x, device=self.tt_device)

    def forward(self, hidden_state: torch.Tensor):
        try:
            import ttnn  # type: ignore
        except Exception:
            ttnn = None

        if (
            ttnn is not None
            and self.tt_device is not None
            and isinstance(hidden_state, ttnn.Tensor)
        ):
            residual = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)
            # Device ReLU with fallback to host
            hidden_state = _tt_relu_with_fallback(residual, self.tt_device, ttnn)

            hidden_state = self._tt_convolution(1, hidden_state)

            # Device ReLU with fallback to host
            hidden_state = _tt_relu_with_fallback(hidden_state, self.tt_device, ttnn)

            hidden_state = self._tt_convolution(2, hidden_state)
            # residual add on device
            hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)
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

    def __init__(
        self,
        channels: int,
        use_batch_norm: bool,
        tt_device=None,
        memcfg=None,
        align_corners: bool = True,
        approx_align_corners: bool = False,
    ):
        super().__init__()
        self.tt_device = tt_device
        self.memcfg = memcfg
        self.align_corners = align_corners
        self.approx_align_corners = approx_align_corners

        self.projection = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.residual_layer1 = DPTPreActResidualLayerTT(channels, use_batch_norm, tt_device=tt_device, memcfg=memcfg)
        self.residual_layer2 = DPTPreActResidualLayerTT(channels, use_batch_norm, tt_device=tt_device, memcfg=memcfg)

        self._tt_proj = None

    def _tt_projection(self, x):
        if self._tt_proj is None:
            self._tt_proj = TTConv2dCached.from_conv(self.projection)
        return self._tt_proj(x, device=self.tt_device)

    def forward(
        self,
        hidden_state,
        residual=None,
        *,
        expected_input_hw: Optional[Tuple[int, int]] = None,
        expected_output_hw: Optional[Tuple[int, int]] = None,
    ):
        try:
            import ttnn  # type: ignore
        except Exception:
            ttnn = None

        if (
            ttnn is not None
            and isinstance(hidden_state, ttnn.Tensor)
            and self.tt_device is not None
        ):
            hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)
            hidden_state = tt_canonicalize_nchw_spatial(
                hidden_state,
                expected_hw=expected_input_hw,
                op_name="dpt_fusion.hidden_state.input",
            )
            if residual is not None:
                residual = _ensure_tt_device_tensor(residual, self.tt_device, ttnn)
                residual = tt_canonicalize_nchw_spatial(
                    residual,
                    expected_hw=expected_input_hw,
                    op_name="dpt_fusion.residual.input",
                )
                # `ttnn.Shape` does not support slicing, so work with a plain
                # Python tuple when comparing spatial dims.
                hs_shape = tuple(hidden_state.shape)
                res_shape = tuple(residual.shape)
                if hs_shape[-2] != res_shape[-2] or hs_shape[-1] != res_shape[-1]:
                    residual = tt_resize_to_nchw(
                        residual,
                        target_hw=(hs_shape[-2], hs_shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                        memory_config=self.memcfg,
                        op_name="dpt_fusion.residual.resize",
                    )
                residual_out = self.residual_layer1(residual)
                hidden_state = ttnn.add(hidden_state, residual_out)

            hidden_state = self.residual_layer2(hidden_state)
            in_h, in_w = _shape4_hw(hidden_state)
            hidden_state = tt_upsample_nchw(
                hidden_state,
                scale_factor=2,
                mode="bilinear",
                align_corners=self.align_corners,
                approx_align_corners=self.approx_align_corners,
                memory_config=self.memcfg,
                expected_input_hw=(in_h, in_w),
                op_name="dpt_fusion.hidden_state.upsample",
            )
            if expected_output_hw is not None:
                hidden_state = tt_canonicalize_nchw_spatial(
                    hidden_state,
                    expected_hw=expected_output_hw,
                    op_name="dpt_fusion.hidden_state.output",
                )
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
        approx_align_corners = bool(getattr(config, "tt_approx_align_corners", False))
        self.layers = nn.ModuleList(
            [
                DPTFeatureFusionLayerTT(
                    channels=channels,
                    use_batch_norm=use_bn,
                    tt_device=tt_device,
                    memcfg=memcfg,
                    approx_align_corners=approx_align_corners,
                )
                for _ in range(len(config.neck_hidden_sizes))
            ]
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        # Reverse to start from the last feature map, mirroring HF.
        hidden_states = hidden_states[::-1]

        fused_hidden_states: List[torch.Tensor] = []
        stage0_in_hw = None
        stage0_out_hw = None
        if len(hidden_states) >= 2:
            next_h, next_w = _shape4_hw(hidden_states[1])
            if next_h % 2 == 0 and next_w % 2 == 0:
                stage0_in_hw = (next_h // 2, next_w // 2)
            stage0_out_hw = (next_h, next_w)
        fused_hidden_state = self.layers[0](
            hidden_states[0],
            expected_input_hw=stage0_in_hw,
            expected_output_hw=stage0_out_hw,
        )
        fused_hidden_states.append(fused_hidden_state)

        for idx, (hidden_state, layer) in enumerate(zip(hidden_states[1:], self.layers[1:]), start=1):
            exp_in_hw = _shape4_hw(hidden_state)
            exp_out_hw = (exp_in_hw[0] * 2, exp_in_hw[1] * 2)
            # For non-terminal fusion layers, anchor expected output to the next
            # residual map when available.
            if idx + 1 < len(hidden_states):
                exp_out_hw = _shape4_hw(hidden_states[idx + 1])
            fused_hidden_state = layer(
                fused_hidden_state,
                hidden_state,
                expected_input_hw=exp_in_hw,
                expected_output_hw=exp_out_hw,
            )
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
        self.tt_approx_align_corners = bool(getattr(config, "tt_approx_align_corners", False))

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
        if index in self._tt_head_convs:
            return self._tt_head_convs[index]
        tt_conv = TTConv2dCached.from_conv(conv)
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
        import ttnn  # type: ignore

        hidden_state = hidden_states[self.config.head_in_index]

        if self.projection is not None:
            if self._tt_proj is None:
                self._tt_proj = TTConv2dCached.from_conv(self.projection)
            hidden_state = self._tt_proj(hidden_state, device=self.tt_device)
            hidden_state = _tt_relu_with_fallback(hidden_state, self.tt_device, ttnn)

        # Head[0] conv
        conv0 = self.head[0]
        tt_conv0 = self._tt_conv_for(0, conv0)
        hidden_state = tt_conv0(hidden_state, device=self.tt_device)

        # Upsample
        hidden_state = tt_upsample_nchw(
            hidden_state,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
            approx_align_corners=self.tt_approx_align_corners,
            memory_config=self.memcfg,
            expected_input_hw=_shape4_hw(hidden_state),
            op_name="dpt_depth_head.upsample",
        )

        # Conv to 32 channels + device ReLU
        conv1 = self.head[2]
        tt_conv1 = self._tt_conv_for(2, conv1)
        hidden_state = tt_conv1(hidden_state, device=self.tt_device)
        hidden_state = _tt_relu_with_fallback(hidden_state, self.tt_device, ttnn)

        # Final 1x1 conv to depth + device ReLU
        conv2 = self.head[4]
        tt_conv2 = self._tt_conv_for(4, conv2)
        hidden_state = tt_conv2(hidden_state, device=self.tt_device)
        hidden_state = _tt_relu_with_fallback(hidden_state, self.tt_device, ttnn)
        hidden_state = _ensure_tt_device_tensor(hidden_state, self.tt_device, ttnn)

        return hidden_state

    def forward(self, hidden_states: List[torch.Tensor]):
        """
        Dispatch to either the TT or torch implementation depending on the
        type of the incoming feature maps and TT device availability.
        """
        try:
            import ttnn  # type: ignore
        except Exception:
            ttnn = None

        # Optionally convert incoming torch tensors to TT tensors to keep fusion entirely on device
        if (
            ttnn is not None
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
            ttnn is not None
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
    `DPTDepthEstimationHead`, with TT-native conv/upsample execution in TT mode.
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
