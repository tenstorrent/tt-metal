# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .config import DPTLargeConfig, DEFAULT_CONFIG
from .vit_backbone import ViTBackboneOutputs
from .tt_configs import TTLayerConfig
from models.common.utility_functions import torch_to_tt_tensor_rm


class DPTReassembleLayerTT(nn.Module):
    """
    TT-aware variant of Hugging Face's `DPTReassembleLayer`.

    It projects from the ViT hidden size to a per-scale channel size, then
    performs up/down sampling depending on the `factor`. On TT devices the
    projection runs via `fallback_ops.Conv2d`, while upsampling falls back to
    bilinear interpolation for now.
    """

    def __init__(self, hidden_size: int, channels: int, factor: float, tt_device=None, memcfg=None):
        super().__init__()
        self.tt_device = tt_device
        self.memcfg = memcfg
        self.factor = factor

        # 1x1 projection from hidden_size -> channels
        self.projection = nn.Conv2d(in_channels=hidden_size, out_channels=channels, kernel_size=1)

        # Up/down sampling depending on factor (CPU path mirrors HF exactly).
        if factor > 1:
            self.resize = nn.ConvTranspose2d(channels, channels, kernel_size=int(factor), stride=int(factor), padding=0)
        elif factor == 1:
            self.resize = nn.Identity()
        else:
            # downsample: stride=int(1 / factor)
            stride = int(1.0 / factor)
            self.resize = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1)

        # TT conv caches
        self._tt_proj = None
        self._tt_resize_conv = None

    def _tt_project(self, x):
        import tt_lib.fallback_ops as fallback_ops  # type: ignore
        import ttnn  # type: ignore

        if self._tt_proj is None:
            wt = torch_to_tt_tensor_rm(self.projection.weight.detach(), self.tt_device, put_on_device=True)
            bs = torch_to_tt_tensor_rm(self.projection.bias.detach(), self.tt_device, put_on_device=True)
            self._tt_proj = fallback_ops.Conv2d(
                weights=wt,
                biases=bs,
                in_channels=self.projection.in_channels,
                out_channels=self.projection.out_channels,
                kernel_size=self.projection.kernel_size[0],
                stride=1,
                padding=0,
            )
        return self._tt_proj(x)

    def _tt_resize(self, x):
        import tt_lib.fallback_ops as fallback_ops  # type: ignore
        import ttnn  # type: ignore

        # For now we approximate ConvTranspose2d with bilinear interpolate on TT.
        if self.factor > 1:
            return fallback_ops.interpolate(
                x,
                scale_factor=self.factor,
                mode="bilinear",
                align_corners=False,
            )

        if isinstance(self.resize, nn.Identity):
            return x

        if self._tt_resize_conv is None:
            # Only downsampling path uses a real Conv2d kernel.
            wt = torch_to_tt_tensor_rm(self.resize.weight.detach(), self.tt_device, put_on_device=True)
            bs = torch_to_tt_tensor_rm(self.resize.bias.detach(), self.tt_device, put_on_device=True)
            self._tt_resize_conv = fallback_ops.Conv2d(
                weights=wt,
                biases=bs,
                in_channels=self.resize.in_channels,
                out_channels=self.resize.out_channels,
                kernel_size=self.resize.kernel_size[0],
                stride=self.resize.stride[0],
                padding=self.resize.padding[0],
            )
        return self._tt_resize_conv(x)

    def forward(self, x):
        # x: torch.Tensor or ttnn.Tensor of shape [B, hidden_size, H, W] / TT layout.
        try:
            import tt_lib.fallback_ops as fallback_ops  # type: ignore
            import ttnn  # type: ignore
        except Exception:
            fallback_ops = None
            ttnn = None

        if fallback_ops is not None and ttnn is not None and isinstance(x, ttnn.Tensor) and self.tt_device is not None:
            x = self._tt_project(x)
            x = self._tt_resize(x)
            return x

        x = self.projection(x)
        x = self.resize(x)
        return x


class DPTReassembly(nn.Module):
    """
    Neck module mirroring Hugging Face's DPT reassemble + 3x3 conv stage.

    It takes backbone outputs and, for each selected layer, applies:

      1. A DPT-style reassemble layer that projects from `hidden_size` to
         `neck_hidden_sizes[i]` and performs up/downsampling according to
         `reassemble_factors[i]`.
      2. A 3x3 convolution mapping from `neck_hidden_sizes[i]` to
         `fusion_hidden_size`.

    The resulting list of feature maps is fed into the fusion head.
    """

    def __init__(
        self,
        config: DPTLargeConfig | None = None,
        proj_channels: Optional[int] = None,
        tt_device=None,
        layer_cfg: TTLayerConfig | None = None,
    ):
        super().__init__()
        self.config = config if config is not None else DPTLargeConfig()
        self.tt_device = tt_device
        self.layer_cfg = layer_cfg
        memcfg = layer_cfg.memcfg() if layer_cfg else None
        hidden_size = self.config.hidden_size
        # Allow small test configs to override the fusion channel width without
        # touching the HF-aligned `config.fusion_hidden_size`. The HF
        # checkpoint uses 256, which is the default here.
        self.fusion_channels = proj_channels if proj_channels is not None else self.config.fusion_hidden_size

        # Readout configuration mirrors HF's DPTReassembleStage for the
        # non-hybrid DPT model. We only implement the "project" path for now,
        # which is what DPT-Large uses.
        self.readout_type = self.config.readout_type
        self.hidden_act = self.config.hidden_act

        self.readout_projects: Optional[nn.ModuleList] = None
        if self.readout_type == "project":
            # One readout projection per reassemble stage.
            self.readout_projects = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * hidden_size, hidden_size),
                    nn.GELU() if self.hidden_act == "gelu" else nn.ReLU(),
                )
                    for _ in range(len(self.config.neck_hidden_sizes))
                ]
            )

        # Reassemble layers: hidden_size -> neck_hidden_sizes[i] with scale factor.
        self.reassemble_layers = nn.ModuleList(
            [
                DPTReassembleLayerTT(
                    hidden_size=hidden_size,
                    channels=self.config.neck_hidden_sizes[i],
                    factor=self.config.reassemble_factors[i],
                    tt_device=tt_device,
                    memcfg=memcfg,
                )
                for i in range(len(self.config.neck_hidden_sizes))
            ]
        )

        # 3x3 convs: neck_hidden_sizes[i] -> fusion_hidden_size.
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.config.neck_hidden_sizes[i],
                    out_channels=self.fusion_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
                for i in range(len(self.config.neck_hidden_sizes))
            ]
        )

        # TT conv caches for the 3x3 convs.
        self._tt_convs = [None for _ in range(len(self.convs))]

    # ------------------------------------------------------------------ weight loading
    def load_from_hf_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load neck weights from a Hugging Face `DPTForDepthEstimation` state dict.
        """
        # Reassemble stage (readout + projection + resize).
        for i, layer in enumerate(self.reassemble_layers):
            # Readout projections (neck.reassemble_stage.readout_projects.i)
            if self.readout_projects is not None:
                rp = self.readout_projects[i][0]  # Linear
                base_r = f"neck.reassemble_stage.readout_projects.{i}.0"
                rp_w = state_dict.get(f"{base_r}.weight", None)
                rp_b = state_dict.get(f"{base_r}.bias", None)
                if rp_w is not None:
                    rp.weight.data.copy_(rp_w)
                if rp_b is not None:
                    rp.bias.data.copy_(rp_b)

            base = f"neck.reassemble_stage.layers.{i}"
            proj_w = state_dict.get(f"{base}.projection.weight", None)
            proj_b = state_dict.get(f"{base}.projection.bias", None)
            if proj_w is not None:
                layer.projection.weight.data.copy_(proj_w)
            if proj_b is not None:
                layer.projection.bias.data.copy_(proj_b)

            # Resize weights only exist for factor != 1.
            if isinstance(layer.resize, (nn.Conv2d, nn.ConvTranspose2d)):
                resize_w = state_dict.get(f"{base}.resize.weight", None)
                resize_b = state_dict.get(f"{base}.resize.bias", None)
                if resize_w is not None:
                    layer.resize.weight.data.copy_(resize_w)
                if resize_b is not None:
                    layer.resize.bias.data.copy_(resize_b)

        # 3x3 convs from neck_hidden_sizes -> fusion_hidden_size.
        for i, conv in enumerate(self.convs):
            key = f"neck.convs.{i}.weight"
            if key in state_dict:
                conv.weight.data.copy_(state_dict[key])

    # ------------------------------------------------------------------ forward
    def forward(self, feats: ViTBackboneOutputs) -> List[torch.Tensor]:
        """
        Args:
            feats: ViTBackboneOutputs with `features[idx+1]` of shape
                [B, hidden_size, H, W] for each idx in `config.output_layers`.

        Returns:
            List of feature maps at multiple scales, each with
            `fusion_hidden_size` channels.
        """
        outputs: List[torch.Tensor] = []

        # Map selected backbone outputs to stage indices 0..len(neck_hidden_sizes)-1.
        max_idx = max(0, self.config.num_hidden_layers - 1)
        safe_layers = [min(i, max_idx) for i in self.config.output_layers]
        for stage_idx, layer_idx in enumerate(safe_layers):
            fmap = feats.features[layer_idx + 1]  # stored with +1 offset

            # If token-level features (with CLS) are available, replicate
            # HF's DPTReassembleStage readout logic on CPU.
            tokens = None
            if hasattr(feats, "tokens") and feats.tokens is not None:
                tokens = feats.tokens.get(layer_idx + 1, None)

            if tokens is not None and self.readout_projects is not None:
                # tokens: [B, N+1, C] with CLS at position 0
                cls_token = tokens[:, 0]  # [B, C]
                patch_tokens = tokens[:, 1:]  # [B, N, C]
                B, N, C = patch_tokens.shape
                # Use spatial shape from fmap to avoid assumptions about image size.
                _, _, H, W = fmap.shape
                if H * W != N:
                    # Fallback to sqrt if shapes are inconsistent, but this
                    # should not happen when backbone and neck are aligned.
                    size = int(torch.sqrt(torch.tensor(N, dtype=torch.float32)))
                    H = W = size
                hidden_state = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

                if self.readout_type == "project":
                    feature_shape = hidden_state.shape  # [B, C, H, W]
                    # Flatten spatial dims: [B, H*W, C]
                    hidden_flat = hidden_state.flatten(2).permute(0, 2, 1)
                    readout = cls_token.unsqueeze(1).expand_as(hidden_flat)
                    # Concatenate token and readout and project.
                    proj_in = torch.cat((hidden_flat, readout), dim=-1)  # [B, H*W, 2C]
                    proj = self.readout_projects[stage_idx]
                    # Ensure dtype matches the projection weights (typically float32).
                    proj_in = proj_in.to(dtype=proj[0].weight.dtype)
                    hidden_proj = proj(proj_in)  # [B, H*W, C]
                    hidden_state = hidden_proj.permute(0, 2, 1).reshape(feature_shape)
                elif self.readout_type == "add":
                    feature_shape = hidden_state.shape
                    hidden_flat = hidden_state.flatten(2)
                    hidden_flat = hidden_flat + cls_token.unsqueeze(-1)
                    hidden_state = hidden_flat.reshape(feature_shape)
                else:
                    raise ValueError(f"Unsupported readout_type: {self.readout_type}")
            else:
                # Fallback: use the already-reshaped fmap when tokens/CLS are
                # not available (e.g., in pure shape tests).
                hidden_state = fmap

            # Optional: push into TT device path post-readout to keep conv/resize on device
            try:
                import ttnn  # type: ignore
                from models.common.utility_functions import torch_to_tt_tensor_rm

                if (
                    self.tt_device is not None
                    and (
                        getattr(self.config, "tt_device_reassembly", False)
                        or getattr(self.config, "tt_perf_neck", False)
                    )
                    and not isinstance(hidden_state, ttnn.Tensor)
                ):
                    hidden_state = torch_to_tt_tensor_rm(hidden_state, self.tt_device, put_on_device=True)
            except Exception:
                pass

            # Reassemble to per-scale channels and resolution.
            x = self.reassemble_layers[stage_idx](hidden_state)

            # 3x3 conv to common fusion_hidden_size.
            conv = self.convs[stage_idx]

            try:
                import tt_lib.fallback_ops as fallback_ops  # type: ignore
                import ttnn  # type: ignore
            except Exception:
                fallback_ops = None
                ttnn = None

            if (
                fallback_ops is not None
                and ttnn is not None
                and isinstance(x, ttnn.Tensor)
                and self.tt_device is not None
            ):
                # Lazily materialize TT conv for this stage.
                if self._tt_convs[stage_idx] is None:
                    wt = torch_to_tt_tensor_rm(conv.weight.detach(), self.tt_device, put_on_device=True)
                    # No bias in HF convs.
                    self._tt_convs[stage_idx] = fallback_ops.Conv2d(
                        weights=wt,
                        biases=None,
                        in_channels=conv.in_channels,
                        out_channels=conv.out_channels,
                        kernel_size=conv.kernel_size[0],
                        stride=1,
                        padding=1,
                    )
                x = self._tt_convs[stage_idx](x)
                outputs.append(x)
            else:
                outputs.append(conv(x))

        return outputs
