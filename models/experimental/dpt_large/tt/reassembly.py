# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .config import DPTLargeConfig
from .vit_backbone import ViTBackboneOutputs
from .tt_configs import TTLayerConfig
from .tt_cnn_ops import (
    TTConv2dCached,
    TTConvTranspose2dCached,
    ensure_tt_device_tensor,
)


def _ensure_tt_device_tensor(x, tt_device, ttnn):
    return ensure_tt_device_tensor(x, tt_device)


class DPTReassembleLayerTT(nn.Module):
    """
    TT-aware variant of Hugging Face's `DPTReassembleLayer`.

    It projects from the ViT hidden size to a per-scale channel size, then
    performs up/down sampling depending on the `factor` using TT-native ops
    when TT tensors are provided.
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
        self._tt_resize_conv_transpose = None

    def _tt_project(self, x):
        if self._tt_proj is None:
            self._tt_proj = TTConv2dCached.from_conv(self.projection)
        return self._tt_proj(x, device=self.tt_device)

    def _tt_resize(
        self,
        x,
        *,
        expected_input_hw: Optional[tuple[int, int]] = None,
        expected_output_hw: Optional[tuple[int, int]] = None,
    ):
        # Preserve HF behavior for upsampling with a learned ConvTranspose2d,
        # but keep execution fully on TT device.
        if self.factor > 1:
            if self._tt_resize_conv_transpose is None:
                self._tt_resize_conv_transpose = TTConvTranspose2dCached.from_conv_transpose(self.resize)
            return self._tt_resize_conv_transpose(
                x,
                device=self.tt_device,
                expected_input_hw=expected_input_hw,
                expected_output_hw=expected_output_hw,
            )

        if isinstance(self.resize, nn.Identity):
            return x

        if self._tt_resize_conv is None:
            self._tt_resize_conv = TTConv2dCached.from_tensors(
                weight_torch=self.resize.weight.detach(),
                bias_torch=self.resize.bias.detach(),
                stride=(int(self.resize.stride[0]), int(self.resize.stride[1])),
                padding=(int(self.resize.padding[0]), int(self.resize.padding[1])),
            )
        return self._tt_resize_conv(x, device=self.tt_device)

    def prefers_device_path(self) -> bool:
        return True

    def expected_output_hw(self, input_hw: tuple[int, int]) -> tuple[int, int]:
        in_h, in_w = int(input_hw[0]), int(input_hw[1])
        if self.factor > 1:
            scale = int(self.factor)
            return in_h * scale, in_w * scale
        if isinstance(self.resize, nn.Identity):
            return in_h, in_w
        if isinstance(self.resize, nn.Conv2d):
            stride_h, stride_w = int(self.resize.stride[0]), int(self.resize.stride[1])
            pad_h, pad_w = int(self.resize.padding[0]), int(self.resize.padding[1])
            dil_h, dil_w = int(self.resize.dilation[0]), int(self.resize.dilation[1])
            k_h, k_w = int(self.resize.kernel_size[0]), int(self.resize.kernel_size[1])
            out_h = ((in_h + 2 * pad_h - dil_h * (k_h - 1) - 1) // stride_h) + 1
            out_w = ((in_w + 2 * pad_w - dil_w * (k_w - 1) - 1) // stride_w) + 1
            return out_h, out_w
        raise RuntimeError(f"Unsupported resize module type: {type(self.resize)!r}")

    def forward(self, x, *, expected_input_hw: Optional[tuple[int, int]] = None):
        # x: torch.Tensor or ttnn.Tensor of shape [B, hidden_size, H, W] / TT layout.
        try:
            import ttnn  # type: ignore
        except Exception:
            ttnn = None

        if ttnn is not None and isinstance(x, ttnn.Tensor) and self.tt_device is not None:
            x = self._tt_project(x)
            expected_output_hw = None
            if expected_input_hw is not None:
                expected_output_hw = self.expected_output_hw(expected_input_hw)
            x = self._tt_resize(
                x,
                expected_input_hw=expected_input_hw,
                expected_output_hw=expected_output_hw,
            )
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
        # Lazy per-stage readout caches for TT device readout ("project" path).
        self._tt_readout_cache: List[Optional[dict]] = [None for _ in range(len(self.config.neck_hidden_sizes))]

    def _tt_readout_to_nchw(
        self,
        *,
        stage_idx: int,
        tokens,
        fmap_hw: tuple[int, int],
        ttnn,
    ):
        """
        Attempt to execute DPT readout on device.

        Returns:
            TT tensor [B, C, H, W] on success, or raises to trigger host fallback.
        """
        if self.tt_device is None:
            raise RuntimeError("TT device is not available for TT readout")
        if not isinstance(tokens, ttnn.Tensor):
            raise TypeError("Expected TT tokens")

        H, W = int(fmap_hw[0]), int(fmap_hw[1])
        shape = tuple(int(v) for v in tuple(getattr(tokens, "shape", ())))
        if len(shape) == 4:
            B, one, Np1, C = shape
            if one != 1:
                raise RuntimeError(f"Unexpected TT token shape: {shape}")
            tokens_tt4 = tokens
        elif len(shape) == 3:
            B, Np1, C = shape
            tokens_tt4 = ttnn.reshape(tokens, (int(B), 1, int(Np1), int(C)))
        else:
            raise RuntimeError(f"Unexpected TT token rank: {shape}")

        Np1 = int(Np1)
        C = int(C)
        if Np1 <= 1:
            raise RuntimeError(f"Unexpected token sequence length: {Np1}")
        if int(H) * int(W) != (Np1 - 1):
            raise RuntimeError(f"Token length mismatch: tokens={Np1 - 1} vs fmap={int(H) * int(W)}")

        # CLS at position 0, patch tokens at 1..Np1-1
        cls_tt4 = ttnn.slice(tokens_tt4, (0, 0, 0, 0), (int(B), 1, 1, int(C)))  # [B,1,1,C]
        patch_tt4 = ttnn.slice(tokens_tt4, (0, 0, 1, 0), (int(B), 1, int(Np1), int(C)))  # [B,1,N,C]

        if self.readout_type == "add":
            # Broadcast cls across the sequence dimension.
            try:
                cls_rep = ttnn.expand(cls_tt4, [-1, -1, int(Np1 - 1), -1])
            except Exception:
                cls_rep = ttnn.repeat(cls_tt4, [1, 1, int(Np1 - 1), 1])
            out_tt4 = ttnn.add(patch_tt4, cls_rep)
        elif self.readout_type == "project":
            if self.readout_projects is None:
                raise RuntimeError("Readout projects are not initialized")
            cache = self._tt_readout_cache[stage_idx]
            if cache is None:
                proj = self.readout_projects[stage_idx]
                linear = proj[0]
                w_full = linear.weight.detach()
                b_full = linear.bias.detach()
                if w_full.shape[1] != 2 * int(C) or w_full.shape[0] != int(C):
                    raise RuntimeError(f"Unexpected readout weight shape: {tuple(w_full.shape)}")
                w_patch = w_full[:, : int(C)]
                w_cls = w_full[:, int(C) :]
                cache = {
                    "w_patch_tt": ttnn.from_torch(
                        torch.transpose(w_patch, -1, -2).contiguous(),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.tt_device,
                    ),
                    "w_cls_tt": ttnn.from_torch(
                        torch.transpose(w_cls, -1, -2).contiguous(),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.tt_device,
                    ),
                    "b_tt": ttnn.from_torch(
                        b_full.contiguous(),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=self.tt_device,
                    ),
                    "act": "gelu" if self.hidden_act == "gelu" else "relu",
                }
                self._tt_readout_cache[stage_idx] = cache

            memcfg = ttnn.DRAM_MEMORY_CONFIG
            patch_proj = ttnn.linear(
                patch_tt4,
                cache["w_patch_tt"],
                bias=cache["b_tt"],
                dtype=ttnn.bfloat16,
                memory_config=memcfg,
            )
            cls_proj = ttnn.linear(
                cls_tt4,
                cache["w_cls_tt"],
                bias=None,
                dtype=ttnn.bfloat16,
                memory_config=memcfg,
            )
            try:
                cls_rep = ttnn.expand(cls_proj, [-1, -1, int(Np1 - 1), -1])
            except Exception:
                cls_rep = ttnn.repeat(cls_proj, [1, 1, int(Np1 - 1), 1])
            out_tt4 = ttnn.add(patch_proj, cls_rep)
            if cache["act"] == "gelu":
                out_tt4 = ttnn.gelu(out_tt4)
            else:
                out_tt4 = ttnn.relu(out_tt4)
        else:
            raise ValueError(f"Unsupported readout_type: {self.readout_type}")

        nhwc = ttnn.reshape(out_tt4, (int(B), int(H), int(W), int(C)))
        return ttnn.permute(nhwc, (0, 3, 1, 2))

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
        try:
            import ttnn  # type: ignore
        except Exception:
            ttnn = None

        # Map selected backbone outputs to stage indices 0..len(neck_hidden_sizes)-1.
        max_idx = max(0, self.config.num_hidden_layers - 1)
        safe_layers = [min(max(int(i), 0), max_idx) for i in self.config.output_layers]
        for stage_idx, layer_idx in enumerate(safe_layers):
            fmap = feats.features[layer_idx + 1]  # stored with +1 offset
            _, _, H, W = fmap.shape

            # If token-level features (with CLS) are available, replicate
            # HF's DPTReassembleStage readout logic on CPU.
            tokens = None
            if hasattr(feats, "tokens") and feats.tokens is not None:
                tokens = feats.tokens.get(layer_idx + 1, None)

            # Prefer a device-side readout when token maps are already on device.
            if (
                ttnn is not None
                and tokens is not None
                and isinstance(tokens, ttnn.Tensor)
                and self.tt_device is not None
                and bool(getattr(self.config, "tt_perf_neck", False))
            ):
                try:
                    hidden_state = self._tt_readout_to_nchw(
                        stage_idx=stage_idx,
                        tokens=tokens,
                        fmap_hw=(int(H), int(W)),
                        ttnn=ttnn,
                    )
                except Exception:
                    hidden_state = None
            else:
                hidden_state = None

            if hidden_state is None and tokens is not None and self.readout_projects is not None:
                # Host fallback readout (reference behavior). If token maps are on
                # device but TT readout failed, convert back to torch here to
                # preserve correctness.
                if ttnn is not None and isinstance(tokens, ttnn.Tensor):
                    tokens_host = tokens.cpu()
                    if hasattr(tokens_host, "layout") and tokens_host.layout == ttnn.TILE_LAYOUT:
                        tokens_host = tokens_host.to(ttnn.ROW_MAJOR_LAYOUT)
                    tokens_host = tokens_host.to_torch()
                    if tokens_host.dim() == 4 and tokens_host.shape[1] == 1:
                        tokens_host = tokens_host[:, 0, :, :]
                    tokens = tokens_host
                # tokens: [B, N+1, C] with CLS at position 0
                cls_token = tokens[:, 0]  # [B, C]
                patch_tokens = tokens[:, 1:]  # [B, N, C]
                B, N, C = patch_tokens.shape
                # Use spatial shape from fmap to avoid assumptions about image size.
                if H * W != N:
                    # Fallback to sqrt if shapes are inconsistent, but this
                    # should not happen when backbone and neck are aligned.
                    size = int(torch.sqrt(torch.tensor(N, dtype=torch.float32)))
                    H = W = size
                if self.readout_type == "project":
                    hidden_flat = patch_tokens  # [B, H*W, C]
                    readout = cls_token.unsqueeze(1).expand_as(hidden_flat)
                    # Concatenate token and readout and project.
                    proj_in = torch.cat((hidden_flat, readout), dim=-1)  # [B, H*W, 2C]
                    proj = self.readout_projects[stage_idx]
                    # Ensure dtype matches the projection weights (typically float32).
                    proj_in = proj_in.to(dtype=proj[0].weight.dtype)
                    hidden_flat = proj(proj_in)  # [B, H*W, C]
                elif self.readout_type == "add":
                    hidden_flat = patch_tokens + cls_token.unsqueeze(1)
                else:
                    raise ValueError(f"Unsupported readout_type: {self.readout_type}")
                hidden_state = hidden_flat.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
            elif hidden_state is None:
                # Fallback: use the already-reshaped fmap when tokens/CLS are
                # not available (e.g., in pure shape tests).
                hidden_state = fmap

            # Optional: push into TT device path post-readout to keep conv/resize on device.
            # For TT runs, all reassembly stages stay on device.
            if (
                ttnn is not None
                and self.tt_device is not None
                and (
                    getattr(self.config, "tt_device_reassembly", False)
                    or getattr(self.config, "tt_perf_neck", False)
                )
                and self.reassemble_layers[stage_idx].prefers_device_path()
                and not isinstance(hidden_state, ttnn.Tensor)
            ):
                hidden_state = ttnn.from_torch(
                    hidden_state,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.tt_device,
                )

            # Reassemble to per-scale channels and resolution.
            x = self.reassemble_layers[stage_idx](
                hidden_state,
                expected_input_hw=(int(H), int(W)),
            )

            # 3x3 conv to common fusion_hidden_size.
            conv = self.convs[stage_idx]

            try:
                import ttnn  # type: ignore
            except Exception:
                ttnn = None

            if (
                ttnn is not None
                and isinstance(x, ttnn.Tensor)
                and self.tt_device is not None
            ):
                # Lazily materialize TT conv for this stage.
                if self._tt_convs[stage_idx] is None:
                    self._tt_convs[stage_idx] = TTConv2dCached.from_conv(conv)
                x = self._tt_convs[stage_idx](x, device=self.tt_device)
                x = _ensure_tt_device_tensor(x, self.tt_device, ttnn)
                outputs.append(x)
            else:
                outputs.append(conv(x))

        return outputs
