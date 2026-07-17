from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torchvision.models import vgg19
from torchvision.models import VGG19_Weights


def _center_crop_or_pad(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Match spatial size of x to ref by symmetric crop or pad.
    Handles off-by-one due to pooling/upsampling.
    """
    _, _, h, w = x.shape
    _, _, rh, rw = ref.shape

    # Crop if bigger
    if h > rh:
        dh = h - rh
        x = x[:, :, dh // 2 : dh // 2 + rh, :]
    if w > rw:
        dw = w - rw
        x = x[:, :, :, dw // 2 : dw // 2 + rw]

    # Pad if smaller
    _, _, h2, w2 = x.shape
    pad_h = max(0, rh - h2)
    pad_w = max(0, rw - w2)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(
            x,
            [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            mode="constant",
            value=0.0,
        )
    return x


def _make_norm(norm: str, num_channels: int, gn_groups: int) -> nn.Module:
    norm = norm.lower()
    if norm in ("bn", "batch", "batchnorm"):
        return nn.BatchNorm2d(num_channels)
    if norm in ("gn", "group", "groupnorm"):
        g = min(gn_groups, num_channels)
        while g > 1 and (num_channels % g) != 0:
            g -= 1
        return nn.GroupNorm(num_groups=max(1, g), num_channels=num_channels)
    raise ValueError(f"Unsupported norm='{norm}'. Use 'batch' or 'group'.")


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "group", gn_groups: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.n1 = _make_norm(norm, out_ch, gn_groups)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.n2 = _make_norm(norm, out_ch, gn_groups)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.n1(self.conv1(x)))
        x = self.act2(self.n2(self.conv2(x)))
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        bilinear: bool,
        norm: str = "group",
        gn_groups: int = 16,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.up_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
            self.up_norm = _make_norm(norm, out_ch, gn_groups)
            self.up_act = nn.ReLU(inplace=True)
            conv_in = out_ch + skip_ch
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
            conv_in = out_ch + skip_ch

        self.conv = DoubleConv(conv_in, out_ch, norm=norm, gn_groups=gn_groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.bilinear:
            x = self.up(x)
            x = self.up_act(self.up_norm(self.up_conv(x)))
        else:
            x = self.up(x)

        x = _center_crop_or_pad(x, skip)
        skip = _center_crop_or_pad(skip, x)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class Bridge(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 1,
        norm: str = "group",
        gn_groups: int = 16,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        if k not in (1, 3):
            raise ValueError("bridge_kernel_size must be 1 or 3")
        pad = 0 if k == 1 else 1
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False)
        self.norm = _make_norm(norm, out_ch, gn_groups)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UNetVGG19(nn.Module):
    """
    VGG19 encoder + 5-stage UNet decoder.

    Memory optimizations:
      - Aggressive activation checkpointing (enc/bridge/dec) when use_checkpoint=True.
      - forward() returns LOGITS (no sigmoid/softmax).
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        bilinear: bool = True,
        use_checkpoint: bool = False,
        norm: str = "group",
        gn_groups: int = 16,
        bridge_kernel_size: int = 1,
        dec_ch: Tuple[int, int, int, int, int] = (64, 128, 256, 512, 512),
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.bilinear = bool(bilinear)
        self.use_checkpoint = bool(use_checkpoint)

        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = vgg19(weights=weights)
        feats = vgg.features

        self.enc1 = nn.Sequential(*[feats[i] for i in range(0, 5)])  # /2
        self.enc2 = nn.Sequential(*[feats[i] for i in range(5, 10)])  # /4
        self.enc3 = nn.Sequential(*[feats[i] for i in range(10, 19)])  # /8
        self.enc4 = nn.Sequential(*[feats[i] for i in range(19, 28)])  # /16
        self.enc5 = nn.Sequential(*[feats[i] for i in range(28, 37)])  # /32

        self.bridge1 = Bridge(64, dec_ch[0], kernel_size=bridge_kernel_size, norm=norm, gn_groups=gn_groups)
        self.bridge2 = Bridge(128, dec_ch[1], kernel_size=bridge_kernel_size, norm=norm, gn_groups=gn_groups)
        self.bridge3 = Bridge(256, dec_ch[2], kernel_size=bridge_kernel_size, norm=norm, gn_groups=gn_groups)
        self.bridge4 = Bridge(512, dec_ch[3], kernel_size=bridge_kernel_size, norm=norm, gn_groups=gn_groups)
        self.bridge5 = Bridge(512, dec_ch[4], kernel_size=bridge_kernel_size, norm=norm, gn_groups=gn_groups)

        self.bottleneck = DoubleConv(dec_ch[4], dec_ch[4], norm=norm, gn_groups=gn_groups)

        self.input_proj = nn.Conv2d(3, dec_ch[0], kernel_size=1, bias=False)
        self.input_norm = _make_norm(norm, dec_ch[0], gn_groups)
        self.input_act = nn.ReLU(inplace=True)

        self.dec4 = UpBlock(dec_ch[4], dec_ch[3], dec_ch[3], bilinear=bilinear, norm=norm, gn_groups=gn_groups)
        self.dec3 = UpBlock(dec_ch[3], dec_ch[2], dec_ch[2], bilinear=bilinear, norm=norm, gn_groups=gn_groups)
        self.dec2 = UpBlock(dec_ch[2], dec_ch[1], dec_ch[1], bilinear=bilinear, norm=norm, gn_groups=gn_groups)
        self.dec1 = UpBlock(dec_ch[1], dec_ch[0], dec_ch[0], bilinear=bilinear, norm=norm, gn_groups=gn_groups)

        self.up0 = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            if bilinear
            else nn.ConvTranspose2d(dec_ch[0], dec_ch[0], kernel_size=2, stride=2, bias=False)
        )
        self.dec0_conv = DoubleConv(dec_ch[0] + dec_ch[0], dec_ch[0], norm=norm, gn_groups=gn_groups)

        out_ch = 1 if self.num_classes == 1 else self.num_classes
        self.final_conv = nn.Conv2d(dec_ch[0], out_ch, kernel_size=1)

    def _ckpt(self, fn, *args: torch.Tensor) -> torch.Tensor:
        """
        PyTorch 2.9: pass use_reentrant explicitly; recommended False. :contentReference[oaicite:4]{index=4}
        Only checkpoint when gradients are enabled (training).
        """
        if not self.use_checkpoint or not torch.is_grad_enabled():
            return fn(*args)
        return checkpoint(fn, *args, use_reentrant=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1x scale "skip"
        x0 = self.input_act(self.input_norm(self.input_proj(x)))

        # Encoder (checkpoint internal activations)
        x1 = self._ckpt(self.enc1, x)
        x2 = self._ckpt(self.enc2, x1)
        x3 = self._ckpt(self.enc3, x2)
        x4 = self._ckpt(self.enc4, x3)
        x5 = self._ckpt(self.enc5, x4)

        # Bridges
        s1 = self._ckpt(self.bridge1, x1)  # /2
        s2 = self._ckpt(self.bridge2, x2)  # /4
        s3 = self._ckpt(self.bridge3, x3)  # /8
        s4 = self._ckpt(self.bridge4, x4)  # /16
        s5 = self._ckpt(self.bridge5, x5)  # /32

        # Bottleneck
        b = self._ckpt(self.bottleneck, s5)

        # Decoder (checkpointed blocks)
        d4 = self._ckpt(self.dec4, b, s4)
        d3 = self._ckpt(self.dec3, d4, s3)
        d2 = self._ckpt(self.dec2, d3, s2)
        d1 = self._ckpt(self.dec1, d2, s1)

        # /2 -> /1
        u0 = self.up0(d1)
        u0 = _center_crop_or_pad(u0, x0)
        x0m = _center_crop_or_pad(x0, u0)

        cat0 = torch.cat([u0, x0m], dim=1)
        d0 = self._ckpt(self.dec0_conv, cat0)

        logits = self.final_conv(d0)
        return logits  # LOGITS

    def encoder_stages(self) -> List[nn.Module]:
        return [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]
