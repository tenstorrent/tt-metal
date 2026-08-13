# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the residual-VAE primitive ops AvgDown3D / DupUp3D.

After the BH bring-up showed that on-device reshape+permute interacts badly
with the H/W shard-axis metadata, the canonical implementation for these
two ops moved to `host_forward` (pure-PyTorch) — the residual blocks
device↔host round-trip the shortcut path each call. So these tests only
need to validate the host_forward math against an inline reference copy
of the diffusers `AvgDown3D` / `DupUp3D`. No device required.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tt_dit.models.vae.vae_wan2_1 import AvgDown3D, DupUp3D


# Host reference impls inlined to avoid pulling in vae_wan2_1_encoder_host.py,
# which imports `AutoencoderMixin` — only available on diffusers main, not the
# pinned 0.35.1 on our dev boxes. These are exact copies of the host module
# bodies from vae_wan2_1_encoder_host.py.
class _RefAvgDown3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x):
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * self.factor, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        x = x.view(B, self.out_channels, self.group_size, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        return x.mean(dim=2)


class _RefDupUp3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x, first_chunk=False):
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0), self.out_channels, self.factor_t, self.factor_s, self.factor_s, x.size(2), x.size(3), x.size(4)
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


@pytest.mark.parametrize(
    "in_C, out_C, ft, fs",
    [
        # Cosmos3 residual encoder dim_mult levels (encoder base_dim=160).
        (160, 160, 1, 2),  # level 0: spatial-only halve
        (160, 320, 2, 2),  # level 1: spatial+temporal halve, channel grow
        (320, 640, 2, 2),  # level 2
        (640, 640, 1, 1),  # level 3 (identity)
        # Cosmos3 residual decoder dim_mult levels (decoder base_dim=256).
        (1024, 512, 2, 2),
        (512, 256, 2, 2),
    ],
    ids=["enc_lvl0_s2", "enc_lvl1_st2", "enc_lvl2_st2", "enc_lvl3_id", "dec_lvl1_st2", "dec_lvl2_st2"],
)
def test_avg_down_3d_host_forward(in_C: int, out_C: int, ft: int, fs: int) -> None:
    """AvgDown3D.host_forward matches the inline reference for Cosmos3 channel configs."""
    torch.manual_seed(0)
    B, T = 1, 4 if ft > 1 else 5
    H = W = 32
    x = torch.randn(B, in_C, T, H, W, dtype=torch.float32)

    ref = _RefAvgDown3D(in_C, out_C, factor_t=ft, factor_s=fs)(x)
    out = AvgDown3D.host_forward(x, in_channels=in_C, out_channels=out_C, factor_t=ft, factor_s=fs)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "in_C, out_C, ft, fs",
    [
        (640, 320, 2, 2),
        (320, 160, 2, 2),
        (160, 160, 1, 2),
        # Decoder mirror with decoder_base_dim=256 channel set.
        (1024, 512, 2, 2),
        (512, 256, 2, 2),
        (256, 256, 1, 2),
    ],
    ids=["up_lvl0_st2", "up_lvl1_st2", "up_lvl2_s2", "dec_up_lvl0", "dec_up_lvl1", "dec_up_lvl2_s"],
)
@pytest.mark.parametrize("first_chunk", [False, True], ids=["full_chunk", "first_chunk"])
def test_dup_up_3d_host_forward(in_C: int, out_C: int, ft: int, fs: int, first_chunk: bool) -> None:
    """DupUp3D.host_forward matches the inline reference for both chunk modes."""
    torch.manual_seed(0)
    B, T = 1, 2
    H = W = 16
    x = torch.randn(B, in_C, T, H, W, dtype=torch.float32)

    ref = _RefDupUp3D(in_C, out_C, factor_t=ft, factor_s=fs)(x, first_chunk=first_chunk)
    out = DupUp3D.host_forward(
        x,
        in_channels=in_C,
        out_channels=out_C,
        factor_t=ft,
        factor_s=fs,
        first_chunk=first_chunk,
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_avg_dup_inverse_cpu() -> None:
    """Sanity: AvgDown3D(DupUp3D(x)) == x for matching configs. Pure host."""
    torch.manual_seed(0)
    x = torch.randn(1, 160, 4, 16, 16)
    up = _RefDupUp3D(in_channels=160, out_channels=160, factor_t=1, factor_s=2)
    down = _RefAvgDown3D(in_channels=160, out_channels=160, factor_t=1, factor_s=2)
    y = down(up(x))
    torch.testing.assert_close(y, x, rtol=1e-4, atol=1e-4)
