# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: VAE Snake1d activation vs diffusers AutoencoderOobleck Snake1d.

Snake1d: y = x + 1/(beta+1e-9) * sin(alpha*x)^2, per-channel alpha/beta.
Validated against the genuine reference implementation with random alpha/beta (the op is
weight-agnostic elementwise math; real VAE Snake params are exercised in the decoder test).
"""

import pytest
import torch

from diffusers.models.autoencoders.autoencoder_oobleck import Snake1d as RefSnake1d

from models.experimental.acestep.tt.snake import Snake1d, Snake1dConfig, snake_alpha_beta_to_lazy
from models.experimental.acestep.tests.test_utils import (
    require_single_device,
    to_ttnn_tensor,
    to_torch,
    assert_pcc,
)

# (channels, time) — audio feature maps in the VAE decoder run C in {2048,1024,512,256,128,2}.
CASES = [(128, 96), (256, 128), (512, 64), (2048, 40)]


@pytest.mark.parametrize("channels,time", CASES)
def test_vae_snake1d(device, channels, time):
    require_single_device(device)
    torch.manual_seed(channels + time)

    # logscale=True (diffusers default): stored alpha/beta are log-domain; effective = exp(param).
    ref = RefSnake1d(channels).eval()
    with torch.no_grad():
        # Non-trivial log-domain params (init is zeros -> exp=1 -> too easy).
        ref.alpha.copy_(torch.randn(1, channels, 1) * 0.3)
        ref.beta.copy_(torch.randn(1, channels, 1) * 0.3)

    x = torch.randn(1, channels, time)  # [B, C, T]
    with torch.no_grad():
        ref_out = ref(x)  # [B, C, T]

    # TT operates in [B, T, C]; transpose in/out.
    x_btc = x.transpose(1, 2).contiguous()  # [B, T, C]
    cfg = Snake1dConfig(
        alpha=snake_alpha_beta_to_lazy(ref.alpha, device),
        beta=snake_alpha_beta_to_lazy(ref.beta, device),
        channels=channels,
    )
    mod = Snake1d(cfg)
    x_tt = to_ttnn_tensor(x_btc.reshape(1, 1, time, channels), device)
    out_tt = mod.forward(x_tt)
    out = to_torch(out_tt, expected_shape=(1, 1, time, channels)).reshape(1, time, channels)
    out_bct = out.transpose(1, 2)  # [B, C, T]

    assert_pcc(ref_out, out_bct, 0.99)
