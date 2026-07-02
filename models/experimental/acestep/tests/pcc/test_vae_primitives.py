# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: TTTv2 audio primitives vs genuine Oobleck VAE weights.

The ACE-Step VAE decoder is built by REUSING three TTTv2 audio primitives from
`models/tt_dit/layers/audio_ops`. These tests validate each primitive independently against the
real Oobleck weights, so a decoder-level regression can be localized to a specific primitive:

  - SnakeBeta (alpha_logscale=True)  == diffusers Snake1d (logscale=True)
  - Conv1dViaConv3d                  == diffusers weight-normed Conv1d (k7 "same", and k1)
  - ConvTranspose1dViaConv3d         == diffusers weight-normed ConvTranspose1d (upsample)

Skipped if the VAE checkpoint isn't downloaded.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc

from models.experimental.acestep.reference.weight_utils import have_pipeline, vae_dir
from models.experimental.acestep.tests.test_utils import require_single_device


def _load_ref_decoder():
    from diffusers import AutoencoderOobleck

    return AutoencoderOobleck.from_pretrained(vae_dir()).eval().decoder


pytestmark = pytest.mark.skipif(not have_pipeline(), reason="ACE-Step VAE not downloaded")


def _to_btc(x_bct):
    return x_bct.transpose(1, 2).contiguous()


@pytest.mark.parametrize("t", [64, 96])
def test_vae_snakebeta_real(device, t):
    """SnakeBeta vs the genuine Oobleck Snake1d (final decoder snake, 128 ch)."""
    require_single_device(device)
    from diffusers.models.autoencoders.autoencoder_oobleck import Snake1d
    from models.tt_dit.layers.audio_ops import SnakeBeta

    ref = _load_ref_decoder().snake1  # Snake1d(128)
    assert isinstance(ref, Snake1d)
    c = ref.alpha.shape[1]
    torch.manual_seed(t)
    x = torch.randn(1, c, t)
    with torch.no_grad():
        ro = ref(x)

    sb = SnakeBeta(c, alpha_logscale=True, mesh_device=device, dtype=ttnn.float32)
    sb.load_torch_state_dict({"alpha": ref.alpha.detach().reshape(-1), "beta": ref.beta.detach().reshape(-1)})
    xt = ttnn.from_torch(_to_btc(x), device=device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    o = ttnn.to_torch(sb.forward(xt)).float()[..., :c].reshape(1, t, c).transpose(1, 2)
    passing, msg = comp_pcc(ro, o, 0.99)
    print(f"VAE_SNAKE_PCC c={c} t={t}: {msg}")
    assert passing, msg


@pytest.mark.parametrize("t", [40, 96])
def test_vae_conv1d_real(device, t):
    """Conv1dViaConv3d vs the genuine input Conv1d (64->2048, k7 'same')."""
    require_single_device(device)
    from models.tt_dit.layers.audio_ops import Conv1dViaConv3d

    ref = _load_ref_decoder().conv1  # weight-normed Conv1d(64,2048,k7,pad3)
    cin, cout, k = ref.in_channels, ref.out_channels, ref.kernel_size[0]
    torch.manual_seed(t)
    x = torch.randn(1, cin, t)
    with torch.no_grad():
        ro = ref(x)

    m = Conv1dViaConv3d(
        in_channels=cin,
        out_channels=cout,
        kernel_size=k,
        stride=1,
        padding_mode="zeros",
        bias=True,
        mesh_device=device,
        dtype=ttnn.float32,
    )
    m.load_torch_state_dict({"weight": ref.weight.detach(), "bias": ref.bias.detach()})
    xt = ttnn.from_torch(_to_btc(x), device=device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    o = ttnn.to_torch(m.forward(xt)).float()[..., :cout].reshape(1, t, cout).transpose(1, 2)
    n = min(ro.shape[-1], o.shape[-1])
    passing, msg = comp_pcc(ro[..., :n], o[..., :n], 0.99)
    print(f"VAE_CONV1D_PCC t={t}: {msg}")
    assert passing, msg


@pytest.mark.parametrize("t", [16, 32])
def test_vae_convtranspose1d_real(device, t):
    """ConvTranspose1dViaConv3d vs the genuine first upsample ConvTranspose1d (2048->1024, k20 s10)."""
    require_single_device(device)
    from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDecoderBlock
    from models.tt_dit.layers.audio_ops import ConvTranspose1dViaConv3d

    dec = _load_ref_decoder()
    blk = [m for m in dec.modules() if isinstance(m, OobleckDecoderBlock)][0]
    ct = blk.conv_t1  # weight-normed ConvTranspose1d(2048,1024,k20,stride10)
    cin, cout, k, s = ct.in_channels, ct.out_channels, ct.kernel_size[0], ct.stride[0]
    torch.manual_seed(t)
    x = torch.randn(1, cin, t)
    with torch.no_grad():
        ro = ct(x)

    m = ConvTranspose1dViaConv3d(
        in_channels=cin, out_channels=cout, kernel_size=k, stride=s, bias=True, mesh_device=device, dtype=ttnn.float32
    )
    m.load_torch_state_dict({"weight": ct.weight.detach(), "bias": ct.bias.detach()})
    xt = ttnn.from_torch(_to_btc(x), device=device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    o = ttnn.to_torch(m.forward(xt)).float()[..., :cout].reshape(1, -1, cout).transpose(1, 2)
    n = min(ro.shape[-1], o.shape[-1])
    passing, msg = comp_pcc(ro[..., :n], o[..., :n], 0.99)
    print(f"VAE_CONVT_PCC t={t}: {msg}")
    assert passing, msg
