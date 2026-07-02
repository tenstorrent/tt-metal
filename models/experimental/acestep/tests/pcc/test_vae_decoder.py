# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: TTNN Oobleck VAE decoder vs diffusers AutoencoderOobleck.decoder (real weights).

Turns DiT audio latents [B,64,T] into a 48kHz stereo waveform [B,2,T*1920]. Built entirely
from validated TTTv2 audio primitives (Conv1dViaConv3d / ConvTranspose1dViaConv3d / SnakeBeta).
This is the stage that makes generated music audible so SongEval can score it.

Weights come from the genuine checkpoint (skipped if the VAE isn't downloaded).
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc

from models.experimental.acestep.reference.weight_utils import have_pipeline, vae_dir
from models.experimental.acestep.tt.vae_decoder import OobleckDecoder, OobleckVAEConfig
from models.experimental.acestep.tests.test_utils import require_single_device


def _effective_decoder_state(ref_decoder) -> dict:
    """Build a state dict of EFFECTIVE (weight-norm folded) weights matching the TT module tree.

    diffusers stores weight_norm as weight_g/weight_v; `.weight` yields g*v/||v||. We read the
    effective `.weight`/`.bias`/`alpha`/`beta` under the same child names the TT modules expect.
    """
    import torch.nn as nn
    from diffusers.models.autoencoders.autoencoder_oobleck import Snake1d

    state = {}
    for name, mod in ref_decoder.named_modules():
        if isinstance(mod, (nn.Conv1d, nn.ConvTranspose1d)):
            state[f"{name}.weight"] = mod.weight.detach()
            if mod.bias is not None:
                state[f"{name}.bias"] = mod.bias.detach()
        elif isinstance(mod, Snake1d):
            state[f"{name}.alpha"] = mod.alpha.detach().reshape(-1)
            state[f"{name}.beta"] = mod.beta.detach().reshape(-1)
    return state


# Latent lengths (25 Hz frames): 40 -> ~1.6s, 120 -> ~4.8s audio.
LATENT_LENGTHS = [40, 120]


@pytest.mark.slow
@pytest.mark.skipif(not have_pipeline(), reason="ACE-Step VAE not downloaded")
@pytest.mark.parametrize("t_latent", LATENT_LENGTHS)
def test_vae_decoder(device, t_latent):
    require_single_device(device)
    from diffusers import AutoencoderOobleck

    vae = AutoencoderOobleck.from_pretrained(vae_dir()).eval()
    ref_decoder = vae.decoder
    cfg = OobleckVAEConfig.from_diffusers(vae.config)

    torch.manual_seed(0)
    latents = torch.randn(1, cfg.decoder_input_channels, t_latent)  # [B, 64, T]
    with torch.no_grad():
        ref_wav = ref_decoder(latents)  # [B, 2, T*1920]

    dec = OobleckDecoder(cfg, mesh_device=device, dtype=ttnn.float32)
    dec.load_torch_state_dict(_effective_decoder_state(ref_decoder))

    lat_btc = latents.transpose(1, 2).contiguous()  # [B, T, 64]
    lat_tt = ttnn.from_torch(lat_btc, device=device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    out_tt = dec.forward(lat_tt)
    out = ttnn.to_torch(out_tt).float()[..., : cfg.audio_channels]  # [B, T*1920, 2]
    out_bct = out.reshape(1, -1, cfg.audio_channels).transpose(1, 2)  # [B, 2, samples]

    n = min(ref_wav.shape[-1], out_bct.shape[-1])
    passing, msg = comp_pcc(ref_wav[..., :n], out_bct[..., :n], 0.97)
    print(f"VAE_DECODER_PCC t_latent={t_latent}: {msg}")
    assert passing, f"VAE decoder PCC {msg} < 0.97 (samples ref={ref_wav.shape[-1]} got={out_bct.shape[-1]})"
