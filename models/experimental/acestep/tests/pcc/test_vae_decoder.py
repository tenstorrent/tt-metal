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


from models.experimental.acestep.tt.model_config import _effective_vae_decoder_state


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

    # Validate the SAME dtype the pipeline ships (bf16 by default; env-overridable). This gates the
    # VAE precision choice: a dtype/blocking change must still clear the 0.97 PCC threshold.
    from models.experimental.acestep.tt.vae_conv_config import apply_vae_conv3d_config, vae_default_dtype

    vae_dtype = vae_default_dtype()
    apply_vae_conv3d_config(device, vae_dtype)
    dec = OobleckDecoder(cfg, mesh_device=device, dtype=vae_dtype)
    dec.load_torch_state_dict(_effective_vae_decoder_state(ref_decoder))

    lat_btc = latents.transpose(1, 2).contiguous()  # [B, T, 64]
    lat_tt = ttnn.from_torch(lat_btc, device=device, dtype=vae_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    out_tt = dec.forward(lat_tt)
    out = ttnn.to_torch(out_tt).float()[..., : cfg.audio_channels]  # [B, T*1920, 2]
    out_bct = out.reshape(1, -1, cfg.audio_channels).transpose(1, 2)  # [B, 2, samples]

    n = min(ref_wav.shape[-1], out_bct.shape[-1])
    passing, msg = comp_pcc(ref_wav[..., :n], out_bct[..., :n], 0.97)
    print(f"VAE_DECODER_PCC t_latent={t_latent}: {msg}")
    assert passing, f"VAE decoder PCC {msg} < 0.97 (samples ref={ref_wav.shape[-1]} got={out_bct.shape[-1]})"
