# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.config import DEFAULT_LANGUAGE
from models.experimental.xtts.reference.xtts_conditioning import MEL_SR, load_reference_audio, wav_to_mel
from models.experimental.xtts.reference.xtts_gpt_generate import START_AUDIO_TOKEN, greedy_generate, wrap_text_ids
from models.experimental.xtts.reference.xtts_inference import XttsReference
from models.experimental.xtts.reference.xtts_mel import SAMPLE_RATE as SPK_SR
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_hifi_decoder import TtHifiDecoder

SEED_CODES = 32
HIFI_TEXT = (
    "Voice synthesis has come a long way, and modern systems can already generate "
    "natural sounding speech with remarkable accuracy. Hey how are you doing?"
)


@pytest.fixture(scope="module")
def hifi_real_inputs(xtts_state_dict):
    """Module fixture with real HiFi-GAN latents, speaker embed, and seed codes."""
    from scipy.signal import resample_poly

    torch.manual_seed(0)
    sd = xtts_state_dict
    wav = load_reference_audio(sample="en_sample.wav")
    cond_mel = wav_to_mel(wav, sd["mel_stats"].cpu())
    q = math.gcd(SPK_SR, MEL_SR)
    spk_wav = torch.from_numpy(
        resample_poly(wav.reshape(-1).numpy().astype("float32"), SPK_SR // q, MEL_SR // q).astype("float32")
    ).unsqueeze(0)

    ref = XttsReference(sd)
    text_ids = wrap_text_ids(preprocess_text(HIFI_TEXT, lang=DEFAULT_LANGUAGE))
    cond_latents = ref._cond_latents(cond_mel)
    g = ref.decoder_full.speaker_embedding(spk_wav)
    codes, _ = greedy_generate(ref.gpt, text_ids, cond_latents, max_new_tokens=SEED_CODES, wrap_text=False)
    return ref, text_ids, cond_latents, g, codes


def _latents_at(ref, text_ids, cond_latents, codes, length):
    """Teacher-force GPT latents tiled to the requested latent length."""
    reps = -(-length // codes.shape[1])
    tiled = codes.repeat(1, reps)[:, :length]
    start = torch.full((1, 1), START_AUDIO_TOKEN, dtype=torch.long)
    mel_ids = torch.cat([start, tiled], dim=1)
    with torch.no_grad():
        latents = ref.gpt(text_ids, mel_ids, cond_latents=cond_latents, return_latent=True)
    return latents[:, 1:]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("latent_len", [32, 320])
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_hifi_decoder(device, hifi_real_inputs, latent_len, pcc, reset_seeds):
    """Compare TTNN HiFi-GAN decoder waveform to the PyTorch reference via PCC."""
    ref, text_ids, cond_latents, g, codes = hifi_real_inputs
    latents = _latents_at(ref, text_ids, cond_latents, codes, latent_len)
    with torch.no_grad():
        ref_out = ref.decoder_full.decoder(latents, g)

    tt_dec = TtHifiDecoder(device, ref.decoder_full.decoder.waveform_decoder.state_dict())
    latents_dev = ttnn.from_torch(latents.float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32)
    g_dev = ttnn.from_torch(
        g.permute(0, 2, 1).contiguous().float(),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.float32,
    )
    tt_out = ttnn.to_torch(tt_dec(latents_dev, g_dev)).float().permute(0, 2, 1)

    assert tt_out.shape == ref_out.shape, f"shape {tuple(tt_out.shape)} != {tuple(ref_out.shape)}"
    does_pass, msg = comp_pcc(ref_out, tt_out, pcc)
    logger.info(comp_allclose(ref_out, tt_out))
    logger.info(f"hifi_decoder latent_len={latent_len}: {msg}")
    assert does_pass, f"hifi_decoder PCC below {pcc}: {msg}"
