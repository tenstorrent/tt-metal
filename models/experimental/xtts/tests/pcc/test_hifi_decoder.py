# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the XTTS-v2 HifiDecoder (latent upsample + HiFi-GAN generator).

Validates the on-device latent linear-upsample (as a resample matmul) chained into
the generator against the pure-PyTorch reference ``HifiDecoder.forward(latents, g)``,
with real coqui/XTTS-v2 weights. Input is a GPT latent ``[1, T, 1024]`` and speaker
embedding ``[1, 512, 1]``; output is the ``[1, T*~4.35*256, 1]`` waveform.

Latents are teacher-forced GPT hidden states from real audio codes (not ``randn``).
Two lengths: 32 (short tile) and 320 (above the demo's 192 / 205 / 240 budgets).

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/pcc/test_hifi_decoder.py
"""

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.config import DEMO
from models.experimental.xtts.reference.xtts_conditioning import MEL_SR, load_reference_audio, wav_to_mel
from models.experimental.xtts.reference.xtts_gpt_generate import START_AUDIO_TOKEN, greedy_generate, wrap_text_ids
from models.experimental.xtts.reference.xtts_inference import XttsReference
from models.experimental.xtts.reference.xtts_mel import SAMPLE_RATE as SPK_SR
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_hifi_decoder import TtHifiDecoder

SEED_CODES = 32  # greedy once, then tiled to each latent_len


@pytest.fixture(scope="module")
def hifi_real_inputs(xtts_state_dict):
    """Real GPT seed codes + speaker embedding, paid once for both lengths."""
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
    text_ids = wrap_text_ids(preprocess_text(DEMO.text.rstrip("."), lang=DEMO.language))
    cond_latents = ref._cond_latents(cond_mel)
    g = ref.decoder_full.speaker_embedding(spk_wav)
    codes, _ = greedy_generate(ref.gpt, text_ids, cond_latents, max_new_tokens=SEED_CODES, wrap_text=False)
    return ref, text_ids, cond_latents, g, codes


def _latents_at(ref, text_ids, cond_latents, codes, length):
    """Real GPT latents ``[1, length, 1024]``: tile the seed codes and teacher-force."""
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
    tt_out = ttnn.to_torch(tt_dec(latents_dev, g_dev)).float().permute(0, 2, 1)  # [1, 1, L]

    assert tt_out.shape == ref_out.shape, f"shape {tuple(tt_out.shape)} != {tuple(ref_out.shape)}"
    does_pass, msg = comp_pcc(ref_out, tt_out, pcc)
    logger.info(comp_allclose(ref_out, tt_out))
    logger.info(f"hifi_decoder latent_len={latent_len}: {msg}")
    assert does_pass, f"hifi_decoder PCC below {pcc}: {msg}"
