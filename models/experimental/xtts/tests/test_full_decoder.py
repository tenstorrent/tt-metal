# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC test for the full XTTS-v2 HifiDecoder.

Validates the complete on-device GAN decoder — reference audio -> mel frontend ->
speaker encoder -> g, and (GPT latents + g) -> latent upsample -> HiFi-GAN generator
-> waveform — against the pure-PyTorch reference, all from real coqui/XTTS-v2 weights.
No torch/host fallback anywhere in the device path.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/test_full_decoder.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_hifi_decoder import XttsHifiDecoderFull
from models.experimental.xtts.tt.xtts_full_decoder import TtXttsHifiDecoder


@pytest.fixture(scope="module")
def xtts_state_dict():
    return load_xtts_state_dict()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("latent_len,ref_samples", [(16, 16000)])
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_full_decoder(device, xtts_state_dict, latent_len, ref_samples, pcc, reset_seeds):
    reference = XttsHifiDecoderFull(xtts_state_dict)

    latents = torch.randn(1, latent_len, 1024) * 0.5  # GPT latents [B, T, 1024]
    ref_wav = torch.randn(1, ref_samples) * 0.1  # reference speaker audio [B, samples]
    with torch.no_grad():
        ref_g = reference.speaker_embedding(ref_wav)  # [1, 512, 1]
        ref_out = reference(latents, ref_wav)  # [1, 1, T_out]

    tt_dec = TtXttsHifiDecoder(device, reference)
    latents_dev = ttnn.from_torch(latents.float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32)
    wav_dev = ttnn.from_torch(
        ref_wav.reshape(1, ref_samples, 1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
    )

    # Check the intermediate speaker embedding, then the full waveform.
    tt_g = ttnn.to_torch(tt_dec.speaker_embedding(wav_dev)).float().reshape(1, 512, 1)
    g_pass, g_msg = comp_pcc(ref_g, tt_g, pcc)
    logger.info(f"speaker embedding g: {g_msg}")

    tt_out = ttnn.to_torch(tt_dec(latents_dev, wav_dev)).float().permute(0, 2, 1)  # [1, 1, T_out]

    assert tt_out.shape == ref_out.shape, f"shape {tuple(tt_out.shape)} != {tuple(ref_out.shape)}"
    does_pass, msg = comp_pcc(ref_out, tt_out, pcc)
    logger.info(comp_allclose(ref_out, tt_out))
    logger.info(f"full_decoder latent_len={latent_len} ref_samples={ref_samples}: {msg}")
    assert g_pass, f"speaker embedding PCC below {pcc}: {g_msg}"
    assert does_pass, f"full_decoder PCC below {pcc}: {msg}"
