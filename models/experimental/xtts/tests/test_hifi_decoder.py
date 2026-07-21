# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the XTTS-v2 HifiDecoder (latent upsample + HiFi-GAN generator).

Validates the on-device latent linear-upsample (as a resample matmul) chained into
the generator against the pure-PyTorch reference ``HifiDecoder.forward(latents, g)``,
with real coqui/XTTS-v2 weights. Input is a GPT latent ``[1, T, 1024]`` and speaker
embedding ``[1, 1, 512]``; output is the ``[1, T*~4.35*256, 1]`` waveform.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/test_hifi_decoder.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_hifi_decoder import XttsHifiDecoderReference
from models.experimental.xtts.tt.xtts_hifi_decoder import TtHifiDecoder


@pytest.fixture(scope="module")
def xtts_state_dict():
    return load_xtts_state_dict()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("latent_len", [16, 32])
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_hifi_decoder(device, xtts_state_dict, latent_len, pcc, reset_seeds):
    reference = XttsHifiDecoderReference(xtts_state_dict)

    # Real GPT latents have std ~2.3 (with large outliers), not ~0.5 — and the decoder's
    # mixed-precision (bf16) stages are far more accurate on that larger, structured
    # signal than on small ~N(0, 0.5) noise. Scale the synthetic latents to the real
    # magnitude so the PCC bar reflects real inference (see the bf16-stage default).
    latents = torch.randn(1, latent_len, 1024) * 2.3  # channels-last [B, T, C]
    g = torch.randn(1, 512, 1) * 0.5  # [B, C, 1]
    with torch.no_grad():
        ref_out = reference(latents, g)  # [1, 1, L_out*256]

    tt_dec = TtHifiDecoder(device, reference.waveform_decoder.state_dict())
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
