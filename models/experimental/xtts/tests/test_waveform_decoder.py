# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the XTTS-v2 HiFi-GAN generator (``waveform_decoder``).

Validates the TTNN ``TtHifiganGenerator`` against the pure-PyTorch reference,
both loaded from the *real* coqui/XTTS-v2 checkpoint (weight-norm folded). The
generator maps a GPT latent ``[1, T, 1024]`` plus a speaker embedding
``[1, 1, 512]`` to a ``[1, T*256, 1]`` waveform.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    # first run downloads ~1.9 GB of XTTS-v2 weights to the HF cache
    pytest models/experimental/xtts/tests/test_waveform_decoder.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_hifigan import build_reference_waveform_decoder
from models.experimental.xtts.tt.xtts_hifigan import TtHifiganGenerator


@pytest.fixture(scope="module")
def xtts_state_dict():
    return load_xtts_state_dict()


def _to_device_blc(torch_ncl: torch.Tensor, device) -> ttnn.Tensor:
    """torch [N, C, L] -> ttnn channels-last [N, L, C] fp32 ROW_MAJOR on device."""
    return ttnn.from_torch(
        torch_ncl.permute(0, 2, 1).contiguous().float(),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.float32,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("latent_len", [16, 32])
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_waveform_decoder(device, xtts_state_dict, latent_len, pcc, reset_seeds):
    # Reference generator (real weights, weight-norm folded, eval).
    reference = build_reference_waveform_decoder(xtts_state_dict)

    latents = torch.randn(1, 1024, latent_len) * 0.5  # GPT-latent-like [N, C, T]
    g = torch.randn(1, 512, 1) * 0.5  # speaker embedding [N, C, 1]
    with torch.no_grad():
        ref_out = reference(latents, g)  # [1, 1, latent_len*256]

    # TTNN generator from the same folded weights.
    tt_gen = TtHifiganGenerator(device, reference.state_dict())
    tt_out = tt_gen(_to_device_blc(latents, device), _to_device_blc(g, device))
    tt_out = ttnn.to_torch(tt_out).float().permute(0, 2, 1)  # [N, L, 1] -> [N, 1, L]

    assert tt_out.shape == ref_out.shape, f"shape {tuple(tt_out.shape)} != {tuple(ref_out.shape)}"
    does_pass, msg = comp_pcc(ref_out, tt_out, pcc)
    logger.info(comp_allclose(ref_out, tt_out))
    logger.info(f"waveform_decoder latent_len={latent_len}: {msg}")
    assert does_pass, f"waveform_decoder PCC below {pcc}: {msg}"
