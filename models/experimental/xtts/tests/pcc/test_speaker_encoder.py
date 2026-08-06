# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the XTTS-v2 speaker encoder body (ResNetSpeakerEncoder).

Validates the TTNN SE-ResNet + attentive-statistics pooling against the pure-
PyTorch reference, both from the real coqui/XTTS-v2 checkpoint. Input is a log-mel
``[1, 64, T]`` (the STFT/mel frontend is a later phase); output is the 512-d
L2-normalized speaker embedding ``g``.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/pcc/test_speaker_encoder.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_speaker_encoder import build_reference_speaker_encoder
from models.experimental.xtts.tt.xtts_speaker_encoder import TtResNetSpeakerEncoder


@pytest.fixture(scope="module")
def xtts_state_dict():
    return load_xtts_state_dict()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mel_len", [200])
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_speaker_encoder(device, xtts_state_dict, mel_len, pcc, reset_seeds):
    reference = build_reference_speaker_encoder(xtts_state_dict)

    # mel magnitudes [B, 64, T] (positive, as a real mel spectrogram would be).
    mel = torch.randn(1, 64, mel_len).abs() + 0.1
    with torch.no_grad():
        ref_g = reference(mel)  # [1, 512]

    tt_enc = TtResNetSpeakerEncoder(device, reference)
    mel_dev = ttnn.from_torch(mel.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    tt_g = ttnn.to_torch(tt_enc(mel_dev)).float()

    assert tt_g.shape == ref_g.shape, f"shape {tuple(tt_g.shape)} != {tuple(ref_g.shape)}"
    does_pass, msg = comp_pcc(ref_g, tt_g, pcc)
    logger.info(comp_allclose(ref_g, tt_g))
    logger.info(f"speaker_encoder mel_len={mel_len}: {msg}")
    assert does_pass, f"speaker_encoder PCC below {pcc}: {msg}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_speaker_encoder_shape_reuse(device, xtts_state_dict, pcc, reset_seeds):
    """One encoder instance, two reference clips of different length — both must be right.

    ``TtConv2d`` caches ttnn.conv2d's preprocessed weights, and those are only valid for the
    parallelization the conv picked from the *first* call's shape. ttnn cannot detect a stale one
    (``is_valid_device_conv_weights`` checks only layout/rank/out_channels/dtype), so before
    ``TtConv2d.forward`` learned to key that cache this returned a plausible but badly wrong
    embedding on the second length — silently, with no error. 200 then 512 is the smallest pair
    that trips it: at 512 layer3 crosses ``_stage_memory_config``'s sharding threshold, so its
    convs get a different parallelization than the cached weights were built for.

    Both lengths are checked against a *fresh* encoder too, so a regression here is attributed to
    the reuse rather than to the lengths themselves.
    """
    reference = build_reference_speaker_encoder(xtts_state_dict)
    lengths = [200, 512]

    def embed(tt_enc, mel_len):
        torch.manual_seed(0)
        mel = torch.randn(1, 64, mel_len).abs() + 0.1
        with torch.no_grad():
            ref_g = reference(mel)
        mel_dev = ttnn.from_torch(mel.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
        return ref_g, ttnn.to_torch(tt_enc(mel_dev)).float()

    shared = TtResNetSpeakerEncoder(device, reference)
    for mel_len in lengths:
        ref_g, reused_g = embed(shared, mel_len)
        _, fresh_g = embed(TtResNetSpeakerEncoder(device, reference), mel_len)

        does_pass, msg = comp_pcc(ref_g, reused_g, pcc)
        logger.info(f"speaker_encoder reused-instance mel_len={mel_len}: {msg}")
        assert does_pass, f"reused encoder at mel_len={mel_len} scored below {pcc}: {msg}"
        # The reused instance must match the fresh one, not merely clear the PCC bar: a stale
        # conv weight shifts the embedding well before it would fail an absolute threshold.
        assert torch.allclose(reused_g, fresh_g, atol=1e-6), (
            f"reused encoder at mel_len={mel_len} disagrees with a fresh one "
            f"(max abs diff {(reused_g - fresh_g).abs().max().item():.3e})"
        )
