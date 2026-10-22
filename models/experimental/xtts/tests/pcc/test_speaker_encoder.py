# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_speaker_encoder import build_reference_speaker_encoder
from models.experimental.xtts.tt.xtts_speaker_encoder import TtResNetSpeakerEncoder


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mel_len", [200])
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_speaker_encoder(device, xtts_state_dict, mel_len, pcc, reset_seeds):
    """Compare TTNN speaker-encoder embedding to the PyTorch reference via PCC."""
    reference = build_reference_speaker_encoder(xtts_state_dict)

    mel = torch.randn(1, 64, mel_len).abs() + 0.1
    with torch.no_grad():
        ref_g = reference(mel)

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
    """Check a reused TTNN speaker encoder matches a fresh instance across mel lengths."""
    reference = build_reference_speaker_encoder(xtts_state_dict)
    lengths = [200, 512]

    def embed(tt_enc, mel_len):
        """Embed a random mel of the given length with both reference and TTNN encoders."""
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
        # Stale TtConv2d weight cache can pass absolute PCC but disagree with a fresh encoder.
        assert torch.allclose(reused_g, fresh_g, atol=1e-6), (
            f"reused encoder at mel_len={mel_len} disagrees with a fresh one "
            f"(max abs diff {(reused_g - fresh_g).abs().max().item():.3e})"
        )
