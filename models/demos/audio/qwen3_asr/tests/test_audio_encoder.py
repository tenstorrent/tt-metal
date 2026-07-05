# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC + regression tests for the ttnn Qwen3-ASR AuT audio encoder (Blackhole / P150).

  test_conv_frontend_pcc      ttnn conv2d frontend  vs golden conv_out
  test_audio_encoder_pcc      full ttnn audio tower  vs golden audio_tower
  test_encoder_length_stability
                              encoder output for a fixed clip is deterministic and
                              unchanged after encoding other-length clips in between
                              (guards the length-keyed state bug that motivates the
                              fixed-prefill workaround — see tt/qwen3_asr_decoder.py).

Golden tensors and the HF snapshot are supplied via env vars; see tests/conftest.py.
Run on a P150 (single Blackhole) box that has them staged, e.g.:

  pytest models/demos/audio/qwen3_asr/tests/test_audio_encoder.py \
    --device-params '{"l1_small_size": 32768}'   # or via the parametrize below
"""
import numpy as np
import pytest
import torch

import ttnn
from models.demos.audio.qwen3_asr.reference import audio_encoder_ref as ref
from models.demos.audio.qwen3_asr.tt import audio_encoder as tt_enc
from tests.ttnn.utils_for_testing import assert_with_pcc

# conv2d frontend reserves an L1 scratch region; 32768 is validated for this model.
QWEN3ASR_L1_SMALL_SIZE = 32768

PCC_CONV = 0.99
PCC_ENCODER = 0.99
PCC_DETERMINISTIC = 0.999  # greedy/static graph -> identical inputs must give ~identical output


@pytest.mark.parametrize("device_params", [{"l1_small_size": QWEN3ASR_L1_SMALL_SIZE}], indirect=True)
def test_conv_frontend_pcc(device, golden, audio_tower_weights):
    """ttnn conv2d frontend (mel -> conv_out, pre positional-embedding) vs golden."""
    from ttnn.model_preprocessing import preprocess_linear_weight

    w = audio_tower_weights
    mel = golden("input_features.npy")
    gold = golden("conv_out.npy").reshape(-1, 1024)

    conv_w = tt_enc.preprocess_conv_weights(w, device)
    conv_out_w = ttnn.to_device(preprocess_linear_weight(w["conv_out.weight"], dtype=ttnn.bfloat16), device)
    out = tt_enc.conv_frontend_tt(mel, conv_w, conv_out_w, None, device)

    assert_with_pcc(gold, out, PCC_CONV)


@pytest.mark.parametrize("device_params", [{"l1_small_size": QWEN3ASR_L1_SMALL_SIZE}], indirect=True)
def test_audio_encoder_pcc(device, golden, audio_tower_weights):
    """Full ttnn audio tower vs golden audio_tower output (host conv frontend + PE feed)."""
    w = audio_tower_weights
    mel = golden("input_features.npy")
    gold = golden("audio_tower.npy")  # (S, 2048)

    # host conv frontend + positional embedding (validated PCC=1.0 in reference)
    conv = ref.conv_frontend(mel, w)  # (n_chunks, 13, 1024)
    pe = ref.sinusoids(1500, 1024)[: conv.shape[1]]
    x_host = (conv + pe.unsqueeze(0)).reshape(-1, 1024)  # (S, 1024)

    params = tt_enc.preprocess_weights(w, device)
    out = tt_enc.encode(x_host, params, device)  # (S, 2048) torch

    assert_with_pcc(gold, out, PCC_ENCODER)


@pytest.mark.parametrize("device_params", [{"l1_small_size": QWEN3ASR_L1_SMALL_SIZE}], indirect=True)
def test_encoder_length_stability(device, golden, audio_tower_weights):
    """Encoding must be deterministic AND free of cross-request state leak across mel lengths.

    The server empties transcripts when requests of different real lengths interleave; this
    test proved the *encoder* is not the culprit (stable across lengths). It stays as a
    regression guard: if a future encoder change introduces length-dependent state, the PCC
    of a fixed clip re-encoded after other lengths drops below 1.0 and this fails.
    """
    w = audio_tower_weights
    mel = golden("input_features.npy")  # (128, ~1200) = ~12s
    A = mel
    Bs = mel[:, :600].contiguous()  # ~6s
    Bl = mel.repeat(1, 3)[:, :3000].contiguous()  # ~30s (the length the server emptied on)

    params = tt_enc.preprocess_weights(w, device)

    eA = tt_enc.encode_mel(A, params, device).float()
    eBl = tt_enc.encode_mel(Bl, params, device).float()
    eA2 = tt_enc.encode_mel(A, params, device).float()  # repeat, no interleave
    _ = tt_enc.encode_mel(Bl, params, device)
    _ = tt_enc.encode_mel(Bs, params, device)
    eA3 = tt_enc.encode_mel(A, params, device).float()  # A after other lengths

    assert torch.isfinite(eBl).all(), "30s encode produced non-finite values"
    assert_with_pcc(eA, eA2, PCC_DETERMINISTIC)  # deterministic
    assert_with_pcc(eA, eA3, PCC_DETERMINISTIC)  # no state leak across lengths
