# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Phase 3b PCC: full TtSpeechEncoder vs HF SeamlessM4Tv2 speech encoder, real weights.

Loads the checkpoint (no processor/tokenizer needed — synthetic input_features),
runs the HF speech encoder for the golden, builds TtSpeechEncoder from the same
state dict, and checks end-to-end encoder-output PCC >= 0.98. Skips if the
checkpoint is unavailable.
"""

import pytest
import torch

import ttnn
from models.demos.seamless_m4t_v2.tt.conformer_encoder import TtSpeechEncoder
from models.demos.seamless_m4t_v2.tt.model_config import SeamlessS2TTConfig, DEFAULT_MODEL_ID


@pytest.fixture(scope="module")
def hf_speech_encoder():
    try:
        from transformers import SeamlessM4Tv2ForSpeechToText

        model = SeamlessM4Tv2ForSpeechToText.from_pretrained(DEFAULT_MODEL_ID).eval().float()
        return model
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"checkpoint unavailable: {e}")


# on-device conv path needs an L1_SMALL scratch region
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("conv_cpu_fallback", [True, False], ids=["host", "device"])
@pytest.mark.parametrize("seq_len", [256])
def test_encoder_pcc(device, hf_speech_encoder, seq_len, conv_cpu_fallback):
    from tests.ttnn.utils_for_testing import assert_with_pcc

    torch.manual_seed(0)
    cfg = SeamlessS2TTConfig.from_hf(conv_cpu_fallback=conv_cpu_fallback)
    feats = torch.randn(1, seq_len, cfg.feature_projection_input_dim)

    with torch.no_grad():
        golden = hf_speech_encoder.speech_encoder(input_features=feats, return_dict=True).last_hidden_state

    state_dict = hf_speech_encoder.speech_encoder.state_dict()
    tt_enc = TtSpeechEncoder(state_dict, cfg, device, dtype=ttnn.bfloat16)
    feats_tt = ttnn.from_torch(feats, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(tt_enc(feats_tt))

    assert out.shape == golden.shape, f"{out.shape} vs {golden.shape}"
    assert_with_pcc(golden, out, pcc=0.98)
