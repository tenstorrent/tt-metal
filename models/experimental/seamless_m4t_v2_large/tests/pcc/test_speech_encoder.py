# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_speech_encoder import (
    forward_torch_speech_encoder,
    load_pretrained_speech_encoder,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_speech_encoder_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_speech_encoder import TTSeamlessM4Tv2SpeechEncoder

PCC_THRESHOLD = 0.99


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_seamless_m4t_v2_speech_encoder_pcc(device, reset_seeds):
    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    torch.manual_seed(0)
    speech_enc, cfg = load_pretrained_speech_encoder(weights_dir, dtype=torch.bfloat16)

    # Short mel sequence (feature dim must match ``feature_projection_input_dim``).
    batch, seq = 1, 48
    n_mels = cfg.feature_projection_input_dim
    input_features = torch.randn(batch, seq, n_mels, dtype=torch.bfloat16)
    attention_mask = torch.ones(batch, seq, dtype=torch.long)

    ref = forward_torch_speech_encoder(
        speech_enc,
        input_features,
        attention_mask,
    ).to(torch.bfloat16)

    params = create_speech_encoder_parameters(speech_enc, device=device)
    tt_model = TTSeamlessM4Tv2SpeechEncoder(
        device,
        params,
        hidden_size=cfg.hidden_size,
        feature_projection_input_dim=cfg.feature_projection_input_dim,
        speech_encoder_attention_heads=cfg.speech_encoder_attention_heads,
        speech_encoder_intermediate_size=cfg.speech_encoder_intermediate_size,
        speech_encoder_layers=cfg.speech_encoder_layers,
        layer_norm_eps=cfg.layer_norm_eps,
        speech_encoder_chunk_size=cfg.speech_encoder_chunk_size,
        speech_encoder_left_chunk_num=cfg.speech_encoder_left_chunk_num,
    )

    tt_x = ttnn.from_torch(
        input_features,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    m1 = ttnn.from_torch(
        attention_mask.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_out = tt_model(tt_x, conv_attention_mask_1d=m1)
    tt_cpu = ttnn.to_torch(tt_out).to(torch.bfloat16)
    pcc_passed, pcc_message = check_with_pcc(ref, tt_cpu, pcc=PCC_THRESHOLD)
    logger.info(pcc_message)
    assert pcc_passed, pcc_message
