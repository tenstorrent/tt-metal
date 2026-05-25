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


from models.experimental.seamless_m4t_v2_large.tt.common import to_torch_replicated_first_shard
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_speech_encoder_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_speech_encoder import TTSeamlessM4Tv2SpeechEncoder
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    mesh_default_device,
    MESH_DEVICE_PARAMETRIZE_TEXT,
    from_torch_bfloat16_tile,
)

PCC_THRESHOLD = 0.99


def _run_speech_encoder_pcc(device, *, seq: int = 48) -> None:
    """Shared PCC body. Mesh-safe readback via ``to_torch_replicated_first_shard``.

    HF's speech encoder has no fixed max audio sequence length — chunked attention
    (``speech_encoder_chunk_size = 20000`` mel frames) lets it process arbitrarily long inputs
    bounded only by DRAM. Parametrising ``seq`` exposes a long-audio regression test that
    exercises the chunked-attention mask path and L1 budget at larger feature lengths.
    """
    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    torch.manual_seed(0)
    speech_enc, cfg = load_pretrained_speech_encoder(weights_dir, dtype=torch.bfloat16)

    # Mel sequence (feature dim must match ``feature_projection_input_dim``).
    batch = 1
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
        matmul_token_rows=64,
    )

    tt_x = from_torch_bfloat16_tile(device, input_features, memory_config=ttnn.L1_MEMORY_CONFIG)
    m1 = from_torch_bfloat16_tile(device, attention_mask, memory_config=ttnn.L1_MEMORY_CONFIG)

    tt_out = tt_model(tt_x, conv_attention_mask_1d=m1)
    tt_cpu = to_torch_replicated_first_shard(tt_out).to(torch.bfloat16)
    pcc_passed, pcc_message = check_with_pcc(ref, tt_cpu, pcc=PCC_THRESHOLD)
    logger.info(pcc_message)
    assert pcc_passed, pcc_message


@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_speech_encoder_pcc(mesh_device, device_params, reset_seeds):
    _ = reset_seeds
    _ = device_params
    with mesh_default_device(mesh_device):
        _run_speech_encoder_pcc(mesh_device)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_speech_encoder_long_audio_pcc(mesh_device, device_params, reset_seeds):
    """PCC check at a longer mel sequence so chunked-attention and adaptor paths are exercised.

    HF has no fixed max audio sequence — speech-encoder accepts any length, with chunked attention
    keyed by ``speech_encoder_chunk_size``. 512 frames (~10 s at 50 fps mel) exercises the
    long-audio path (chunked 1D matmul when mel > 128, relative-position DRAM offload).
    """
    _ = reset_seeds
    _ = device_params
    with mesh_default_device(mesh_device):
        _run_speech_encoder_pcc(mesh_device, seq=256)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_speech_encoder_very_long_mel_pcc(mesh_device, device_params, reset_seeds):
    """PCC at ~13 s mel length (671 frames) — matches full demo T2ST wav fed into S2TT."""
    _ = reset_seeds
    _ = device_params
    with mesh_default_device(mesh_device):
        _run_speech_encoder_pcc(mesh_device, seq=672)
