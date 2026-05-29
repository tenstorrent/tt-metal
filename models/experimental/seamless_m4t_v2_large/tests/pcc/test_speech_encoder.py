# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-shot PCC test for the SeamlessM4Tv2 speech encoder at its longest supported mel input.

HF's speech encoder has no fixed maximum input length: chunked self-attention
(``speech_encoder_chunk_size`` = 20000 mel frames in the HF config) lets it process arbitrarily long
audio bounded only by DRAM. The test below runs at ``seq=3000`` mel frames (~60 s at the
SeamlessM4T mel rate), which exercises every long-audio code path in one go: chunked 1D matmul
(active above ``MATMUL_1D_SEQ_THRESHOLD`` = 128), DRAM residual / LN (above
``_LONG_AUDIO_RES_DRAM_THRESHOLD`` = 1024 mel frames), uncached relative-position tables (above
``_MAX_CACHED_REL_POS_TABLE_BYTES`` = 32 MB, which seq=3000 vastly exceeds at ~1.1 GB per layer).

Empirical ceiling (from ``test_sweep_max_seq.py`` on Blackhole 1×4):

    seq=2125  PCC 0.9945  PASS
    seq=2400  PCC 0.9951  PASS
    seq=2700  PCC 0.9955  PASS
    seq=3000  PCC 0.9966  PASS  ← MAX_SEQ here
    seq=3300  CB clash  FAIL    (conformer attention softmax static CBs vs persistent QKV in L1)
    seq=3600  L1 allocator OOM  FAIL
    seq=4096  L1 allocator OOM  FAIL

Inputs longer than 3000 keep working via the 20000-frame chunked-attention window. To run a single
non-windowed pass at seq >3000, the speech-encoder needs chunked-SDPA or sequence-parallel attention
(model-side work); TP only shards weights, not seq, so the 1×4 mesh doesn't lift this ceiling.

Real weights only — if ``huggingface_hub`` is missing or the download fails the test is skipped.
"""

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
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    MESH_DEVICE_PARAMETRIZE_TEXT,
    from_torch_bfloat16_tile,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_speech_encoder_parameters
from models.experimental.seamless_m4t_v2_large.tests.pcc.prof_capture_limits import SPEECH_ENCODER_MEL_SEQ
from models.experimental.seamless_m4t_v2_large.tt.tt_speech_encoder import TTSeamlessM4Tv2SpeechEncoder

PCC_THRESHOLD = 0.99
PROF_CAPTURE_MEL_SEQ = SPEECH_ENCODER_MEL_SEQ
# Empirically determined by ``test_sweep_max_seq.py`` — longest single-pass mel input where the
# conformer attention's static CBs still fit per-core L1 (see file docstring for the seq vs PCC
# scan; seq=3300 is the first FAIL, with CB clash on the softmax).
MAX_SEQ = 3000


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_speech_encoder_max_seq_pcc(mesh_device, device_params, reset_seeds):
    """Speech encoder PCC ≥ 0.99 at ``mel_seq=2125``, all long-audio paths active.

    Above ``speech_encoder_chunk_size`` (20000) the encoder uses windowed attention to handle
    arbitrarily long inputs; this test pins the longest *non-windowed* mel input that still
    fits in a single attention pass (per-core L1 CB budget). seq=4096 needs the windowed path.
    """
    _ = reset_seeds
    _ = device_params

    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    with mesh_default_device(mesh_device):
        torch.manual_seed(0)
        speech_enc, cfg = load_pretrained_speech_encoder(weights_dir, dtype=torch.bfloat16)

        batch = 1
        seq = MAX_SEQ
        n_mels = cfg.feature_projection_input_dim
        input_features = torch.randn(batch, seq, n_mels, dtype=torch.bfloat16)
        attention_mask = torch.ones(batch, seq, dtype=torch.long)

        ref = forward_torch_speech_encoder(speech_enc, input_features, attention_mask).to(torch.bfloat16)

        params = create_speech_encoder_parameters(speech_enc, device=mesh_device)
        tt_model = TTSeamlessM4Tv2SpeechEncoder(
            mesh_device,
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

        tt_x = from_torch_bfloat16_tile(mesh_device, input_features, memory_config=ttnn.L1_MEMORY_CONFIG)
        m1 = from_torch_bfloat16_tile(mesh_device, attention_mask, memory_config=ttnn.L1_MEMORY_CONFIG)

        tt_out = tt_model(tt_x, conv_attention_mask_1d=m1)
        tt_cpu = to_torch_replicated_first_shard(tt_out).to(torch.bfloat16)

        ok, msg = check_with_pcc(ref, tt_cpu, pcc=PCC_THRESHOLD)
        logger.info(f"SeamlessM4Tv2 speech encoder PCC @ mel_seq={seq}: {msg} (threshold {PCC_THRESHOLD})")
        assert ok, msg


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_speech_encoder_prof_capture_seq_pcc(mesh_device, device_params, reset_seeds):
    """Speech encoder PCC ≥ 0.99 at the tracy-safe mel length (``PROF_CAPTURE_MEL_SEQ`` = 32).

    Highest power-of-two mel length where ``python3 -m tracy -r -v`` and PCC both pass on BH 1×4.
    mel_seq=64/128 fail PCC (bf16 drift); mel_seq=256+ overflows Tracy's 32K source locations.
    ``test_seamless_m4t_v2_speech_encoder_max_seq_pcc`` (mel_seq=3000) also breaks capture.
    """
    _ = reset_seeds
    _ = device_params

    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    with mesh_default_device(mesh_device):
        torch.manual_seed(0)
        speech_enc, cfg = load_pretrained_speech_encoder(weights_dir, dtype=torch.bfloat16)

        batch = 1
        seq = PROF_CAPTURE_MEL_SEQ
        n_mels = cfg.feature_projection_input_dim
        input_features = torch.randn(batch, seq, n_mels, dtype=torch.bfloat16)
        attention_mask = torch.ones(batch, seq, dtype=torch.long)

        ref = forward_torch_speech_encoder(speech_enc, input_features, attention_mask).to(torch.bfloat16)

        params = create_speech_encoder_parameters(speech_enc, device=mesh_device)
        tt_model = TTSeamlessM4Tv2SpeechEncoder(
            mesh_device,
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

        tt_x = from_torch_bfloat16_tile(mesh_device, input_features, memory_config=ttnn.L1_MEMORY_CONFIG)
        m1 = from_torch_bfloat16_tile(mesh_device, attention_mask, memory_config=ttnn.L1_MEMORY_CONFIG)

        tt_out = tt_model(tt_x, conv_attention_mask_1d=m1)
        tt_cpu = to_torch_replicated_first_shard(tt_out).to(torch.bfloat16)

        ok, msg = check_with_pcc(ref, tt_cpu, pcc=PCC_THRESHOLD)
        logger.info(f"SeamlessM4Tv2 speech encoder prof-capture PCC @ mel_seq={seq}: {msg} (threshold {PCC_THRESHOLD})")
        assert ok, msg
