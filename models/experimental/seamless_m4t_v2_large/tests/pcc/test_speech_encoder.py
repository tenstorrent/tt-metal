# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Speech encoder PCC at long mel sequences and a Tracy-safe shorter length.

Tracy profiling (``test_seamless_m4t_v2_speech_encoder_max_seq_pcc``)::

    python -m tracy -p --op-support-count 100000 -r -v -m pytest \\
        models/experimental/seamless_m4t_v2_large/tests/pcc/test_speech_encoder.py::test_seamless_m4t_v2_speech_encoder_max_seq_pcc \\
        -k 1x4 -sv

    tt-perf-report generated/profiler/reports/<run>/ops_perf_results_<run>.csv \\
        --start-signpost start --end-signpost stop

Real weights only — skipped when download fails.
"""

import os

import pytest
import torch
import ttnn
from loguru import logger

try:
    from tracy import signpost
except ImportError:

    def signpost(header, message=None):  # noqa: ARG001
        pass


from tracy.common import PROFILER_LOGS_DIR
from tracy.process_ops_logs import get_device_data_generate_report

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
MAX_SEQ = 4096


def _drain_device_profiler(device: ttnn.Device) -> None:
    """Flush device profiler ops accumulated since the last drain."""
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        return
    ttnn.ReadDeviceProfiler(device)
    try:
        get_device_data_generate_report(PROFILER_LOGS_DIR, None, None, None, export_csv=False, cleanup_device_log=True)
    except Exception:
        pass


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_speech_encoder_max_seq_pcc(mesh_device, device_params, reset_seeds):
    """PCC ≥ 0.99 at ``mel_seq=MAX_SEQ`` (long-audio paths: chunked matmul, DRAM residual, uncached rel-pos)."""
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
        token_rows = batch * seq
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
            matmul_token_rows=token_rows,
        )

        tt_x = from_torch_bfloat16_tile(mesh_device, input_features, memory_config=ttnn.L1_MEMORY_CONFIG)
        m1 = from_torch_bfloat16_tile(mesh_device, attention_mask, memory_config=ttnn.L1_MEMORY_CONFIG)

        tt_model.pre_warm(batch, seq)
        _ = tt_model(tt_x, conv_attention_mask_1d=m1)
        ttnn.synchronize_device(mesh_device)
        _drain_device_profiler(mesh_device)

        signpost("start")
        tt_out = tt_model(tt_x, conv_attention_mask_1d=m1)
        ttnn.synchronize_device(mesh_device)
        signpost("stop")

        tt_cpu = to_torch_replicated_first_shard(tt_out).to(torch.bfloat16)

        ok, msg = check_with_pcc(ref, tt_cpu, pcc=PCC_THRESHOLD)
        logger.info(f"SeamlessM4Tv2 speech encoder PCC @ mel_seq={seq}: {msg} (threshold {PCC_THRESHOLD})")
        assert ok, msg


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_m4t_v2_speech_encoder_prof_capture_seq_pcc(mesh_device, device_params, reset_seeds):
    """PCC ≥ 0.99 at ``PROF_CAPTURE_MEL_SEQ`` (Tracy-safe mel length on BH 1×4)."""
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
