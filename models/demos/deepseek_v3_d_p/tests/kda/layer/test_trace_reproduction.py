# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in real decoder-stream KDA probes for issue #55420."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.kda.trace_utils import load_decoder_stream_probe
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device({"l1_small_size": 24576, "trace_region_size": 33554432}),
    pytest.mark.skipif(
        not os.getenv("KDA_REAL_DECODER_TRACE_ROOT"), reason="set KDA_REAL_DECODER_TRACE_ROOT for local probes"
    ),
]


@pytest.mark.parametrize("layer_idx", [5, 13, 20])
def test_kda_decoder_stream_probe(device: ttnn.Device, layer_idx: int) -> None:
    """Match issue #55420: FP32 normalized host reference, BF16 device input."""
    sequence = int(os.getenv("KDA_REAL_TRACE_SEQUENCE", "1024"))
    hidden, weights, config = load_decoder_stream_probe(
        Path(os.environ["KIMI_K3_CKPT"]), Path(os.environ["KDA_REAL_DECODER_TRACE_ROOT"]), layer_idx, sequence
    )
    expected, expected_state = kda_forward_reference(hidden, weights, config)
    layer = ttKDA(device, config, weights, layer_idx=layer_idx)
    state = layer.allocate_state(batch_size=1)
    hidden_tt = ttnn.from_torch(
        hidden, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output, state = layer.forward(hidden_tt, state)
    actual = ttnn.to_torch(output).reshape_as(expected)
    actual_state = ttnn.to_torch(state.recurrent).reshape_as(expected_state.recurrent)
    passed, message = comp_pcc(expected, actual, 0.999)
    state_passed, state_message = comp_pcc(expected_state.recurrent, actual_state, 0.999)
    print(
        "KDA_DECODER_PROBE="
        + json.dumps(
            {
                "layer": layer_idx,
                "sequence": sequence,
                "host_reference_dtype": str(hidden.dtype),
                "output": str(message),
                "state": str(state_message),
                "output_max_abs_error": float((expected - actual).abs().max()),
                "state_max_abs_error": float((expected_state.recurrent - actual_state).abs().max()),
            }
        )
    )
    repetitions = int(os.getenv("KDA_LAYER_PERF_REPETITIONS", "0"))
    if repetitions:
        from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _trace_wall_samples_ms

        def validate_replay(replay_state: KdaState, replay_output: ttnn.Tensor) -> dict[str, float]:
            replay = ttnn.to_torch(replay_output).reshape_as(expected)
            recurrent = ttnn.to_torch(replay_state.recurrent).reshape_as(expected_state.recurrent)
            output_ok, output_pcc = comp_pcc(expected, replay, 0.999)
            state_ok, state_pcc = comp_pcc(expected_state.recurrent, recurrent, 0.999)
            assert torch.isfinite(replay).all() and torch.isfinite(recurrent).all()
            assert output_ok, output_pcc
            assert state_ok, state_pcc
            return {"output_pcc": float(output_pcc), "state_pcc": float(state_pcc)}

        samples, trace_accuracy = _trace_wall_samples_ms(
            device, layer, hidden_tt, repetitions, validate_first_replay=validate_replay
        )
        print("KDA_TRACE_REPLAY_ACCURACY=" + json.dumps(trace_accuracy))
        print("KDA_LAYER_TIMINGS_MS=" + json.dumps(samples))
    assert torch.isfinite(actual).all() and torch.isfinite(actual_state).all()
    assert passed, message
    assert state_passed, state_message
