# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-layer prefill+decode performance test for Devstral-2-123B (Ministral3) on TTNN.

Builds a 1-layer model, runs prefill (compile + trace replay) and decode (compile + trace
replay), then logs wall-clock metrics. Does not invoke Tracy or write perf CSV/JSON reports.

Run::

    pytest models/experimental/devstral2_123B_instruct/tests/perf/test_perf.py
"""

from __future__ import annotations

import json

import pytest
from loguru import logger

import ttnn
from models.experimental.devstral2_123B_instruct.demo.text_demo import _mesh_device_param
from models.experimental.devstral2_123B_instruct.tests.perf.test_e2e_performant import _run_devstral2_perf
from models.experimental.devstral2_123B_instruct.tt.model_args import DEVSTRAL2_LARGE_L1_SMALL_SIZE

NUM_LAYERS = 1
PROMPT_LEN = 128
PREFILL_ITERS = 3
DECODE_ITERS = 32


@pytest.mark.timeout(0)
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 100_000_000,
            "num_command_queues": 1,
            "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
        }
    ],
    indirect=True,
)
def test_perf_device_bare_metal_devstral2_123B_instruct_single_layer(mesh_device):
    """Measure 1-layer prefill+decode trace performance and log timings."""

    model_name = f"devstral2_123B_instruct_L{NUM_LAYERS}"

    results = _run_devstral2_perf(
        mesh_device,
        num_layers=NUM_LAYERS,
        prompt_len=PROMPT_LEN,
        decode_iters=DECODE_ITERS,
        prefill_iters=PREFILL_ITERS,
    )

    settings = f"L{NUM_LAYERS}_prefill{results['padded_prompt_len']}_" f"x{PREFILL_ITERS}_decode{DECODE_ITERS}"
    logger.info(f"{model_name} (batch=1, {settings}) trace perf:\n" f"{json.dumps(results, indent=2)}")
    logger.info(
        f"{model_name}: "
        f"TTFT={results['ttft_ms']:.1f}ms, "
        f"prefill compile={results['prefill_compile_time_s']*1000:.0f}ms, "
        f"prefill replay={results['prefill_replay_time_s']*1000:.1f}ms "
        f"({results['prefill_throughput_tok_per_s']:.1f} tok/s), "
        f"decode compile={results['decode_compile_time_s']*1000:.0f}ms, "
        f"decode replay={results['decode_replay_time_s']*1000:.2f}ms "
        f"({results['decode_throughput_tok_per_s_per_user']:.2f} tok/s/user)"
    )
