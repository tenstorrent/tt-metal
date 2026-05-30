# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-layer prefill+decode performance test for Mistral-Small-4-119B multimodal on TTNN.

Thin wrapper around ``_run_mistral_perf`` (see ``test_e2e_performant.py``) that exercises
the L1V1 (1 text layer + 1 vision layer) smoke configuration. Logs wall-clock metrics;
does not invoke Tracy or write perf CSV/JSON reports.

Run::

    pytest models/experimental/mistral_small_4_119b/tests/perf/test_perf.py
"""

from __future__ import annotations

import json

import pytest
from loguru import logger

from models.experimental.mistral_small_4_119b.tests.perf.test_e2e_performant import (
    _e2e_perf_device_params,
    _mesh_device_param,
    _run_mistral_perf,
)

NUM_LAYERS = 1
NUM_VISION_LAYERS = 1
PROMPT_LEN = 128
PREFILL_ITERS = 3
DECODE_ITERS = 32


@pytest.mark.timeout(0)
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", [_e2e_perf_device_params()], indirect=True)
def test_perf_device_bare_metal_mistral_small_4_119b_single_layer(mesh_device):
    """Measure 1-text-layer + 1-vision-layer multimodal prefill+decode trace performance."""
    results = _run_mistral_perf(
        mesh_device,
        num_text_layers=NUM_LAYERS,
        num_vision_layers=NUM_VISION_LAYERS,
        prompt_len=PROMPT_LEN,
        decode_iters=DECODE_ITERS,
        prefill_iters=PREFILL_ITERS,
    )

    model_name = f"mistral_small_4_119b_L{NUM_LAYERS}V{NUM_VISION_LAYERS}"
    settings = (
        f"L{NUM_LAYERS}V{NUM_VISION_LAYERS}_prefill{results['padded_prompt_len']}_"
        f"x{PREFILL_ITERS}_decode{DECODE_ITERS}"
    )
    logger.info(f"{model_name} (batch=1, {settings}) trace perf:\n{json.dumps(results, indent=2)}")
    logger.info(
        f"{model_name}: "
        f"TTFT={results['ttft_ms']:.1f}ms, "
        f"vision compile+load={results['vision_compile_time_s']*1000:.0f}ms, "
        f"vision replay={results['vision_replay_time_s']*1000:.1f}ms, "
        f"prefill compile={results['prefill_compile_time_s']*1000:.0f}ms, "
        f"prefill replay={results['prefill_replay_time_s']*1000:.1f}ms "
        f"({results['prefill_throughput_tok_per_s']:.1f} tok/s), "
        f"decode compile={results['decode_compile_time_s']*1000:.0f}ms, "
        f"decode capture={results['decode_capture_time_s']*1000:.0f}ms, "
        f"decode replay={results['decode_replay_time_s']*1000:.2f}ms, "
        f"steady-state={results['steady_state_decode_throughput_tok_per_s']:.2f} tok/s/user, "
        f"end-to-end={results['end_to_end_throughput_tok_per_s']:.2f} tok/s/user"
    )
