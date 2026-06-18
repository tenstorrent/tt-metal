# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TT-only ISL performance sweep for Devstral-2-123B (88 layers).

Traced chunked prefill + decode trace, matching ``demo/text_demo.py`` defaults.
Decode replay count uses ``DEVSTRAL2_MAX_NEW_TOKENS`` (default 100), capped via
``DEVSTRAL2_ISL_PERF_DECODE_REPLAY_CAP`` (default 32).

Run::

    pytest models/experimental/devstral2_123B_instruct/tests/perf/test_isl_sweep_perf.py -k sanity -v
    pytest models/experimental/devstral2_123B_instruct/tests/perf/test_isl_sweep_perf.py -k sweep -v

Set ``DEVSTRAL2_HF_LOCAL_ONLY=1`` when HF shards and the TT weight cache are already
populated. Results are written under ``tests/isl_sweep_perf_outputs/``.
"""

from __future__ import annotations

import pytest
from loguru import logger

import ttnn
from models.experimental.devstral2_123B_instruct.demo.decode_trace_2cq import num_command_queues_for_decode
from models.experimental.devstral2_123B_instruct.demo.text_demo import _mesh_device_param
from models.experimental.devstral2_123B_instruct.tests.perf.isl_perf_common import (
    PREFILL_SANITY_SEQ_LENGTHS,
    PREFILL_SWEEP_SEQ_LENGTHS,
    isl_sweep_timeout_seconds,
    run_isl_perf_sweep,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import DEVSTRAL2_LARGE_L1_SMALL_SIZE


def _isl_perf_device_params():
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": 100_000_000,
        "num_command_queues": num_command_queues_for_decode(),
        "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    }


@pytest.mark.timeout(isl_sweep_timeout_seconds(PREFILL_SANITY_SEQ_LENGTHS))
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", [_isl_perf_device_params()], indirect=True)
def test_devstral2_123B_instruct_isl_perf_sanity(mesh_device):
    """Short ISL perf smoke (32, 128) on full 88-layer TT model."""
    rows = run_isl_perf_sweep(mesh_device, PREFILL_SANITY_SEQ_LENGTHS, label="sanity")
    for row in rows:
        assert row["ttft_s"] > 0, f"ISL={row['isl']}: TTFT must be positive"
        assert row["prefill_tok_per_s"] > 0, f"ISL={row['isl']}: prefill tok/s must be positive"
        if row["decode_replay_iters"] > 0:
            assert row["decode_tok_per_s_per_user"] > 0, f"ISL={row['isl']}: decode tok/s/u must be positive"


@pytest.mark.timeout(isl_sweep_timeout_seconds(PREFILL_SWEEP_SEQ_LENGTHS))
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", [_isl_perf_device_params()], indirect=True)
def test_devstral2_123B_instruct_isl_perf_sweep(mesh_device):
    """Full ISL perf sweep: 32 … 262144 (powers of two). Expect hours on BH Loudbox."""
    rows = run_isl_perf_sweep(mesh_device, PREFILL_SWEEP_SEQ_LENGTHS, label="sweep")
    logger.info(f"Completed ISL perf sweep with {len(rows)} points")
