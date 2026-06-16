# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-layer prefill+decode workload for Tracy / device-perf profiling.

Mirrors the single-layer model perf setup (1 decoder layer, partial Hub weights, prefill 128
tokens then one decode step at position 128) without PCC assertions.

Each measured iteration runs **both** prefill and decode inside the ``start``/``stop``
signpost window. Run prefill before decode so phase-split tooling can separate prefill vs
decode kernels in the ops log.

Standalone Tracy capture::

    python -m tracy -p -v -r --dump-device-data-mid-run \\
        pytest models/experimental/devstral2_123B_instruct/tests/perf/test_profile_single_layer_prefill_decode.py \\
        ::test_profile_single_layer_prefill_decode -v

Device perf CSV/JSON dump (wraps the command above via ``run_device_perf``)::

    pytest models/experimental/devstral2_123B_instruct/tests/perf/test_device_perf_single_layer_prefill_decode.py \\
        -v -m models_device_performance_bare_metal

Analyze the generated ``ops_perf_results_*.csv`` with ``tt-perf-report``. The file
contains ``start``/``stop`` signposts; you must pass both explicitly (default mode
only anchors on the last signpost and shows no device ops)::

    tt-perf-report generated/profiler/devstral2_123B_instruct_L1_prefill_decode/reports/.../ops_perf_results_*.csv \\
        --start-signpost start --end-signpost stop

Or ignore signposts to include warmup + measured iteration (noisier)::

    tt-perf-report .../ops_perf_results_*.csv --ignore-signposts
"""

from __future__ import annotations

import os

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.devstral2_123B_instruct.tests._devstral_weights import DEVSTRAL2_TEST_MAX_SEQ_LEN
from models.experimental.devstral2_123B_instruct.tests.model_test_helpers import (
    current_pos_to_tt,
    input_ids_to_tt,
    setup_devstral_ministral3_partial_one_layer,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import DEVSTRAL2_LARGE_L1_SMALL_SIZE

PREFILL_SEQ_LEN = 128
DECODE_POS = PREFILL_SEQ_LEN
NUM_WARMUP_ITERS = 1


def _mesh_device_param():
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "N150x4": (1, 4),
        "P150x4": (1, 4),
        "T3K": (1, 8),
        "TG": (8, 4),
    }.get(os.environ.get("MESH_DEVICE"), (1, 8))


def _tracy_signpost_available() -> bool:
    try:
        from tracy import signpost  # noqa: F401

        return True
    except ImportError:
        return False


def _run_prefill_decode_step(tt_model, mesh_device, *, input_ids_prefill, input_ids_decode, decode_pos: int):
    tt_model(input_ids_to_tt(input_ids_prefill, mesh_device), mode="prefill", start_pos=0)
    current_pos_tt = current_pos_to_tt(torch.tensor([decode_pos], dtype=torch.long), mesh_device)
    return tt_model(
        input_ids_to_tt(input_ids_decode, mesh_device),
        mode="decode",
        current_pos=current_pos_tt,
    )


@torch.no_grad()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
        }
    ],
    indirect=True,
)
@pytest.mark.timeout(0)
def test_profile_single_layer_prefill_decode(mesh_device, batch_size):
    """Prefill 128 + decode 1 on a 1-layer ``TtMinistral3Model`` (profile target for device perf)."""
    fixtures = setup_devstral_ministral3_partial_one_layer(
        mesh_device,
        max_seq_len=max(DEVSTRAL2_TEST_MAX_SEQ_LEN, DECODE_POS + 1),
    )
    text_cfg = fixtures.text_cfg
    tt_model = fixtures.tt_model

    torch.manual_seed(42)
    gen = torch.Generator(device="cpu").manual_seed(42)
    total_len = PREFILL_SEQ_LEN + 1
    input_ids_full = torch.randint(0, text_cfg.vocab_size, (batch_size, total_len), dtype=torch.long, generator=gen)
    input_ids_prefill = input_ids_full[:, :PREFILL_SEQ_LEN]
    input_ids_decode = input_ids_full[:, PREFILL_SEQ_LEN : PREFILL_SEQ_LEN + 1]

    use_signpost = _tracy_signpost_available()
    if use_signpost:
        from tracy import signpost
    else:
        logger.info("tracy.signpost unavailable; running profile workload without signpost markers.")

    for _ in range(NUM_WARMUP_ITERS):
        out = _run_prefill_decode_step(
            tt_model,
            mesh_device,
            input_ids_prefill=input_ids_prefill,
            input_ids_decode=input_ids_decode,
            decode_pos=DECODE_POS,
        )
        out.deallocate(True)
        ttnn.synchronize_device(mesh_device)

    if use_signpost:
        signpost("start")

    tt_out = _run_prefill_decode_step(
        tt_model,
        mesh_device,
        input_ids_prefill=input_ids_prefill,
        input_ids_decode=input_ids_decode,
        decode_pos=DECODE_POS,
    )
    ttnn.synchronize_device(mesh_device)

    if use_signpost:
        signpost("stop")

    tt_out.deallocate(True)
    logger.info(
        f"Profile workload complete: prefill_seq_len={PREFILL_SEQ_LEN}, decode_pos={DECODE_POS}, "
        f"signposts={'on' if use_signpost else 'off'}"
    )
