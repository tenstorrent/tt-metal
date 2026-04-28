# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

try:
    from tracy import signpost
except ImportError:

    def signpost(*_args, **_kwargs):
        return None


MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 1
SEQ_LEN = 512


def _to_ttnn_ids(ids: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        ids.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def _make_synthetic_inputs(tokenizer):
    low = 100
    high = max(low + 1, min(tokenizer.vocab_size, 50000))
    input_ids = torch.randint(low, high, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
    return input_ids, attention_mask, token_type_ids


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_bge_m3_tracy_forward_only_bs1_seq512(mesh_device):
    """
    Forward-only BGE-M3 perf test for TT-NN Visualizer.

    Warmup happens before Tracy signposts.
    The signposted region measures only TT model forward passes and excludes
    torch<->tt conversion overhead.

    By default this runs a visualizer-safe forward loop (no trace capture).
    Optional trace replay can be enabled for non-visualizer runs with:
      BGE_M3_USE_TRACE_REPLAY=1

    Example:
      BGE_M3_TRACE_ITERS=1 python -m tracy -p -r -v -m pytest \
        models/demos/wormhole/bge_m3/tests/perf/test_tracy_forward_only_bs1_seq512.py::test_bge_m3_tracy_forward_only_bs1_seq512
    """
    default_iters = "1" if ttnn.CONFIG.enable_logging else "10"
    iterations = int(os.environ.get("BGE_M3_TRACE_ITERS", default_iters))
    use_trace_replay = _env_flag("BGE_M3_USE_TRACE_REPLAY")
    if use_trace_replay and ttnn.CONFIG.enable_logging:
        logger.warning("Disabling trace replay because TTNN logging is enabled (visualizer mode).")
        use_trace_replay = False
    if ttnn.CONFIG.enable_logging and iterations > 1:
        logger.warning(
            "TTNN logging is enabled and iterations > 1 may overflow profiler markers. "
            "Use BGE_M3_TRACE_ITERS=1 for visualizer report generation."
        )

    logger.info(
        f"Building BGE-M3 for batch={BATCH_SIZE}, seq_len={SEQ_LEN}, "
        f"iterations={iterations}, trace_replay={use_trace_replay}"
    )

    model_args, model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=BATCH_SIZE,
        max_seq_len=SEQ_LEN,
        dtype=ttnn.bfloat8_b,
        hf_model_name=MODEL_NAME,
    )
    tokenizer = model_args.tokenizer

    input_ids, attention_mask, token_type_ids = _make_synthetic_inputs(tokenizer)
    tt_input_ids = _to_ttnn_ids(input_ids, mesh_device=mesh_device)
    tt_attention_mask = _to_ttnn_ids(attention_mask, mesh_device=mesh_device)
    tt_token_type_ids = _to_ttnn_ids(token_type_ids, mesh_device=mesh_device)

    trace_captured = False
    tt_output = None
    try:
        # Compile/warmup pass (outside measured signpost window).
        tt_output = model(
            input_ids=tt_input_ids,
            attention_mask=tt_attention_mask,
            token_type_ids=tt_token_type_ids,
            position_ids=None,
        )
        ttnn.synchronize_device(mesh_device)

        if use_trace_replay:
            # Capture a forward trace after warmup to replay during the profiled window.
            tt_output = model.capture_trace(
                input_ids=tt_input_ids,
                attention_mask=tt_attention_mask,
                token_type_ids=tt_token_type_ids,
                position_ids=None,
                mesh_device=mesh_device,
                cq_id=0,
            )
            trace_captured = True

            signpost("start")
            for _ in range(iterations):
                tt_output = model.execute_trace(blocking=False, synchronize=False)
            ttnn.synchronize_device(mesh_device)
            signpost("stop")
        else:
            signpost("start")
            for _ in range(iterations):
                tt_output = model(
                    input_ids=tt_input_ids,
                    attention_mask=tt_attention_mask,
                    token_type_ids=tt_token_type_ids,
                    position_ids=None,
                )
            ttnn.synchronize_device(mesh_device)
            signpost("stop")
    finally:
        if trace_captured:
            model.release_trace()

    assert tt_output is not None
