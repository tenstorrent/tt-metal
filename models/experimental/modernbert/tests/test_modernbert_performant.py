# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Trace and two command queues, separately and together."""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.modernbert.common import load_config
from models.experimental.modernbert.runner.performant_runner import ModernBertPerformantRunner

WARMUP_ITERS = 3
TIMED_ITERS = 20


def _assert_responds_to_new_input(runner, device, batch_size, seq_len):
    """A trace bakes in buffer addresses, so a misrouted input does not fail
    loudly - it replays the previous contents and scores a perfect PCC against the
    last iteration. Feed a different input and require the output to move."""
    baseline = ttnn.to_torch(runner.runner_infra.output_tensor).clone()
    config = load_config()
    torch.manual_seed(1)
    other_ids = torch.randint(low=1, high=config.vocab_size - 1, size=(batch_size, seq_len), dtype=torch.int32)
    runner.run(input_ids=other_ids)
    ttnn.synchronize_device(device)
    changed = ttnn.to_torch(runner.runner_infra.output_tensor)
    drift = (changed - baseline).abs().max().item()
    assert drift > 1e-3, f"output did not respond to a new input (max delta {drift})"
    return drift


def _steady_state(runner, device):
    for _ in range(WARMUP_ITERS):
        runner.run()
    ttnn.synchronize_device(device)
    t0 = time.time()
    for _ in range(TIMED_ITERS):
        runner.run()
    ttnn.synchronize_device(device)
    return (time.time() - t0) / TIMED_ITERS


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 33554432}],
    indirect=True,
)
@pytest.mark.parametrize("seq_len, batch_size", [(256, 1)])
def test_modernbert_trace_only(device, seq_len, batch_size):
    """Trace on a single command queue. No staging buffer and no events, so a
    failure here is trace capture/replay and nothing else."""
    runner = ModernBertPerformantRunner(device, device_batch_size=batch_size, sequence_length=seq_len, mode="trace")
    try:
        runner.setup()
        runner.run()
        ttnn.synchronize_device(device)
        p = runner.validate()
        drift = _assert_responds_to_new_input(runner, device, batch_size, seq_len)
        runner.run()
        ttnn.synchronize_device(device)
        per_iter = _steady_state(runner, device)
    finally:
        runner.release()

    logger.info(
        f"ModernBERT trace-only batch={batch_size} seq={seq_len}: "
        f"{per_iter * 1000:.2f} ms/inference, PCC={p:.8f}, input-response delta={drift:.4f}"
    )
    assert per_iter > 0


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("seq_len, batch_size", [(256, 1)])
def test_modernbert_2cq_only(device, seq_len, batch_size):
    """Two command queues with no trace: cq1 stages the next input while cq0 runs
    the model. A failure here is the event handshake, not trace."""
    runner = ModernBertPerformantRunner(device, device_batch_size=batch_size, sequence_length=seq_len, mode="2cq")
    try:
        runner.setup()
        runner.run()
        ttnn.synchronize_device(device)
        p = runner.validate()
        drift = _assert_responds_to_new_input(runner, device, batch_size, seq_len)
        runner.run()
        ttnn.synchronize_device(device)
        per_iter = _steady_state(runner, device)
    finally:
        runner.release()

    logger.info(
        f"ModernBERT 2cq-only batch={batch_size} seq={seq_len}: "
        f"{per_iter * 1000:.2f} ms/inference, PCC={p:.8f}, input-response delta={drift:.4f}"
    )
    assert per_iter > 0


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 33554432, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("seq_len, batch_size", [(256, 1), (256, 8)])
def test_modernbert_performant(device, seq_len, batch_size):
    runner = ModernBertPerformantRunner(device, device_batch_size=batch_size, sequence_length=seq_len)
    try:
        runner.setup()

        # 1. correctness of the traced path
        runner.run()
        ttnn.synchronize_device(device)
        p = runner.validate()

        # 2. the trace reads what cq1 last wrote, not a stale buffer
        drift = _assert_responds_to_new_input(runner, device, batch_size, seq_len)

        # restore the input the PCC gate was measured against
        runner.run()
        ttnn.synchronize_device(device)

        # 3. steady state, traced
        per_iter = _steady_state(runner, device)

        # 4. the same loop without trace, uploading an input every iteration too.
        # test_modernbert_perf.py writes one input tensor and reuses it, so it never
        # pays for an upload; comparing trace against that measures the upload as
        # much as the dispatch. Both loops here go through the identical cq1 path,
        # so the difference is trace and nothing else.
        for _ in range(WARMUP_ITERS):
            runner._stage(runner.tt_inputs_host)
            ttnn.deallocate(runner.runner_infra.model(runner.tt_input))
        ttnn.synchronize_device(device)

        t0 = time.time()
        for _ in range(TIMED_ITERS):
            runner._stage(runner.tt_inputs_host)
            ttnn.deallocate(runner.runner_infra.model(runner.tt_input))
        ttnn.synchronize_device(device)
        untraced_per_iter = (time.time() - t0) / TIMED_ITERS
    finally:
        runner.release()

    logger.info(
        f"ModernBERT trace+2cq batch={batch_size} seq={seq_len}: "
        f"{per_iter * 1000:.2f} ms/inference, "
        f"{batch_size / per_iter:.1f} sequences/s, "
        f"{batch_size * seq_len / per_iter:,.0f} tokens/s, "
        f"PCC={p:.8f}, input-response delta={drift:.4f}"
    )
    logger.info(
        f"ModernBERT same loop untraced batch={batch_size} seq={seq_len}: "
        f"{untraced_per_iter * 1000:.2f} ms/inference; "
        f"trace is {(untraced_per_iter - per_iter) / untraced_per_iter * 100:+.1f}%"
    )
    assert per_iter > 0
