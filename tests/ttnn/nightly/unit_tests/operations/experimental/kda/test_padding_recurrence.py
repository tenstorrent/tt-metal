# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prepared terms and direct scan match physically trimmed native execution."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole()


def test_padding_preparation_and_scan(device):
    generator = torch.Generator().manual_seed(183)
    sequence, heads, width = 256, 2, 32
    q, k, v = [torch.randn(1, sequence, heads * width, generator=generator).bfloat16() for _ in range(3)]
    gate = (-0.02 * torch.rand(q.shape, generator=generator)).bfloat16()
    beta = torch.sigmoid(torch.randn(heads, sequence // 32, 32, 1, generator=generator))
    initial = 0.02 * torch.randn(heads, width, width, generator=generator)

    def upload(value, dtype):
        return ttnn.from_torch(value, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    inputs = [upload(t, ttnn.bfloat16) for t in (q, k, v, gate)]
    beta_tt, initial_tt = upload(beta, ttnn.float32), upload(initial, ttnn.float32)

    def scalar(value):
        return ttnn.from_torch(
            torch.tensor([value], dtype=torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    start, end = scalar(0), scalar(sequence)

    def run(start_bound=start, end_bound=end):
        prepared = ttnn.experimental.kda.prepare_chunk_recurrence(
            *inputs, beta_tt, heads, actual_start=start_bound, actual_end=end_bound
        )
        output, state = ttnn.experimental.kda.recurrent_chunk_scan(
            *prepared, initial_tt, actual_start=start_bound, actual_end=end_bound, tail_entry_states=initial_tt
        )
        summary = ttnn.experimental.kda.summarize_chunk_recurrence(
            *prepared, actual_start=start_bound, actual_end=end_bound
        )
        return (*prepared, output, state, *summary)

    for _ in range(2):
        for tensor in run():
            ttnn.deallocate(tensor)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    outputs = run()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for length in (256, 32, 96, 160, 32, 256):
            source = scalar(length)
            ttnn.copy(source, end)
            ttnn.deallocate(source)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            trimmed = [upload(t[:, :length], ttnn.bfloat16) for t in (q, k, v, gate)]
            trimmed_beta = upload(beta[:, : length // 32], ttnn.float32)
            expected_terms = ttnn.experimental.kda.prepare_chunk_recurrence(
                *trimmed, trimmed_beta, heads, actual_start=start
            )
            expected_output, expected_state = ttnn.experimental.kda.recurrent_chunk_scan(
                *expected_terms, initial_tt, actual_start=start, tail_entry_states=initial_tt
            )
            expected_summary = ttnn.experimental.kda.summarize_chunk_recurrence(*expected_terms, actual_start=start)
            for observed, expected in zip(outputs[:8], (*expected_terms, expected_output), strict=True):
                torch.testing.assert_close(
                    ttnn.to_torch(observed)[:, : length // 32], ttnn.to_torch(expected), rtol=0, atol=0
                )
            torch.testing.assert_close(ttnn.to_torch(outputs[8]), ttnn.to_torch(expected_state), rtol=0, atol=0)
            # Dynamic summaries use the PR's BF16 transport boundary.
            for observed, expected in zip(outputs[9:11], expected_summary[:2], strict=True):
                torch.testing.assert_close(ttnn.to_torch(observed), ttnn.to_torch(expected).bfloat16(), rtol=0, atol=0)
            # Rebind cached programs to fresh scalar addresses while the captured
            # scalars describe a different length. Stale addresses cannot pass.
            source = scalar(sequence)
            ttnn.copy(source, end)
            ttnn.deallocate(source)
            fresh_start, fresh_end = scalar(1024), scalar(1024 + length)
            rebound = run(fresh_start, fresh_end)
            for observed, expected in zip(rebound[:8], (*expected_terms, expected_output), strict=True):
                torch.testing.assert_close(
                    ttnn.to_torch(observed)[:, : length // 32], ttnn.to_torch(expected), rtol=0, atol=0
                )
            torch.testing.assert_close(ttnn.to_torch(rebound[8]), ttnn.to_torch(expected_state), rtol=0, atol=0)
            for observed, expected in zip(rebound[9:11], expected_summary[:2], strict=True):
                torch.testing.assert_close(ttnn.to_torch(observed), ttnn.to_torch(expected).bfloat16(), rtol=0, atol=0)
            for tensor in (*rebound, fresh_start, fresh_end):
                ttnn.deallocate(tensor)
            for tensor in (*trimmed, trimmed_beta, *expected_terms, expected_output, expected_state, *expected_summary):
                ttnn.deallocate(tensor)
    finally:
        ttnn.release_trace(device, trace)
        for tensor in (*outputs, *inputs, beta_tt, initial_tt, start, end):
            ttnn.deallocate(tensor)


@pytest.mark.parametrize("width", [32, 128])
def test_padding_scan_preserves_valid_group_states(device, width):
    """Shortening one trace preserves group indices and the fixed-slot final carry."""
    from tests.ttnn.nightly.unit_tests.operations.experimental.kda.test_prepare_chunk_recurrence import (
        _device_inputs,
        _host_inputs,
        _run,
    )
    from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

    heads, groups, chunks_per_group = 2, 4, 2
    inputs = _device_inputs(_host_inputs(heads, groups * chunks_per_group, width, width, seed=1912), device)
    prepared = _run(inputs, heads)
    prepared_host = [ttnn.to_torch(t) for t in prepared]
    grouped = [ttnn.reshape(t, (heads * groups, chunks_per_group, *tuple(t.shape)[2:])) for t in prepared]
    initial_host = torch.randn(heads, groups, width, width, generator=torch.Generator().manual_seed(219)) * 0.1

    def upload(host, dtype=ttnn.float32):
        return ttnn.from_torch(host.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    initial = upload(initial_host.reshape(heads * groups, width, width))
    tail = upload(initial_host[:, 0])
    start, end = make_actual_start(device, 0), make_actual_start(device, 256)

    def run():
        return ttnn.experimental.kda.recurrent_chunk_scan(
            *grouped, initial, groups_per_head=groups, actual_start=start, actual_end=end, tail_entry_states=tail
        )

    for _ in range(2):
        for tensor in run():
            ttnn.deallocate(tensor)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    output, states = run()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for length in (256, 32, 96, 160, 224, 128, 32, 256):
            source = make_actual_start(device, length)
            ttnn.copy(source, end)
            ttnn.deallocate(source)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            observed = ttnn.to_torch(states).reshape(heads, groups, width, width)
            active_groups = (length + 63) // 64
            for group in range(active_groups):
                begin = group * chunks_per_group
                count = min(chunks_per_group, length // 32 - begin)
                terms = [
                    upload(host[:, begin : begin + count], tensor.dtype)
                    for host, tensor in zip(prepared_host, prepared, strict=True)
                ]
                seed = upload(initial_host[:, group])
                expected_output, expected_state = ttnn.experimental.kda.recurrent_chunk_scan(
                    *terms, seed, actual_start=start, tail_entry_states=seed
                )
                expected = ttnn.to_torch(expected_state)
                torch.testing.assert_close(observed[:, group], expected, rtol=0, atol=0)
                if group == active_groups - 1:
                    torch.testing.assert_close(observed[:, -1], expected, rtol=0, atol=0)
                for tensor in (*terms, seed, expected_output, expected_state):
                    ttnn.deallocate(tensor)
    finally:
        ttnn.release_trace(device, trace)
        for tensor in (output, states, initial, tail, start, end, *grouped, *inputs):
            ttnn.deallocate(tensor)
