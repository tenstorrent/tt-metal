# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prepared terms and direct scan match physically trimmed native execution."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = run_for_blackhole()


@pytest.mark.parametrize("device_params", [{"trace_region_size": 4_000_000}], indirect=True)
def test_padding_preparation_and_scan(device, device_params):
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
