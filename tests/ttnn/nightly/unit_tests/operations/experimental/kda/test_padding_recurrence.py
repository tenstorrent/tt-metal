# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prepared terms and direct scan match physically trimmed native execution."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical

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


_PREPARED_TERMS = ("v_beta", "kd", "q_decay", "intra", "k_dec_t", "final_decay", "t_inv")
# Terms whose rows are query rows. A padded query row only reaches its own row of
# these terms, and padded output rows are unspecified, so only valid rows compare.
_QUERY_ROW_TERMS = frozenset({"q_decay", "intra", "output"})


def test_padding_unaligned_end_ignores_padding_values(device):
    """With an unaligned end, the value in the padded rows does not change the result.

    The reference runs the same operations at the 32-aligned ceiling on host inputs
    whose padded G, beta, K, and V rows are zero: identity recurrence steps, so the
    aligned path computes exactly the result of stopping at the end. Observed runs
    fill the padded rows of every input, Q included, with each value below and must
    match it bit for bit. NaN and inf expose any arithmetic that reads a padded row
    instead of replacing it, including masking by multiplication (inf * 0 = NaN); a
    large finite value exposes blends that keep a dependence on the discarded value;
    random values model stale buffer contents.
    """
    generator = torch.Generator().manual_seed(4211)
    sequence, heads, width = 256, 2, 32
    q, k, v = [torch.randn(1, sequence, heads * width, generator=generator).bfloat16() for _ in range(3)]
    gate = (-0.02 * torch.rand(q.shape, generator=generator)).bfloat16()
    beta = torch.sigmoid(torch.randn(heads, sequence // 32, 32, 1, generator=generator))
    initial = 0.02 * torch.randn(heads, width, width, generator=generator)
    fills = {"zero": 0.0, "nan": float("nan"), "inf": float("inf"), "large": 1e30, "random": None}

    def upload(value, dtype):
        return ttnn.from_torch(value, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    def scalar(value):
        return ttnn.from_torch(
            torch.tensor([value], dtype=torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    def fill_rows(tensor, rows, fill):
        tensor[:, rows:] = torch.randn(tensor[:, rows:].shape, generator=generator) if fill is None else fill

    def with_padding(length, fill, *, query):
        """Fill rows at or after ``length`` in G, beta, K, V and, when ``query``, Q."""
        padded = [t.clone() for t in (q, k, v, gate)]
        for tensor in padded if query else padded[1:]:
            fill_rows(tensor, length, fill)
        padded_beta = beta.clone().reshape(heads, sequence, 1)
        fill_rows(padded_beta, length, fill)
        return padded, padded_beta.reshape(beta.shape)

    inputs = [upload(t, ttnn.bfloat16) for t in (q, k, v, gate)]
    beta_tt, initial_tt = upload(beta, ttnn.float32), upload(initial, ttnn.float32)
    start, end = scalar(0), scalar(sequence)

    def run(flat, beta_input, end_bound):
        """Return named results: prepared terms, scan output and state, head summary.

        start=0 on one device has no separated tail, so the tail summary slots are
        unspecified and are released here.
        """
        prepared = ttnn.experimental.kda.prepare_chunk_recurrence(
            *flat, beta_input, heads, actual_start=start, actual_end=end_bound
        )
        output, state = ttnn.experimental.kda.recurrent_chunk_scan(
            *prepared, initial_tt, actual_start=start, actual_end=end_bound, tail_entry_states=initial_tt
        )
        head_a, head_b, tail_a, tail_b = ttnn.experimental.kda.summarize_chunk_recurrence(
            *prepared, actual_start=start, actual_end=end_bound
        )
        ttnn.deallocate(tail_a)
        ttnn.deallocate(tail_b)
        return {
            **dict(zip(_PREPARED_TERMS, prepared, strict=True)),
            "output": output,
            "state": state,
            "head_a": head_a,
            "head_b": head_b,
        }

    def comparable(name, tensor, length):
        """The defined part of a result: valid chunks, and valid rows of query-row terms."""
        host = ttnn.to_torch(tensor)
        if name in ("state", "head_a", "head_b"):
            return host
        host = host[:, : -(-length // 32)]
        if name in _QUERY_ROW_TERMS:
            host = host.reshape(host.shape[0], -1, host.shape[-1])[:, :length]
        return host

    for _ in range(2):
        for tensor in run(inputs, beta_tt, end).values():
            ttnn.deallocate(tensor)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    observed = run(inputs, beta_tt, end)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        # One row, one short of a tile, one past a tile, mid-sequence, one short of
        # capacity, then aligned lengths to show the same capture still serves them.
        for length in (1, 2, 31, 33, 97, 255, 256, 64):
            reference_inputs, reference_beta = with_padding(length, 0.0, query=False)
            reference_tt = [upload(t, ttnn.bfloat16) for t in reference_inputs]
            reference_beta_tt = upload(reference_beta, ttnn.float32)
            ceiling = scalar(-(-length // 32) * 32)
            reference = run(reference_tt, reference_beta_tt, ceiling)
            expected = {name: comparable(name, tensor, length) for name, tensor in reference.items()}
            for tensor in (*reference_tt, reference_beta_tt, ceiling, *reference.values()):
                ttnn.deallocate(tensor)

            for fill_name, fill in fills.items():
                padded, padded_beta = with_padding(length, fill, query=True)
                for tensor, host in zip((*inputs, beta_tt), (*padded, padded_beta), strict=True):
                    source = upload(host, tensor.dtype)
                    ttnn.copy(source, tensor)
                    ttnn.deallocate(source)
                source = scalar(length)
                ttnn.copy(source, end)
                ttnn.deallocate(source)
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                for name, tensor in observed.items():
                    assert_bit_identical(
                        expected[name],
                        comparable(name, tensor, length),
                        name=f"length={length} fill={fill_name} {name}",
                    )
    finally:
        ttnn.release_trace(device, trace)
        for tensor in (*observed.values(), *inputs, beta_tt, initial_tt, start, end):
            ttnn.deallocate(tensor)
