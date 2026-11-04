# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Active-only group composition, including empty ranks, with known affine pairs."""

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import run_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    CHUNK_SIZE,
    device_protocol,
    host_protocol,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical, make_actual_start

pytestmark = run_for_blackhole()


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("width", [32, 128])
def test_padding_prefix_empty_and_partial_groups(mesh_device, width):
    # Two heads, four 64-row groups each. Every active summary is (I, (g+1)I): (0, (g+1)I) in complement form.
    eye = torch.eye(width)
    e = torch.zeros(8, width, width).bfloat16()
    b = torch.stack([(g + 1) * eye for _ in range(2) for g in range(4)]).bfloat16()
    initial = torch.stack([10 * eye, 20 * eye])

    def upload(tensor, dtype):
        return ttnn.from_torch(tensor, device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    # Two SP ranks need a tail slot; an unsplit start never reads it, so it is NaN.
    head = torch.cat([e, b], dim=-1)
    summaries_tt = upload(torch.stack([head, torch.full_like(head, float("nan"))]), ttnn.bfloat16)
    initial_tt = upload(initial, ttnn.float32)
    start = make_actual_start(mesh_device, 0)
    end = make_actual_start(mesh_device, 512)

    def run():
        reduced = ttnn.experimental.kda.reduce_affine_transforms(
            summaries_tt, 4, actual_start=start, actual_end=end, local_rows=256, sequence_parallel_axis=0
        )
        entries = ttnn.experimental.kda.affine_exclusive_scan(
            summaries_tt,
            initial_tt,
            4,
            actual_start=start,
            actual_end=end,
            local_rows=256,
            sequence_parallel_axis=0,
            tail_entry_states=initial_tt,
        )
        return reduced, entries

    for _ in range(2):
        for tensor in run():
            ttnn.deallocate(tensor)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outputs = run()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        # Hand-verifiable active group counts, rank 0 then rank 1.
        for length, counts in [(512, (4, 4)), (32, (1, 0)), (96, (2, 0)), (256, (4, 0)), (288, (4, 1)), (32, (1, 0))]:
            source = make_actual_start(mesh_device, length)
            ttnn.copy(source, end)
            ttnn.deallocate(source)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            shards = [[ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)] for out in outputs]
            for device_index in range(8):
                active = counts[device_index // 4]
                # 0, 1, 1+2, 1+2+3, 1+2+3+4; no model formula copied here.
                total = (0, 1, 3, 6, 10)[active]
                # The composition of identity multipliers is exact, so E = A - I is exactly zero; an empty rank
                # publishes the identity map, (0, 0) in complement form, which is the same formula with total 0.
                packed = shards[0][device_index].reshape(2, width, 2 * width)
                torch.testing.assert_close(packed[..., :width], torch.zeros(2, width, width), rtol=0, atol=0)
                torch.testing.assert_close(packed[..., width:], (total * eye).repeat(2, 1, 1), rtol=0, atol=0)
                for head in range(2):
                    for group in range(active):
                        expected = initial[head] + (0, 1, 3, 6)[group] * eye
                        torch.testing.assert_close(shards[1][device_index][head * 4 + group], expected, rtol=0, atol=0)
    finally:
        ttnn.release_trace(mesh_device, trace)
        for tensor in (*outputs, summaries_tt, initial_tt, start, end):
            ttnn.deallocate(tensor)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("dim", [32, 128], ids=["one-value-block", "value-block-multicast"])
@pytest.mark.parametrize(
    "start,end",
    [(0, None), (32, None), (64, None), (0, 64), (32, 64), (64, 96)],
    ids=["full", "split", "boundary", "rank1-empty", "rank0-only-split", "rank1-only"],
)
def test_single_group_reduction_publishes_the_summary_unchanged(mesh_device, dim, start, end) -> None:
    """With one group per head, reduce_affine_transforms publishes the summary's complement-form [E | B] bit for bit,
    including the identity, (0, 0), of a rank without a valid chunk; the single-group SP path gathers the summary
    directly, so both paths hand the prefix the same transform."""
    batch_heads, num_chunks = 4, 2
    local_rows = num_chunks * CHUNK_SIZE
    protocol = device_protocol(host_protocol(batch_heads, num_chunks, dim, dim, seed=1907), mesh_device)
    start_tt = make_actual_start(mesh_device, start)
    end_tt = None if end is None else make_actual_start(mesh_device, end)
    bounds = dict(actual_start=start_tt, actual_end=end_tt, sequence_parallel_axis=0)
    outputs = []
    try:
        summary = ttnn.experimental.kda.summarize_chunk_recurrence(*protocol, **bounds)
        outputs.append(summary)
        reduced = ttnn.experimental.kda.reduce_affine_transforms(summary, 1, local_rows=local_rows, **bounds)
        outputs.append(reduced)
        assert tuple(summary.shape) == tuple(reduced.shape) == (1, batch_heads, dim, 2 * dim)
        for device_index, (want, got) in enumerate(
            zip(ttnn.get_device_tensors(summary), ttnn.get_device_tensors(reduced), strict=True)
        ):
            assert_bit_identical(
                ttnn.to_torch(want), ttnn.to_torch(got), name=f"single-group head on device {device_index}"
            )
    finally:
        for tensor in (*protocol, *outputs, start_tt, end_tt):
            if tensor is not None:
                ttnn.deallocate(tensor)
