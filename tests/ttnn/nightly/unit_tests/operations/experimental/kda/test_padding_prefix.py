# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Active-only group composition, including empty ranks, with known affine pairs."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = run_for_blackhole()


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("width", [32, 128])
def test_padding_prefix_empty_and_partial_groups(mesh_device, width):
    # Two heads, four 64-row groups each. Every active summary is (I, (g+1)I).
    eye = torch.eye(width)
    a = eye.repeat(8, 1, 1).bfloat16()
    b = torch.stack([(g + 1) * eye for _ in range(2) for g in range(4)]).bfloat16()
    initial = torch.stack([10 * eye, 20 * eye])

    def upload(tensor, dtype):
        return ttnn.from_torch(tensor, device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    a_tt, b_tt, initial_tt = upload(a, ttnn.bfloat16), upload(b, ttnn.bfloat16), upload(initial, ttnn.float32)
    start = make_actual_start(mesh_device, 0)
    end = make_actual_start(mesh_device, 512)

    def run():
        reduced = ttnn.experimental.kda.reduce_affine_transforms(
            a_tt, b_tt, 4, actual_start=start, actual_end=end, local_rows=256, sequence_parallel_axis=0
        )
        entries = ttnn.experimental.kda.affine_exclusive_scan(
            a_tt,
            b_tt,
            initial_tt,
            4,
            actual_start=start,
            actual_end=end,
            local_rows=256,
            sequence_parallel_axis=0,
            tail_a=a_tt,
            tail_b=b_tt,
            tail_entry_states=initial_tt,
        )
        return (*reduced, entries)

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
                torch.testing.assert_close(shards[0][device_index], eye.repeat(2, 1, 1), rtol=0, atol=0)
                torch.testing.assert_close(shards[1][device_index], (total * eye).repeat(2, 1, 1), rtol=0, atol=0)
                for head in range(2):
                    for group in range(active):
                        expected = initial[head] + (0, 1, 3, 6)[group] * eye
                        torch.testing.assert_close(shards[2][device_index][head * 4 + group], expected, rtol=0, atol=0)
    finally:
        ttnn.release_trace(mesh_device, trace)
        for tensor in (*outputs, a_tt, b_tt, initial_tt, start, end):
            ttnn.deallocate(tensor)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "start,end",
    [(0, None), (32, None), (64, None), (0, 64), (32, 64), (64, 96)],
    ids=["full", "split", "boundary", "rank1-empty", "rank0-only-split", "rank1-only"],
)
def test_packed_head_summary_matches_reduced_and_packed_pair(mesh_device, start, end):
    """The packed head summary must equal reduce_affine_transforms' pair packed as [A | B] in BF16, including the
    identity a rank without a head contributes."""
    from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
        device_protocol,
        host_protocol,
    )
    from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical

    batch_heads, num_chunks, dim = 4, 2, 32
    local_rows = num_chunks * 32
    protocol = device_protocol(host_protocol(batch_heads, num_chunks, dim, dim, seed=1907), mesh_device)
    start_tt = make_actual_start(mesh_device, start)
    end_tt = None if end is None else make_actual_start(mesh_device, end)
    bounds = dict(actual_start=start_tt, actual_end=end_tt, sequence_parallel_axis=0)

    head_a, head_b, tail_a, tail_b = ttnn.experimental.kda.summarize_chunk_recurrence(*protocol, **bounds)
    reduced = ttnn.experimental.kda.reduce_affine_transforms(head_a, head_b, 1, local_rows=local_rows, **bounds)
    expected = ttnn.typecast(
        ttnn.concat(
            [
                ttnn.reshape(reduced[0], (1, batch_heads, dim, dim)),
                ttnn.reshape(reduced[1], (1, batch_heads, dim, dim)),
            ],
            dim=3,
        ),
        ttnn.bfloat16,
    )
    (packed,) = ttnn.experimental.kda.summarize_chunk_recurrence(
        *protocol, memory_config=ttnn.DRAM_MEMORY_CONFIG, packed_head=True, **bounds
    )
    assert tuple(packed.shape) == (1, batch_heads, dim, 2 * dim)
    for device_index, (want, got) in enumerate(
        zip(ttnn.get_device_tensors(expected), ttnn.get_device_tensors(packed), strict=True)
    ):
        assert_bit_identical(ttnn.to_torch(want), ttnn.to_torch(got), name=f"packed head on device {device_index}")
    for tensor in (*protocol, head_a, head_b, tail_a, tail_b, *reduced, expected, packed, start_tt):
        ttnn.deallocate(tensor)
    if end_tt is not None:
        ttnn.deallocate(end_tt)
