# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mesh chronology summary slots under trace replay; separate from module-scoped device tests.

Every head and tail slot is checked: a slot with a valid segment against the segment oracle, every other slot as the
exact identity (I, 0) written with init.
"""

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import run_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    device_protocol,
    host_protocol,
    pack_summary,
    summary_oracle,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_bit_identical

pytestmark = run_for_blackhole()

_CHUNKS_PER_GROUP = 4


def _scalar(device: ttnn.MeshDevice, value: int) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )


def _rank(mesh_device: ttnn.MeshDevice, shard_index: int, axis: int) -> int:
    columns = tuple(mesh_device.shape)[1]
    return shard_index // columns if axis == 0 else shard_index % columns


def _assert_slots(
    host: tuple[torch.Tensor, ...],
    summaries: torch.Tensor,
    groups: int,
    segments: tuple[tuple[int, int], tuple[int, int]],
    context: str,
    oracles: dict[tuple[int, int, int], torch.Tensor],
) -> None:
    """``segments`` holds this rank's (begin, end) local chunk ranges of its head and tail segments. ``oracles``
    caches each (folded head, begin, end) segment's expected summary across calls on the same ``host``."""
    dim = host[1].shape[-1]
    identity = pack_summary(torch.eye(dim), torch.zeros(dim, dim)).bfloat16()
    for folded_head in range(summaries.shape[1]):
        group = folded_head % groups
        group_begin = group * _CHUNKS_PER_GROUP
        for slot, (begin, end) in enumerate(segments):
            begin, end = max(begin, group_begin) - group_begin, min(end, group_begin + _CHUNKS_PER_GROUP) - group_begin
            name = f"{context} group={group} slot={slot}"
            actual = summaries[slot, folded_head]
            if begin >= end:
                assert_bit_identical(identity, actual.bfloat16(), name=f"{name} identity")
                continue
            key = (folded_head, begin, end)
            if key not in oracles:
                segment = tuple(tensor[folded_head : folded_head + 1, begin:end] for tensor in host)
                oracles[key] = pack_summary(*summary_oracle(segment))[0].bfloat16().float()
            assert_accurate(
                oracles[key],
                actual.float(),
                name=name,
                pcc_threshold=0.999,
                nrmse_threshold=0.006,
            )


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize(
    "groups,dim",
    [(1, 32), (1, 128), (4, 32)],
    ids=["g1-one-value-block", "g1-value-blocks", "g4-one-value-block"],
)
@pytest.mark.parametrize("axis", [0, 1])
def test_chronology_summaries_single_capture(
    mesh_device: ttnn.MeshDevice, groups: int, dim: int, device_params: dict, axis: int
) -> None:
    """One trace replays every 32-aligned actual_start over the mesh span and one chunk past it, moving the split
    through every rank and every in-group offset."""
    device = mesh_device
    chunks = groups * _CHUNKS_PER_GROUP
    host = host_protocol(2 * groups, _CHUNKS_PER_GROUP, dim, dim, seed=981)
    inputs = device_protocol(host, device)
    oracles = {}
    actual_start = _scalar(device, 0)

    def run():
        return ttnn.experimental.kda.summarize_chunk_recurrence(
            *inputs, groups_per_head=groups, actual_start=actual_start, sequence_parallel_axis=axis, tail=True
        )

    for _ in range(2):
        ttnn.deallocate(run())
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    summaries = run()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        assert summaries.dtype == ttnn.bfloat16
        assert tuple(summaries.shape) == (2, 2 * groups, dim, 2 * dim)
        partitions = tuple(mesh_device.shape)[axis]
        for start_chunk in range(partitions * chunks + 1):
            source = _scalar(device, 32 * start_chunk)
            ttnn.copy(source, actual_start)
            ttnn.deallocate(source)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            offset = start_chunk % chunks
            first_rank = (start_chunk // chunks) % partitions
            for shard_index, shard in enumerate(ttnn.get_device_tensors(summaries)):
                split = offset != 0 and _rank(mesh_device, shard_index, axis) == first_rank
                # A split rank's first chunks - offset are the chronological head; the rest is the wrapped tail.
                wrap = chunks - offset if split else chunks
                _assert_slots(
                    host,
                    ttnn.to_torch(shard).float(),
                    groups,
                    ((0, wrap), (wrap, chunks if split else wrap)),
                    f"G={groups} K={dim} axis={axis} start_chunk={start_chunk} device={shard_index}",
                    oracles,
                )
    finally:
        ttnn.release_trace(device, trace)


# Hand-derived segments for clipped intervals on 4 ranks (mesh axis 1) of 4 * G chunks, keyed by G. Each case maps a
# rank to its (head, tail) local chunk ranges; unlisted ranks hold no valid chunk.
_FULL = ((0, 4), (0, 0))
_CLIPPED = {
    4: {
        # Split at rank 1 chunk 6 (10 head chunks), interval of 5 chunks: only the head's first 5 chunks are valid.
        "end-inside-head": (16 + 6, 5, {1: ((0, 5), (0, 0))}),
        # Same split, 61 chunks: ranks 2, 3, 0 full; rank 1 keeps its 10 head chunks and the first 3 tail chunks.
        "end-inside-tail": (
            16 + 6,
            61,
            {1: ((0, 10), (10, 13)), 2: ((0, 16), (0, 0)), 3: ((0, 16), (0, 0)), 0: ((0, 16), (0, 0))},
        ),
        # Unsplit start, 20 chunks: rank 0 full, rank 1 four chunks, ranks 2 and 3 empty.
        "empty-ranks": (0, 20, {0: ((0, 16), (0, 0)), 1: ((0, 4), (0, 0))}),
    },
    # One group per head: the production layout the distributed prefix gathers, where every rank's head slot is read.
    1: {
        # Split at rank 1 chunk 1 (3 head chunks), interval of 2 chunks.
        "end-inside-head": (4 + 1, 2, {1: ((0, 2), (0, 0))}),
        # Split at rank 1 chunk 2 (2 head chunks), 15 chunks: ranks 2, 3, 0 full; rank 1 keeps 1 of its 2 tail chunks.
        "end-inside-tail": (4 + 2, 15, {1: ((0, 2), (2, 3)), 2: _FULL, 3: _FULL, 0: _FULL}),
        # Unsplit start, 6 chunks: rank 0 full, rank 1 two chunks, ranks 2 and 3 empty.
        "empty-ranks": (0, 6, {0: _FULL, 1: ((0, 2), (0, 0))}),
    },
}


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("groups,dim", [(4, 32), (1, 128)], ids=["g4-one-value-block", "g1-value-blocks"])
def test_chronology_summaries_clipped_intervals_single_capture(
    mesh_device: ttnn.MeshDevice, device_params: dict, groups: int, dim: int
):
    """actual_end empties whole groups and ranks; their slots must hold the identity, and init=False must leave every
    defined slot unchanged."""
    device, axis = mesh_device, 1
    host = host_protocol(2 * groups, _CHUNKS_PER_GROUP, dim, dim, seed=983)
    inputs = device_protocol(host, device)
    oracles = {}
    span_chunks = 4 * groups * _CHUNKS_PER_GROUP
    actual_start, actual_end = _scalar(device, 0), _scalar(device, 32 * span_chunks)

    def run(init: bool):
        return ttnn.experimental.kda.summarize_chunk_recurrence(
            *inputs,
            groups_per_head=groups,
            actual_start=actual_start,
            actual_end=actual_end,
            sequence_parallel_axis=axis,
            tail=True,
            init=init,
        )

    for _ in range(2):
        ttnn.deallocate(run(True))
        ttnn.deallocate(run(False))
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    filled, unfilled = run(True), run(False)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for case, (start_chunk, length, ranks) in _CLIPPED[groups].items():
            for tensor, value in ((actual_start, start_chunk), (actual_end, start_chunk + length)):
                source = _scalar(device, 32 * value)
                ttnn.copy(source, tensor)
                ttnn.deallocate(source)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            for shard_index, (filled_shard, unfilled_shard) in enumerate(
                zip(ttnn.get_device_tensors(filled), ttnn.get_device_tensors(unfilled), strict=True)
            ):
                segments = ranks.get(_rank(mesh_device, shard_index, axis), ((0, 0), (0, 0)))
                context = f"{case} device={shard_index}"
                filled_host, unfilled_host = ttnn.to_torch(filled_shard), ttnn.to_torch(unfilled_shard)
                _assert_slots(host, filled_host.float(), groups, segments, context, oracles)
                for folded_head in range(2 * groups):
                    group_begin = (folded_head % groups) * _CHUNKS_PER_GROUP
                    for slot, (begin, end) in enumerate(segments):
                        if max(begin, group_begin) < min(end, group_begin + _CHUNKS_PER_GROUP):
                            assert_bit_identical(
                                filled_host[slot, folded_head],
                                unfilled_host[slot, folded_head],
                                name=f"{context} init=False defined slot={slot} head={folded_head}",
                            )
    finally:
        ttnn.release_trace(device, trace)
