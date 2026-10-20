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


def _bf16_spacing(values: torch.Tensor) -> torch.Tensor:
    """BF16 spacing at each value's binade (8 significant bits), an upper bound on one rounding's error times two."""
    return torch.exp2(torch.floor(torch.log2(values.abs().clamp_min(torch.finfo(torch.bfloat16).tiny))) - 7)


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
@pytest.mark.parametrize("dim", [32, 128], ids=["one-value-block", "value-block-multicast"])
@pytest.mark.parametrize(
    "start,end",
    [(0, None), (32, None), (64, None), (0, 64), (32, 64), (64, 96)],
    ids=["full", "split", "boundary", "rank1-empty", "rank0-only-split", "rank1-only"],
)
def test_packed_head_summary_matches_reduced_and_packed_pair(mesh_device, dim, start, end) -> None:
    """The packed head must carry reduce_affine_transforms' pair in complement form, [A - I | B] in BF16: B and the
    off-diagonal exactly, the diagonal within one rounding of each path, and (0, 0) for a rank without a valid chunk
    (its A is the identity)."""
    batch_heads, num_chunks = 4, 2
    local_rows = num_chunks * CHUNK_SIZE
    protocol = device_protocol(host_protocol(batch_heads, num_chunks, dim, dim, seed=1907), mesh_device)
    start_tt = make_actual_start(mesh_device, start)
    end_tt = None if end is None else make_actual_start(mesh_device, end)
    bounds = dict(actual_start=start_tt, actual_end=end_tt, sequence_parallel_axis=0)
    outputs = []
    try:
        outputs += ttnn.experimental.kda.summarize_chunk_recurrence(*protocol, **bounds)
        head_a, head_b = outputs[:2]
        reduced = ttnn.experimental.kda.reduce_affine_transforms(head_a, head_b, 1, local_rows=local_rows, **bounds)
        outputs += reduced
        (packed,) = ttnn.experimental.kda.summarize_chunk_recurrence(
            *protocol, memory_config=ttnn.DRAM_MEMORY_CONFIG, packed_head=True, **bounds
        )
        outputs.append(packed)
        assert tuple(packed.shape) == (1, batch_heads, dim, 2 * dim)
        eye = torch.eye(dim)
        off_diagonal = ~eye.bool()
        for device_index, shards in enumerate(
            zip(*(ttnn.get_device_tensors(tensor) for tensor in (*reduced, packed)), strict=True)
        ):
            transform_a, transform_b, got = (ttnn.to_torch(shard).reshape(batch_heads, dim, -1) for shard in shards)
            complement, got_b = got[..., :dim], got[..., dim:]
            name = f"packed head on device {device_index}"
            # The reduced pair holds the summary's BF16 values, so B and the off-diagonal of E = A - I carry over exactly.
            assert_bit_identical(transform_b.bfloat16(), got_b, name=f"{name}, B")
            assert_bit_identical(
                transform_a[:, off_diagonal].bfloat16(), complement[:, off_diagonal], name=f"{name}, E off the diagonal"
            )
            # On the diagonal the two paths round the same FP32 A differently: the summary rounds A near 1, the packed
            # head rounds A - 1 near 0, which keeps the decay that BF16 A loses. They agree within both roundings.
            a_diagonal = torch.diagonal(transform_a, dim1=1, dim2=2).double()
            e_diagonal = torch.diagonal(complement, dim1=1, dim2=2).double()
            gap = (1 + e_diagonal - a_diagonal).abs()
            bound = (_bf16_spacing(a_diagonal) + _bf16_spacing(e_diagonal)) / 2
            assert (gap <= bound).all(), f"{name}: diagonal E differs from A - I by {(gap / bound).max():.2f} roundings"
            if torch.equal(transform_a, eye.expand_as(transform_a)) and not transform_b.any():
                # A rank without a valid chunk publishes the identity map, (0, 0) in complement form.
                assert not got.any(), f"{name}: identity summary is not packed as (0, 0)"
    finally:
        for tensor in (*protocol, *outputs, start_tt, end_tt):
            if tensor is not None:
                ttnn.deallocate(tensor)
