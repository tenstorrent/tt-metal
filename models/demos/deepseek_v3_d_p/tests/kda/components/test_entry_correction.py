# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.entry_correction must form every valid chunk's output and the final state from the summary's
emitted terms and the entry state, switching to the tail's entry state at the chronological reset chunk."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.components.test_relay_affine_prefix import (
    _relay,
    _relay_columns,
    _Transforms,
)
from models.demos.deepseek_v3_d_p.tt.kda.entry_correction import EntryCorrection
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import _RelayFinalCarry
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, make_actual_start

pytestmark = run_for_blackhole()

_ROWS, _HEADS, _KEY = 640, 24, 128
_CHUNKS = _ROWS // 32


def _rank_chunks(sp_size: int, start: int, end: int | None) -> dict[int, tuple[int, int]]:
    """Chronology by hand: per rank, its valid chunks and the chunk its split tail starts at (0: no tail). Only the
    rank holding the first token splits, and only when the interval reaches back into it."""
    offset = start % _ROWS
    first_rank = (start // _ROWS) % sp_size
    head_rows = _ROWS - offset
    length = None if end is None else end - start
    chunks = {}
    for rank in range(sp_size):
        step = (rank - first_rank) % sp_size
        if length is None:
            valid_rows = _ROWS
        elif step == 0:
            valid_rows = min(length, head_rows) + max(0, length - head_rows - (sp_size - 1) * _ROWS)
        else:
            valid_rows = min(max(0, length - head_rows - (step - 1) * _ROWS), _ROWS)
        reset = head_rows // 32 if step == 0 and offset != 0 and valid_rows > head_rows else 0
        chunks[rank] = ((valid_rows + 31) // 32, reset)
    return chunks


_MESH = pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4",
        )
    ],
    indirect=True,
)


def _correction(mesh_device) -> EntryCorrection:
    grid = mesh_device.compute_with_storage_grid_size()
    return EntryCorrection(
        mesh_device,
        sequence_parallel_axis=0,
        batch_heads=_HEADS,
        num_chunks=_CHUNKS,
        key_dim=_KEY,
        local_rows=_ROWS,
        cores=[ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][:110],
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )


def _replicate(mesh_device, host, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        host,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@_MESH
# From the sequence start no rank's chunks wrap, so every chunk enters at the group's entry state; "bounded" ends the
# interval two chunks before the last rank's end. A start 352 rows into rank 1 splits that rank at chunk 9: its chunks
# from the reset on, and its final state, take the tail's entry state. Over the 110 cores a head's chunks split five per
# core, so the reset falls inside the second core's chunks, leaving the later cores none of the group's; "split-bounded"
# ends the interval three chunks into the tail.
@pytest.mark.parametrize(
    "start, length_delta",
    [(0, None), (0, -64), (992, None), (992, -256)],
    ids=["full", "bounded", "split", "split-bounded"],
)
def test_entry_correction_matches_reference(mesh_device, start, length_delta):
    generator = torch.Generator().manual_seed(start + (length_delta or 0))
    sp_size, tp_size = tuple(mesh_device.shape)
    zero = 0.1 * torch.randn(_HEADS, _CHUNKS, 32, _KEY, generator=generator)
    identity = 0.1 * torch.randn(_HEADS, _CHUNKS, 32, _KEY, generator=generator)
    final_a, final_b, entry, tail = (0.1 * torch.randn(_HEADS, _KEY, _KEY, generator=generator) for _ in range(4))
    correction = _correction(mesh_device)
    to_device = lambda host, memory_config=ttnn.DRAM_MEMORY_CONFIG: _replicate(mesh_device, host, memory_config)

    end = None if length_delta is None else start + sp_size * _ROWS + length_delta
    output, final = correction(
        to_device(zero, correction.terms_memory_config),
        to_device(identity, correction.terms_memory_config),
        to_device(final_a),
        to_device(final_b),
        to_device(entry),
        to_device(tail),
        make_actual_start(mesh_device, start),
        None if end is None else make_actual_start(mesh_device, end),
    )

    from_entry = zero + (identity - zero) @ entry.unsqueeze(1)
    from_tail = zero + (identity - zero) @ tail.unsqueeze(1)
    chunks = _rank_chunks(sp_size, start, end)
    for index, (local_output, local_final) in enumerate(
        zip(ttnn.get_device_tensors(output), ttnn.get_device_tensors(final), strict=True)
    ):
        rank = index // tp_size
        valid, reset = chunks[rank]
        expected = from_entry.clone()
        if reset:
            expected[:, reset:] = from_tail[:, reset:]
        expected_final = final_a @ (tail if reset else entry) + final_b
        got = ttnn.to_torch(local_output).float()
        assert_accurate(
            expected[:, :valid], got[:, :valid], name=f"output rank={rank}", pcc_threshold=0.9999, nrmse_threshold=0.01
        )
        assert_accurate(
            expected_final,
            ttnn.to_torch(local_final),
            name=f"final rank={rank}",
            pcc_threshold=0.9999,
            nrmse_threshold=1e-3,
        )


@_MESH
# The relay leaves the last rank's final state in its landing; the first rank's correction waits for it there and
# takes its split tail's seed from it (one core per head reads it and hands it to the head's other cores). Two calls
# with the final carry's SP-line barrier between them: the second must see the arrival count the first consumed, also
# after an unsplit call whose tail reads nothing. The relay's head cores go column by column, so head h's state is not
# on the landing's h-th shard core in row-major order.
@pytest.mark.parametrize("starts", [(992, 992), (640, 992)], ids=["split-twice", "unsplit-then-split"])
def test_entry_correction_takes_tail_from_relay_landing(mesh_device, starts):
    generator = torch.Generator().manual_seed(sum(starts))
    sp_size, tp_size = tuple(mesh_device.shape)
    zero = 0.1 * torch.randn(_HEADS, _CHUNKS, 32, _KEY, generator=generator)
    identity = 0.1 * torch.randn(_HEADS, _CHUNKS, 32, _KEY, generator=generator)
    final_a, final_b = (0.1 * torch.randn(_HEADS, _KEY, _KEY, generator=generator) for _ in range(2))
    transforms = _Transforms(mesh_device, seed=starts[0])
    relay = _relay(mesh_device, cores=_relay_columns(mesh_device))
    final_carry = _RelayFinalCarry.shared(mesh_device, 0, relay.num_links)
    correction = _correction(mesh_device)
    terms = [_replicate(mesh_device, host, correction.terms_memory_config) for host in (zero, identity)]
    final_terms = [_replicate(mesh_device, host) for host in (final_a, final_b)]

    results = []
    for start in starts:
        actual_start = make_actual_start(mesh_device, start)
        entry, prefix_final, _ = relay(
            transforms.packed, transforms.initial_tt, actual_start, leave_final_in_landing=True
        )
        output, final = correction(
            *terms,
            *final_terms,
            entry,
            prefix_final,
            actual_start,
            None,
            tail_relay=relay,
        )
        # The SP-line barrier before the next call (the model runs the final carry after the correction).
        final_carry(final, prefix_final, actual_start, None, _ROWS)
        results.append((start, entry, prefix_final, output, final))

    for start, entry, prefix_final, output, final in results:
        first_rank = (start // _ROWS) % sp_size
        last_rank = (first_rank + sp_size - 1) % sp_size
        # The relay still leaves the final state on the last chronological rank: the tail's seed.
        tail = ttnn.to_torch(ttnn.get_device_tensors(prefix_final)[last_rank * tp_size])
        chunks = _rank_chunks(sp_size, start, None)
        devices = zip(
            ttnn.get_device_tensors(entry),
            ttnn.get_device_tensors(output),
            ttnn.get_device_tensors(final),
            strict=True,
        )
        for index, (local_entry, local_output, local_final) in enumerate(devices):
            rank = index // tp_size
            valid, reset = chunks[rank]
            seed = ttnn.to_torch(local_entry)
            expected = zero + (identity - zero) @ seed.unsqueeze(1)
            if reset:
                expected[:, reset:] = (zero + (identity - zero) @ tail.unsqueeze(1))[:, reset:]
            expected_final = final_a @ (tail if reset else seed) + final_b
            name = f"start={start} rank={rank}"
            # The relay's late states are larger than the reference test's seeds: through the FPU the first and last
            # ranks' final states land at NRMSE ~1.15e-3, the same with the tail's seed read from DRAM.
            final_nrmse = 2e-3
            assert_accurate(
                expected[:, :valid],
                ttnn.to_torch(local_output).float()[:, :valid],
                name=f"output {name}",
                pcc_threshold=0.9999,
                nrmse_threshold=0.01,
            )
            assert_accurate(
                expected_final,
                ttnn.to_torch(local_final),
                name=f"final {name}",
                pcc_threshold=0.9999,
                nrmse_threshold=final_nrmse,
            )
