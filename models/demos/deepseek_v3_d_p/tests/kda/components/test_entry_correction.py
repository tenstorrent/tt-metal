# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.entry_correction must form every valid chunk's output and the final state from the summary's
emitted terms and the entry state, switching to the tail's entry state at the chronological reset chunk: accurately,
bit-identically call after call, and within its device-time reference."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    DeviceMismatch,
    assert_device_time,
    closed_form_tensor,
    kda_mesh_param,
    median_device_us,
)
from models.demos.deepseek_v3_d_p.tt.kda.config import kimi_k3_program_config
from models.demos.deepseek_v3_d_p.tt.kda.entry_correction import EntryCorrection
from tests.ttnn.profiling.realtime_profiler_utils import require_realtime_profiler
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, make_actual_start

pytestmark = run_for_blackhole()

# Kimi-K3's per-chip workload on every SKU: 640 rows of 24 heads, on the model's 110 cores.
_ROWS, _HEADS, _KEY = 640, 24, 128
_CHUNKS = _ROWS // 32
_CORES = 110
# The model's entry correction runs at the scan's fidelity.
_MATH_FIDELITY = kimi_k3_program_config(
    active_seq_len_local=_ROWS, tp_ccl_topology=ttnn.Topology.Linear
).recurrence.scan_math_fidelity

# SP takes mesh axis 0. The op runs on each chip alone; the meshes only change the chronology.
_MESH = pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        kda_mesh_param((8, 4), "mesh-8x4", "SP8xTP4"),
        kda_mesh_param((8, 1), "ring", "loudbox-SP8xTP1"),
        kda_mesh_param((2, 4), "mesh-2x4", "loudbox-SP2xTP4"),
        kda_mesh_param((4, 1), "ring", "quietbox-SP4xTP1"),
    ],
    indirect=True,
)

# From the sequence start no rank's chunks wrap, so every chunk enters at the group's entry state; "bounded" ends the
# interval two chunks before the last rank's end. A start 352 rows into rank 1 splits that rank at chunk 9: its chunks
# from the reset on, and its final state, take the tail's entry state. Over the 110 cores a head's chunks split five per
# core, so the reset falls inside the second core's chunks, leaving the later cores none of the group's; "split-bounded"
# ends the interval three chunks into the tail.
_INTERVALS = {"full": (0, None), "bounded": (0, -64), "split": (992, None), "split-bounded": (992, -256)}


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


def _interval(sp_size: int, name: str) -> tuple[int, int | None]:
    start, length_delta = _INTERVALS[name]
    return start, None if length_delta is None else start + sp_size * _ROWS + length_delta


class _Correction:
    """The op on the model's cores, and its inputs on device in the summary's terms layout."""

    def __init__(self, mesh_device, zero, identity, final_a, final_b, entry, tail, math_fidelity):
        grid = mesh_device.compute_with_storage_grid_size()
        self.mesh_device = mesh_device
        self.op = EntryCorrection(
            mesh_device,
            sequence_parallel_axis=0,
            batch_heads=_HEADS,
            num_chunks=_CHUNKS,
            key_dim=_KEY,
            local_rows=_ROWS,
            cores=[ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][:_CORES],
            math_fidelity=math_fidelity,
        )
        # Each core's chunk rows of F(0), then of F(I).
        blocks = _HEADS * self.op.cores_per_head
        terms = torch.stack([zero.reshape(blocks, -1, _KEY), identity.reshape(blocks, -1, _KEY)], dim=1)
        self.inputs = [
            self._replicated(terms, self.op.terms_memory_config),
            *(self._replicated(t) for t in (final_a, final_b, entry, tail)),
        ]

    def _replicated(self, host, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            host,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def __call__(self, actual_start, actual_end):
        return self.op(*self.inputs, actual_start, actual_end)


def _random_case(mesh_device, seed: int, math_fidelity=ttnn.MathFidelity.HiFi4):
    generator = torch.Generator().manual_seed(seed)
    zero = 0.1 * torch.randn(_HEADS, _CHUNKS, 32, _KEY, generator=generator)
    identity = 0.1 * torch.randn(_HEADS, _CHUNKS, 32, _KEY, generator=generator)
    final_a, final_b, entry, tail = (0.1 * torch.randn(_HEADS, _KEY, _KEY, generator=generator) for _ in range(4))
    host = (zero, identity, final_a, final_b, entry, tail)
    return host, _Correction(mesh_device, *host, math_fidelity=math_fidelity)


@_MESH
@pytest.mark.parametrize("interval", list(_INTERVALS))
def test_entry_correction_matches_reference(mesh_device, interval):
    sp_size, tp_size = tuple(mesh_device.shape)
    start, end = _interval(sp_size, interval)
    (zero, identity, final_a, final_b, entry, tail), correction = _random_case(mesh_device, seed=start + (end or 0))
    output, final = correction(
        make_actual_start(mesh_device, start), None if end is None else make_actual_start(mesh_device, end)
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


_DETERMINISM_CALLS = 3


@_MESH
def test_entry_correction_determinism(mesh_device):
    """Every interval's calls return bit-identical outputs (over each rank's valid chunks) and final states."""
    sp_size, tp_size = tuple(mesh_device.shape)
    terms = closed_form_tensor(_HEADS, _CHUNKS, 32, _KEY, offset=-0.125)
    states = [closed_form_tensor(_HEADS, _KEY, _KEY, offset=offset) for offset in (-0.1, 0.05, 0.02, -0.03)]
    correction = _Correction(mesh_device, terms, terms.flip(-1), *states, math_fidelity=_MATH_FIDELITY)
    mismatch = DeviceMismatch(mesh_device)
    for interval in _INTERVALS:
        start, end = _interval(sp_size, interval)
        actual_start = make_actual_start(mesh_device, start)
        actual_end = None if end is None else make_actual_start(mesh_device, end)
        # Chunks past a rank's valid ones are left unwritten: compare only the valid ones.
        valid = torch.zeros(sp_size * tp_size * _HEADS, _CHUNKS, 32, _KEY)
        for rank, (chunks, _) in _rank_chunks(sp_size, start, end).items():
            valid[rank * tp_size * _HEADS : (rank + 1) * tp_size * _HEADS, :chunks] = 1
        valid = ttnn.from_torch(
            valid,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        first_output, first_final = None, None
        for _ in range(_DETERMINISM_CALLS):
            output, final = correction(actual_start, actual_end)
            output = ttnn.where(valid, output, 0.0)
            if first_output is None:
                first_output, first_final = output, final
                continue
            mismatch.compare(first_output, output, f"{interval} output")
            mismatch.compare(first_final, final, f"{interval} final")
            for tensor in (output, final):
                ttnn.deallocate(tensor)
        for tensor in (first_output, first_final, valid, actual_start, actual_end):
            if tensor is not None:
                ttnn.deallocate(tensor)
    mismatch.assert_none()


# Device time of one call: its program's slowest chip, the median over single calls separated by a device synchronize.
_PERF_SAMPLES = 15
_PERF_MARGIN = 0.05
# A full chunk that starts 320 rows into the second rank, as the layer gates place it: that rank splits.
_PERF_START = _ROWS + 320
# Device-time references in us, at the model's fidelity. The op runs on each chip alone.
_PERF_REFERENCE_US = {"SP8xTP4": 29.5, "loudbox-SP8xTP1": 24.5, "quietbox-SP4xTP1": 24.4}


@pytest.mark.parametrize(
    "mesh_device,device_params,layout",
    [
        pytest.param(*case.values, case.id, marks=case.marks, id=case.id)
        for case in (
            kda_mesh_param((8, 4), "mesh-8x4", "SP8xTP4"),
            kda_mesh_param((8, 1), "ring", "loudbox-SP8xTP1"),
            kda_mesh_param((4, 1), "ring", "quietbox-SP4xTP1"),
        )
    ],
    indirect=["mesh_device", "device_params"],
)
def test_entry_correction_kimi_k3_perf(mesh_device, layout):
    """The model's entry correction: 24 heads per chip over 110 cores at the model's fidelity, the terms in L1."""
    require_realtime_profiler("the entry_correction device-time gate")
    _, correction = _random_case(mesh_device, seed=_PERF_START, math_fidelity=_MATH_FIDELITY)
    actual_start = make_actual_start(mesh_device, _PERF_START)

    def call():
        outputs = correction(actual_start, None)
        ttnn.synchronize_device(mesh_device)
        return outputs

    median_us = median_device_us(mesh_device, call, "entry_correction", _PERF_SAMPLES)
    assert_device_time(median_us, _PERF_REFERENCE_US[layout], _PERF_MARGIN, f"entry_correction Kimi-K3 {layout}")
