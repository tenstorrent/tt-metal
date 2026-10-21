# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.kda.entry_correction must form every valid chunk's output and the final state from the summary's
emitted terms and the entry state."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.kda.entry_correction import EntryCorrection
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, make_actual_start

pytestmark = run_for_blackhole()

_ROWS, _HEADS, _KEY = 640, 24, 128
_CHUNKS = _ROWS // 32


@pytest.mark.parametrize(
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
# From the sequence start no rank's chunks wrap, so every chunk enters at the group's entry state; the bounded interval
# ends two chunks before the last rank's end.
@pytest.mark.parametrize("end_chunks", [None, _CHUNKS - 2], ids=["full", "bounded"])
def test_entry_correction_matches_reference(mesh_device, end_chunks):
    generator = torch.Generator().manual_seed(0 if end_chunks is None else end_chunks)
    sp_size, _ = tuple(mesh_device.shape)
    zero = 0.1 * torch.randn(_HEADS, _CHUNKS, 32, _KEY, generator=generator)
    identity = 0.1 * torch.randn(_HEADS, _CHUNKS, 32, _KEY, generator=generator)
    final_a, final_b, entry, tail = (0.1 * torch.randn(_HEADS, _KEY, _KEY, generator=generator) for _ in range(4))
    grid = mesh_device.compute_with_storage_grid_size()
    correction = EntryCorrection(
        mesh_device,
        sequence_parallel_axis=0,
        batch_heads=_HEADS,
        num_chunks=_CHUNKS,
        key_dim=_KEY,
        local_rows=_ROWS,
        cores=[ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][:110],
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )

    def to_device(host, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            host,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    end = None if end_chunks is None else make_actual_start(mesh_device, ((sp_size - 1) * _CHUNKS + end_chunks) * 32)
    output, final = correction(
        to_device(zero, correction.terms_memory_config),
        to_device(identity, correction.terms_memory_config),
        to_device(final_a),
        to_device(final_b),
        to_device(entry),
        to_device(tail),
        make_actual_start(mesh_device, 0),
        end,
    )

    expected = zero + (identity - zero) @ entry.unsqueeze(1)
    expected_final = final_a @ entry + final_b
    tp_size = mesh_device.shape[1]
    for index, (local_output, local_final) in enumerate(
        zip(ttnn.get_device_tensors(output), ttnn.get_device_tensors(final), strict=True)
    ):
        rank = index // tp_size
        valid = _CHUNKS if end_chunks is None or rank < sp_size - 1 else end_chunks
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
