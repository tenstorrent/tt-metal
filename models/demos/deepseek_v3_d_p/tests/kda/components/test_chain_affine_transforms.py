# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The fused distributed-prefix chain must match the op-by-op reference bit for bit."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import _AffineTransform, _distributed_prefix
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

_ROWS, _HEADS, _KEY, _VALUE = 640, 24, 128, 128


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
@pytest.mark.parametrize("start", [0, 640, 672, 3200], ids=["aligned", "rotated", "split", "late-split"])
def test_chain_affine_transforms_matches_reference(mesh_device, start):
    torch.manual_seed(start)
    sp_size = tuple(mesh_device.shape)[0]
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=tuple(mesh_device.shape))

    def per_rank(shape):
        return ttnn.from_torch(
            0.1 * torch.randn((sp_size * shape[0],) + shape[1:]),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=mapper,
        )

    transform = _AffineTransform(per_rank((_HEADS, _KEY, _KEY)), per_rank((_HEADS, _KEY, _VALUE)))
    initial = ttnn.from_torch(
        0.1 * torch.randn(_HEADS, _KEY, _VALUE),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    actual_start = make_actual_start(mesh_device, start)
    selections = ChronologicalSelections(
        ttnn.experimental.kda.chronological_selections(actual_start, 0, _ROWS, _HEADS, _KEY, _VALUE)
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    results = {}
    for fused in (False, True):
        states = _distributed_prefix(
            transform,
            initial,
            sequence_parallel_axis=0,
            selections=selections,
            compute_config=compute_config,
            actual_start=actual_start,
            local_rows=_ROWS,
            fused=fused,
        )
        results[fused] = [
            ttnn.to_torch(state, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)) for state in states
        ]
    for name, reference, fused in zip(("entry", "final"), results[False], results[True]):
        assert torch.equal(reference, fused), f"{name} state differs from the op-by-op reference"
