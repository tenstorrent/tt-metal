# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The distributed-prefix chain must apply each rank's transform in chronological order."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import _AffineTransform, _distributed_prefix
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, make_actual_start

pytestmark = run_for_blackhole()

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
@pytest.mark.parametrize("start", [0, 640, 672, 3232], ids=["aligned", "rotated", "split", "late-split"])
def test_chain_affine_transforms_matches_reference(mesh_device, start):
    generator = torch.Generator().manual_seed(start)
    sp_size, tp_size = tuple(mesh_device.shape)
    a = 0.1 * torch.randn(sp_size, _HEADS, _KEY, _KEY, generator=generator)
    b = 0.1 * torch.randn(sp_size, _HEADS, _KEY, _VALUE, generator=generator)
    initial = 0.1 * torch.randn(_HEADS, _KEY, _VALUE, generator=generator)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(sp_size, tp_size))

    def per_rank(host):
        return ttnn.from_torch(
            host.reshape((-1,) + host.shape[2:]),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=mapper,
        )

    entry, final = _distributed_prefix(
        _AffineTransform(per_rank(a), per_rank(b)),
        ttnn.from_torch(
            initial,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        sequence_parallel_axis=0,
        compute_config=ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False
        ),
        actual_start=make_actual_start(mesh_device, start),
        local_rows=_ROWS,
    )

    # The transforms cross the mesh as BF16; the carry stays FP32.
    a, b = a.bfloat16().float(), b.bfloat16().float()
    first_rank = (start // _ROWS) % sp_size
    carry = initial
    entries = {}
    for step in range(sp_size):
        rank = (first_rank + step) % sp_size
        entries[rank] = carry
        carry = a[rank] @ carry + b[rank]
    for index, (local_entry, local_final) in enumerate(
        zip(ttnn.get_device_tensors(entry), ttnn.get_device_tensors(final), strict=True)
    ):
        rank, tp = divmod(index, tp_size)
        assert_accurate(entries[rank], ttnn.to_torch(local_entry), name=f"entry rank={rank} tp={tp}")
        assert_accurate(carry, ttnn.to_torch(local_final), name=f"final rank={rank} tp={tp}")


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
@pytest.mark.parametrize(
    "case, message",
    [("fp32-transforms", "transforms must be BFLOAT16"), ("bf16-dest", "fp32_dest_acc_en must be enabled")],
    ids=["fp32-transforms", "bf16-dest"],
)
def test_chain_affine_transforms_rejects_lossy_configurations(mesh_device, case, message, expect_error):
    sp_size = tuple(mesh_device.shape)[0]

    def replicated(shape, dtype):
        return ttnn.from_torch(
            torch.zeros(shape),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    transform_dtype = ttnn.float32 if case == "fp32-transforms" else ttnn.bfloat16
    with expect_error(RuntimeError, message):
        ttnn.experimental.kda.chain_affine_transforms(
            replicated((sp_size, 1, 32, 64), transform_dtype),
            replicated((1, 32, 32), ttnn.float32),
            actual_start=make_actual_start(mesh_device),
            local_rows=32,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                mesh_device.arch(), fp32_dest_acc_en=case != "bf16-dest", packer_l1_acc=False
            ),
        )
