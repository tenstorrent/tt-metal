# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The distributed-prefix chain must apply each rank's transform in chronological order."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import _AffineTransform, _distributed_prefix, _pack_transform
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    make_actual_start,
)

pytestmark = run_for_blackhole()

_ROWS, _HEADS, _KEY, _VALUE = 640, 24, 128, 128


def _meshes(**overrides):
    return [
        pytest.param(
            (8, 4),
            torus_xy_device_params(**overrides),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4",
        ),
        pytest.param(
            (2, 4),
            fabric_1d_device_params(**overrides),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="SP2xTP4",
        ),
    ]


@pytest.mark.parametrize("mesh_device,device_params", _meshes(), indirect=True)
# Only the rank holding the first token orders the chain; "split" checks that an offset within that rank is ignored.
# Starts past the last rank wrap modulo the mesh, so every start applies to both meshes.
@pytest.mark.parametrize(
    "start", [0, 640, 672, 3232, 4480], ids=["aligned", "rotated", "split", "late-split", "last-rank"]
)
def test_chain_affine_transforms_matches_reference(mesh_device, start):
    generator = torch.Generator().manual_seed(start)
    sp_size, tp_size = tuple(mesh_device.shape)
    # BF16-representable transforms make the BF16 transport exact, so the op-by-op chain below sees the same values.
    a = (0.1 * torch.randn(sp_size, _HEADS, _KEY, _KEY, generator=generator)).bfloat16().float()
    b = (0.1 * torch.randn(sp_size, _HEADS, _KEY, _VALUE, generator=generator)).bfloat16().float()
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

    def replicated(host, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            host,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    entry, final = _distributed_prefix(
        _pack_transform(_AffineTransform(per_rank(a), per_rank(b))),
        replicated(initial, ttnn.float32),
        sequence_parallel_axis=0,
        compute_config=compute_config,
        actual_start=make_actual_start(mesh_device, start),
        local_rows=_ROWS,
    )

    first_rank = (start // _ROWS) % sp_size
    # The op-by-op chain the kernel replaced: per rank a BF16 -> FP32 typecast, an FP32 matmul and an FP32 add.
    working = ttnn.L1_MEMORY_CONFIG
    op_by_op_carry = replicated(initial.reshape(1, _HEADS, _KEY, _VALUE), ttnn.float32, working)
    op_by_op_entries = {}
    for step in range(sp_size):
        rank = (first_rank + step) % sp_size
        op_by_op_entries[rank] = op_by_op_carry
        transform_a = ttnn.typecast(replicated(a[rank : rank + 1], ttnn.bfloat16, working), ttnn.float32)
        transform_b = ttnn.typecast(replicated(b[rank : rank + 1], ttnn.bfloat16, working), ttnn.float32)
        product = ttnn.matmul(
            transform_a, op_by_op_carry, memory_config=working, dtype=ttnn.float32, compute_kernel_config=compute_config
        )
        op_by_op_carry = ttnn.add(product, transform_b, memory_config=working)

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
        # Same arithmetic as the op-by-op chain, so the states match it bit for bit.
        assert_bit_identical(
            ttnn.to_torch(ttnn.get_device_tensors(op_by_op_entries[rank])[index]).reshape(_HEADS, _KEY, _VALUE),
            ttnn.to_torch(local_entry),
            name=f"entry vs op-by-op rank={rank} tp={tp}",
        )
        assert_bit_identical(
            ttnn.to_torch(ttnn.get_device_tensors(op_by_op_carry)[index]).reshape(_HEADS, _KEY, _VALUE),
            ttnn.to_torch(local_final),
            name=f"final vs op-by-op rank={rank} tp={tp}",
        )


# Only this test captures a trace, and the default trace region is empty.
@pytest.mark.parametrize("mesh_device,device_params", _meshes(trace_region_size=1 << 20), indirect=True)
def test_chain_affine_transforms_trace_replay_follows_actual_start(mesh_device):
    """One capture serves every start: the chronology is derived on device from actual_start's contents."""
    generator = torch.Generator().manual_seed(0)
    sp_size, tp_size = tuple(mesh_device.shape)
    a = (0.1 * torch.randn(sp_size, _HEADS, _KEY, _KEY, generator=generator)).bfloat16().float()
    b = (0.1 * torch.randn(sp_size, _HEADS, _KEY, _VALUE, generator=generator)).bfloat16().float()
    initial = 0.1 * torch.randn(_HEADS, _KEY, _VALUE, generator=generator)

    def replicated(host, dtype):
        return ttnn.from_torch(
            host,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    transforms = replicated(torch.cat([a, b], dim=-1), ttnn.bfloat16)
    initial_tt = replicated(initial, ttnn.float32)
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False
    )

    def chain(actual_start):
        return ttnn.experimental.kda.chain_affine_transforms(
            transforms, initial_tt, actual_start=actual_start, local_rows=_ROWS, compute_kernel_config=compute_config
        )

    actual_start = make_actual_start(mesh_device, 0)
    for tensor in chain(actual_start):
        ttnn.deallocate(tensor)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    entry, final = chain(actual_start)
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        for start in [0, 672, 4480, 3232, 640, 0]:
            source = make_actual_start(mesh_device, start)
            ttnn.copy(source, actual_start)
            ttnn.deallocate(source)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            eager_entry, eager_final = chain(make_actual_start(mesh_device, start))
            for index, tensors in enumerate(
                zip(*(ttnn.get_device_tensors(t) for t in (entry, final, eager_entry, eager_final)), strict=True)
            ):
                rank, tp = divmod(index, tp_size)
                replay_entry, replay_final, expected_entry, expected_final = (ttnn.to_torch(t) for t in tensors)
                label = f"start={start} rank={rank} tp={tp}"
                assert_bit_identical(expected_entry, replay_entry, name=f"replayed entry {label}")
                assert_bit_identical(expected_final, replay_final, name=f"replayed final {label}")
            ttnn.deallocate(eager_entry)
            ttnn.deallocate(eager_final)
    finally:
        ttnn.release_trace(mesh_device, trace)
