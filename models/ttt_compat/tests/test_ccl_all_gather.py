# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import ttnn
from models.ttt_compat.tt.ccl import TT_CCL, tt_all_gather


@pytest.mark.parametrize(
    "mesh_shape, cluster_axis, gathers",
    [
        ((1, 1), None, False),
        ((1, 1), 0, False),
        ((1, 1), 1, False),
        ((1, 2), 0, False),
        ((1, 2), 1, True),
        ((1, 2), None, True),
        ((1, 8), 1, True),
        ((2, 1), 0, True),
        ((2, 1), 1, False),
        ((2, 1), None, True),
        ((8, 1), 0, True),
        ((2, 4), 0, True),
        ((2, 4), 1, True),
        ((2, 4), None, True),
    ],
)
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_all_gather_dispatches_target_axis(monkeypatch, mesh_shape, cluster_axis, gathers, sharded, dtype):
    tensor = Mock(dtype=ttnn.bfloat16)
    collectives = Mock()
    collectives.get_num_links.return_value = 2
    all_gather = Mock()
    convert = Mock(return_value=tensor)
    monkeypatch.setattr(ttnn.experimental, "all_gather_async", all_gather)
    monkeypatch.setattr(ttnn, "to_memory_config", convert)

    output = tt_all_gather(
        tensor,
        SimpleNamespace(shape=mesh_shape),
        collectives,
        cluster_axis=cluster_axis,
        dim=3,
        sharded=sharded,
        dtype=dtype,
    )

    if not gathers:
        assert output is tensor
        all_gather.assert_not_called()
        convert.assert_not_called()
        assert tensor.mock_calls == []
        assert collectives.mock_calls == []
        return

    assert output is all_gather.return_value
    all_gather.assert_called_once()
    assert all_gather.call_args.args == (tensor,)
    assert all_gather.call_args.kwargs.get("cluster_axis") == cluster_axis
    assert all_gather.call_args.kwargs["num_links"] == 2
    collectives.get_num_links.assert_called_once_with(cluster_axis)
    semaphore_args = () if cluster_axis is None else (cluster_axis,)
    collectives.get_and_cycle_ag_semaphore_handles.assert_called_once_with(*semaphore_args)
    collectives.get_and_cycle_barrier_semaphore_handle.assert_called_once_with(*semaphore_args)
    tensor.deallocate.assert_called_once_with(True)


@pytest.mark.parametrize("mesh_device", [(1, 2), (2, 1)], ids=["1x2", "2x1"], indirect=True)
# A partial open on a larger system mesh can leave fabric peers unopened.
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}],
    indirect=True,
)
@pytest.mark.parametrize(
    "rows, width, dtype, layout",
    [
        (32, 4096, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        (64, 128, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        (32, 128, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
def test_all_gather_1d_mesh(mesh_device, rows, width, dtype, layout):
    """Different width shards must be gathered along the non-singleton axis."""
    active_axis = 1 if mesh_device.shape[0] == 1 else 0
    collectives = TT_CCL(mesh_device)
    reference = torch.arange(rows * width).reshape(1, 1, rows, width).remainder(31).to(torch.bfloat16)
    tensor = ttnn.from_torch(
        reference,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_shape = tuple(tensor.shape)
    output = tt_all_gather(
        tensor,
        mesh_device,
        collectives,
        cluster_axis=active_axis,
        dim=3,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print(f"mesh={tuple(mesh_device.shape)} axis={active_axis} input={input_shape} output={tuple(output.shape)}")
    assert tuple(output.shape) == tuple(reference.shape)
    # The orthogonal singleton axis must return the original, still live input.
    assert tt_all_gather(output, mesh_device, collectives, cluster_axis=1 - active_axis, dim=3, dtype=dtype) is output
    shards = ttnn.get_device_tensors(output)
    assert len(shards) == mesh_device.get_num_devices()
    for shard in shards:
        assert torch.equal(ttnn.to_torch(shard), reference)


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}],
    indirect=True,
)
def test_distributed_rmsnorm_gathers_column_statistics(mesh_device):
    # Exercise an actual caller through the compatibility import used by demos.
    # Different shard magnitudes distinguish global RMS from per-chip RMS.
    from models.tt_transformers.tt.ccl import tt_distributed_rmsnorm

    width = 128
    epsilon = 1e-5
    source = torch.ones((1, 1, 32, width), dtype=torch.bfloat16)
    source[..., width // 2 :] = 3
    tensor = ttnn.from_torch(
        source,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gamma = ttnn.from_torch(
        torch.ones((1, 1, width // 32, 32), dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output = tt_distributed_rmsnorm(
        tensor,
        epsilon,
        gamma,
        mesh_device,
        TT_CCL(mesh_device),
        ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    expected = source.float() * torch.rsqrt(source.float().square().mean(dim=-1, keepdim=True) + epsilon)
    for shard, reference in zip(ttnn.get_device_tensors(output), expected.chunk(2, dim=3)):
        torch.testing.assert_close(ttnn.to_torch(shard).float(), reference, rtol=0.01, atol=0.005)
