# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
from models.demos.llama3_70b_galaxy.tt.model_config import get_core_ranges

NUM_RANDOM_ITERS = 64


def _torch_reference_unpack(grammar_bitmask: torch.Tensor) -> torch.Tensor:
    structured_output_arange = torch.arange(32, dtype=torch.int32, device=grammar_bitmask.device)
    unpacked = torch.bitwise_right_shift(grammar_bitmask[:, :, None], structured_output_arange[None, None, :]) & 1
    unpacked = unpacked.reshape(grammar_bitmask.shape[0], -1).to(torch.float32)
    return torch.where(unpacked != 0, torch.tensor(0.0), torch.tensor(-1e9))


@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
def test_unpack_bitmask_with_subcore_grids(mesh_device):
    batch_size = 2
    packed_vocab_dim = 8

    bitmask_arange_tt = ttnn.from_torch(
        torch.arange(32, dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    transformer = SimpleNamespace(
        args=SimpleNamespace(
            sub_core_grids=ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
                ]
            )
        ),
        mesh_device=mesh_device,
        bitmask_arange=bitmask_arange_tt,
    )

    g = torch.Generator().manual_seed(0)
    for i in range(NUM_RANDOM_ITERS):
        print(f"Iteration {i}")
        grammar_bitmask = torch.randint(0, 2**31 - 1, (batch_size, packed_vocab_dim), dtype=torch.int32, generator=g)
        grammar_bitmask_tt = ttnn.from_torch(
            grammar_bitmask,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        unpacked_tt = TtTransformer.unpack_bitmask(transformer, grammar_bitmask_tt)
        unpacked_tt_torch = ttnn.to_torch(ttnn.get_device_tensors(unpacked_tt)[0]).to(torch.float32)

        expected = _torch_reference_unpack(grammar_bitmask)
        assert torch.equal(unpacked_tt_torch, expected), f"Mismatch at iteration {i}"


@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
def test_unpack_bitmask_sharded_with_subdevices_and_subcore_grids(mesh_device):
    # Match Galaxy-style worker/prefetcher core partitioning.
    active_sender_cores, _, _, _, _, worker_cores_range_set, _, _ = get_core_ranges(
        num_reader_cores=12, num_global_cb_receivers=2, is_functional_test=False
    )
    sender_core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(core_coord, core_coord) for core_coord in active_sender_cores]
    )
    prefetcher_sub_device = ttnn.SubDevice([sender_core_range_set])
    worker_sub_device = ttnn.SubDevice([worker_cores_range_set])
    decode_manager = mesh_device.create_sub_device_manager([prefetcher_sub_device, worker_sub_device], 0)
    mesh_device.load_sub_device_manager(decode_manager)
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)])

    batch_size = 2
    packed_vocab_dim = 64  # divisible across Galaxy mesh sharding
    cluster_shape = tuple(mesh_device.shape)

    bitmask_arange_tt = ttnn.from_torch(
        torch.arange(32, dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    transformer = SimpleNamespace(
        args=SimpleNamespace(
            sub_core_grids=worker_cores_range_set,
        ),
        mesh_device=mesh_device,
        bitmask_arange=bitmask_arange_tt,
    )

    g = torch.Generator().manual_seed(1)
    for i in range(NUM_RANDOM_ITERS):
        grammar_bitmask = torch.randint(0, 2**31 - 1, (batch_size, packed_vocab_dim), dtype=torch.int32, generator=g)
        grammar_bitmask_tt = ttnn.from_torch(
            grammar_bitmask,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, None), mesh_shape=cluster_shape),
        )

        unpacked_tt = TtTransformer.unpack_bitmask(transformer, grammar_bitmask_tt)

        expected_full = _torch_reference_unpack(grammar_bitmask)
        num_cols = cluster_shape[1]
        for idx, dev_tensor in enumerate(ttnn.get_device_tensors(unpacked_tt)):
            row = idx // num_cols
            local_unpacked = ttnn.to_torch(dev_tensor).to(torch.float32)
            shard_width = local_unpacked.shape[-1]
            expected_local = expected_full[:, row * shard_width : (row + 1) * shard_width]
            assert torch.equal(
                local_unpacked, expected_local
            ), f"Mismatch at iter {i} on device {idx} (row={row}, col={idx % num_cols})"
