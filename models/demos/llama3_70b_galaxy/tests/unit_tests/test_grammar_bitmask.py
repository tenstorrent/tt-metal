# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer


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

    grammar_bitmask = torch.randint(0, 2**31 - 1, (batch_size, packed_vocab_dim), dtype=torch.int32)
    grammar_bitmask_tt = ttnn.from_torch(
        grammar_bitmask,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    bitmask_arange_tt = ttnn.arange(
        start=0,
        end=32,
        step=1,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
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

    unpacked_tt = TtTransformer.unpack_bitmask(transformer, grammar_bitmask_tt)
    unpacked_tt_torch = ttnn.to_torch(ttnn.get_device_tensors(unpacked_tt)[0]).to(torch.float32)

    expected = _torch_reference_unpack(grammar_bitmask)
    assert torch.equal(unpacked_tt_torch, expected)
