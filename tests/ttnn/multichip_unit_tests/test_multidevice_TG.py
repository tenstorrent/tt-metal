# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
import tempfile
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc


from ttnn import (
    ShardTensorToMesh,
    ReplicateTensorToMesh,
    ConcatMeshToTensor,
    ListMeshToTensor,
    TensorToMesh,
    MeshToTensor,
)


@pytest.mark.parametrize(
    "device_mesh",
    [
        32,
    ],
    indirect=True,
)
def test_galaxy_matmul_1d_fracture(device_mesh):
    from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

    act_pt = torch.randn(1, 1, 32, 8192)
    weights_pt = torch.randn(1, 1, 8192, 32768)
    act = ttnn.from_torch(
        act_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    weights = ttnn.from_torch(
        weights_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ShardTensorToMesh(device_mesh, dim=3),
    )

    gt = act_pt @ weights_pt

    act = ttnn.to_device(act, device_mesh)
    weights = ttnn.to_device(weights, device_mesh)

    compute_kernel_attn = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    out = ttnn.matmul(
        act,
        weights,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=4, x=8),
        compute_kernel_config=compute_kernel_attn,
    )
    out = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(device_mesh, dim=3))

    out_pass, out_pcc = comp_pcc(gt, out, pcc=0.99)
    print(out_pcc)
    assert out_pass


class ShardTensor2dMesh(TensorToMesh):
    def __init__(self, device_mesh, dims, cluster_shape):
        super().__init__(device_mesh)
        self.dims = dims
        self.cluster_shape = cluster_shape

    def map(self, tensor: torch.tensor):
        # Returns list of tensors to map to row-major ordering of chips in cluster
        tensors_grid_y = None
        if self.dims[1] == None:
            tensors_grid_y = [tensor.clone() for _ in range(self.cluster_shape[1])]
        else:
            tensors_grid_y = torch.chunk(tensor, self.cluster_shape[1], dim=self.dims[1])

        tensors_grid_all = None
        if self.dims[0] == None:
            tensors_grid_all = [t.clone() for t in tensors_grid_y for _ in range(self.cluster_shape[0])]
        else:
            tensors_grid_all = [
                tt for t in tensors_grid_y for tt in torch.chunk(t, self.cluster_shape[0], dim=self.dims[0])
            ]

        return list(tensors_grid_all)

    def config(self):
        return {
            "strategy": "shard",
            "shard_dim": f"{self.dims[0]}",
        }


class ConcatMesh2DToTensor(MeshToTensor):
    def __init__(self, device_mesh, dims, cluster_shape):
        self.dims = dims
        self.cluster_shape = cluster_shape
        self.device_mesh = device_mesh

    def compose(self, tensor: ttnn.Tensor) -> torch.Tensor:
        tt_shards = [ttnn.to_torch(tt_input_tensor) for tt_input_tensor in ttnn.get_device_tensors(tensor)]

        row_concat = []
        for cluster_row in range(self.cluster_shape[1]):
            start = cluster_row * self.cluster_shape[0]
            end = start + self.cluster_shape[0]
            row_concat.append(torch.cat(tt_shards[start:end], dim=self.dims[0]))
        all_concat = torch.cat(row_concat, dim=self.dims[1])
        return all_concat


@pytest.mark.parametrize(
    "device_mesh",
    [
        32,
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "M,K,N",
    [
        (32, 8192, 32768),  # Llama3-70B decode FF1
        (32, 32768, 8192),  # Llama3-70B decode FF2
        (512, 8192, 32768),  # Llama3-70B prefill FF1
        (512, 32768, 8192),  # Llama3-70B prefill FF2
        (32, 16 * 1024, 64 * 1024),  # Llama3-400B decode FF1
        (32, 64 * 1024, 16 * 1024),  # Llama3-400B decode FF2
        # (512, 16*1024, 64*1024),# Llama3-400B prefill FF1 # Skipped, OOM
        (512, 64 * 1024, 16 * 1024),  # Llama3-400B prefill FF2
    ],
)
@pytest.mark.parametrize(
    "cluster_shape",
    [
        (8, 4),
        (4, 8),
    ],
)
def test_galaxy_matmul_2d_fracture(M, K, N, cluster_shape, device_mesh):
    from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

    act_pt = torch.randn(1, 1, M, K)
    weights_pt = torch.randn(1, 1, K, N)

    act = ttnn.from_torch(
        act_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ShardTensor2dMesh(device_mesh, dims=(3, None), cluster_shape=cluster_shape),
    )
    weights = ttnn.from_torch(
        weights_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ShardTensor2dMesh(device_mesh, dims=(2, 3), cluster_shape=cluster_shape),
    )

    gt = act_pt @ weights_pt

    act = ttnn.to_device(act, device_mesh)
    weights = ttnn.to_device(weights, device_mesh)

    compute_kernel_attn = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    out = ttnn.matmul(
        act,
        weights,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=4, x=8),
        compute_kernel_config=compute_kernel_attn,
    )

    out = ttnn.to_torch(out, mesh_composer=ConcatMesh2DToTensor(device_mesh, dims=(1, 3), cluster_shape=cluster_shape))
    out = torch.sum(out, dim=1)

    out_pass, out_pcc = comp_pcc(gt, out, pcc=0.99)
    print(out_pcc)
    assert out_pass
