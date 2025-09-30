# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import numpy as np
from loguru import logger

from models.common.utility_functions import comp_pcc, comp_allclose_and_pcc


def setup_ccl_semahpores(mesh_device):
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)]
    return ccl_semaphore_handles


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("hidden_dim", [8192, 16384])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("norm_type", ["layer_norm", "rms_norm"])
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
def test_distributed_norm_comparison(mesh_device, batch_size, seq_len, hidden_dim, eps, norm_type):
    torch.manual_seed(42)

    # Generate random input data
    input_shape = (batch_size, 1, seq_len, hidden_dim)
    torch_input = torch.randn(input_shape)

    torch_weight = torch.randn(hidden_dim)
    torch_bias = torch.randn(hidden_dim)

    # Quantize to bfloat16, as that's the standard dataformat for activations in neural networks
    torch_input = torch_input.to(torch.bfloat16)
    torch_weight = torch_weight.to(torch.bfloat16)
    torch_bias = torch_bias.to(torch.bfloat16)

    # PyTorch reference implementation
    if norm_type == "layer_norm":
        torch_norm = torch.nn.LayerNorm(normalized_shape=hidden_dim, eps=eps)
    elif norm_type == "rms_norm":
        torch_norm = torch.nn.RMSNorm(normalized_shape=hidden_dim, eps=eps)
    torch_norm.weight.data = torch_weight.clone().float()
    if norm_type == "layer_norm":
        torch_norm.bias.data = torch_bias.clone().float()

    with torch.no_grad():
        # Torch reference operates on float32 tensors to compute in high precision, commonly seen in LLM and DiT models.
        torch_output = torch_norm(torch_input.float()).type_as(torch_input)

    # TTNN implementation
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )

    N_DEV = 8
    ttnn_weight = ttnn.from_torch(
        torch_weight.reshape(N_DEV, 1, -1, 32),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    if norm_type == "layer_norm":
        ttnn_bias = ttnn.from_torch(
            torch_bias.reshape(N_DEV, 1, -1, 32),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

    # Use highest precision compute kernel config for comparison
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    """
    Distributed Layernorm operates in 3 parts.
    1. Compute local sum(x) and sum(x**2). Write into stats tensor.
    2. Gather stats tensor across devices.
    3. Reduce stats tensor. Compute variance and mean. Do normalization.
    """
    if norm_type == "layer_norm":
        ttnn_stats = ttnn.layer_norm_pre_all_gather(
            ttnn_input, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16
        )
    elif norm_type == "rms_norm":
        ttnn_stats = ttnn.rms_norm_pre_all_gather(
            ttnn_input, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16
        )
    ccl_semaphore_handles = setup_ccl_semahpores(mesh_device)
    ttnn.synchronize_device(mesh_device)
    ttnn_stats_gathered = ttnn.experimental.all_gather_async(
        ttnn_stats,
        dim=3,
        multi_device_global_semaphore=ccl_semaphore_handles,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        cluster_axis=1,
    )

    if norm_type == "layer_norm":
        ttnn_output = ttnn.layer_norm_post_all_gather(
            ttnn_input,
            ttnn_stats_gathered,
            epsilon=eps,
            weight=ttnn_weight,
            bias=ttnn_bias,
            compute_kernel_config=compute_kernel_config,
        )
    elif norm_type == "rms_norm":
        ttnn_output = ttnn.rms_norm_post_all_gather(
            ttnn_input,
            ttnn_stats_gathered,
            epsilon=eps,
            weight=ttnn_weight,
            compute_kernel_config=compute_kernel_config,
        )

    ttnn_output_torch = ttnn.to_torch(ttnn_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

    # Calculate metrics
    pcc_passed, pcc_value = comp_pcc(torch_output, ttnn_output_torch, pcc=0.99)
