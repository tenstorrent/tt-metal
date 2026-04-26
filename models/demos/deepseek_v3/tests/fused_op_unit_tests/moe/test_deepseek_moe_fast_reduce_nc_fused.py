# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Unit test for deepseek_moe_fast_reduce_nc_fused — a single fused kernel that
replaces the four-op chain:
    permute(scores, (3,1,0,2)) → to_layout(TILE) → mul(activation, scores)
    → deepseek_moe_fast_reduce_nc

Runs on a single Tenstorrent device. Emulates the 16×8 mesh (128 devices)
by 128 sequential iterations (same pattern as test_deepseek_moe_fast_reduce_nc_single.py).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

# REFERENCE_MESH_SHAPE = (1, 1)
REFERENCE_MESH_SHAPE = (16, 8)
NUM_REFERENCE_MESH_DEVICES = REFERENCE_MESH_SHAPE[0] * REFERENCE_MESH_SHAPE[1]


def _bf16_to_float(bf16):
    """Convert a torch.bfloat16 scalar or tensor to float32 using numpy reinterpretation"""
    import numpy as np

    if isinstance(bf16, float):
        return float(bf16)
    # Convert torch tensor to numpy uint16, upcast to uint32 << 16
    arr = bf16.detach().cpu().numpy().astype(np.uint16)
    arr32 = arr.astype(np.uint32) << 16
    return arr32.view(np.float32)


def _torch_golden_scale_and_fast_reduce_nc(
    torch_unsqueezed_global,
    torch_expert_scores_global,
    *,
    num_replicated_devices,
):
    """Reference: permute + broadcast mul + split-width sum along expert dim (dim 0)."""
    topk_experts_weights = torch_expert_scores_global.permute(3, 1, 0, 2)
    scaled = torch_unsqueezed_global * topk_experts_weights
    hidden_size = scaled.shape[-1]
    split_size = hidden_size // num_replicated_devices
    assert hidden_size % split_size == 0
    num_chunks = hidden_size // split_size
    goldens = []
    for i in range(num_chunks):
        chunk = scaled[:, :, :, i * split_size : (i + 1) * split_size]
        goldens.append(chunk.sum(dim=0, keepdim=True))
    return goldens


# @pytest.mark.requires_device(["N150", "N300"])
@pytest.mark.parametrize("batches_per_device", [32])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("device_params", [{}], indirect=True, ids=["single_device_default"])
def test_deepseek_moe_fast_reduce_nc_separated(
    mesh_device,
    batches_per_device,
    select_experts_k,
    seq,
    hidden_size,
):
    if mesh_device.get_num_devices() != 1:
        pytest.skip(
            f"Single-device fused variant: expected 1 device, got {mesh_device.get_num_devices()} "
            "(use e.g. MESH_DEVICE=N150)."
        )

    torch.manual_seed(2005)

    cluster_axis = 0
    ref_mesh_shape = REFERENCE_MESH_SHAPE
    num_dispatch_devices = ref_mesh_shape[cluster_axis]
    num_replicated_devices = NUM_REFERENCE_MESH_DEVICES // num_dispatch_devices
    batch = batches_per_device * num_dispatch_devices
    tokens_per_dispatch_device = batch // num_dispatch_devices

    # Memory configs (identical to test_deepseek_moe_fast_reduce_nc_single.py)
    activation_memory_config = ttnn.L1_MEMORY_CONFIG
    scaled_output_memory_config = ttnn.L1_MEMORY_CONFIG

    fast_reduce_output_memory_config = ttnn.MemoryConfig(
        ttnn.BufferType.L1,
        ttnn.NdShardSpec(
            ttnn.Shape([1, 32, 128]),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 5), ttnn.CoreCoord(2, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 5), ttnn.CoreCoord(3, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(6, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 5), ttnn.CoreCoord(6, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )

    replicate_mapper = ttnn.replicate_tensor_to_mesh_mapper(mesh_device)

    # Global tensors (same dimensions as the original multi-chip test)
    torch_unsqueezed_global = torch.rand((select_experts_k, 1, batch, hidden_size), dtype=torch.bfloat16) - 0.5
    torch_expert_scores_global = torch.rand((batch, 1, seq, select_experts_k), dtype=torch.bfloat16)
    torch_expert_scores_global = torch_expert_scores_global / torch_expert_scores_global.sum(dim=-1, keepdim=True)

    torch_goldens = _torch_golden_scale_and_fast_reduce_nc(
        torch_unsqueezed_global,
        torch_expert_scores_global,
        num_replicated_devices=num_replicated_devices,
    )

    pcc_threshold = 0.988

    for virtual_device_idx in range(NUM_REFERENCE_MESH_DEVICES):
        mesh_row = virtual_device_idx // ref_mesh_shape[1]
        t0 = mesh_row * tokens_per_dispatch_device
        t1 = t0 + tokens_per_dispatch_device

        u_slice = torch_unsqueezed_global[:, :, t0:t1, :].contiguous()
        s_slice = torch_expert_scores_global[t0:t1, :, :, :].contiguous()

        # Activation: TILE layout, L1 — same as unsqueezed_output in unfused path
        tt_activation = ttnn.from_torch(
            u_slice,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=activation_memory_config,
            mesh_mapper=replicate_mapper,
        )

        # Scores: ROW_MAJOR layout, DRAM — passed directly (no permute/tilize in Python)
        tt_scores_dram = ttnn.from_torch(
            s_slice,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_mapper,
        )

        # permute, to_layout, mul, deepseek_moe_fast_reduce_nc
        topk_experts_weights = tt_scores_dram
        tt_unsqueezed_output = tt_activation
        topk_experts_weights = ttnn.permute(
            topk_experts_weights, (3, 1, 0, 2), memory_config=scaled_output_memory_config
        )
        topk_experts_weights = ttnn.to_layout(
            topk_experts_weights, layout=ttnn.TILE_LAYOUT, memory_config=scaled_output_memory_config
        )
        tt_scaled_output = ttnn.mul(
            tt_unsqueezed_output, topk_experts_weights, memory_config=scaled_output_memory_config
        )
        tt_fast_reduce_output_tensors = ttnn.experimental.deepseek_moe_fast_reduce_nc(
            tt_scaled_output,
            dim=0,
            split_size=int(tt_scaled_output.shape[-1] // num_replicated_devices),
            output_memory_config=fast_reduce_output_memory_config,
        )

        # PCC check
        for i, tt_out in enumerate(tt_fast_reduce_output_tensors):
            tt_host = ttnn.to_torch(tt_out, dtype=torch.bfloat16)
            golden_slice = torch_goldens[i][:, :, t0:t1, :]
            ok, msg = comp_pcc(golden_slice, tt_host, pcc=pcc_threshold)
            logger.info(f"virtual_dev={virtual_device_idx} mesh_row={mesh_row} chunk={i}: {msg}")
            assert ok, f"virtual_dev={virtual_device_idx} chunk={i} failed: {msg}"

        ttnn.deallocate(tt_activation)
        ttnn.deallocate(tt_scores_dram)
        ttnn.deallocate(topk_experts_weights)
        ttnn.deallocate(tt_scaled_output)
        for t in tt_fast_reduce_output_tensors:
            ttnn.deallocate(t)


# @pytest.mark.requires_device(["N150", "N300"])
@pytest.mark.parametrize("batches_per_device", [32])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("device_params", [{}], indirect=True, ids=["single_device_default"])
def test_deepseek_moe_fast_reduce_nc_fused(
    mesh_device,
    batches_per_device,
    select_experts_k,
    seq,
    hidden_size,
):
    if mesh_device.get_num_devices() != 1:
        pytest.skip(
            f"Single-device fused variant: expected 1 device, got {mesh_device.get_num_devices()} "
            "(use e.g. MESH_DEVICE=N150)."
        )

    torch.manual_seed(2005)

    cluster_axis = 0
    ref_mesh_shape = REFERENCE_MESH_SHAPE
    num_dispatch_devices = ref_mesh_shape[cluster_axis]
    num_replicated_devices = NUM_REFERENCE_MESH_DEVICES // num_dispatch_devices
    batch = batches_per_device * num_dispatch_devices
    tokens_per_dispatch_device = batch // num_dispatch_devices

    # Memory configs (identical to test_deepseek_moe_fast_reduce_nc_single.py)
    activation_memory_config = ttnn.L1_MEMORY_CONFIG

    fast_reduce_output_memory_config = ttnn.MemoryConfig(
        ttnn.BufferType.L1,
        ttnn.NdShardSpec(
            ttnn.Shape([1, 32, 128]),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 5), ttnn.CoreCoord(2, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 5), ttnn.CoreCoord(3, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(6, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 5), ttnn.CoreCoord(6, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )

    replicate_mapper = ttnn.replicate_tensor_to_mesh_mapper(mesh_device)

    # Global tensors (same dimensions as the original multi-chip test)
    torch_unsqueezed_global = torch.rand((select_experts_k, 1, batch, hidden_size), dtype=torch.bfloat16) - 0.5
    torch_expert_scores_global = torch.rand((batch, 1, seq, select_experts_k), dtype=torch.bfloat16)
    torch_expert_scores_global = torch_expert_scores_global / torch_expert_scores_global.sum(dim=-1, keepdim=True)

    torch_goldens = _torch_golden_scale_and_fast_reduce_nc(
        torch_unsqueezed_global,
        torch_expert_scores_global,
        num_replicated_devices=num_replicated_devices,
    )

    pcc_threshold = 0.988

    for virtual_device_idx in range(NUM_REFERENCE_MESH_DEVICES):
        mesh_row = virtual_device_idx // ref_mesh_shape[1]
        t0 = mesh_row * tokens_per_dispatch_device
        t1 = t0 + tokens_per_dispatch_device

        u_slice = torch_unsqueezed_global[:, :, t0:t1, :].contiguous()
        s_slice = torch_expert_scores_global[t0:t1, :, :, :].contiguous()

        # Activation: TILE layout, L1 — same as unsqueezed_output in unfused path
        tt_activation = ttnn.from_torch(
            u_slice,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=activation_memory_config,
            mesh_mapper=replicate_mapper,
        )

        # Scores: ROW_MAJOR layout, DRAM — passed directly (no permute/tilize in Python)
        tt_scores_dram = ttnn.from_torch(
            s_slice,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_mapper,
        )

        # Single fused call replacing permute + to_layout + mul + deepseek_moe_fast_reduce_nc
        tt_fused_outputs = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            tt_activation,
            reduce_dim=0,
            split_size=int(hidden_size // num_replicated_devices),
            output_memory_config=fast_reduce_output_memory_config,
            scores_tensor=tt_scores_dram,
        )
        # with garbage, 2026-04-17 03:59:47.239 | INFO     | z_shortcut.test_deepseek_moe_fast_reduce_nc_fused:test_deepseek_moe_fast_reduce_nc_fused:197 - virtual_dev=0 mesh_row=0 chunk=0: 0.9999966324541119

        for i, tt_out in enumerate(tt_fused_outputs):
            tt_host = ttnn.to_torch(tt_out, dtype=torch.bfloat16)
            golden_slice = torch_goldens[i][:, :, t0:t1, :]
            ok, msg = comp_pcc(golden_slice, tt_host, pcc=pcc_threshold)
            logger.info(f"virtual_dev={virtual_device_idx} mesh_row={mesh_row} chunk={i}: {msg}")
            assert ok, f"virtual_dev={virtual_device_idx} chunk={i} failed: {msg}"

        ttnn.deallocate(tt_activation)
        ttnn.deallocate(tt_scores_dram)
        for t in tt_fused_outputs:
            ttnn.deallocate(t)
