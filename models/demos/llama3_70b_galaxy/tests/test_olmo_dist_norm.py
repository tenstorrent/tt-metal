# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Minimal test for OLMo distributed RMSNorm on prefill path.

Run with:
    export HF_MODEL=~/models/models--allenai--Olmo-3.1-32B-Think
    pytest models/demos/llama3_70b_galaxy/tests/test_olmo_dist_norm.py -v -x
"""

import torch
import pytest
import ttnn
from loguru import logger


@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],  # Galaxy TG mesh shape
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_dist_rmsnorm_prefill(mesh_device):
    """Test distributed RMSNorm directly for prefill."""
    from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
    from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
    from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL, tt_distributed_rmsnorm

    # Load model config
    model_args = TtOlmoModelArgs(mesh_device, max_batch_size=1, max_seq_len=128)
    model_args.n_layers = 1
    model_args.use_prefetcher = False

    # Setup prefetcher and CCL
    prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="prefill")
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
    tt_ccl = TT_CCL(
        mesh_device, model_args, prefetcher_setup.worker_sub_device_id, mode="prefill", is_qwen=False, is_olmo=True
    )

    # Create test input - 4D, sharded across 4 cols on last dim
    batch_size, seq_len = 1, 128
    dim = model_args.dim  # 5120
    torch.manual_seed(42)
    torch_input = torch.randn(1, 1, seq_len, dim)
    logger.info(f"Input shape: {torch_input.shape}")

    # Create TT input - sharded across cols
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=model_args.cluster_shape),
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    logger.info(f"TT input shape: {tt_input.shape}, memory_config: {tt_input.memory_config()}")

    # Get norm weight
    state_dict = model_args.load_state_dict()
    norm_weight = state_dict["layers.0.attention_norm.weight"]
    logger.info(f"Norm weight shape: {norm_weight.shape}")

    # Create distributed weight - same as in RMSNorm.__init__
    norm_weight_4d = norm_weight.unsqueeze(0).view(1, 1, dim // 32, 32)
    gamma = ttnn.as_tensor(
        norm_weight_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=list(mesh_device.shape)),
    )
    logger.info(f"Gamma shape: {gamma.shape}")

    # Compute kernel config
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Try distributed RMSNorm
    logger.info("Calling tt_distributed_rmsnorm...")
    output, _ = tt_distributed_rmsnorm(
        tt_input,
        epsilon=1e-6,
        gamma=gamma,
        mesh_device=mesh_device,
        compute_kernel_config=compute_kernel_config,
        tt_ccl=tt_ccl,
    )
    logger.info(f"Output shape: {output.shape}")

    # Convert back and check for NaN/Inf
    output_torch = ttnn.to_torch(
        output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    assert not torch.isnan(output_torch).any(), "Output contains NaN"
    assert not torch.isinf(output_torch).any(), "Output contains Inf"

    tt_ccl.close()
    logger.info("Distributed RMSNorm test PASSED!")
