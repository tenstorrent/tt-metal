# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from tracy import signpost

from models.experimental.marko_ubench.modules.reference.pytorch_routed_expert import PytorchRoutedExpert
from models.experimental.marko_ubench.modules.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "batch_size, seq_len, emb_dim, hidden_dim",
    [
        # (1, 32, 7 * 1024, 2 * 1024),
        # (1, 512, 7 * 1024, 2 * 1024),
        (1, 1024, 7 * 1024, 2 * 1024),
    ],
)
def test_routed_expert_pcc(mesh_device, batch_size, seq_len, emb_dim, hidden_dim, reset_seeds):
    """Test PCC (Pearson Correlation Coefficient) between PyTorch and TTNN routed expert implementations.

    Args:
        mesh_device: TTNN mesh device fixture with shape (1, 1)
        batch_size: Batch size for input tensor
        seq_len: Sequence length for input tensor
        emb_dim: Embedding dimension
        hidden_dim: Hidden dimension
        reset_seeds: Fixture to reset random seeds
    """
    # Add Tracy signpost for profiling identification
    signpost(f"test_routed_expert_pcc_b{batch_size}_s{seq_len}_e{emb_dim}_h{hidden_dim}")

    # Create compute kernel config with optimized low-precision settings
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,  # Best accuracy with no performance penalty
    )

    # Create PyTorch reference module with float32 weights
    torch_module = PytorchRoutedExpert(emb_dim=emb_dim, hidden_dim=hidden_dim)
    torch_module.eval()

    # Create TTNN module with low-precision configuration
    ttnn_module = TtRoutedExpert(
        device=mesh_device,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        torch_module=torch_module,
        weights_dtype=ttnn.bfloat4_b,
        activations_dtype=ttnn.bfloat8_b,
        compute_kernel_config=compute_kernel_config,
    )

    # Create random input tensor (float32 for PyTorch)
    torch_input = torch.randn(batch_size, seq_len, emb_dim, dtype=torch.float32)

    # PyTorch forward pass
    with torch.no_grad():
        torch_output = torch_module(torch_input)

    # Convert input to TTNN (bfloat16, will be converted to bfp8 in forward)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # TTNN forward pass
    ttnn_output = ttnn_module.forward(ttnn_input)

    # Convert TTNN output back to PyTorch
    output_tensor = ttnn.to_torch(ttnn_output)

    # Reshape to match PyTorch output shape (TTNN may add batch dimensions)
    output_tensor = output_tensor.reshape(torch_output.shape)

    # Calculate PCC (relaxed threshold for low precision)
    passing, pcc_message = assert_with_pcc(torch_output, output_tensor, 0.988)

    # Get actual data types and config from the module
    weights_dtype = ttnn_module.gate_proj.dtype
    activations_dtype = ttnn_module.activations_dtype

    logger.info(f"Configuration:")
    logger.info(f"  Weights: {weights_dtype}")
    logger.info(f"  Activations: {activations_dtype}")
    logger.info(f"  Math fidelity: {compute_kernel_config.math_fidelity}")
    logger.info(f"  FP32 dest acc: {compute_kernel_config.fp32_dest_acc_en}")
    logger.info(f"  Packer L1 acc: {compute_kernel_config.packer_l1_acc}")
    logger.info(f"  Math approx mode: {compute_kernel_config.math_approx_mode}")
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"PCC check failed: {pcc_message}"
