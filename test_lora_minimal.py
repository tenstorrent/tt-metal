import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import torch_random


def get_lora_matmul_config(M, N, lora_rank):
    K = lora_rank // 32
    configs = {
        (640, 640): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 8),
            in0_block_w=1 if K == 1 else 2 if K == 2 else 4,
            per_core_M=3,
            per_core_N=4,
            out_subblock_h=1,
            out_subblock_w=1 if K == 1 else 2 if K == 2 else 4,
            transpose_mcast=False,
            fused_activation=None,
        ),
        (1280, 1280): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 8),
            in0_block_w=1 if K == 1 else 2 if K == 2 else 4,
            per_core_M=5,
            per_core_N=8,
            out_subblock_h=1,
            out_subblock_w=1 if K == 1 else 2 if K == 2 else 4,
            transpose_mcast=False,
            fused_activation=None,
        ),
        (2048, 640): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 8),
            in0_block_w=1 if K == 1 else 2 if K == 2 else 4,
            per_core_M=8,
            per_core_N=4,
            out_subblock_h=1,
            out_subblock_w=1 if K == 1 else 2 if K == 2 else 4,
            transpose_mcast=False,
            fused_activation=None,
        ),
        (2048, 1280): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 8),
            in0_block_w=1 if K == 1 else 2 if K == 2 else 4,
            per_core_M=8,
            per_core_N=8,
            out_subblock_h=1,
            out_subblock_w=1 if K == 1 else 2 if K == 2 else 4,
            transpose_mcast=False,
            fused_activation=None,
        ),
        (640, 2560): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 8),
            in0_block_w=1 if K == 1 else 2 if K == 2 else 4,
            per_core_M=3,
            per_core_N=16,
            out_subblock_h=1,
            out_subblock_w=1 if K == 1 else 2 if K == 2 else 4,
            transpose_mcast=False,
            fused_activation=None,
        ),
        (2560, 640): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 8),
            in0_block_w=1 if K == 1 else 2 if K == 2 else 4,
            per_core_M=10,
            per_core_N=4,
            out_subblock_h=1,
            out_subblock_w=1 if K == 1 else 2 if K == 2 else 4,
            transpose_mcast=False,
            fused_activation=None,
        ),
        (1280, 5120): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 8),
            in0_block_w=1 if K == 1 else 2 if K == 2 else 4,
            per_core_M=5,
            per_core_N=32,
            out_subblock_h=1,
            out_subblock_w=1 if K == 1 else 2 if K == 2 else 4,
            transpose_mcast=False,
            fused_activation=None,
        ),
        (5120, 1280): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 8),
            in0_block_w=1 if K == 1 else 2 if K == 2 else 4,
            per_core_M=20,
            per_core_N=8,
            out_subblock_h=1,
            out_subblock_w=1 if K == 1 else 2 if K == 2 else 4,
            transpose_mcast=False,
            fused_activation=None,
        ),
    }

    return configs.get((M, N), None)


MATRIX_DIMENSIONS = [
    (640, 640),
    (1280, 1280),
    (2048, 640),
    (2048, 1280),
    (640, 2560),
    (2560, 640),
    (1280, 5120),
    (5120, 1280),
]

LORA_RANKS = [32, 64, 128]


@pytest.mark.parametrize("input_dim,output_dim", MATRIX_DIMENSIONS)
@pytest.mark.parametrize("lora_rank", LORA_RANKS)
def test_lora_matmul(device, input_dim, output_dim, lora_rank):
    torch_A = torch_random((input_dim, lora_rank), -0.1, 0.1, dtype=torch.float32)
    torch_B = torch_random((lora_rank, output_dim), -0.1, 0.1, dtype=torch.float32)
    torch_og_weights = torch_random((input_dim, output_dim), -0.1, 0.1, dtype=torch.float32)

    torch_lora_reconstructed = torch.matmul(torch_A, torch_B)
    torch_weights = torch.add(torch_og_weights, torch_lora_reconstructed)

    ttnn_A = ttnn.from_torch(
        torch_A.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_B = ttnn.from_torch(
        torch_B.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_og_weights = ttnn.from_torch(
        torch_og_weights.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    program_config = get_lora_matmul_config(input_dim, output_dim, lora_rank)
    assert program_config is not None, f"No program config found for {input_dim}x{output_dim}"

    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    ttnn_lora_reconstructed = ttnn.matmul(
        ttnn_A,
        ttnn_B,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        program_config=program_config,
        compute_kernel_config=compute_config,
    )
    tt_lora_reconstructed_torch = ttnn.to_torch(ttnn_lora_reconstructed).squeeze(0).squeeze(0)

    passes, pcc_message = assert_with_pcc(torch_lora_reconstructed, tt_lora_reconstructed_torch, 0.999)
    assert passes, f"PCC failed: {pcc_message}"

    ttnn_weights = ttnn.add_(ttnn_og_weights, ttnn_lora_reconstructed)
    tt_weights_torch = ttnn.to_torch(ttnn_weights).squeeze(0).squeeze(0)

    passes, pcc_message = assert_with_pcc(torch_weights, tt_weights_torch, 0.999)
    assert passes, f"PCC failed: {pcc_message}"


ADD_DIMENSIONS = [
    (1024, 1280),
    (1024, 5120),
    (96, 1280),
    (1024, 3840),
    (4096, 640),
    (96, 640),
    (4096, 2560),
    (4096, 1920),
    (32, 1280),
    (32, 640),
]


@pytest.mark.parametrize("input_dim,output_dim", ADD_DIMENSIONS)
def test_add(device, input_dim, output_dim):
    torch_A = torch_random((input_dim, output_dim), -0.1, 0.1, dtype=torch.float32)
    torch_B = torch_random((input_dim, output_dim), -0.1, 0.1, dtype=torch.float32)

    ttnn_A = ttnn.from_torch(
        torch_A.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_B = ttnn.from_torch(
        torch_B.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    torch_weights = torch.add(torch_A, torch_B)
    ttnn_weights = ttnn.add(ttnn_A, ttnn_B)
    tt_weights_torch = ttnn.to_torch(ttnn_weights).squeeze(0).squeeze(0)

    passes, pcc_message = assert_with_pcc(torch_weights, tt_weights_torch, 0.999)
    assert passes, f"PCC failed: {pcc_message}"
