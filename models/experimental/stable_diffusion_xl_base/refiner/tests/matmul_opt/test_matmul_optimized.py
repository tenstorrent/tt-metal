import torch
import ttnn
import math


def test_RESNET_LINEAR1_32x1536x384(device):
    in0_shape = [1, 1536]
    in1_shape = [1536, 384]
    bias_shape = [384]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (6, 2)
    per_core_M = 1
    per_core_N = (384 // 32) // (grid_size[0] * grid_size[1])
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=12,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        mcast_in0=True,
        fuse_batch=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


def test_RESNET_LINEAR1_32x1536x768(device):
    in0_shape = [1, 1536]
    in1_shape = [1536, 768]
    bias_shape = [768]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (6, 4)
    per_core_M = 1
    per_core_N = (768 // 32) // (grid_size[0] * grid_size[1])
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=24,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        mcast_in0=True,
        fuse_batch=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


def test_RESNET_LINEAR1_32x1536x1536(device):
    in0_shape = [1, 1536]
    in1_shape = [1536, 1536]
    bias_shape = [1536]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (6, 8)
    per_core_M = 1
    per_core_N = (1536 // 32) // (grid_size[0] * grid_size[1])
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=12,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        mcast_in0=True,
        fuse_batch=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


def test_RESNET_LINEAR2_4096x384x768(device):
    in0_shape = [1, 1, 4096, 384]
    in1_shape = [384, 768]
    bias_shape = [768]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (4096 // 32) // grid_size[0]
    per_core_N = (768 // 32) // grid_size[1]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=3,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=4,
        out_subblock_w=1,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


def test_RESNET_LINEAR2_1024x768x1536(device):
    in0_shape = [1, 1, 1024, 768]
    in1_shape = [768, 1536]
    bias_shape = [1536]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (1024 // 32) // grid_size[0]
    per_core_N = (1536 // 32) // grid_size[1]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=6,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=6,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


def test_RESNET_LINEAR2_256x3072x1536(device):
    in0_shape = [1, 1, 256, 3072]
    in1_shape = [3072, 1536]
    bias_shape = [1536]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (256 // 32) // grid_size[0]
    per_core_N = (1536 // 32) // grid_size[1]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=6,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=6,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


def test_RESNET_LINEAR2_1024x3072x1536(device):
    in0_shape = [1, 1, 1024, 3072]
    in1_shape = [3072, 1536]
    bias_shape = [1536]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (1024 // 32) // grid_size[0]
    per_core_N = (1536 // 32) // grid_size[1]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=12,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=3,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


def test_RESNET_LINEAR2_1024x2304x1536(device):
    in0_shape = [1, 1, 1024, 2304]
    in1_shape = [2304, 1536]
    bias_shape = [1536]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (1024 // 32) // grid_size[0]
    per_core_N = (1536 // 32) // grid_size[1]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=6,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=6,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


def test_RESNET_LINEAR2_4096x2304x768(device):
    in0_shape = [1, 1, 4096, 2304]
    in1_shape = [2304, 768]
    bias_shape = [768]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (4096 // 32) // grid_size[0]
    per_core_N = (768 // 32) // grid_size[1]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=6,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=2,
        out_subblock_w=3,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


def test_RESNET_LINEAR2_4096x1536x768(device):
    in0_shape = [1, 1, 4096, 1536]
    in1_shape = [1536, 768]
    bias_shape = [768]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (4096 // 32) // grid_size[0]
    per_core_N = (768 // 32) // grid_size[1]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=8,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=4,
        out_subblock_w=1,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


def test_RESNET_LINEAR2_4096x1152x768(device):
    in0_shape = [1, 1, 4096, 1152]
    in1_shape = [1152, 768]
    bias_shape = [768]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (4096 // 32) // grid_size[0]
    per_core_N = (768 // 32) // grid_size[1]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=6,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=2,
        out_subblock_w=3,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


def test_RESNET_LINEAR2_16384x1152x384(device):
    in0_shape = [1, 1, 16384, 1152]
    in1_shape = [1152, 384]
    bias_shape = [384]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 6)
    per_core_M = (16384 // 32) // grid_size[0]
    per_core_N = (384 // 32) // grid_size[1]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(6, 8),
        in0_block_w=2,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=2,
        out_subblock_w=2,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )


def test_RESNET_LINEAR2_16384x768x384(device):
    in0_shape = [1, 1, 16384, 768]
    in1_shape = [768, 384]
    bias_shape = [384]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (6, 8)
    per_core_M = (16384 // 32) // grid_size[1]
    per_core_N = (384 // 32) // grid_size[0]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=2,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=2,
        out_subblock_w=2,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


# 64 cores


def test_GEGLU_LINEAR1_4096x768x3072(device):
    in0_shape = [1, 1, 4096, 768]
    in1_shape = [768, 3072]
    bias_shape = [3072]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (4096 // 32) // grid_size[0]
    per_core_N = (3072 // 32) // grid_size[1]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=6,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=6,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def test_GEGLU_LINEAR1_1024x1536x6144(device):
    in0_shape = [1, 1, 1024, 1536]
    in1_shape = [1536, 6144]
    bias_shape = [6144]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (1024 // 32) // grid_size[0]
    per_core_N = (6144 // 32) // grid_size[1]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=8,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=8,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def test_GEGLU_LINEAR1_256x1536x6144(device):
    in0_shape = [1, 1, 256, 1536]
    in1_shape = [1536, 6144]
    bias_shape = [6144]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (256 // 32) // grid_size[0]
    per_core_N = (6144 // 32) // grid_size[1]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=4,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=6,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def test_FF_LINEAR_4096x3072x768(device):
    in0_shape = [1, 1, 4096, 3072]
    in1_shape = [3072, 768]
    bias_shape = [768]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (4096 // 32) // grid_size[0]
    per_core_N = (768 // 32) // grid_size[1]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=6,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=2,
        out_subblock_w=3,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def test_FF_LINEAR_1024x6144x1536(device):
    in0_shape = [1, 1, 1024, 6144]
    in1_shape = [6144, 1536]
    bias_shape = [1536]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (1024 // 32) // grid_size[0]
    per_core_N = (1536 // 32) // grid_size[1]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=16,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=6,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def test_FF_LINEAR_256x6144x1536(device):
    in0_shape = [1, 1, 256, 6144]
    in1_shape = [6144, 1536]
    bias_shape = [1536]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (8, 8)
    per_core_M = (256 // 32) // grid_size[0]
    per_core_N = (1536 // 32) // grid_size[1]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=12,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=1,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


# 40 cores


def test_GEGLU_LINEAR1_4096x768x3072_40_CORES(device):
    in0_shape = [1, 1, 4096, 768]
    in1_shape = [768, 3072]
    bias_shape = [3072]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (5, 8)
    per_core_M = (4096 // 32) // grid_size[1]
    per_core_N = math.ceil((3072 / 32) / grid_size[0])
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=6,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=5,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def test_GEGLU_LINEAR1_1024x1536x6144_40_CORES(device):
    in0_shape = [1, 1, 1024, 1536]
    in1_shape = [1536, 6144]
    bias_shape = [6144]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (5, 8)
    per_core_M = (1024 // 32) // grid_size[1]
    per_core_N = math.ceil((6144 / 32) / grid_size[0])
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=6,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=3,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def test_GEGLU_LINEAR1_256x1536x6144_40_CORES(device):
    in0_shape = [1, 1, 256, 1536]
    in1_shape = [1536, 6144]
    bias_shape = [6144]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (5, 8)
    per_core_M = (256 // 32) // 1
    per_core_N = math.ceil((6144 // 32) / (grid_size[0] * grid_size[1]))
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=6,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=5,
        mcast_in0=True,
        fuse_batch=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def test_FF_LINEAR_4096x3072x768_40_CORES(device):
    in0_shape = [1, 1, 4096, 3072]
    in1_shape = [3072, 768]
    bias_shape = [768]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (5, 8)
    per_core_M = (4096 // 32) // grid_size[1]
    per_core_N = math.ceil((768 / 32) / grid_size[0])
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=6,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=5,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def test_FF_LINEAR_1024x6144x1536_40_CORES(device):
    in0_shape = [1, 1, 1024, 6144]
    in1_shape = [6144, 1536]
    bias_shape = [1536]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (5, 8)
    per_core_M = (1024 // 32) // grid_size[1]
    per_core_N = math.ceil((1536 / 32) / grid_size[0])
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=6,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=5,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def test_FF_LINEAR_256x6144x1536_40_CORES(device):
    in0_shape = [1, 1, 256, 6144]
    in1_shape = [6144, 1536]
    bias_shape = [1536]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()
    bias_torch = torch.randn(bias_shape).bfloat16().float()

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    bias_tt = ttnn.from_torch(
        bias_torch, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (5, 8)
    per_core_M = (256 // 32) // grid_size[1]
    per_core_N = math.ceil((1536 / 32) / grid_size[0])
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=12,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=5,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.linear(
        in0_tt,
        in1_tt,
        bias=bias_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def test_ATTENTION_QKV_4096x768x2304_40_CORES(device):
    in0_shape = [1, 1, 4096, 768]
    in1_shape = [768, 2304]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()

    l1_interleaved_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    dram_interleaved_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=l1_interleaved_config,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_interleaved_config,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (5, 8)
    per_core_M = (4096 // 32) // grid_size[1]
    per_core_N = math.ceil((2304 / 32) / grid_size[0])
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=4,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=5,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.matmul(
        in0_tt,
        in1_tt,
        memory_config=l1_interleaved_config,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def test_ATTENTION_QKV_1024x1536x4608_40_CORES(device):
    in0_shape = [1, 1, 1024, 1536]
    in1_shape = [1536, 4608]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()

    l1_interleaved_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    dram_interleaved_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=l1_interleaved_config,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_interleaved_config,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (5, 8)
    per_core_M = (1024 // 32) // grid_size[1]
    per_core_N = math.ceil((4608 / 32) / grid_size[0])
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=8,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=4,
        out_subblock_w=1,
        transpose_mcast=False,
        fused_activation=None,
    )

    out = ttnn.matmul(
        in0_tt,
        in1_tt,
        memory_config=l1_interleaved_config,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )


def test_ATTENTION_QKV_256x1536x4608_40_CORES(device):
    in0_shape = [1, 1, 256, 1536]
    in1_shape = [1536, 4608]

    in0_torch = torch.randn(in0_shape).bfloat16().float()
    in1_torch = torch.randn(in1_shape).bfloat16().float()

    l1_interleaved_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )
    dram_interleaved_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )

    in0_tt = ttnn.from_torch(
        in0_torch,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=l1_interleaved_config,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_interleaved_config,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid_size = (5, 8)
    per_core_M = (256 // 32) // 1
    per_core_N = math.ceil((4608 / 32) / (grid_size[0] * grid_size[1]))
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=6,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_subblock_h=1,
        out_subblock_w=4,
        mcast_in0=True,
        fuse_batch=False,
        fused_activation=None,
    )

    out = ttnn.matmul(
        in0_tt,
        in1_tt,
        memory_config=l1_interleaved_config,
        dtype=ttnn.bfloat16,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
