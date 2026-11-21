# The GDN a/b projection (5120 -> 96, dense) as one matmul: which program config is fast?

import pytest
import torch

import ttnn


@pytest.mark.parametrize("cfg", ["auto", "w4", "w8", "w16", "w32", "w40"])
def test_ab(device, cfg):
    x = ttnn.from_torch(
        torch.randn(1, 1, 5120).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    w = ttnn.from_torch(torch.randn(5120, 96).bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    pc = None
    if cfg != "auto":
        bw = int(cfg[1:])
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(3, 1),
            in0_block_w=bw,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    ref = ttnn.to_torch(ttnn.linear(x, w, compute_kernel_config=ckc)).float()
    y = ttnn.linear(x, w, compute_kernel_config=ckc, program_config=pc)
    err = ((ttnn.to_torch(y).float() - ref).norm() / ref.norm()).item()
    for _ in range(5):
        ttnn.linear(x, w, compute_kernel_config=ckc, program_config=pc)
    ttnn.synchronize_device(device)
    print(f"reinit=ab {cfg} err {err:.4f}")
