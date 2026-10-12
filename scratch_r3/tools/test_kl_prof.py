# Regression audit #58817, headroom: one-tile sub blocks with several k steps per block in production configs: the
# DeepSeek V3 d_p / GPT-OSS 120B MoE gate prefill (1D, 11x10 grid, in0_block_w 23, 1x1, HiFi4, fp32 DEST) and the
# Qwen3.6 decode out-proj shape (2D 8x6 grid, 1x1, in0_block_w 4). Device profiler, one launch per repetition.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
import math
import pytest
import torch
import ttnn


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


def run(device, m, k, n, pc, ckc, dt1=ttnn.bfloat16):
    torch.manual_seed(0)
    a = ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.1, dtype=dt1, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.matmul(a, b, program_config=pc, compute_kernel_config=ckc)
    ttnn.synchronize_device(device)
    out.deallocate(); a.deallocate(); b.deallocate()


def test_kl_gate_prefill(device):
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10), in0_block_w=23, out_subblock_h=1, out_subblock_w=1,
        out_block_h=1, out_block_w=4, per_core_M=1, per_core_N=4, fuse_batch=True, fused_activation=None, mcast_in0=False,
    )
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    run(device, 640, 736, 128, pc, ckc)


@pytest.mark.parametrize("n", [2048, 4096])
def test_kl_qwen_outproj_decode(device, n):
    cg = (8, 6); m, k = 32, 4096
    pcn = max(1, math.ceil(n / 32 / cg[0]))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=cg, in0_block_w=min(4, max(1, k // 32 // cg[0])), out_subblock_h=1, out_subblock_w=1,
        per_core_M=max(1, math.ceil(m / 32 / cg[1])), per_core_N=pcn, out_block_w=max(1, pcn // 2), transpose_mcast=False,
        fused_activation=None, fuse_batch=False,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    run(device, m, k, n, pc, ckc, ttnn.bfloat8_b)
