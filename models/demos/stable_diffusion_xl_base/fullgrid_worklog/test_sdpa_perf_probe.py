# SDPA fidelity perf probe (run under test_profile_probe: PROBE_CMD='pytest <this file>' PROBE_OPS=Scaled).
import pytest
import torch

import ttnn


@pytest.mark.parametrize(
    "shape,kchunk",
    [((1, 20, 1024, 64), 1024), ((1, 10, 4096, 64), 512), ((1, 20, 1024, 64), 96), ((1, 10, 4096, 64), 96)],
)
def test_sdpa_perf(device, shape, kchunk):
    B, H, S, D = shape
    q, k, v = (torch.randn(shape) for _ in range(3))
    if kchunk == 96:  # cross-attention: 77 encoder tokens padded to 96
        k, v = (torch.randn((B, H, 96, D)) for _ in range(2))
        kchunk = 128 if S == 1024 else 128
    tq, tk, tv = [ttnn.from_torch(t, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT) for t in (q, k, v)]
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(11, 10), q_chunk_size=128, k_chunk_size=kchunk, exp_approx_mode=False
    )
    for fid in ("LoFi", "HiFi2", "HiFi3", "HiFi4"):
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, fid),
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        for _ in range(3):
            out = ttnn.transformer.scaled_dot_product_attention(
                tq,
                tk,
                tv,
                is_causal=False,
                program_config=pc,
                compute_kernel_config=ckc,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(out)
    ttnn.synchronize_device(device)
