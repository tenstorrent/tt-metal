# LOCAL EXPERIMENT (kmabee): single-chip repro for the SDPA matmul throughput gap (LLK issue draft).
# SDPA at Gemma4-31B's global-attention shape per device at chunk 8192 (8 Q heads x 1024 rows, 1 KV head, head dim
# 512, prefix 8192), streaming compute path (fp32_dest_acc_en=False), LoFi, q_chunk 128 / k_chunk 256; next to a
# plain ttnn.matmul with the same Q@K^T operand shapes. Run under tracy with G4X_SDPA_ZONES=1 for MM-ACQ / MM-LOOP
# compute zones; the signposts bracket the measured iterations.
import pytest
import torch
import ttnn
from tracy import signpost

Q_HEADS, SQ, SK, D = 8, 1024, 8192, 512


@pytest.mark.parametrize("q_chunk, k_chunk", [(128, 256)])
def test_g4x_sdpa_matmul_rate(device, q_chunk, k_chunk):
    torch.manual_seed(0)
    q = torch.randn(1, Q_HEADS, SQ, D)
    k = torch.randn(1, 1, SK, D)
    v = torch.randn(1, 1, SK, D)
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    grid = device.compute_with_storage_grid_size()
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk, exp_approx_mode=False
    )
    lofi = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )

    def sdpa():
        return ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, is_causal=False, program_config=program_config, compute_kernel_config=lofi
        )

    # The reference: a compute-bound regular matmul with the same operand dtypes, fidelity, DST mode and 2x4
    # subblocks as the SDPA matmuls: 1024 x 4096 x 4096 on an 8 x 8 grid, 4 x 16 output tiles per core, so
    # 8,192 tile-matmuls per core (SDPA: 32,768 per core on 64 cores).
    a = ttnn.from_torch(torch.randn(1, 1, 1024, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(1, 1, 4096, 4096), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    mm_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=8,
        out_subblock_h=2,
        out_subblock_w=4,
        per_core_M=4,
        per_core_N=16,
        transpose_mcast=False,
        fused_activation=None,
    )

    def qk_matmul():
        return ttnn.matmul(a, b, program_config=mm_config, compute_kernel_config=lofi)

    out = sdpa()  # compile + accuracy check
    ref = torch.nn.functional.scaled_dot_product_attention(q, k.expand(1, Q_HEADS, SK, D), v.expand(1, Q_HEADS, SK, D))
    pcc = torch.corrcoef(torch.stack((ttnn.to_torch(out).float().flatten(), ref.flatten())))[0, 1].item()
    assert pcc > 0.97, pcc  # LoFi + bfp8 K/V on random data lands near 0.98
    qk_matmul()
    ttnn.synchronize_device(device)
    signpost("start")
    for _ in range(3):
        sdpa()
        qk_matmul()
    ttnn.synchronize_device(device)
    signpost("stop")
