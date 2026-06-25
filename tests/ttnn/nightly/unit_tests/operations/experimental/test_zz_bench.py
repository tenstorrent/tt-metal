import os
import pytest
import torch
import ttnn


@pytest.mark.timeout(1200)
def test_bench(device):
    # Production-default path: NO explicit MinimalMatmulConfig (factory picks 8/8/8 + fp32->sb2x2).
    # Dataflow variant is selected purely by TT_MM_* env flags. Shape from env. Loops FC_REPS times;
    # tracy captures per-invocation device kernel duration. One PCC check at the end (BENCH_PCC=1).
    M = int(os.environ.get("FL_M", "4096"))
    K = int(os.environ.get("FL_K", "4096"))
    N = int(os.environ.get("FL_N", "4096"))
    reps = int(os.environ.get("FC_REPS", "20"))
    torch.manual_seed(0)
    ti = torch.randn((M, K), dtype=torch.bfloat16)
    wi = torch.randn((K, N), dtype=torch.bfloat16)
    tt_i = ttnn.from_torch(ti, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_w = ttnn.from_torch(wi, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    # Optional explicit block (FC_MB set) -> disables the gate -> unicast at that block. Else default
    # path (gate auto-fires). TT_MM_NUM_SLICES applies either way.
    kw = {}
    if "FC_MB" in os.environ:
        kw["config"] = ttnn.MinimalMatmulConfig(
            M_block_size=int(os.environ["FC_MB"]),
            K_block_size=int(os.environ["FC_KB"]),
            N_block_size=int(os.environ["FC_NB"]),
            subblock_h=int(os.environ["FC_SBH"]),
            subblock_w=int(os.environ["FC_SBW"]),
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        )
    out = None
    for _ in range(reps):
        if out is not None:
            out.deallocate()
        out = ttnn.experimental.minimal_matmul(tt_i, tt_w, compute_kernel_config=cc, **kw)
        ttnn.synchronize_device(device)
    if os.environ.get("BENCH_PCC", "0") == "1":
        got = ttnn.to_torch(out).float()
        ref = ti.float() @ wi.float()
        pcc = torch.corrcoef(torch.stack([ref.flatten().float(), got.flatten().float()]))[0, 1].item()
        print(f"BENCH_PCC M{M}K{K}N{N} pcc={pcc:.5f}")
        assert pcc > 0.99, f"PCC too low: {pcc}"
    out.deallocate()
