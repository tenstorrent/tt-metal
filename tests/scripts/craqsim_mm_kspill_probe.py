# Single-core K-spill probe for ttnn.experimental.quasar.matmul (1D mcast factory), sized for the emulator.
# usage: python mm_kspill_probe.py <K> [N=64] [M=32]   -> K=2048 gives 31 partials round trips with in0_block_w=2; K=128 gives 1; K=64 gives 0
import sys, torch, ttnn
from models.experimental.llama32_1b_quasar.utility_functions import comp_pcc

K = int(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
M = int(sys.argv[3]) if len(sys.argv) > 3 else 32
Q = ttnn._ttnn.operations.experimental.quasar
torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
g = ttnn.CoreCoord(1, 1)  # force a single core so the shape is identical on sim and emulator
print(f"PROBE M={M} K={K} N={N} grid=1x1 device_grid={dev.compute_with_storage_grid_size()}", flush=True)


def to_dev(t):
    return ttnn.to_device(
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


x = torch.randn(1, 1, M, K).to(torch.bfloat16)
w = torch.randn(K, N).to(torch.bfloat16)
xt, wt = to_dev(x), to_dev(w)
Mt, Kt, Nt = M // 32, K // 32, N // 32
in0_block_w = 2 if Kt % 2 == 0 else 1
pc = Q.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=g,
    in0_block_w=in0_block_w,
    out_subblock_h=1,
    out_subblock_w=Nt,
    out_block_h=Mt,
    out_block_w=Nt,
    per_core_M=Mt,
    per_core_N=Nt,
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=True,
)
print(
    f"PROBE k_blocks={Kt//in0_block_w} partials_round_trips={Kt//in0_block_w - 1} out_block_tiles={Mt*Nt}", flush=True
)
ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
)
out = ttnn.experimental.quasar.matmul(xt, wt, program_config=pc, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
got = ttnn.to_torch(out).float()
ref = x.float() @ w.float()
print("PROBE_RESULT", f"K={K}", comp_pcc(ref, got, 0.99), flush=True)
ttnn.close_device(dev)
