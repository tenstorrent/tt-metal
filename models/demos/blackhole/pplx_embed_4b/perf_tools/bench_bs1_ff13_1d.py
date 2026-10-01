# bs1 FF1 / FF3 ([512, 2560] x [2560, 9728] bfp8 x bfp4, LoFi) on the 1D in0-multicast matmul over 12x10 with the
# model's DRAM width-sharded weights (each core reads its N slice from its bank: the 1D factory's IN1_DRAM_WIDTH_SHARDED
# path, NEGATIVE_RESULTS 69), in0 width-sharded over 10 / 5 cores, 3 / 4 N tiles per core. Prints PCC vs torch and
# wall us; under TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=<dir>, device_kernel_us.py <dir> gives device us
# per config in print order. The model's 2D call is bench_bs1_mm_ablate.py FF1 (69.6 us device).
import sys

import torch

import ttnn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_common_traced import make_traced

D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 << 20)
tr = make_traced(D)
ckc = ttnn.init_device_compute_kernel_config(
    D.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
)
L1, B8, B4 = ttnn.L1_MEMORY_CONFIG, ttnn.bfloat8_b, ttnn.bfloat4_b
grid = ttnn.CoreCoord(12, 10)
K, N = 2560, 9728
torch.manual_seed(0)
xh, wh = torch.randn(1, 1, 512, K), torch.randn(1, 1, K, N) * 0.03
cr = lambda n: ttnn.num_cores_to_corerangeset(n, grid, row_wise=True)
banks = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
w = ttnn.from_torch(
    wh,
    dtype=B4,
    layout=ttnn.TILE_LAYOUT,
    device=D,
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(banks, [K, N // 8], ttnn.ShardOrientation.ROW_MAJOR),
    ),
)
wq = ttnn.to_torch(w).float()
for nsh, bws, pns in ((10, (8,), (3, 4)), (5, (8, 16), (3, 4))):
    x = ttnn.from_torch(
        xh,
        dtype=B8,
        layout=ttnn.TILE_LAYOUT,
        device=D,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cr(nsh), [512, K // nsh], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    gold = ttnn.to_torch(x).float() @ wq
    for pn in pns:
        for bw in bws:
            for sh, sw in ((8, 1), (2, pn), (1, pn)):
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=grid,
                    in0_block_w=bw,
                    out_subblock_h=sh,
                    out_subblock_w=sw,
                    out_block_h=16,
                    out_block_w=pn,
                    per_core_M=16,
                    per_core_N=pn,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
                fn = lambda: ttnn.linear(x, w, dtype=B8, compute_kernel_config=ckc, program_config=pc, memory_config=L1)
                try:
                    o = ttnn.to_torch(fn()).float()
                    pcc = torch.corrcoef(torch.stack([o.flatten(), gold.flatten()]))[0, 1].item()
                    print(f"RES in0_WS{nsh} pn{pn} bw{bw} sb{sh}x{sw} pcc {pcc:.5f} wall {tr(fn):.1f}", flush=True)
                except Exception as e:
                    print(f"RES in0_WS{nsh} pn{pn} bw{bw} sb{sh}x{sw} FAILED {str(e)[:200]}", flush=True)
    ttnn.deallocate(x)
ttnn.close_device(D)
