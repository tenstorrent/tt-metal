# bs1 projections at the model's calls (capture_qkv_call.py 1, CAP_N=6144,2560,9728): the legacy 2D-multicast matmul on
# 12x8, bfp4 weights DRAM width-sharded over the 8 banks, LoFi, l1_acc. QKV / FF1 read a 10x8 block-sharded bfp8 in0,
# WO / FF2 an L1-interleaved one. Run once per kernel variant (mm_legacy_variants.py) under the device profiler:
#   cd <dir without a ttnn/ tree>; TT_METAL_KERNEL_PATH=<out>/<variant> TT_METAL_CACHE=<out>/cache_<variant> \
#     TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=<prof> python bench_bs1_mm_ablate.py [op ...]
#   device_kernel_us.py <prof> QKV WO FF1 FF2
# Prints each op's PCC against torch (a patched variant shows ~0: proof the patch compiled in) and wall us per call.
# MM_SWEEP="op:in0_block_w,sbh,sbw;..." overrides a config.
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common_traced import make_traced  # noqa: E402

B8, B4, BF = ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat16


def grid(x, y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})


def mc(layout, buf, g, shape):
    return ttnn.MemoryConfig(layout, buf, ttnn.ShardSpec(g, shape, ttnn.ShardOrientation.ROW_MAJOR))


IN0_BS = mc(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, grid(10, 8), [64, 256])
L1 = ttnn.L1_MEMORY_CONFIG
# name: K, N, in0 memory, out dtype, in0_block_w, subblock h, w, per_core_N, fuse_batch
OPS = {
    "QKV": (2560, 6144, IN0_BS, B8, 8, 1, 4, 16, True),
    "WO": (4096, 2560, L1, B8, 16, 2, 1, 7, True),
    "FF1": (2560, 9728, IN0_BS, B8, 8, 2, 2, 26, True),
    "FF2": (9728, 2560, L1, BF, 38, 2, 1, 7, False),
}


def main():
    names = sys.argv[1:] or list(OPS)
    over = dict(kv.split(":") for kv in os.getenv("MM_SWEEP", "").split(";") if kv)
    D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 << 20)
    traced = make_traced(D)
    ckc = ttnn.init_device_compute_kernel_config(
        D.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    try:
        torch.manual_seed(0)
        for name in names:
            K, N, in0_mc, odt, bw, sbh, sbw, pn, fb = OPS[name]
            if name in over:
                bw, sbh, sbw = (int(v) for v in over[name].split(","))
            xh, wh = torch.randn(1, 1, 512, K), torch.randn(1, 1, K, N) * 0.03
            x = ttnn.from_torch(xh, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=in0_mc)
            w_mc = mc(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, grid(8, 1), [K, -(-N // 256) * 32])
            w = ttnn.from_torch(wh, dtype=B4, layout=ttnn.TILE_LAYOUT, device=D, memory_config=w_mc)
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(12, 8),
                in0_block_w=bw,
                out_subblock_h=sbh,
                out_subblock_w=sbw,
                out_block_h=2,
                out_block_w=pn,
                per_core_M=2,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=fb,
            )
            fn = lambda: ttnn.linear(x, w, dtype=odt, compute_kernel_config=ckc, program_config=pc, memory_config=L1)
            o = ttnn.to_torch(fn()).float()
            gold = ttnn.to_torch(x).float() @ ttnn.to_torch(w).float()
            pcc = torch.corrcoef(torch.stack([o.flatten(), gold.flatten()]))[0, 1].item()
            us = traced(fn)
            print(f"RES {name:4s} bw{bw} sb{sbh}x{sbw} pcc {pcc:.5f} wall {us:6.1f} us", flush=True)
            for t in (x, w):
                ttnn.deallocate(t)
    finally:
        ttnn.close_device(D)


if __name__ == "__main__":
    main()
