# bs1 FF1 + FF3 + SwiGLU product at the model's call (captured with capture_qkv_call.py, CAP_N=9728): in0 bfp8
# block-sharded on 10x8, bfp4 weights DRAM width-sharded over the 8 banks, legacy 2D multicast on 12x8 (per core 2 x 26
# tiles), LoFi. Arms:
#   stock      FF1 / FF3 out L1 interleaved (2x2 subblocks), ttnn.mul(silu(a), b)   (the model before this change)
#   cheap_I    same outputs, silu_mul mode 3 (single-pass SwiGLU, bfp8-sized sigmoid)
#   cheap_BS   FF1 / FF3 out block-sharded on their 12x8 grid (needs 1x2 subblocks), silu_mul mode 3 on the shards
# Prints each arm's SwiGLU output error against an fp32 torch SwiGLU of the device's own FF1 / FF3 outputs, and with
# TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=<dir> each op's device time (device_kernel_us.py <dir>; three traces
# per arm in the order FF1, FF3, product).
import sys

import torch

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.silu_mul import silu_mul

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_common_traced import make_traced  # noqa: E402

D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 << 20)
traced = make_traced(D)
L1, B8, B4 = ttnn.L1_MEMORY_CONFIG, ttnn.bfloat8_b, ttnn.bfloat4_b


def grid(x, y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})


def sharded(layout, buf, g, shape):
    return ttnn.MemoryConfig(layout, buf, ttnn.ShardSpec(g, shape, ttnn.ShardOrientation.ROW_MAJOR))


in0_mc = sharded(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, grid(10, 8), [64, 256])
w_mc = sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, grid(8, 1), [2560, 1216])
out_bs = sharded(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, grid(12, 8), [64, 832])


def pc(sbh, sbw):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(12, 8),
        in0_block_w=8,
        out_subblock_h=sbh,
        out_subblock_w=sbw,
        out_block_h=2,
        out_block_w=26,
        per_core_M=2,
        per_core_N=26,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


ckc = ttnn.init_device_compute_kernel_config(
    D.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
)
torch.manual_seed(0)
x = ttnn.from_torch(torch.randn(1, 1, 512, 2560), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=in0_mc)
w1, w3 = (
    ttnn.from_torch(
        torch.randn(1, 1, 2560, 9728) * 0.03, dtype=B4, layout=ttnn.TILE_LAYOUT, device=D, memory_config=w_mc
    )
    for _ in range(2)
)
ARMS = {
    "stock": (L1, pc(2, 2), lambda a, b: ttnn.mul(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=B8, memory_config=L1)),
    "cheap_I": (L1, pc(2, 2), lambda a, b: silu_mul(a, b, out_dtype=B8, memory_config=L1, mode=3)),
    "cheap_BS": (out_bs, pc(1, 2), lambda a, b: silu_mul(a, b, out_dtype=B8, memory_config=L1, mode=3)),
}  # fmt: skip
try:
    for name, (mc, p, prod) in ARMS.items():
        ff = lambda w: (
            lambda: ttnn.linear(x, w, dtype=B8, compute_kernel_config=ckc, program_config=p, memory_config=mc)
        )
        a, b = ff(w1)(), ff(w3)()
        o = ttnn.to_torch(prod(a, b)).float()
        aq, bq = ttnn.to_torch(a).float(), ttnn.to_torch(b).float()
        gold = torch.nn.functional.silu(aq) * bq
        rel = ((o - gold).norm() / gold.norm()).item()
        pcc = torch.corrcoef(torch.stack([o.flatten(), gold.flatten()]))[0, 1].item()
        t1, t3, tp = traced(ff(w1)), traced(ff(w3)), traced(lambda: prod(a, b))
        print(
            f"RES {name:9s} FF1 {t1:6.1f} FF3 {t3:6.1f} product {tp:6.1f} us wall  rel_rmse {rel:.5f} pcc {pcc:.6f}",
            flush=True,
        )
        for t in (a, b):
            ttnn.deallocate(t)
finally:
    ttnn.close_device(D)
