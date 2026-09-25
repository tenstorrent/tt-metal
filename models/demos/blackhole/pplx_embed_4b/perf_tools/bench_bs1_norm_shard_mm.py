# bs1: can QKV / FF1 / FF3 (legacy 2D mcast, 12x8) read the RMSNorm's 10x8 block-sharded output directly, instead of
# ShardedToInterleaved + interleaved in0? The 2D factory sizes its in0 senders from the shard grid's width, so a
# 10-column shard feeds a 12-column matmul as long as in0_block_w divides the 8-tile shard width (the model's
# in0_block_w is 10). Times each arm as a traced chain and checks the outputs against the current path.
# Usage: TT_VISIBLE_DEVICES=<chip> bench_bs1_norm_shard_mm.py
import math
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common_traced import make_traced

B4, B8, T, M, K = ttnn.bfloat4_b, ttnn.bfloat8_b, 32, 512, 2560
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
traced = make_traced(D)
ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)
L1 = ttnn.L1_MEMORY_CONFIG
# the prefill RMSNorm's layout at bs1 (get_prefill_block_sharded_norm_config): 2x8 tiles per core on 10x8
NORM_SHARD = ttnn.create_sharded_memory_config(
    shape=(2 * T, 8 * T),
    core_grid=ttnn.CoreGrid(y=8, x=10),
    strategy=ttnn.ShardStrategy.BLOCK,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)


def sharded_w(N):
    pad = math.ceil(N / (T * 8)) * (T * 8)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
    spec = ttnn.ShardSpec(grid, (K, pad // 8), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.from_torch(
        torch.randn(1, 1, K, N) * 0.02,
        dtype=B4,
        layout=ttnn.TILE_LAYOUT,
        device=D,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec),
    )


def pc(bw, sb, pn, fuse_batch):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(12, 8),
        in0_block_w=bw,
        out_subblock_h=sb[0],
        out_subblock_w=sb[1],
        per_core_M=2,
        per_core_N=pn,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=fuse_batch,
    )


def mm(x, w, c):
    return ttnn.matmul(x, w, program_config=c, compute_kernel_config=ckc, memory_config=L1, dtype=B8)


def pcc(a, b):
    a, b = ttnn.to_torch(a).float().flatten(), ttnn.to_torch(b).float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


try:
    torch.manual_seed(0)
    xi = ttnn.from_torch(torch.randn(1, 1, M, K), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    xs = ttnn.to_memory_config(xi, NORM_SHARD)
    s2i_us = traced(lambda: ttnn.sharded_to_interleaved(xs, L1))
    print(f"[S2I alone] {s2i_us:6.1f} us", flush=True)
    # (name, N, per_core_N, subblock, model fuse_batch) as the bs1 profile records them (in0_block_w 10)
    cases = [("QKV", 6144, 16, (1, 4), True), ("FF1", 9728, 26, (2, 2), False)]
    ws = {}
    for name, N, pn, sb, fb in cases:
        w = ws[name] = sharded_w(N)
        cur = pc(10, sb, pn, fb)
        ref = mm(ttnn.sharded_to_interleaved(xs, L1), w, cur)
        arms = [
            ("current: S2I + mm(in0 L1 interleaved, bw10)", lambda: mm(ttnn.sharded_to_interleaved(xs, L1), w, cur)),
            ("  mm alone (interleaved, bw10)", lambda: mm(xi, w, cur)),
            ("  mm alone (interleaved, bw8)", lambda: mm(xi, w, pc(8, sb, pn, fb))),
        ]
        for bw in (8, 4):
            arms.append((f"sharded in0, bw{bw}", lambda bw=bw: mm(xs, w, pc(bw, sb, pn, True))))
        print(f"[{name} K={K} N={N} per_core_N={pn} sb={sb[0]}x{sb[1]}]", flush=True)
        for tag, fn in arms:
            try:
                p = pcc(fn(), ref)
                print(f"    {tag:46s} {traced(fn):6.1f} us  pcc {p:.6f}", flush=True)
            except Exception as e:
                print(f"    {tag:46s} FAILED: {str(e).splitlines()[0][:150]}", flush=True)
    # the MLP side as the model runs it: one norm output feeds FF1 and FF3
    w1, w3 = ws["FF1"], sharded_w(9728)
    cur, sh = pc(10, (2, 2), 26, False), pc(8, (2, 2), 26, True)

    def ff_cur():
        x = ttnn.sharded_to_interleaved(xs, L1)
        a, b = mm(x, w1, cur), mm(x, w3, cur)
        ttnn.deallocate(b)
        return a

    def ff_sh():
        a, b = mm(xs, w1, sh), mm(xs, w3, sh)
        ttnn.deallocate(b)
        return a

    print(f"[FF1+FF3 chain] current {traced(ff_cur):6.1f} us   sharded bw8 {traced(ff_sh):6.1f} us", flush=True)
    print("[done]")
finally:
    ttnn.close_device(D)
