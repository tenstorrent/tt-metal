# QKV projection + fused heads op, the current path (minimal_matmul into an L1 QKV intermediate, then the heads op
# writing Q / K / V) vs the head-major path (minimal_matmul_split writing Q / K / V head-major directly, then the heads op
# normalising / rotating Q and K in place): Q / K / V bit-identity and traced us per call of each pair. bs8 / 16 shapes
# (Q in L1 when HQ_L1=1, K / V in L1). Usage: bench_qkv_head_major.py <batch>
import os
import statistics
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_mm_ablate import preset  # noqa: E402

from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm import op as hop  # noqa: E402
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm.constants import (  # noqa: E402
    make_norm_constants,
)
from models.tt_transformers.tt.common import get_rot_transformation_mat  # noqa: E402

NH, NKV, DH, EPS, S = 32, 8, 128, 1e-6, 512
B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
M, K, N, blocks, _, wlayout, _, *rest = preset("qkv", B)
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
L1, DR, B8 = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat8_b
try:
    torch.manual_seed(0)
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=blocks[0], K_block_size=blocks[1], N_block_size=blocks[2], subblock_h=blocks[3],
        subblock_w=blocks[4], compute_with_storage_grid_size=ttnn.CoreCoord(12, 10),
    )  # fmt: skip
    ckc = ttnn.init_device_compute_kernel_config(
        D.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    pad = -(-N // 256) * 256
    w_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            [K, pad // 8],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, K, N) * 0.02, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=D, memory_config=w_mc
    )
    x = ttnn.from_torch(torch.randn(1, 1, M, K), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    GQ, GK, SC, EP = make_norm_constants(torch.rand(DH) + 0.5, torch.rand(DH) + 0.5, EPS, D)
    ang = torch.rand(1, 1, S, DH) * 6.28
    tl = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    cos, sin, T = tl(torch.cos(ang)), tl(torch.sin(ang)), tl(get_rot_transformation_mat(32))
    qmc = L1 if os.getenv("HQ_L1", "0") == "1" else DR
    alloc = lambda shp, mc: ttnn.allocate_tensor_on_device(ttnn.Shape(shp), B8, ttnn.TILE_LAYOUT, D, mc)
    mk = lambda: (alloc([B, NH, S, DH], qmc), alloc([B, NKV, S, DH], L1), alloc([B, NKV, S, DH], L1))
    # the two paths run one after the other, each with its own Q / K / V (both sets in L1 at once do not fit at bs16)
    qkv_a = mk() if os.getenv("HM_MATMUL_ONLY") != "1" else None
    qkv_b = None
    heads_kw = dict(num_heads=NH, num_kv_heads=NKV, memory_config=qmc, kv_memory_config=L1, rot_cos=cos, rot_sin=sin,
                    trans_mat=T, q_dtype=B8, kv_dtype=B8, norm_eps=EPS, use_v3=True)  # fmt: skip
    St, Wt = S // 32, DH // 32
    chunk_heads = 8
    n_chunks = (NH + 2 * NKV) // chunk_heads
    hm = [St, Wt]
    for c in range(n_chunks):
        h0 = c * chunk_heads
        if h0 < NH:
            hm += [h0, NH]
        else:
            hm += [0, NKV]
    outs_b = None

    def old():
        y = ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ckc, config=cfg, memory_config=L1, dtype=B8)
        r = hop.nlp_create_qkv_heads_norm_headsplit(
            ttnn.reshape(y, [B, 1, S, N]), GQ, GK, SC, EP, out_tensors=qkv_a, batch_offset=0, **heads_kw
        )
        ttnn.deallocate(y)
        return r

    def new():
        ttnn.experimental.minimal_matmul_split(
            x, w, chunks=n_chunks, config=cfg, memory_config=L1, dtype=B8, compute_kernel_config=ckc,
            output_tensors=outs_b, head_major=hm,
        )  # fmt: skip
        return hop.nlp_create_qkv_heads_norm_headsplit(
            qkv_b[0], GQ, GK, SC, EP, out_tensors=qkv_b, batch_offset=0, inplace=True, **heads_kw
        )

    if os.getenv("HM_MATMUL_ONLY") == "1":  # the head-major matmul alone vs the row-major one, rearranged on the host
        y = ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ckc, config=cfg, memory_config=L1, dtype=B8)
        ref = ttnn.to_torch(y).float().reshape(B, S, NH + 2 * NKV, DH).permute(0, 2, 1, 3)
        ttnn.deallocate(y)
        qkv_b = mk()
        outs_b = [qkv_b[0]] * (NH // chunk_heads) + [qkv_b[1], qkv_b[2]]
        ttnn.experimental.minimal_matmul_split(
            x, w, chunks=n_chunks, config=cfg, memory_config=L1, dtype=B8, compute_kernel_config=ckc,
            output_tensors=outs_b, head_major=hm,
        )  # fmt: skip
        for i, (lo, hi) in enumerate(((0, NH), (NH, NH + NKV), (NH + NKV, NH + 2 * NKV))):
            got = ttnn.to_torch(qkv_b[i]).float()
            exp = ref[:, lo:hi]
            bad = (got != exp).reshape(B, hi - lo, S // 32, 32, DH // 32, 32).any(-1).any(-2)
            idx = bad.nonzero()[:6].tolist()
            print(
                f"RES matmul-only {'QKV'[i]} equal {torch.equal(got, exp)} bad tiles {int(bad.sum())} first [b,h,s,j] {idx}",
                flush=True,
            )
        raise SystemExit

    def timed(label, fn):
        for _ in range(2):
            fn()
        ttnn.synchronize_device(D)
        n = 6
        tid = ttnn.begin_trace_capture(D, cq_id=0)
        for _ in range(n):
            fn()
        ttnn.end_trace_capture(D, tid, cq_id=0)
        ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
        ts = []
        for _ in range(7):
            t0 = time.perf_counter()
            ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) / n * 1e6)
        ttnn.release_trace(D, tid)
        print(
            f"RES B{B} Q={'L1' if qmc is L1 else 'DRAM'} {label:36s} {statistics.median(ts):8.1f} us per pair",
            flush=True,
        )

    try:
        old()
        old_fits = True
    except RuntimeError:  # bs16 with Q in L1: the QKV intermediate does not fit beside Q / K / V (the point of this)
        old_fits = False
        [ttnn.deallocate(t) for t in qkv_a]
        qmc_ref = DR
        qkv_a = (alloc([B, NH, S, DH], DR), alloc([B, NKV, S, DH], L1), alloc([B, NKV, S, DH], L1))
        heads_kw_ref = dict(heads_kw, memory_config=DR)
        y = ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ckc, config=cfg, memory_config=L1, dtype=B8)
        hop.nlp_create_qkv_heads_norm_headsplit(
            ttnn.reshape(y, [B, 1, S, N]), GQ, GK, SC, EP, out_tensors=qkv_a, batch_offset=0, **heads_kw_ref
        )
        ttnn.deallocate(y)
        print(f"RES B{B} old path does not fit with Q in L1 (reference values from Q in DRAM)", flush=True)
    ref = [ttnn.to_torch(t).float() for t in qkv_a]
    if old_fits:
        timed("matmul + heads op", old)
    [ttnn.deallocate(t) for t in qkv_a]
    qkv_b = mk()
    outs_b = [qkv_b[0]] * (NH // chunk_heads) + [qkv_b[1], qkv_b[2]]
    new()
    for i, name in enumerate(("Q", "K", "V")):
        got = ttnn.to_torch(qkv_b[i]).float()
        print(
            f"RES B{B} {name} bit-identical {torch.equal(ref[i], got)} max|diff| {float((ref[i] - got).abs().max()):.4g}",
            flush=True,
        )
    timed("head-major matmul + in-place heads", new)
    timed(
        "head-major matmul only",
        lambda: ttnn.experimental.minimal_matmul_split(
            x, w, chunks=n_chunks, config=cfg, memory_config=L1, dtype=B8, compute_kernel_config=ckc,
            output_tensors=outs_b, head_major=hm,
        ),
    )  # fmt: skip
    timed(
        "in-place heads only",
        lambda: hop.nlp_create_qkv_heads_norm_headsplit(
            qkv_b[0], GQ, GK, SC, EP, out_tensors=qkv_b, batch_offset=0, inplace=True, **heads_kw
        ),
    )
    [ttnn.deallocate(t) for t in qkv_b]
    timed(
        "row-major matmul only (L1 out)",
        lambda: ttnn.deallocate(
            ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ckc, config=cfg, memory_config=L1, dtype=B8)
        ),
    )
finally:
    ttnn.close_device(D)
