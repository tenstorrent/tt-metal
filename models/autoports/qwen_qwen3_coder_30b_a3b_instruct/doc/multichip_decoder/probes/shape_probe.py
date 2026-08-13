# Shape/legality/scaling probes for the multichip plan. Single device is enough.
import statistics
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.optimized_decoder import (
    _DRAM_BANKS,
    EXPERT_IN0_BLOCK_W_DOWN,
    EXPERT_IN0_BLOCK_W_GATE_UP,
    _bank_row,
    _dram_sharded_program_config,
    _expert_compute_kernel_config,
    _tuned_sparse_matmul_config,
    _width_sharded_l1,
)

dev = ttnn.open_device(device_id=0, trace_region_size=100_000_000, l1_small_size=32768)
H, I = 2048, 768


def timed(fn, reps=17, iters=20):
    o = fn()
    ttnn.synchronize_device(dev)
    ttnn.deallocate(o)

    def cap(r):
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(r):
            ttnn.deallocate(fn())
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) * 1e6)
        ttnn.release_trace(dev, tid)
        return statistics.median(ts)

    a, b = cap(1), cap(reps)
    return (b - a) / (reps - 1)


def expert_case(E, inter, batch, active_total, use_nnz):
    """gate_up + down sparse matmuls with E expert slots, `active_total` nonzeros."""
    gate_up = ttnn.from_torch(
        torch.randn(1, E, H, 2 * inter),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    down = ttnn.from_torch(
        torch.randn(1, E, inter, H),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x = ttnn.from_torch(
        torch.randn(1, batch, 1, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sp = torch.zeros(1, 1, batch, E)
    per = active_total // batch
    for b in range(batch):
        sp[0, 0, b, torch.randperm(E)[:per]] = 1.0
    sparsity = ttnn.from_torch(sp, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    nnz = per * batch if use_nnz else None
    cc = _expert_compute_kernel_config(dev)
    gu_cfg = _tuned_sparse_matmul_config(1, 2 * inter, H, EXPERT_IN0_BLOCK_W_GATE_UP)
    dn_cfg = _tuned_sparse_matmul_config(1, H, inter, EXPERT_IN0_BLOCK_W_DOWN)
    tile = ttnn.Tile([32, 32])

    def f_gu():
        return ttnn.sparse_matmul(
            x,
            gate_up,
            sparsity=sparsity,
            nnz=nnz,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=tile,
            program_config=gu_cfg,
            compute_kernel_config=cc,
            dtype=ttnn.bfloat16,
        )

    di = ttnn.from_torch(
        torch.randn(batch, E, 1, inter),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def f_dn():
        return ttnn.sparse_matmul(
            di,
            down,
            sparsity=sparsity,
            nnz=nnz,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=tile,
            program_config=dn_cfg,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            compute_kernel_config=cc,
            dtype=ttnn.bfloat16,
        )

    try:
        t_gu = timed(f_gu)
    except Exception as e:
        t_gu = f"ERR {str(e)[:110]}"
    try:
        t_dn = timed(f_dn)
    except Exception as e:
        t_dn = f"ERR {str(e)[:110]}"
    for t in (gate_up, down, x, sparsity, di):
        ttnn.deallocate(t)
    return t_gu, t_dn


print("P|=== expert sparse_matmul scaling (batch=1, M=1) ===")
for E, inter, act, use_nnz, label in [
    (128, 768, 8, True, "baseline single-die E=128 nnz=8"),
    (32, 768, 2, True, "EP4 E=32 nnz=2 (mean load)"),
    (32, 768, 4, True, "EP4 E=32 nnz=4 (tail load)"),
    (32, 768, 8, True, "EP4 E=32 nnz=8 (worst case)"),
    (32, 768, 2, False, "EP4 E=32 nnz=None (dynamic)"),
    (128, 768, 8, False, "E=128 nnz=None (dynamic)"),
    (128, 192, 8, True, "TP4 expert-intermediate E=128 I=192"),
]:
    gu, dn = expert_case(E, inter, 1, act, use_nnz)
    g = f"{gu:.2f}" if isinstance(gu, float) else gu
    d = f"{dn:.2f}" if isinstance(dn, float) else dn
    print(f"P|{label:42s} gate_up={g}us down={d}us")

print("P|=== TP=4 attention shape legality ===")
for nh, nkv, kqkv, nqkv, ko, no, tag in [
    (32, 4, 2048, 5120, 4096, 2048, "single-die"),
    (8, 1, 2048, 1280, 1024, 2048, "TP4 per-die"),
]:
    try:
        w = ttnn.from_torch(
            torch.randn(1, 1, kqkv, nqkv),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(_bank_row(_DRAM_BANKS), [kqkv, nqkv // _DRAM_BANKS], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        xs = ttnn.from_torch(
            torch.randn(1, 1, 32, kqkv),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=_width_sharded_l1(kqkv),
        )
        f = lambda: ttnn.linear(
            xs,
            w,
            program_config=_dram_sharded_program_config(kqkv, nqkv),
            memory_config=_width_sharded_l1(nqkv),
            dtype=ttnn.bfloat16,
        )
        t = timed(f)
        print(f"P|{tag} qkv K={kqkv} N={nqkv} DRAM-sharded OK t={t:.2f}us")
        out = f()
        outl = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            outl, num_heads=nh, num_kv_heads=nkv, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        print(
            f"P|{tag} nlp_create_qkv_heads_decode nh={nh} nkv={nkv} -> q{list(q.shape)} k{list(k.shape)} v{list(v.shape)}"
        )
        for t_ in (out, outl, q, k, v, w, xs):
            ttnn.deallocate(t_)
    except Exception as e:
        print(f"P|{tag} qkv ERR {str(e)[:200]}")
    try:
        w = ttnn.from_torch(
            torch.randn(1, 1, ko, no),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(_bank_row(_DRAM_BANKS), [ko, no // _DRAM_BANKS], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        xs = ttnn.from_torch(
            torch.randn(1, 1, 32, ko),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=_width_sharded_l1(ko),
        )
        f = lambda: ttnn.linear(
            xs,
            w,
            program_config=_dram_sharded_program_config(ko, no),
            memory_config=_width_sharded_l1(no),
            dtype=ttnn.bfloat16,
        )
        t = timed(f)
        print(f"P|{tag} wo K={ko} N={no} DRAM-sharded OK t={t:.2f}us")
        ttnn.deallocate(w)
        ttnn.deallocate(xs)
    except Exception as e:
        print(f"P|{tag} wo ERR {str(e)[:200]}")

ttnn.close_device(dev)
