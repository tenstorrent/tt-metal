#!/usr/bin/env python3
"""Scratch, not for merge. SDPA op-level device time at production Blackhole shapes, run identically on each side's own
ttnn build. Per case: setup from a fixed seed, N_WARM warm-up calls (the first output saved for the bits check when
SDPA_BITS is set), a signpost, N_MEAS measured calls, then ReadDeviceProfiler. Configs: G8 = (8, 8) grid (main's model
path), GF = full grid (#57395's prefill model path), GFS = full grid with the Q block first in sub_core_grids (#57395's
decode model path from batch 8), CPB16 / CPB4 = max_cores_per_head_batch (main / #57395's DeepSeek MLA decode)."""
import gc
import hashlib
import math
import os

import torch
import ttnn
from tracy import signpost

N_WARM, N_MEAS = 1, int(os.environ.get("SDPA_NMEAS", "5"))
BITS = os.environ.get("SDPA_BITS", "")


def ckc(fid, approx, fp32, l1acc):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, fid), math_approx_mode=approx, fp32_dest_acc_en=fp32, packer_l1_acc=l1acc
    )


def pcfg(dev, grid, q, k, approx=False, cpb=None):
    g = dev.compute_with_storage_grid_size()
    kw = {}
    if grid == "G8":
        gs = (8, 8)
    elif grid == "GJ":  # tt_dit joint attention grid
        gs = (g.x, g.y - 1)
    else:
        gs = (g.x, g.y)
    if grid == "GFS":
        kw["sub_core_grids"] = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3)),
                ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(g.x - 1, 3)),
                ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(g.x - 1, g.y - 1)),
            ]
        )
    if cpb is not None:
        kw["max_cores_per_head_batch"] = cpb
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=gs, q_chunk_size=q, k_chunk_size=k, exp_approx_mode=approx, **kw
    )


def T(dev, x, dtype, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(x, dtype=dtype, layout=layout, device=dev, memory_config=mem)


def rnd(*shape):
    return torch.randn(shape, dtype=torch.bfloat16)


def llama_prefill(nq, nkv, S, grid):
    ch = 256 if S >= 2048 else 64

    def setup(dev):
        q, k, v = rnd(1, nq, S, 128), rnd(1, nkv, S, 128), rnd(1, nkv, S, 128)
        return [T(dev, q, ttnn.bfloat8_b), T(dev, k, ttnn.bfloat8_b), T(dev, v, ttnn.bfloat8_b)]

    def run(dev, a):
        return ttnn.transformer.scaled_dot_product_attention(
            a[0], a[1], a[2], is_causal=True, scale=1 / math.sqrt(128),
            program_config=pcfg(dev, grid, ch, ch), compute_kernel_config=ckc("HiFi4", False, True, True))

    return setup, run


def llama_chunked(nq, nkv, S, start, grid):
    def setup(dev):
        total = start + S
        nb = total // 32
        q = rnd(1, nq, S, 128)
        k, v = rnd(nb, nkv, 32, 128), rnd(nb, nkv, 32, 128)
        pt = torch.randperm(nb, generator=torch.Generator().manual_seed(1)).to(torch.int32).reshape(1, nb)
        return [T(dev, q, ttnn.bfloat8_b), T(dev, k, ttnn.bfloat8_b), T(dev, v, ttnn.bfloat8_b),
                T(dev, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)]

    def run(dev, a):
        ch = min(256, start & -start)
        return ttnn.transformer.chunked_scaled_dot_product_attention(
            input_tensor_q=a[0], input_tensor_k=a[1], input_tensor_v=a[2], page_table_tensor=a[3],
            chunk_start_idx=start, program_config=pcfg(dev, grid, ch, ch),
            compute_kernel_config=ckc("HiFi4", False, True, True))

    return setup, run


def llama_decode(nq, nkv, B, pos, grid, fid):
    def setup(dev):
        bpu = (pos + 1 + 31) // 32
        bpu = ((bpu + 7) // 8) * 8
        nb = B * bpu
        q = rnd(1, B, nq, 128)
        k, v = rnd(nb, nkv, 32, 128), rnd(nb, nkv, 32, 128)
        pt = torch.randperm(nb, generator=torch.Generator().manual_seed(1)).to(torch.int32).reshape(B, bpu)
        qmem = ttnn.create_sharded_memory_config(
            shape=(32, 128), core_grid=ttnn.CoreGrid(y=4, x=8), strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        tq = ttnn.to_memory_config(T(dev, q, ttnn.bfloat16), qmem)
        cp = torch.full((B,), pos, dtype=torch.int32)
        return [tq, T(dev, k, ttnn.bfloat8_b), T(dev, v, ttnn.bfloat8_b),
                T(dev, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), T(dev, cp, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)]

    def run(dev, a):
        cfg = ckc("HiFi2", True, True, True) if fid == "HiFi2" else ckc("HiFi4", False, True, True)
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            a[0], a[1], a[2], page_table_tensor=a[3], cur_pos_tensor=a[4], scale=1 / math.sqrt(128),
            program_config=pcfg(dev, grid, 0, 0), compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return setup, run


def causal_bfp8(S, grid):
    def setup(dev):
        q, k, v = rnd(1, 32, S, 128), rnd(1, 8, S, 128), rnd(1, 8, S, 128)
        return [T(dev, q, ttnn.bfloat8_b), T(dev, k, ttnn.bfloat8_b), T(dev, v, ttnn.bfloat8_b)]

    def run(dev, a):
        return ttnn.transformer.scaled_dot_product_attention(
            a[0], a[1], a[2], is_causal=True, scale=1 / math.sqrt(128),
            program_config=pcfg(dev, grid, 128, 128, approx=True), compute_kernel_config=ckc("HiFi2", False, False, False))

    return setup, run


def mla_decode(pos, cpb):
    users, nh, d, dv, S = 4, 128, 576, 512, 8192

    def setup(dev):
        nb = users * S // 32
        q = rnd(1, users, nh, d) * 0.1
        cache = rnd(nb, 1, 32, d) * 0.1
        pt = torch.randperm(nb, generator=torch.Generator().manual_seed(1)).to(torch.int32).reshape(users, S // 32)
        grid = dev.compute_with_storage_grid_size()
        cores = ttnn.num_cores_to_corerangeset(64, grid, row_wise=True)
        qmem = ttnn.create_sharded_memory_config(
            shape=[32, d], core_grid=cores, strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        tq = ttnn.to_memory_config(T(dev, q, ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG), qmem)
        cp = torch.full((users,), pos, dtype=torch.int32)
        setup.omem = ttnn.create_sharded_memory_config(
            shape=[32, dv], core_grid=cores, strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        return [tq, T(dev, cache, ttnn.bfloat16), T(dev, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                T(dev, cp, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)]

    def run(dev, a):
        return ttnn.transformer.paged_flash_multi_latent_attention_decode(
            a[0], a[1], page_table_tensor=a[2], cur_pos_tensor=a[3], head_dim_v=dv, scale=256 ** -0.5,
            program_config=pcfg(dev, "GF", 0, 128, cpb=cpb), compute_kernel_config=ckc("HiFi4", False, False, False),
            memory_config=setup.omem)

    return setup, run


def mla_prefill(nh, S, kdt):
    def setup(dev):
        q, k = rnd(1, nh, S, 576), rnd(1, 1, S, 576)
        return [T(dev, q, ttnn.bfloat16), T(dev, k, getattr(ttnn, kdt))]

    def run(dev, a):
        return ttnn.transformer.flash_mla_prefill(
            a[0], a[1], 512, is_causal=True, scale=256 ** -0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pcfg(dev, "GF", 128, 64), compute_kernel_config=ckc("HiFi4", False, False, False))

    return setup, run


def joint(H, N, L, approx):
    def setup(dev):
        xs = [rnd(1, H, n, 128) for n in (N, N, N, L, L, L)]
        setup.md5 = hashlib.md5(b"".join(x.view(torch.int16).numpy().tobytes() for x in xs)).hexdigest()[:12]
        return [T(dev, x, ttnn.bfloat16) for x in xs]

    def run(dev, a):
        return ttnn.transformer.joint_scaled_dot_product_attention(
            *a, joint_strategy="rear", program_config=pcfg(dev, "GJ", 128, 512, approx=approx),
            compute_kernel_config=ckc("HiFi2", False, False, False))

    return setup, run


CASES = {}
for S in (1024, 2048, 4096, 8192):
    for g in ("G8", "GF"):
        CASES[f"l8b_prefill_S{S}_{g}"] = llama_prefill(32, 8, S, g)
for g in ("G8", "GF"):
    CASES[f"l8b_chunked_S4096_at4096_{g}"] = llama_chunked(32, 8, 4096, 4096, g)
    for S in (4096, 8192):
        CASES[f"l70b_tp4_prefill_S{S}_{g}"] = llama_prefill(16, 2, S, g)
for pos in (127, 2047, 8191):
    CASES[f"l8b_decode_b1_p{pos}_HiFi2_G8"] = llama_decode(32, 8, 1, pos, "G8", "HiFi2")
for B, poss in ((8, (127, 2047, 8191)), (32, (127, 2047, 4095))):
    for pos in poss:
        for g in ("G8", "GFS"):
            CASES[f"l8b_decode_b{B}_p{pos}_HiFi2_{g}"] = llama_decode(32, 8, B, pos, g, "HiFi2")
for g in ("G8", "GFS"):
    CASES[f"l8b_decode_b32_p2047_HiFi4_{g}"] = llama_decode(32, 8, 32, 2047, g, "HiFi4")
    CASES[f"l70b_tp4_decode_b32_p2047_HiFi2_{g}"] = llama_decode(16, 2, 32, 2047, g, "HiFi2")
for pos in (511, 2047, 8191):
    for c in (16, 4):
        CASES[f"dsv3_mla_decode_p{pos}_CPB{c}"] = mla_decode(pos, c)
for nh in (16, 32):
    for S in (1024, 4096):
        CASES[f"dsv3_mla_prefill_nh{nh}_S{S}_bfp8"] = mla_prefill(nh, S, "bfloat8_b")
CASES["dsv3_mla_prefill_nh32_S4096_bf16"] = mla_prefill(32, 4096, "bfloat16")
for S in (2048, 8192):
    CASES[f"dsv3_mla_prefill_nh16_S{S}_bfp8"] = mla_prefill(16, S, "bfloat8_b")
CASES["dsv3_mla_prefill_nh16_S4096_bf16"] = mla_prefill(16, 4096, "bfloat16")
for H in (6, 12, 24):
    CASES[f"flux_joint_H{H}_N4096_L512"] = joint(H, 4096, 512, False)
CASES["flux_joint_H24_N4096_L512_approx"] = joint(24, 4096, 512, True)
for S in (1024, 2048, 4096, 8192):
    CASES[f"causal_bfp8_S{S}_q128_GF"] = causal_bfp8(S, "GF")


def first(o):
    return o[0] if isinstance(o, (list, tuple)) else o


def free(o):
    for t in o if isinstance(o, (list, tuple)) else [o]:
        try:
            t.deallocate()
        except Exception:  # noqa: BLE001
            pass


def main():
    names = os.environ.get("SDPA_CASES", "").split() or list(CASES)
    so = [l.split()[-1] for l in open("/proc/self/maps") if "_ttnn" in l and l.rstrip().endswith(".so")]
    print(f"HOST ttnn {ttnn.__file__} so {so[:1]}", flush=True)
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    print(f"GRID {g.x}x{g.y}", flush=True)
    try:
        for name in names:
            setup, run = CASES[name]
            gc.collect()
            torch.manual_seed(0)
            a = None
            try:
                a = setup(dev)
                o = run(dev, a)
                ttnn.synchronize_device(dev)
                if BITS:
                    outs = [ttnn.to_torch(t) for t in (o if isinstance(o, (list, tuple)) else [o])]
                    torch.save(outs, os.path.join(BITS, f"{name}.pt"))
                    if hasattr(setup, "md5"):
                        with open(os.path.join(BITS, f"{name}.inputs"), "w") as f:
                            f.write(setup.md5 + "\n")
                    h = hashlib.md5(b"".join(x.contiguous().view(torch.int16).numpy().tobytes() for x in outs))
                    print(f"BITS {name} {h.hexdigest()[:12]}", flush=True)
                free(o)
                for _ in range(N_WARM - 1):
                    free(run(dev, a))
                ttnn.synchronize_device(dev)
                ttnn.ReadDeviceProfiler(dev)
                signpost(header=name)
                for _ in range(N_MEAS):
                    free(run(dev, a))
                ttnn.synchronize_device(dev)
                ttnn.ReadDeviceProfiler(dev)
                free(a)
                print(f"CASE {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                if a is not None:
                    free(a)
                try:
                    ttnn.synchronize_device(dev)
                except Exception:  # noqa: BLE001
                    pass
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
