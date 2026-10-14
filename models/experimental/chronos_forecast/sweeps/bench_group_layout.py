# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Device time of the grouped-path layout choices.

    python models/experimental/chronos_forecast/sweeps/bench_group_layout.py [permute chunk sdpa]

permute: tiled permute / transpose vs a row-major round trip at the full batch (B=1024, T=133, d=768).
chunk:   the row-major permute round trip at one L1 chunk shape, bf16 vs bf8_b input and DRAM vs L1 placement.
sdpa:    block SDPA time vs block size.

With no arguments all three sections run.
"""

import sys
import time

import torch
import ttnn

from models.experimental.chronos_forecast.tt import program_configs

B, T, D, H, DH = 1024, 133, 768, 12, 64


def bench(dev, fn, iters=5):
    out = fn()
    ttnn.synchronize_device(dev)
    ttnn.deallocate(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        ttnn.synchronize_device(dev)
        ttnn.deallocate(out)
    return (time.perf_counter() - t0) / iters * 1e3


def rm_permute(x, dims=(1, 0, 2), mem=None):
    kw = {} if mem is None else {"memory_config": mem}
    r = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, **kw)
    p = ttnn.permute(r, dims, **kw)
    ttnn.deallocate(r)
    t = ttnn.to_layout(p, ttnn.TILE_LAYOUT, **kw)
    ttnn.deallocate(p)
    return t


def bench_permute(dev):
    x = ttnn.from_torch(torch.randn(B, T, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    xf = ttnn.from_torch(torch.randn(T, B, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    print(f"[LAYOUT] tiled permute (B,T,d)->(T,B,d): {bench(dev, lambda: ttnn.permute(x, (1, 0, 2))):.2f} ms")
    print(f"[LAYOUT] tiled permute (T,B,d)->(B,T,d): {bench(dev, lambda: ttnn.permute(xf, (1, 0, 2))):.2f} ms")
    print(f"[LAYOUT] tiled transpose(0,1) (B,T,d): {bench(dev, lambda: ttnn.transpose(x, 0, 1)):.2f} ms")
    print(f"[LAYOUT] row-major permute round trip (B,T,d): {bench(dev, lambda: rm_permute(x)):.2f} ms")
    print(f"[LAYOUT] row-major permute round trip (T,B,d): {bench(dev, lambda: rm_permute(xf)):.2f} ms")
    got = ttnn.to_torch(rm_permute(x))
    ref = ttnn.to_torch(ttnn.permute(x, (1, 0, 2)))
    print(f"[LAYOUT] row-major path matches tiled: {torch.equal(got, ref)} shape={tuple(got.shape)}")
    ttnn.deallocate(x)
    ttnn.deallocate(xf)


def bench_chunk(dev):
    for dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
        for shape in ((64, 133, 768), (133, 64, 768)):
            for mem_name, mem in (("dram", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG)):
                x = ttnn.from_torch(
                    torch.randn(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem
                )
                try:
                    tiled = bench(dev, lambda: ttnn.permute(x, (1, 0, 2), memory_config=mem))
                    rm = bench(dev, lambda: rm_permute(x, mem=mem))
                    y = rm_permute(x, mem=mem)
                    ok = torch.equal(ttnn.to_torch(y), ttnn.to_torch(ttnn.permute(x, (1, 0, 2))))
                    print(
                        f"[CHUNK] {dtype} {shape} {mem_name}: tiled {tiled:.3f} ms, rm {rm:.3f} ms, "
                        f"exact={ok} out_dtype={y.dtype}"
                    )
                    ttnn.deallocate(y)
                except Exception as e:  # noqa: BLE001
                    print(f"[CHUNK] {dtype} {shape} {mem_name}: FAILED {str(e).splitlines()[0][:160]}")
                ttnn.deallocate(x)


def bench_sdpa(dev):
    for s in (32, 64, 128, 256):
        n = T * B // s
        q = ttnn.from_torch(torch.randn(n, H, s, DH), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        same = (torch.arange(s) // 4)[:, None] == (torch.arange(s) // 4)[None, :]
        mask = ttnn.from_torch(
            ((~same) * -1e9).reshape(1, 1, s, s), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
        )
        qc, kc = program_configs.sdpa_chunk_sizes(s, s)
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
            q_chunk_size=qc,
            k_chunk_size=kc,
            exp_approx_mode=False,
        )
        kcfg = program_configs.compute_kernel_config(packer_l1_acc=False)
        ms = bench(
            dev,
            lambda: ttnn.transformer.scaled_dot_product_attention(
                q, q, q, is_causal=False, attn_mask=mask, program_config=cfg, compute_kernel_config=kcfg
            ),
        )
        print(f"[SDPA] block={s} batch={n}: {ms:.2f} ms")
        ttnn.deallocate(q)
        ttnn.deallocate(mask)


SECTIONS = {"permute": bench_permute, "chunk": bench_chunk, "sdpa": bench_sdpa}


if __name__ == "__main__":
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    dev.enable_program_cache()
    try:
        for name in sys.argv[1:] or list(SECTIONS):
            SECTIONS[name](dev)
    finally:
        ttnn.close_mesh_device(dev)
