# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Device time of the grouped-path permutes (tiled vs row-major round trip) and block SDPA vs block size."""

import time

import torch
import ttnn

from models.experimental.chronos_forecast.tt import program_configs

B, T, TP, D, H, DH = 1024, 133, 160, 768, 12, 64


def bench(fn, iters=5):
    out = fn()
    ttnn.synchronize_device(dev)
    ttnn.deallocate(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        ttnn.synchronize_device(dev)
        ttnn.deallocate(out)
    return (time.perf_counter() - t0) / iters * 1e3


def rm_permute(x, dims, out_shape=None):
    r = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    p = ttnn.permute(r, dims)
    ttnn.deallocate(r)
    t = ttnn.to_layout(p, ttnn.TILE_LAYOUT)
    ttnn.deallocate(p)
    return t


dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
dev.enable_program_cache()
try:
    x = ttnn.from_torch(torch.randn(B, T, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    xf = ttnn.from_torch(torch.randn(T, B, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    print(f"[LAYOUT] tiled permute (B,T,d)->(T,B,d): {bench(lambda: ttnn.permute(x, (1, 0, 2))):.2f} ms")
    print(f"[LAYOUT] tiled permute (T,B,d)->(B,T,d): {bench(lambda: ttnn.permute(xf, (1, 0, 2))):.2f} ms")
    print(f"[LAYOUT] tiled transpose(0,1) (B,T,d): {bench(lambda: ttnn.transpose(x, 0, 1)):.2f} ms")
    print(f"[LAYOUT] row-major permute round trip (B,T,d): {bench(lambda: rm_permute(x, (1, 0, 2))):.2f} ms")
    print(f"[LAYOUT] row-major permute round trip (T,B,d): {bench(lambda: rm_permute(xf, (1, 0, 2))):.2f} ms")
    got = ttnn.to_torch(rm_permute(x, (1, 0, 2)))
    ref = ttnn.to_torch(ttnn.permute(x, (1, 0, 2)))
    print(f"[LAYOUT] row-major path matches tiled: {torch.equal(got, ref)} shape={tuple(got.shape)}")
    ttnn.deallocate(x)
    ttnn.deallocate(xf)

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
            lambda: ttnn.transformer.scaled_dot_product_attention(
                q, q, q, is_causal=False, attn_mask=mask, program_config=cfg, compute_kernel_config=kcfg
            )
        )
        print(f"[SDPA] block={s} batch={n}: {ms:.2f} ms")
        ttnn.deallocate(q)
        ttnn.deallocate(mask)
finally:
    ttnn.close_mesh_device(dev)
