# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""rotary_embedding (HF rotate_half) vs rotary_embedding_llama (interleaved) at (1024,12,133,64)."""

import time

import torch
import ttnn

B, H, S, D = 1024, 12, 133, 64


def bench(fn, iters=5):
    out = fn()
    ttnn.synchronize_device(dev)
    ttnn.deallocate(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1000


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def up(t, **kw):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw
    )


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    try:
        torch.manual_seed(0)
        x = torch.randn(B, H, S, D)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
        freqs = torch.arange(S).float()[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]
        rot = torch.cat([-x[..., D // 2 :], x[..., : D // 2]], dim=-1)
        ref = x * cos + rot * sin

        perm = torch.stack([torch.arange(D // 2), torch.arange(D // 2) + D // 2], dim=-1).flatten()
        x_i, cos_i, sin_i, ref_i = x[..., perm], cos[..., perm], sin[..., perm], ref[..., perm]

        tx, tcos, tsin = up(x), up(cos), up(sin)
        ms = bench(lambda: ttnn.experimental.rotary_embedding(tx, tcos, tsin, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        out = ttnn.to_torch(ttnn.experimental.rotary_embedding(tx, tcos, tsin))[:, :, :S]
        print(f"rotary_embedding       {ms:7.3f} ms pcc={pcc(ref, out):.6f}", flush=True)

        trans = torch.zeros(1, 1, 32, 32)
        trans[..., torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1
        trans[..., torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1
        txi, tcosi, tsini, ttrans = up(x_i), up(cos_i), up(sin_i), up(trans)
        for fid in (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2):
            ckc = ttnn.types.BlackholeComputeKernelConfig(
                math_fidelity=fid, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False
            )
            fn = lambda: ttnn.experimental.rotary_embedding_llama(
                txi, tcosi, tsini, ttrans, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckc
            )
            ms = bench(fn)
            out = ttnn.to_torch(fn())[:, :, :S]
            print(f"rotary_embedding_llama {str(fid):>22} {ms:7.3f} ms pcc={pcc(ref_i, out):.6f}", flush=True)
    finally:
        ttnn.close_device(dev)
