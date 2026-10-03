# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Sweep SDPA chunk sizes / dtypes for Chronos time attention (B=1024, H=12, S=133, Dh=64)."""

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


def run(dtype, fid, qc, kc, exp_approx=True):
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
        q_chunk_size=qc,
        k_chunk_size=kc,
        exp_approx_mode=exp_approx,
    )
    q, k, v = (ttnn.typecast(t, dtype) if dtype != ttnn.bfloat16 else t for t in (tq, tk, tv))
    fn = lambda: ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=1.0, program_config=pc, compute_kernel_config=ckc
    )
    ms = bench(fn)
    out = fn()
    got = ttnn.to_torch(out)[:8].float()
    ttnn.deallocate(out)
    return ms, pcc(ref, got)


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    torch.manual_seed(0)
    qh, kh, vh = (torch.randn(B, H, S, D) * 0.4 for _ in range(3))
    ref = torch.softmax(qh[:8] @ kh[:8].transpose(-1, -2), dim=-1) @ vh[:8]
    tq, tk, tv = (
        ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for t in (qh, kh, vh)
    )
    try:
        configs = [(qc, kc) for qc in (32, 64, 96, 128, 160) for kc in (32, 64, 96, 128, 160)]
        for dtype, fid in [
            (ttnn.bfloat16, ttnn.MathFidelity.HiFi2),
            (ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
            (ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
        ]:
            for qc, kc in configs:
                try:
                    ms, p = run(dtype, fid, qc, kc)
                    print(f"{str(dtype):>18} {str(fid):>22} q{qc:<4} k{kc:<4} {ms:7.2f} ms pcc={p:.6f}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"{str(dtype):>18} {str(fid):>22} q{qc:<4} k{kc:<4} ERR {str(e).splitlines()[0][:120]}",
                        flush=True,
                    )
    finally:
        ttnn.close_device(dev)
