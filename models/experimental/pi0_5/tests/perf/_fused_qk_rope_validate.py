# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone PCC + timing validation for ttnn.experimental.rotary_embedding_fused_qk.

Reference = two separate rotary_embedding calls (q, k). Fused = the new single-dispatch op.
PCC must be ~1.0 (identical GPT-J/rotate_half math). Timing via wall-clock median over many iters.
"""
import statistics
import time

import torch
import ttnn


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 1.0 if torch.allclose(a, b, atol=1e-4) else 0.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=24576)
    L1 = ttnn.L1_MEMORY_CONFIG
    nh, nkv, hd, seq = 8, 1, 256, 32
    torch.manual_seed(0)

    def t(shape):
        return ttnn.from_torch(
            torch.randn(*shape) * 0.3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1
        )

    q = t((1, nh, seq, hd))
    k = t((1, nkv, seq, hd))
    cos = t((1, 1, seq, hd))
    sin = t((1, 1, seq, hd))

    # reference: two separate calls
    q_ref = ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=L1)
    k_ref = ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=L1)

    # fused
    q_f, k_f = ttnn.experimental.rotary_embedding_fused_qk(q, k, cos, sin, memory_config=L1)

    q_ref_t, k_ref_t = ttnn.to_torch(q_ref), ttnn.to_torch(k_ref)
    q_f_t, k_f_t = ttnn.to_torch(q_f), ttnn.to_torch(k_f)
    pcc_q, pcc_k = _pcc(q_ref_t, q_f_t), _pcc(k_ref_t, k_f_t)
    print(f"[fused-qk RoPE] PCC q={pcc_q:.6f}  k={pcc_k:.6f}")

    def bench(fn, iters=300):
        for _ in range(20):
            fn()
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e6)
        return statistics.median(ts)

    def ref_path():
        a = ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=L1)
        b = ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=L1)
        ttnn.deallocate(a)
        ttnn.deallocate(b)

    def fused_path():
        a, b = ttnn.experimental.rotary_embedding_fused_qk(q, k, cos, sin, memory_config=L1)
        ttnn.deallocate(a)
        ttnn.deallocate(b)

    ref_us = bench(ref_path)
    fused_us = bench(fused_path)
    print(
        f"[fused-qk RoPE] wall: 2x rotary_embedding={ref_us:.2f} us  fused={fused_us:.2f} us  "
        f"saved={ref_us - fused_us:.2f} us ({100 * (ref_us - fused_us) / ref_us:.1f}%)"
    )
    ttnn.close_mesh_device(dev)
    assert pcc_q > 0.999 and pcc_k > 0.999, f"PCC too low: q={pcc_q}, k={pcc_k}"
    print("PASS")


if __name__ == "__main__":
    main()
