# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC + timing validation for ttnn.experimental.nlp_create_qkv_heads_rope.

Reference = nlp_create_qkv_heads(qkv) -> q,k,v ; then rotary_embedding(q), rotary_embedding(k).
Fused = nlp_create_qkv_heads_rope(qkv) -> (q_rot, k_rot, v) in one dispatch.
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
    qkv_w = (nh + 2 * nkv) * hd
    torch.manual_seed(0)

    def t(shape):
        return ttnn.from_torch(
            torch.randn(*shape) * 0.3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1
        )

    qkv = t((1, 1, seq, qkv_w))
    cos = t((1, 1, seq, hd))
    sin = t((1, 1, seq, hd))

    # reference
    qr, kr, vr = ttnn.experimental.nlp_create_qkv_heads(
        qkv, num_heads=nh, num_kv_heads=nkv, transpose_k_heads=False, memory_config=L1
    )
    qr = ttnn.experimental.rotary_embedding(qr, cos, sin, memory_config=L1)
    kr = ttnn.experimental.rotary_embedding(kr, cos, sin, memory_config=L1)

    # fused
    qf, kf, vf = ttnn.experimental.nlp_create_qkv_heads_rope(qkv, cos, sin, nh, nkv, memory_config=L1)

    pcc_q = _pcc(ttnn.to_torch(qr), ttnn.to_torch(qf))
    pcc_k = _pcc(ttnn.to_torch(kr), ttnn.to_torch(kf))
    pcc_v = _pcc(ttnn.to_torch(vr), ttnn.to_torch(vf))
    print(f"[create+rope] PCC q={pcc_q:.6f}  k={pcc_k:.6f}  v={pcc_v:.6f}")

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
        a, b, c = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=nh, num_kv_heads=nkv, transpose_k_heads=False, memory_config=L1
        )
        a = ttnn.experimental.rotary_embedding(a, cos, sin, memory_config=L1)
        b = ttnn.experimental.rotary_embedding(b, cos, sin, memory_config=L1)
        for x in (a, b, c):
            ttnn.deallocate(x)

    def fused_path():
        a, b, c = ttnn.experimental.nlp_create_qkv_heads_rope(qkv, cos, sin, nh, nkv, memory_config=L1)
        for x in (a, b, c):
            ttnn.deallocate(x)

    ref_us = bench(ref_path)
    fused_us = bench(fused_path)
    print(
        f"[create+rope] wall: create+2xrope={ref_us:.2f} us  fused={fused_us:.2f} us  "
        f"saved={ref_us - fused_us:.2f} us ({100 * (ref_us - fused_us) / ref_us:.1f}%)"
    )
    ttnn.close_mesh_device(dev)
    assert pcc_q > 0.999 and pcc_k > 0.999 and pcc_v > 0.999, f"PCC low: q={pcc_q} k={pcc_k} v={pcc_v}"
    print("PASS")


if __name__ == "__main__":
    main()
