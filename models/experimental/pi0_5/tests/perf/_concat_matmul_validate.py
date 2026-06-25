# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated PCC + timing for ttnn.experimental.concat_heads_matmul.

Reference = nlp_concat_heads(attn) -> [1,1,32,2048] ; then ttnn.linear(., weight).
Fused = concat_heads_matmul(attn, weight) in one dispatch.
"""
import statistics
import time

import torch
import ttnn


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 1.0 if torch.allclose(a, b, atol=1e-3) else 0.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=24576)
    L1 = ttnn.L1_MEMORY_CONFIG
    nh, hd, seq, N = 8, 256, 32, 1024
    K = nh * hd
    torch.manual_seed(0)
    attn = ttnn.from_torch(
        torch.randn(1, nh, seq, hd) * 0.3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1
    )
    weight = ttnn.from_torch(
        torch.randn(K, N) * 0.02, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1
    )

    # exact denoise O-proj config (LoFi via program_config path)
    from models.experimental.pi0_5.tt.tt_pipeline.denoise_block import _denoise_tuned_pcfg
    from models.experimental.pi0_5.tt.tt_pipeline.modeling.bs import matmul_pcfg

    g = dev.compute_with_storage_grid_size()
    pc = _denoise_tuned_pcfg(1, K // 32, N // 32, g.x, g.y) or matmul_pcfg(1, K // 32, N // 32, g.x, g.y, in0_block_w=8)

    # reference: concat + tuned matmul (compute_kernel_config=None -> matmul default, LoFi w/ pc)
    c = ttnn.experimental.nlp_concat_heads(attn, memory_config=L1)
    out_ref = ttnn.linear(c, weight, memory_config=L1, program_config=pc)

    # fused, same tuned config
    out_f = ttnn.experimental.concat_heads_matmul(attn, weight, memory_config=L1, program_config=pc)
    print("ref shape", list(out_ref.shape), "fused shape", list(out_f.shape))
    pcc = _pcc(ttnn.to_torch(out_ref), ttnn.to_torch(out_f))
    print(f"[concat_matmul] PCC = {pcc:.6f}")

    def bench(fn, it=300):
        for _ in range(20):
            fn()
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(it):
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e6)
        return statistics.median(ts)

    def ref_path():
        cc = ttnn.experimental.nlp_concat_heads(attn, memory_config=L1)
        o = ttnn.linear(cc, weight, memory_config=L1)
        ttnn.deallocate(cc)
        ttnn.deallocate(o)

    def fused_path():
        o = ttnn.experimental.concat_heads_matmul(attn, weight, memory_config=L1)
        ttnn.deallocate(o)

    r = bench(ref_path)
    f = bench(fused_path)
    print(f"[concat_matmul] wall: concat+matmul={r:.2f} us  fused={f:.2f} us  saved={r-f:.2f} us ({100*(r-f)/r:.1f}%)")
    ttnn.close_mesh_device(dev)
    assert pcc > 0.99, f"PCC too low: {pcc}"
    print("PASS")


if __name__ == "__main__":
    main()
