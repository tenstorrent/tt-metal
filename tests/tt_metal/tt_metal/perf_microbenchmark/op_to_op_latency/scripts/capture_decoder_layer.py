#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Representative Llama decoder-layer op sequence (random tensors, no weights) under ttnn graph
capture. Dumps the capture JSON for /tmp/raw_analyzer.py. Structure mirrors a decoder layer:

  rms_norm -> {q,k,v proj (3 siblings read the same norm)} -> attention(qk^T, softmax, @v)
  -> o proj -> residual add(x, o) -> rms_norm -> {gate, up proj (2 siblings)} -> silu*up
  -> down proj -> residual add(r1, d)

Interesting dependency structure to recover: q/k/v are mutually independent (read h1), gate/up
independent (read h2); residuals read tensors from many ops back (long consumer distance).

NOTE: attention is done manually (matmul/softmax/matmul) rather than fused SDPA to keep the API
robust for a first prototype; exact head/GQA layout omitted. Fidelity is at the op-DEPENDENCY level.
API details (open_device vs mesh, graph.RunMode, rms_norm signature) to be confirmed post-build.
"""
import json
import torch
import ttnn

H = 4096  # hidden
NH, DH = 32, 128  # heads, head_dim (MHA: nqh == nkv)
INTER = 14336  # MLP intermediate (Llama-3-8B)
B, S = 1, 32  # batch, seq (small prefill-like chunk)
OUT = "/tmp/decoder_capture.json"


def main():
    device = ttnn.open_device(device_id=0)
    try:

        def rand(*shape):
            return ttnn.from_torch(
                torch.randn(*shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        x = rand(B, S, H)
        # weights (random)
        Wq, Wk, Wv, Wo = rand(H, H), rand(H, H), rand(H, H), rand(H, H)
        Wg, Wu, Wd = rand(H, INTER), rand(H, INTER), rand(INTER, H)
        ln1_w, ln2_w = rand(H), rand(H)

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)

        h1 = ttnn.rms_norm(x, weight=ln1_w)
        q = ttnn.matmul(h1, Wq)  # 3 siblings read h1 (mutually independent)
        k = ttnn.matmul(h1, Wk)
        v = ttnn.matmul(h1, Wv)

        def to_heads(t):  # [B,S,H] -> [B,NH,S,DH]
            return ttnn.transpose(ttnn.reshape(t, (B, S, NH, DH)), 1, 2)

        ah = ttnn.transformer.scaled_dot_product_attention(  # FUSED attention (one device op)
            to_heads(q), to_heads(k), to_heads(v), is_causal=True
        )
        attn = ttnn.reshape(ttnn.transpose(ah, 1, 2), (B, S, H))  # [B,NH,S,DH] -> [B,S,H]
        o = ttnn.matmul(attn, Wo)
        r1 = ttnn.add(x, o)  # residual: reads x (far back) + o
        h2 = ttnn.rms_norm(r1, weight=ln2_w)
        g = ttnn.matmul(h2, Wg)  # gate/up siblings read h2 (independent)
        u = ttnn.matmul(h2, Wu)
        gu = ttnn.multiply(ttnn.silu(g), u)
        d = ttnn.matmul(gu, Wd)
        r2 = ttnn.add(r1, d)  # residual: reads r1 (far back) + d
        ttnn.synchronize_device(device)

        captured = ttnn.graph.end_graph_capture()
        json.dump(captured, open(OUT, "w"))
        print(f"captured {len(captured)} nodes -> {OUT}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
