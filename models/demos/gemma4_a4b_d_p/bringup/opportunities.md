# Performance opportunities

Profile: rung last, chunk 51200->56320, layers 30. Wall 743 ms, device 742 ms (warm, slowest chip per section).

Pick entries by marking `[x]`. Each picked entry becomes a perf task behind a switch, gated on warm device time
and on every accuracy gate. Changes that trade accuracy for speed need an explicit decision here.

| Pick | Rank | Section | Device ms | Share | Chip spread | Known issues | Repo map |
|---|---|---|---|---|---|---|---|
| [ ] | 1 | `attention` | 381.1 | 51.4% | 0.2 ms | - | Chunked causal attention; SDPA program config; Sliding-window + global attention, partial RoPE; GQA KV layout and address table; KV cache write at a chunk offset, contiguous cache; GQA with fewer KV heads than chips, chunked prefill |
| [ ] | 2 | `experts` | 212.2 | 28.6% | 0.5 ms | Dense-EP experts. | MoE, expert parallel (EP); MoE, tensor parallel, decode-style |
| [ ] | 3 | `router` | 43.1 | 5.8% | 0.1 ms | Dense-EP experts. | MoE, expert parallel (EP); MoE, tensor parallel, decode-style |
| [ ] | 4 | `mlp` | 40.1 | 5.4% | 0.3 ms | - | MoE, expert parallel (EP); MoE, tensor parallel, decode-style |
| [ ] | 5 | `ffn_residual` | 11.8 | 1.6% | 0.1 ms | - | - |
| [ ] | 6 | `ffn_combine` | 6.8 | 0.9% | 0.0 ms | Dense-EP experts. | MoE, expert parallel (EP); MoE, tensor parallel, decode-style |
| [ ] | 7 | `attn_residual` | 6.7 | 0.9% | 0.0 ms | - | Chunked causal attention; SDPA program config; Sliding-window + global attention, partial RoPE; RoPE, interleaved (Meta) order; GQA KV layout and address table; KV cache write at a chunk offset, contiguous cache; GQA with fewer KV heads than chips, chunked prefill |
| [ ] | 8 | `post_attn_norm` | 5.7 | 0.8% | 0.0 ms | - | Chunked causal attention; SDPA program config; Sliding-window + global attention, partial RoPE; RoPE, interleaved (Meta) order; GQA KV layout and address table; Device profile without Tracy; KV cache write at a chunk offset, contiguous cache; GQA with fewer KV heads than chips, chunked prefill |
| [ ] | 9 | `moe_norm` | 5.7 | 0.8% | 0.0 ms | Dense-EP experts. | MoE, expert parallel (EP); MoE, tensor parallel, decode-style |
| [ ] | 10 | `post_mlp_norm` | 5.7 | 0.8% | 0.0 ms | - | MoE, expert parallel (EP); MoE, tensor parallel, decode-style; Device profile without Tracy |
| [ ] | 11 | `post_moe_norm` | 5.7 | 0.8% | 0.0 ms | Dense-EP experts. | MoE, expert parallel (EP); MoE, tensor parallel, decode-style; Device profile without Tracy |
| [ ] | 12 | `post_ffn_norm` | 5.7 | 0.8% | 0.0 ms | - | Device profile without Tracy |
