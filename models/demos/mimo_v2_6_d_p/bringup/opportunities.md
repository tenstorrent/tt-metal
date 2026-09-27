# Performance opportunities

Profile: rung last, chunk 51200->56320, layers 6. Wall 1280 ms, device 1122 ms (warm, slowest chip per section).

Pick entries by marking `[x]`. Each picked entry becomes a perf task behind a switch, gated on warm device time
and on every accuracy gate. Changes that trade accuracy for speed need an explicit decision here.

| Pick | Rank | Section | Device ms | Share | Chip spread | Known issues | Repo map |
|---|---|---|---|---|---|---|---|
| [ ] | 1 | `experts` | 958.6 | 85.4% | 0.2 ms | Dense-EP experts. | MoE, expert parallel (EP); MoE, tensor parallel, decode-style; Sigmoid + correction-bias (noaux_tc) router on device; noaux_tc router with a large correction bias (selection precision-bound); Routed experts at a chosen fidelity without a ttnn change; All-device model with fp8/mxfp4 weights, dense + MoE graphs, shared module builders |
| [ ] | 2 | `attention` | 142.2 | 12.7% | 0.3 ms | - | Chunked causal attention; SDPA program config; Sliding-window + global attention, partial RoPE; GQA KV layout and address table; Contiguous KV cache at a chunk offset; GQA with fewer KV heads than chips |
| [ ] | 3 | `router` | 7.1 | 0.6% | 0.0 ms | Dense-EP experts. | MoE, expert parallel (EP); MoE, tensor parallel, decode-style; Sigmoid + correction-bias (noaux_tc) router on device; noaux_tc router with a large correction bias (selection precision-bound); All-device model with fp8/mxfp4 weights, dense + MoE graphs, shared module builders |
| [ ] | 4 | `mlp` | 6.0 | 0.5% | 0.0 ms | - | MoE, expert parallel (EP); MoE, tensor parallel, decode-style; Sigmoid + correction-bias (noaux_tc) router on device |
| [ ] | 5 | `attn_norm` | 2.3 | 0.2% | 0.0 ms | - | Chunked causal attention; SDPA program config; Sliding-window + global attention, partial RoPE; RoPE, interleaved (Meta) order; GQA KV layout and address table; Contiguous KV cache at a chunk offset; GQA with fewer KV heads than chips; A worked bring-up with perf (Gemma-4 26B-A4B); noaux_tc router with a large correction bias (selection precision-bound) |
| [ ] | 6 | `ffn_norm` | 2.3 | 0.2% | 0.0 ms | - | A worked bring-up with perf (Gemma-4 26B-A4B); noaux_tc router with a large correction bias (selection precision-bound) |
| [ ] | 7 | `attn_residual` | 1.7 | 0.2% | 0.0 ms | - | Chunked causal attention; SDPA program config; Sliding-window + global attention, partial RoPE; RoPE, interleaved (Meta) order; GQA KV layout and address table; Contiguous KV cache at a chunk offset; GQA with fewer KV heads than chips |
| [ ] | 8 | `ffn_residual` | 1.4 | 0.1% | 0.0 ms | - | - |
| [ ] | 9 | `mlp_residual` | 0.3 | 0.0% | 0.0 ms | - | MoE, expert parallel (EP); MoE, tensor parallel, decode-style; Sigmoid + correction-bias (noaux_tc) router on device |
