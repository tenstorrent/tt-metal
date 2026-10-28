# Round 2 proposals (G2-r2)

Ranked by expected full-depth saving (analysis.md). Order-only items come first, since the accuracy row-bias budget is about half used (R1-4 moved ragged by 0.0107 of 0.02). Every precision change needs the v1.14 variance table and the full check_gates; over 0.02 on any row means an X gate (D5).

| id | change | expected | kind / risk |
|---|---|---|---|
| R2-1 | embed config: pm 103 is prime, so use 8 grid rows or pad pm to 104 (emb_o, emb_r) | about -1.3 ms | order-only |
| R2-2 | causal mask pre-expanded to [B,1,32,32] for the BinaryNg add (unfused) | about -1.5 ms | order-only (same values) |
| R2-3 | decoder activations in L1 (16.8 MB tensors), then encoder in batch chunks if it pays | about -6 to -9 ms | order-only; risk L1 OOM next to the matmul CBs |
| R2-4 | drop the dead v / q columns: one 2560 [q k v \| qR kR] projection, or narrower splits; cross q 1536 -> 1024 | about -4 ms | order-only if the split layout is supported |
| R2-5 | N = 512 linears (wo, o, co) on all 110 cores (1D mcast or another grid), and the 6 cross-kv projections of enc batched into one N = 9216 linear | about -2 to -3 ms | order-only (in0_block_w unchanged) |
| R2-6 | fp32 probabilities straight into P@V, no typecast | about -3 to -4 ms | precision change, low risk |
| R2-7 | fused RoPE (`rotary_embedding_llama`, interleaved q/k columns), dropping the qR/kR columns | about -25 to -30 ms | precision change (fp32 dst for x cos + xR sin); op constraints |
| R2-8 | RoPE as mul + addcmul (fallback if R2-7 fails) | about -5 to -6 ms | precision change, low risk |
| R2-9 | score path through SDPA (scale 1, attn_mask; D3's R1-2 rules: F1/F9 table, every batch, chunk count 1 or even) | about -25 ms | precision change, high risk (the fused mask softmax already failed drift) |
| R2-10 | host prepare (R1-6 carried over): RoPE tables / DRoPE MLPs, vectorized prepare (0.32 s per paper forward vs 0.14 s of device time) | wall only | does not count as device time |

Not proposed: packing the encoder rows (96 vs 65; net about 0 without SDPA), and a residual add in the epilogue (no ttnn.linear support).
