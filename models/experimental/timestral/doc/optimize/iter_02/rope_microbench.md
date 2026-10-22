# R2-2 rotary microbenchmark

One Q (or K) tensor at the paper shape, `(1024, 12, 96, 64)` bf16, DRAM interleaved; median of 7 synchronized runs.
The llama op runs on head_dim in `rope_pair_order` (the CPU check confirms that it equals the permuted rotate-half result exactly, and that q . k is unchanged).

| precision | op | ms | PCC | max abs err |
|---|---|---|---|---|
| accuracy (HiFi2) | `rotary_embedding` (rotate-half) | 2.080 | 0.999996 | 6.2e-2 |
| accuracy (HiFi2) | `rotary_embedding_llama`, `(R, H, S, Dh)` | 1.686 | 0.999992 | 6.6e-2 |
| accuracy (HiFi2) | `rotary_embedding_llama`, `(R * H, 1, S, Dh)` | 1.678 | 0.999992 | 6.6e-2 |
| performance (LoFi) | `rotary_embedding` (rotate-half) | 2.085 | 0.999996 | 6.2e-2 |
| performance (LoFi) | `rotary_embedding_llama`, `(R, H, S, Dh)` | 1.685 | 0.999901 | 2.5e-1 |
| performance (LoFi) | `rotary_embedding_llama`, `(R * H, 1, S, Dh)` | 1.677 | 0.999901 | 2.5e-1 |

- At this shape the llama op takes its reload path (36 tile rows per core > 8), so each input tile row still re-reads its cos / sin tiles. That is why it saves about 0.4 ms per call rather than the 1.0–1.3 ms the proposal estimated.
- Folding heads into the batch saves nothing, and neither does sharding cos / sin, because the sharded prefill factory falls back to reloading when batch > 1.
- The time is identical at LoFi and HiFi2 (the op is not compute-bound), but LoFi loses input mantissa bits in the ±1 rotation matmul. The model therefore runs RoPE at HiFi4 under both precisions.
