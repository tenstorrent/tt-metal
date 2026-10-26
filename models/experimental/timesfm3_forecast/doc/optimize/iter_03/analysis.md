# Round 3 -> 4 analysis (iter_03; analyst: Opus, low effort, read-only; jobs 771 bench+profile, 772 multivariate profile)

Baseline 8da76a2d9cb: paper/accuracy device 423.6 ms, long 129.0, multivariate 858.4; performance paper 408.6.
Estimate 434.7 ms (+2.6 %): the profiled layer is layer 0, whose standalone pre-norm (0.80 ms) runs only once in
steady state (later layers take the pending post_add_norm output), so 19 x layer overcounts ~14 ms.

## Paper, one layer (21.84 ms; utilization vs 359 TFLOPs)
| op | role | ms | util |
|---|---|---|---|
| minimal_matmul 1280->4608 | QKV | 4.81 | 44.8 % |
| rope_qk_norm_qkv (generic) | head split + RoPE + QK norm + v copy | 4.12 | 293 GB/s (57 %) |
| SDPA causal | | 1.91 | ~422 GB/s |
| post_add_norm x3 (generic) | post-norms + residual + next pre-norm | 4.85 | ~415 GB/s (81 %) |
| wo (K 1536) / w_ov / ff0 / ff1 | | 1.50 / 1.26 / 1.31 / 1.28 | 46-48 % |
| LayerNorm | pre-norm, layer 0 only | 0.80 | 417 GB/s |
Matmuls 10.15 ms (46.5 %); op-to-op gaps ~6 us per layer.

## Cut layer (14.86 ms)
QKV + rope_qk_norm 8.86 and SDPA 1.52 are needed (every token's K, V). ~4.0 ms is data movement: query slice
(untilize/slice/tilize 2.02), residual take_slots 1.29, reshapes 0.78. Head: two reshapes 0.94 of 0.99 ms.

## Multivariate (44.64 ms/layer, +22.8 vs paper)
Two (1,0,2) permutes in _var_attention_packed 10.86 ms (56-70 GB/s), second QKV 4.69, NlpCreateHeads 3.09, variate
QK norms 1.90, SDPA 1.98, wo 1.49, minus w_ov 1.26.

## Exit criterion (contract §4)
Not met: matmuls ~46 % of device time (bar > 65 %), non-matmul ~54 % (bar < 15 %). The non-matmul remainder is
mostly memory-bound at ~81 % of DRAM; the analyst judges the criterion unreachable without C++ (norm in the matmul
epilogue). The diminishing-returns exit (< 5 % per round) is the operative one.
