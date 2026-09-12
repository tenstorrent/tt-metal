# Precision-locked projection search

All rows use real checkpoint weights and recorded HF inputs at B1. Times include input/output movement and a warmed trace. PCC is against the quantized-weight operation control; whole-decoder HF PCC, not these values, selects precision. Full sweeps and exact errors are in projection_results.csv. Gate microbench includes its old SiLU epilogue; final separate gate uses the identical unactivated up geometry and SiLU fused into multiplication.

| Layer | Role | Readers | Fastest measured cores/block | Traced us | Quantized-op PCC |
| --- | --- | ---: | --- | ---: | ---: |
| 0 | linear_attn.packed | 1 | 20 / 8 | 174.845 | 0.9999979 |
| 0 | linear_attn.packed | 2 | 10 / 4 | 150.017 | 0.9999978 |
| 0 | linear_attn.packed | 3 | 80 / 2 | 134.626 | 0.9999977 |
| 3 | mlp.down_proj | 1 | 16 / 17 | 173.633 | 0.9999963 |
| 3 | mlp.down_proj | 2 | 8 / 17 | 148.105 | 0.9999963 |
| 3 | mlp.down_proj | 3 | 32 / 17 | 150.519 | 0.9999963 |
| 3 | mlp.gate_proj | 1 | 10 / 4 | 267.955 | 0.9999954 |
| 3 | mlp.gate_proj | 2 | 10 / 4 | 195.215 | 0.9999954 |
| 3 | mlp.gate_proj | 3 | 80 / 2 | 168.574 | 0.9999949 |
| 3 | mlp.gate_up | 1 | 80 / 2 | 547.291 | 0.9999958 |
| 3 | mlp.gate_up | 2 | 10 / 2 | 294.826 | 0.9999958 |
| 3 | mlp.gate_up | 3 | 80 / 2 | 256.683 | 0.9999958 |
| 3 | mlp.up_proj | 1 | 5 / 4 | 185.887 | 0.9999954 |
| 3 | mlp.up_proj | 2 | 10 / 4 | 149.437 | 0.9999954 |
| 3 | mlp.up_proj | 3 | 80 / 2 | 138.146 | 0.9999951 |
| 3 | self_attn.o_proj | 1 | 6 / 16 | 74.923 | 1.0000000 |
| 3 | self_attn.o_proj | 2 | 48 / 4 | 67.569 | 1.0000000 |
| 3 | self_attn.o_proj | 3 | 48 / 4 | 66.618 | 1.0000000 |
| 3 | self_attn.qkvg | 1 | 20 / 8 | 152.353 | 0.9999970 |
| 3 | self_attn.qkvg | 2 | 10 / 4 | 131.872 | 0.9999976 |
| 3 | self_attn.qkvg | 3 | 80 / 2 | 119.709 | 0.9999971 |

Final whole-layer selection differs from isolated down: R3/core32/block17 avoids surrounding movement and beats isolated R2/core8/block17. See movement_down3_l{0,3}.json and final_matmul_rows.csv for actual default kernel shares. Larger legal blocks, fewer storage cores, and all reader counts remain in the CSV even when L1 capacity or native validation rejects them.
