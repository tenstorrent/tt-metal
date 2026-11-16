# moe_decode: batch-1 routed-expert MoE decode

Python `ProgramDescriptor` ops (`op.py`) with their kernels in `kernels/`, in the style of
`models/demos/gemma4/custom_ops`. Batch 1 only: batch 2-32 uses the expert-parallel path (WS1). Built for the QB2
1x4 Blackhole mesh with tensor-parallel experts (each chip holds its slice of every expert's intermediate dimension).

## Programs per decode step and layer

| Program | Cores | What |
|---|---|---|
| `router_topk_b1` | 1 | FP32 router logits of row 0 -> top-k, softmax over the picks, x per-expert scale, written as the packed routing row `[w_0..w_{k-1} \| id_0..id_{k-1}]` (bf16, the `RouterTopk.op_packed` format). Integer arithmetic (the data-movement cores have no FPU). |
| `MoeDecodeB1.gate_up_local` | 3 per DRAM bank + reducers | Bank readers sit on their bank's row and read only that bank: K slice b of their column pairs for each routed expert (one contiguous chunk per expert). `custom_mm` on 1x32 activation tiles gives fp32 partial (g, u) tiles, pushed to one reducer core per (slot, pair) unit. A reducer sums the banks (SFPU row adds), adds the gate / up bias (optional), applies GeGLU or SwiGLU and the routing weight, and writes one 1x32 h tile. |
| `MoeDecodeB1.down` | one per output column, on its bank's row | The k experts' h rows are concatenated along K, so one `custom_mm` over K = k x I/32 tiles gives `sum_s w_s h_s Wd_s`. The optional down bias rides in a zero-padding row of the weights (the reader sets h there to w_s). Writes row 0 of the output tile. |

Weights are DRAM width-sharded, one shard per bank (`gate_up_local_layout`, `down_layout`; uneven splits when the
tile count does not divide by the banks, e.g. H 2880). Every bank serves the same bytes for any routing.

## Contract

- `x`: `[1, 1, 1, H]` bf16 TILE interleaved (row 0 used).
- `route`: `[1, 1, 32, 2k]` bf16 ROW_MAJOR, row 0 = k weights then k global expert ids (exact below 256).
- Output: `[1, 1, 1, H]` bf16 TILE L1; only row 0 is written (the batch-1 consumers read row 0).
- Limits: H and the per-chip I multiples of 32; 3 bank readers need (2 * I/32 / 3) <= 4 output tiles per DST batch
  (looped), k <= 32; `custom_mm` is LoFi with fp32 accumulation.

## Measured (QB2, BFP8, `perf/tools/moe_decode_bench.py`)

Gemma 4 26B (H 2816, I 704 -> 192 per chip, k 8): 13.8 MB of expert weights per chip in 34.4 us = 400 GB/s per chip
(gate|up 24.2 us, down 13.1 us). In the model the batch-1 router + experts take 6 programs.

## Tests

`tests/test_moe_decode.py`: vs torch per chip for Gemma (E 128, k 8, GeGLU) and gpt-oss (H 2880, I 2880 -> 768 per
chip, E 32, k 4, SwiGLU, biases) shapes, BFP8 and BFP4; and `router_topk_b1` vs the HF router tail.
