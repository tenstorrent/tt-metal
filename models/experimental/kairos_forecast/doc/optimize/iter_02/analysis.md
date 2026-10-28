# Round 2 analysis (paper/accuracy, 512 rows; profile job 1253 on 3ee133f0653, reduced 2+2, eager)

Written from the read-only analyst's report (Opus, low effort). Reduced kernel time 50.25 ms; full-depth estimate 136.88 ms, against the traced bench's 137.03 ms (job 1252).

- **Time by role (reduced):** linears 13.62, RoPE mul/mul/add 8.14 (36 ops), head splits 5.74, embed 4.86, attention matmuls 3.57, mask add 3.42, softmax 2.91, residual adds 2.06, norms 1.92, head 1.74, typecast 1.46, concat heads 0.83.
- **Top ops:**
  - emb_h 1.94 ms (60 % of HiFi4 peak).
  - **emb_o 1.59 ms and emb_r 0.85 ms on 80 cores at 23-34 %:** pm = 103 is prime, so linear_config picks out_block_h 1.
  - Encoder FF wi / wo about 1.0 ms each, at 55-57 %.
  - Encoder fp32 mask add and softmax 1.0 ms each, DRAM-bound at about 300 GB/s.
  - QKV linears 0.85 ms (49 %), NlpCreateHeads 0.72 ms, QK^T 0.65 ms (8 %).
- **Per encoder block (12.49 ms):**
  - The fp32 score path is 3.55 ms (QK 0.65, mask 1.0, softmax 1.0, typecast 0.5, PV 0.4).
  - RoPE is 2.03 ms; the second QKV linear plus its split is 1.57 ms; the linears are 4.0 ms.
- **Oddities:**
  - The decoder causal mask add is 6x slow (a [1,1,32,32] broadcast over B*8 in BinaryNg).
  - The dead v / q columns cost about 0.5 ms per encoder block.
  - The N = 512 linears use 80 of 110 cores.
  - All eltwise ops are DRAM-interleaved and DRAM-bound.
  - The BinaryNg shape columns in the CSV are stale program-cache labels.
