# Phi-3.5 Mini advisor challenger

Outcome: **no change**. The frozen shipped batch-32 decoder remains the fastest measured decoder.

The incumbent was measured before advisor capture with three independent harness runs:
0.657757, 0.656791, and 0.657110 ms. Its best repeat is 0.656791 ms and its
same-configuration spread (the tie floor) is 0.000966 ms.

## Advice and reconciliation

Phi has one dense layer kind across all 32 layers. The batch-32 capture used the shipped
BFP4 attention/gate-up/down projection weights, BFP8 KV cache, and LoFi compute state.
The advisor considered and selected DRAM sharding for all five projection matmuls; the
incumbent already ships that strategy. Its novel material disagreement was the
RMSNorm/residual boundary, totaling 2.296% of the frozen device window. The advisor's
traced LoFi compute configuration was treated as captured state, not advice.

The literal 11-core block-sharded norm could not feed the shipped DRAM-sharded matmul
without an explicit conversion because 96 K tiles are not divisible by its 9-tile shards.
With the required conversion restored, its best of three repeats was 0.667707 ms.
Bracketing the advisor geometry produced:

| Candidate | Batch-32 repeats (ms) | Best | Result |
|---|---|---:|---|
| Frozen incumbent, 16-core width norm | 0.657757, 0.656791, 0.657110 | 0.656791 | ship |
| 8-core block norm chain | 0.668536, 0.668677, 0.667588 | 0.667588 | reject, +1.64% |
| 11-core block norm chain | 0.668339, 0.669732, 0.667707 | 0.667707 | reject, +1.66% |
| 12-core block norm chain | not measurable | — | illegal dispatch-core placement |

All other reconciled structural rows were below the 1% chain threshold. There are no
sparse-matmul or SSM layer kinds in this model, so no layer kind was uncapturable.

## Combination and correctness

Only one material chain survived screening, so there was no pairwise combination to
measure. The best measured set is the incumbent itself. After restoring it byte-for-byte,
the real-weight batch-32 trace-replay oracle passed at PCC 0.9999923310 against the
required 0.995 bar.

The shipped `tt/optimized_decoder.py` is intentionally unchanged: every legal
advisor-derived candidate was slower than the frozen incumbent by more than the noise
floor.
