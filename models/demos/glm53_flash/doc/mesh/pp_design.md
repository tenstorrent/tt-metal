# Pipeline parallelism for GLM-5.3-Flash on the 8x4 Galaxy: design study (2026-10-22)

Question (user, 2026-10-22): could we go the PP route, as the DeepSeek-V4.1-Flash "TTNN (TP=4) + PP (PP=8)"
numbers suggest (B=8: 28-30 tok/s per user from 32 K to 512 K; TTFT 15 / 31 / 58 / 143 s at 32 K / 64 K / 128 K /
256 K; versus 14-16 tok/s for their "TTNN TP=8, EP=4" baseline at B=4 with k=5 speculation)? This note estimates
what PP=8 x TP4 would do for GLM-5.3-Flash against the measured TP4 x DP8 hybrid layout, from the numbers in this
doc and the decode ledger (`_agents/decode_ledger/LEDGER.md`). Nothing here was run.

## 1. The two layouts

| | TP4 x DP8 hybrid (today) | PP8 x TP4 (proposed) |
|---|---|---|
| dense weights (attention, mHC, shared expert, head) | one copy per row, TP4: ~8.2 GiB per chip | row r holds layers of stage r only, TP4: ~1.0 GiB per chip |
| routed experts (~300 GB bfp8) | one copy on the mesh: 72 experts per chip, intermediate split over the 8 rows (9.5 GB per chip) | one copy: row r holds the 288 experts of its ~6 layers, full width, 72 per chip (9.5 GB per chip) |
| per-layer collectives | TP all-reduce over 4 columns + row all-gather x2 + row reduce-scatter (the experts' row split) | TP all-reduce over 4 columns only |
| inter-stage traffic | none | one activation hop per layer group: 4 streams x 4096 bf16 = 32 KB per user |
| users per step | 4 per row (32 users) decode together in one 45-layer step | each row runs its stage on the micro-batch in flight; 8 micro-batches = 8 steps in flight |
| per-user latency per token | sum of 45 layers (83 ms at 32 users) | sum of the 8 stage times ~ the same chain minus the row collectives |
| throughput at B >= 8 micro-batches | 32 users / 83 ms | the same user count per round; all rows busy |
| caches (latent + pooled index, 11 DSA layers) | a user's cache for every DSA layer, 1/4 per column chip of its row | a user's cache for the DSA layers of stage r on row r, 1/4 per column chip; every row holds caches for all users |

PP on one Galaxy is naturally an SPMD ring shift: every row runs the same 6-layer program with its own stage's
weights (the layer dimension sharded over the rows at upload), one trace serves all 8 stages at once, and the step
ends with a ring shift of the activations to row r+1 (row 7 -> row 0 carries the new token ids to the embedding).
The per-slot metadata (positions, KDA state, DSA page tables) already lives per row.

## 2. Decode estimate

The decode step is latency-bound (batch 1: 80 ms, 4 users per row: 83 ms, 32 users per row: 115 ms). Per KDA+MoE
layer at 32 users (EP32 profile, hybrid comparable): mHC 475 us, experts ~340 us, KDA ~240 us, row collectives
~70-100 us plus skew, TP all-reduce ~50 us, dispatch gaps ~64 us; DSA layers ~1.9 ms (108 ops).

| item | hybrid 32 users | PP8 estimate | note |
|---|---|---|---|
| row all-gather x2 + reduce-scatter per layer | ~80 us x 45 = 3.6 ms | 0 | removed |
| inter-stage hop | 0 | 8 hops x ~20-40 us = 0.2-0.3 ms per token | 128 KB per hop at 4 users |
| stage imbalance | 0 | 45 layers / 8 = 5.625 -> 6/6/6/6/6/5/5/5 layers, 11 DSA layers at 1-2 per row: the slowest row sets the step; ~7-15 % unless boundaries are chosen by latency | balance tool from the ledger's per-layer times |
| per-op latency (mHC, KDA, DSA, experts) | ~75 ms | ~75 ms | identical ops; the ledger's A3/A1/B5/B7 items apply to both |
| **step at 32 users** | **83 ms** | **~80-85 ms** | no gain at equal batch |

PP does not improve per-user tok/s at the same batch: the chain is the same 45 layers of the same ops. The
DeepSeek page's 2x is against their TP8/EP4 baseline, not against a hybrid layout, and their layer has no mHC. The
decode gap to the 33 / 19 tok/s targets is per-op latency (mHC 0.47 ms, experts, KDA, DSA fusion), which is layout-
independent, and MTP at low batch.

## 3. Memory estimate (the real decode-side win)

Per chip: dense 8.2 -> ~1.0 GiB. Caches: today each column chip holds 1/4 of its row's 4 users' caches for all 11
DSA layers (11.7 GiB at 1 M x 32); under PP it holds 1/4 of ALL 32 users' caches for its stage's 1-2 DSA layers:
(1.5 / 11) x 8 = 1.1x on average, 1.45x on a row with 2 DSA layers (17 GiB at 1 M x 32).

| | hybrid | PP8 (row with 2 DSA layers) |
|---|---|---|
| dense weights | 8.2 GiB | 1.0 GiB |
| experts | 9.5 GB | 9.5 GB |
| caches at 1 M x 32 users | 11.7 GiB | ~17 GiB |
| total of 31.3 GiB | 29.4 | ~27.5 |

So PP frees ~2 GiB on the worst row at 1 M x 32 and ~7 GiB on rows with one DSA layer: the planner's envelope grows
(roughly 40-48 users at 1 M instead of 32, more at shorter ISL) or the stacked expert copy always fits.

## 4. Prefill estimate (the real TTFT-side win)

Today one prompt is sequence-parallel over the 8 rows: each row runs its 2 K-row slab of a 16 K chunk through all 45
layers with TP4, the latent is all-gathered over the rows per layer, the KDA scan is sequence-parallel, and the
whole-prefix copy is rebuilt per chunk per column; 2.2-2.4 K tok/s on 32 chips = ~70 tok/s per chip, far from
compute bound (the 16 K-chunk profile is dominated by the indexer scoring, the prefix copy and small-M matmuls).
Under PP a row runs the full 16 K-row chunk through its 6 layers (8x the M per matmul), no cross-row collective per
layer, and 8 chunks of one or several prompts are in flight.

| | hybrid SP | PP8 |
|---|---|---|
| matmul M per row per layer | 2 K rows | 16 K rows |
| cross-row collectives per layer | latent all-gather + KDA scan exchange | none (one hop per chunk per stage) |
| prompts in flight | 1 | up to 8 chunks (several prompts) |
| expected aggregate rate | 2.2-2.4 K tok/s | 2-4x (estimate; needs the per-row 6-layer prefill timing) |
| per-prompt TTFT at 1 M | 460 s | 460 s / (2-4) if one prompt fills the pipeline with its own chunks; chunk i+1 can start on row 0 when chunk i reaches row 1 |

The DeepSeek PP TTFTs (15 s at 32 K, 143 s at 256 K for 8 users) are per-batch and not directly comparable; our
per-prompt 64 K TTFT is 27 s today.

## 5. Costs and open points

1. Stage balance by latency, not layer count: DSA layers are ~1.5x a KDA layer; the slowest row sets the step.
2. Caches concentrate on rows with 2 DSA layers (section 3); the planner needs a per-row model.
3. A new runtime: per-row stage weights (layer-sharded upload), one SPMD trace with a row-shift collective, a
   host-side micro-batch scheduler (slot = (micro-batch, user)), prefill chunk pipelining, the embedding on row 0 and
   the head on row 7 (or the head replicated: 1.3 GB of bfp8 vocab per row is affordable), MTP as a row-7 extension.
4. Below 8 micro-batches in flight the rows idle: batch 1-7 keeps the per-user latency but loses throughput against
   DP8; at B >= 8 the two are equivalent in throughput.
5. The B1 pod pipeline in this tree (`models/demos/deepseek_v3_b1`: MPI ranks, fabric-socket activation FIFOs, one
   DecoderStage per layer, 4/16/64 ranks) is the multi-host design and much heavier than a single-mesh row shift;
   the page's "TTNN (TP=4) + PP (PP=8)" implementation is not in this tree (ask where it lives before reusing).

## 6. Recommendation

Not now. Keep TP4 x DP8 for the optimization phase: the decode gap is per-op latency and MTP, both layout-independent,
and PP would not change the 83 ms step. Plan PP8 x TP4 as the capacity layout after that: ~1.3-1.5x the users at 1 M
and a 2-4x prefill throughput ceiling, with the SPMD row-shift as the implementation route. First measurement when the
chips are free: the per-row 6-layer prefill rate on a 16 K chunk (one row, TP4) against the SP slab rate, which fixes
the prefill estimate in section 4, and the per-row stage times from the ledger for the balance tool.
