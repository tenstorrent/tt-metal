# Speculative decoding scope, Oct 8 2026

Feasible with the installed checkpoint and existing Metal primitives, but not
enabled in our TP4 runtime. The throughput priority is a batched verifier that
preserves useful user concurrency. A batch-one MTP demonstration would not
establish that goal. This is source and checkpoint-metadata analysis; no new
speculative hardware result, accuracy qualification or runtime promotion.

**User decision:** keep speculation out of the main demo whenever it reduces
total throughput. It remains opt-in until matched workload measurements show
more committed output tokens/s at the same offered concurrency, including
drafting, verification, commit/reseed, sampling, and any user scheduling in
waves. Lower latency for fewer users does not meet this condition. Enable only
qualified winning operating points; ordinary decode remains the fallback.

## What already exists

- The pinned checkpoint `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` has
  `mtp_num_hidden_layers=1`, shared embeddings, and 15 `mtp.*` tensors in
  `model-00018-of-00018.safetensors`. Reading only the index and safetensors
  header confirmed **424,699,392 parameters / 849,398,784 BF16 bytes**. No new
  weight download is needed. The drafter is one full-attention layer plus an
  embedding/hidden fusion projection, sharing the base embedding and LM head.
- [Metal PR #55548](https://github.com/tenstorrent/tt-metal/pull/55548) merged
  Oct 2 as `54bb6f982e3fce9ab534b4666ea56be60fca9346`. Its merge is already an
  ancestor of our pinned native base `a08819ddbe23077f8037d3802303939064868ff6`.
  Source reuse does not require rebasing or changing the running installation.
  The relevant model files are `models/demos/blackhole/qwen36/tt/mtp.py`,
  `spec_decode.py`, `spec_sampling.py`, `attention/tp.py`, and `gdn/tp.py`.
- That implementation targets **batch 1**, with MTP prompt-cache warming,
  grouped multi-position SDPA, per-token recurrent-state retention, commit,
  draft reseeding, and rejection sampling. Its published short-context 2.62x
  result is not a forecast for our B32/32K workload. Its greedy comparison
  permits near-tie differences, so its "lossless" headline is not evidence of
  strict token equivalence to our current numerical path.
- Native `spec_multi_pos_tiles` supports multiple candidate positions sharing
  one KV scan per group, with independent causal bounds. Its one-KV-head-per-chip
  restriction matches our TP4 geometry. The source policy uses groups of four;
  larger groups consume L1 and reduce available attention workers.

## Changes needed in our runtime

1. **Draft head and hidden-state interface.** Load/map MTP tensors into TP4;
   reuse our embedding/head without duplicate weights. Expose pre-final-norm
   hidden states from prefill and verification. Warm MTP KV over the prompt
   with the correct `(base_hidden_i, token_i+1)` pairing; retain that alignment
   through accepted/rejected drafts and subsequent turns. Account for the
   added prefill and LM-head work, including repeated head bandwidth.
2. **Batched verification, explicit user and time axes.** Add a verify entry
   point instead of passing speculative positions as independent users to
   `decode_forward`. Batch projections across `B*(K+1)` rows; apply causal SDPA
   with one page-table ownership mapping per real user, and group candidate
   queries to reuse that user's KV. Never share unrelated users' KV. Candidate
   writes inside an existing 32-token page must preserve the prompt prefix;
   padded/inactive rows require a safe sink or masked writes.
3. **Multi-token GDN and selected-prefix state.** Preserve the current FP32
   recurrence math and BF16 activations. The current custom kernel accepts one
   token per user and updates state in place. Verification needs a time loop,
   independent acceptance lengths, convolution history selection, and stable
   addresses for trace replay. The upstream fused primitive restricts
   `B*value_heads <= compute_cores`: 120 cores / 12 heads allows at most ten
   users per call on this Galaxy. Our B16/B32 targets therefore need work in
   waves or a generalized kernel, not a direct replacement call.
4. **Acceptance, commit and scheduler contract.** Per-user accepted-prefix
   counts must advance token position, RoPE, token history/RNG and recurrent
   state consistently; rejected KV positions remain outside the visible
   frontier and are overwritten. Handle zero acceptance, full acceptance, EOS,
   cancellation, context limits and slot reuse. Our current generator advances
   positions by one and returns one token per user. The vLLM adapter also needs
   variable committed-token counts and ownership of draft/verify scheduling.
5. **Sampling and trace lifecycle.** Reuse the upstream CPU rejection sampler
   as an oracle, then avoid full-vocabulary host readback in a throughput path.
   It uses argmax drafts (a delta proposal), so sampling acceptance is target
   probability mass, not the greedy match rate. Qualify our exact temperature,
   top-k/top-p and penalty semantics. Allocate all state/scratch and warm every
   shape before capturing draft, verify, commit and reseed traces.

These are model/runtime additions plus a batched recurrent-kernel adaptation
and wider projection support. They do not require a new model architecture,
training a drafter, another Galaxy, or lowering precision.

## Concurrency: execution rows and memory are different limits

Our fast DRAM-sharded projection family currently requires one 32-row tile
(`M == 1`). If retained unchanged, the row budget is:

| Draft tokens K | Target verification rows per user | Users per 32-row projection pass |
|---:|---:|---:|
| 0 | 1 | 32 |
| 1 | 2 | 16 |
| 3 | 4 | 8 |
| 7 | 8 | 4 |

These are projection-row limits, not proof that those speculative modes are
implemented, and not a requirement to evict resident conversations. Processing
users in waves preserves residency but may lose aggregate throughput. Keeping
**32 users with K=3 requires 128 verification rows**: a wider/fallback matmul
family or an efficient tiling scheme, measured including any extra weight reads.
The same constraint is already relevant to our unfinished B64 ordinary decode.

At the currently measured operating points, K=3 requires 128 rows for 32K/B32,
64 rows for 128K/B16, and 32 rows for 256K/B8. Thus 256K/B8 is a useful prototype
geometry, while the primary 32K/B32 objective needs wider verification.

Memory accounting below is per chip, with TP4, BFP8 KV and FP32 recurrence:

- Base recurrent state: `48 * 12 * 128 * 128 * 4` = **36 MiB/user**.
  Three-token BF16 convolution history adds **0.703125 MiB/user**.
- MTP KV adds one full-attention layer: **6.25% of the base KV storage**, or
  17/68/136 MiB per user at 32K/128K/256K. It does not multiply by draft depth
  for a linear draft chain; only the candidate-tail reserve grows with depth.
- Sharding the MTP matrices four ways and replicating norm vectors costs about
  **202.55 MiB/chip** at BF16, before padding or temporary copies. No duplicate
  embedding or LM-head allocation is included.
- A simple K=3 implementation retaining four complete candidate state planes
  adds **144 MiB/user** of recurrent snapshots, plus convolution history.
  At B32/32K, the snapshots, MTP KV and BF16 MTP weights total approximately
  **5.32 GiB extra per chip**, excluding activations/logits/traces and scratch.
- An alternative preserves the initial state and replays only the accepted
  GDN prefix from retained operands, using one extra state plane. That rough
  extra-memory figure falls to **1.88 GiB/chip** at B32/32K, plus operands and
  workspace, but adds recurrence and commit latency. It is an unimplemented
  tradeoff, not a free optimization. Do not attempt numerical inversion of the
  delta recurrence to recover rejected states.

If resident capacity were entirely limited by per-user KV/state memory, four
snapshot planes would retain roughly 65%/84%/89% of ordinary capacity at
32K/128K/256K, before fixed MTP weights and additional workspace. Replay would
retain roughly 85%/91%/93%. These ratios are not measured maximum user counts:
existing pool reservations, fragmentation and trace/workspace peaks matter.
At exactly the context limit, reserve verification/bonus positions or shorten
K; speculation does not extend the model's 262,144-token context.

## Throughput sensitivity, not a measured prediction

Let `a_j` be acceptance probability at depth j conditional on reaching it.
For K proposals, expected committed tokens per round are
`A = 1 + sum(product(a_1..a_i), i=1..K)` including correction/bonus.
At fixed user count, speedup is
`A * ordinary_step_time / (draft + verify + commit + reseed + sampling time)`.
With a different active batch, compare `B*A/cycle_time` to the measured baseline
TPS; do not multiply the original B32 throughput by a B8 latency speedup.

Illustration for K=3 and a complete cycle costing 1.7 ordinary decode steps:

| Conditional acceptance at each depth | Committed tokens/round | Same-concurrency speedup | Galaxy TPS extrapolation from 32K/B32 |
|---:|---:|---:|---:|
| 30% | 1.417 | 0.83x | 2337 |
| 50% | 1.875 | 1.10x | 3092 |
| 70% | 2.533 | 1.49x | 4177 |
| 85% | 3.187 | 1.87x | 5255 |

Neither acceptance nor cycle cost has been measured here. At 70% acceptance,
costs of 1.5-1.9 ordinary steps imply **+33-69%**, about **3.7-4.7K output
tok/s/Galaxy** against the current 2803 projection. This conditional engineering
target requires preserving concurrency. Long-context KV reuse can help, but
wide queries, draft KV scans, sequential recurrence, snapshots and head sampling
can remove the benefit. High-temperature evaluation may accept far fewer tokens
than greedy coding. No performance credit belongs to unaccepted draft tokens.

All Galaxy figures here are eight-times-TP4 extrapolations, not physical
eight-replica results. The base measurement is 350.417 output tok/s at 32K/B32.
Calculations, checkpoint tensor metadata and assumptions are retained in
[the scope receipt](speculative-decode-scope-v1.json).

## First experiment and decision gates

1. Reuse the existing MTP reference and sampling tests; qualify the head against
   the installed weights with our precision. Capture real acceptance on
   representative coding and GPQA sampling settings, by depth, including zero
   acceptance and EOS. An assumed 70% is not a qualification criterion.
2. Implement a K=3/B8 verifier first to fit the current projection-row budget.
   Test every accepted prefix, page boundaries, uneven user contexts, two
   consecutive requests and slot reuse. Compare FP32 state and next-token
   logits after rejection to ordinary decode, then run the reference evals.
   A sequential-T fallback is a correctness baseline, not a promised speedup.
3. Benchmark draft, verification, commit, reseed and sampling separately at
   16K/32K and 128K/256K; count committed output only, plus added prefill/TTFT
   and actual peak DRAM. Use the existing global lock and persistent bounded
   runner. Do not interrupt the current capacity/profile/shared-QK experiments.
4. Before promoting for B32 throughput, qualify 128-row projection/verification
   and B32 recurrence. Compare ordinary B32 with speculative B8/B16/B32 at the
   same total offered concurrency. Select K dynamically or fall back to ordinary
   decode when drafting loses throughput. Finish physical 8xTP4 and serving
   qualification with the selected policy; component correctness does not
   satisfy the outstanding 89.2% GPQA gate.
