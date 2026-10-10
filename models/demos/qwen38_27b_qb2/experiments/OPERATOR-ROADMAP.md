# Operator optimization scope and priorities

October 10, 2026 UTC. Primary workload: B16/32K on one TP4 replica; 16K is
secondary, with 128K/256K regression checks at feasible concurrency. Retain
BFP8 weights/KV, BF16 activations and FP32 recurrent state. Eight replicas
require a separate physical Galaxy measurement; TP4 results are not multiplied
and labeled measured Galaxy throughput. This continues the Metal path.

## Evidence and boundaries

- Qualified shared-QK baseline: 67.245 ms/token, 14.871 tokens/s/user and
  237.94 aggregate decode tokens/s/TP4. Full GPQA: 178/198 (89.90%); all
  questions remain in the denominator, including five output-budget cutoffs.
- New direct-preparation/epilogue candidate: three completed B16/32K measurements
  give 60.422 ms/token, 16.550 tokens/s/user and 264.80 aggregate decode
  tokens/s. After-control drift/hash comparison and full GPQA
  remain pending at this snapshot. This is not a promoted configuration.
  [Raw candidate snapshot](../galaxy-evidence/operator-scope-v1/candidate-sweep.json)
  retains source/config hashes and every measured sample.
- Full timing profile covers 64 layers, four ranks, three replays and all
  38 operation types / 5001 device-operation rows per rank/replay. Device
  spans reconcile with fenced host steps within 0.403-0.461%. These counts
  are not program counts. Profile overhead is 7.46%; never subtract it
  uniformly from individual operations or add family medians as wall time.
- The full profile is the shared-QK baseline. Current candidate two-layer
  profiles locate opportunities remaining after preparation/epilogue fusion;
  multiplying them by layer count is an extrapolation, explicitly labeled.
- No physical DRAM-utilization, active-compute or NoC-congestion counter
  conclusion follows from wait-inclusive RISC intervals. No new full decode
  P0 run is needed to choose the next interventions. A matched prefill stage
  profile is still needed before assigning prefill compute-stage savings.

The [complete inventory and matmul table](../galaxy-evidence/operator-scope-v1/INVENTORY.md)
are generated from the retained CSV and checked against every rank/replay's
independent timing summary. [Raw capture](../galaxy-evidence/p0-priority-v1/README.md).

## Matmuls are optimized, with uneven remaining room

The selected path already uses DRAM-sharded weights, L1-sharded activations,
per-role core/block choices, 2-3 workers per DRAM bank, packed projections,
FP32 destination accumulation and HiFi2. These are not untuned generic matmuls.
The current defaults retain earlier selected topology choices; that does not
establish each choice as optimal for today's BFP8/B16 workload.

| Projection family | Profiled kernel sum/step | Encoded-weight bandwidth / assumed 512 GB/s peak |
|---|---:|---:|
| MLP gate/up | 7.59 ms | 83% |
| MLP down | 4.35 ms | 68% |
| GDN packed projection | 3.35 ms | 70% |
| Attention/GDN output projections | 1.83 ms | 57% |
| Full-attention packed projection | 0.87 ms | 70% |
| Vocabulary head chunks | 0.90 ms | about 71-74% |

The aggregate is about 74%. These estimates count one read of every declared
padded BFP8 matrix tile, not actual bus transactions, activations or complete
compute costs. Thus they are bandwidth-floor comparisons, not a calibrated
combined compute/memory roofline. Output and down projections have a stronger
case for retuning than the already faster gate/up projection. Even ideal
weight streaming alone cannot eliminate the full-model bandwidth gap.

## Prioritized decode experiments

Savings below are engineering targets unless explicitly measured. They refer
to a full B16/32K step, exclude prefill and are not independently additive.
Ownership columns identify overlapping operations. A 1-ms saving from the
67.245-ms baseline gives about 1.5% more decode throughput; use
`speedup = old_step / (old_step - saved_ms)` for larger changes.

| Order | Change and ownership | Expected benefit | Evidence, effort and risk |
|---|---|---|---|
| 0 | Finish direct preparation + fused GDN output epilogue qualification | 6.8 ms; ~11% decode. All three B16 measurements agree. | Already running full control/candidate/control and GPQA. Exact four-rank real-layer state/output checks passed. No precision change. Finish this before selecting a new serving baseline. |
| 1 | Compact GDN convolution/history, compact packed-projection consumption, and gate preparation | Combined target 4-6 ms beyond the current fusion candidate, roughly 7-11% of that candidate's decode throughput. | Three convolution-output tilizations account for ~3.50 ms; projection layouts add ~2.68 ms, both from representative-layer extrapolation. Replacement kernels have their own cost. Standalone prototype exists; physical test queued. Integration remains. Moderate implementation risk, low intended arithmetic risk. |
| 2 | Retune output/down matmuls first, then GDN packed projection; include input/output conversions | Target 1.5-3.5 ms across matmuls, ~2-6% over the original baseline. This is a target range, not a demonstrated 90%-bandwidth result. | Vary reader count, bank/worker placement, K blocking and output sharding together. Measure the complete projection boundary. Avoid losing more to padding/resharding than the kernel saves. Existing gate/up is lower priority. Medium effort/risk. |
| 3 | Production SDPA reader/compute delivery: chunk size, worker distribution, tagged lookahead and bank-local delivery | Target 1-2.7 ms, ~2-4% of the baseline. | SDPA is 12.63 ms; modeled KV-only floor is 8.91 ms at peak or 9.90 ms at 90%. Required math/reduction adds cost. Existing remote-delivery prototype loses to production and must not be promoted. Medium/high effort; preserve accurate exponentiation, FP32 accumulation and page ownership. |
| 4 | Recurrence/preparation scheduling with shared Q/K and FP32 state | Target 0.5-1.5 ms after the current fusion; remeasure the boundary before credit. | Baseline generic kernels total 5.29 ms, including preparation. Sweep state work partition/placement and buffering; retain the existing operation order first. Long-horizon 4096-step changing-input reference, rebinding and trace tests are required. Medium/high risk. |
| 5 | Fuse residual add + RMSNorm and finish MLP SwiGLU in compatible L1 layouts | Target 0.5-1.5 ms beyond existing compact MLP/residual paths. | Norm kernels total 1.08 ms and binary/unary kernels 2.36 ms, shared across several candidates. Do not count the whole family. Current packed MLP already fuses SiLU into multiply. Further projection epilogues compete for pack/SFPU/register resources. Medium risk to rounding. |
| 6 | All-reduce placement, buffer reuse and tiled projection/collective overlap | Target 0.3-0.8 ms plus any separately measured producer overlap. | All-reduce totals 2.61 ms. Existing two-link, persistent-buffer, direct-all-reduce path is already selected. Alternative fused collective paths exist in source but are not qualified substitutes. Changing reduction order requires accuracy checks. Medium/high integration risk. |
| 7 | Q/K norm + RoPE + head-layout boundary; write final cache/attention layout directly | Target 0.3-0.7 ms. | Batched RoPE wrapper is ~1.01 ms across 16 layers, but rotation arithmetic itself only 0.137 ms in the full profile. Savings mostly belong to the layout budget. Preserve rotary dimensions, per-user positions and BFP8 cache packing. Medium effort/risk. |
| 8 | Vocabulary head layout and sampling candidate delivery | Target 0.1-0.3 ms excluding matmul savings already in row 2. | Head math is ~0.90 ms; disjoint sampler ~0.48 ms, top-k kernel 0.272 ms. Streaming candidates must preserve exact top-k/top-p, penalties and RNG semantics. Small payoff relative to correctness risk. |
| 9 | Remaining cache updates, token/history maintenance, embedding and scalar ops | Usually <0.1 ms individually; fold into existing work only when nearly free. | Fused KV update 0.071 ms, embedding 0.007 ms, position increment 0.003 ms. History copy savings belong to row 1. These are not current standalone tuning priorities. |

For rows 1-3, reserve 10-30 minutes of accelerator time for a bounded
standalone sweep after compilation, then roughly 1-2 hours for matched
full-model B16 controls across 32K/16K and roughly 1-2 hours for full GPQA
when arithmetic/scheduling changes. These are planning allowances, not a
promise that kernel development or a queued job finishes within that time.
Core-placement changes require physical timing; a simulator cannot rank them.

The 70% useful-byte requirement corresponds to approximately 35.95 ms/token
or 27.8 tokens/s/user under the existing traffic model. The current candidate
is around 60.42 ms: about 24.5 ms still separates it from that target. The
incremental rows above do not prove that gap is fully recoverable. Reaching
the requirement likely needs a larger persistent GDN/decoder tile pipeline
that removes additional materializations and overlaps compatible stages.
Its gains replace overlapping rows above; they cannot be added a second time.

## Scope across every current operator family

| Current family | Optimization scope | Existing work / boundary |
|---|---|---|
| Matmul | Bank/worker mapping, blocks, padding, pack format, fused epilogues, L1 consumers | Already DRAM-sharded and role-tuned. Retune weaker families in row 2. |
| SDPA | KV delivery, page lookup, split/reduction balance, chunk sizes, accurate partial-query implementation | Split-KV/double buffering already exist. Approximate partial-query path is not acceptable. |
| Tilize/untilize, reshape, slice, pad/fill-pad, typecast, concat, transpose, reshard, interleaved/sharded conversions | Keep compact row/head geometry across boundaries; reader/writer address transforms; persistent correctly sized output buffers | Full-profile sum 22.58 ms. This is a shared opportunity pool, not removable time or an extra additive row. |
| Generic GDN preparation/state kernels | Eliminate redundant operand reads, normalized shared-Q/K reuse, state bank placement and compute/dataflow pipeline | Shared-QK and in-place FP32 state already selected. Decode and prefill math paths differ. |
| Causal convolution | Compute retained row only; read old history once, update disjoint ranges in place, write compact tiles | Native packed-batch convolution already replaces per-user calls at B8/B16/B32. Compact replacement unqualified. |
| Gated RMSNorm | Fuse recurrence output preparation, gating/norm and z multiplication | Current candidate covers this. Do not claim its gain again under general eltwise fusion. |
| RMSNorm, binary/unary, reduce | Fuse residual/gates/SwiGLU where dependency and rounding permit; retain activations in L1 | Some activation fusion already exists. Full-row/head reductions still require completed inputs. |
| All-reduce/all-gather | Persistent buffers, layout alignment, link/worker placement, progress/credit-based tiled overlap | Communication shares NoC and depends on complete partial sums. No free overlap assumption. |
| QKV-head creation, RoPE, paged fused cache update | Align producer output with final consumer/cache; fuse neighboring transformations | KV update arithmetic itself is small; do not risk page mapping for negligible gain. |
| Top-k route prep/finish, top-k, sampling, seed | Avoid unnecessary full-vocabulary materialization; preserve exact sampling semantics | Device sampler already avoids per-step full-logit host readback. Low priority. |
| Copy/indexed-fill, plus-one, embeddings | Remove copies made unnecessary by a fused producer; retain trace-address and history invariants | Trace-owned buffers, slot reuse and independent-user ownership remain mandatory. |

## Prefill and serving: a separate priority lane

The B16/32K workload with 128 output tokens currently spends about 98.5 s in
prefill versus 7.67 s in steady decode. Another 10% decode speedup saves only
about 0.7 s of that request; a 10% prefill speedup saves about 9 s. Both matter,
but their effect on short-response end-to-end throughput is very different.

| Priority | Work | Expected benefit and gate |
|---|---|---|
| P1, already queued | B16 prefill budget 32K -> 64K with matched 32K/64K/32K controls | Earlier 5,321 -> 5,870 input tokens/s, about 10%. Expect roughly 99 -> 90 s TTFT and ~9% all-in gain for 128 output tokens if reproduced. B32 allocation previously failed; do not adopt it globally. Changed chunking needs GPQA. |
| P2, profile first | Batched full-attention cache fills and prefill attention calls | Source loops over users for cache writes and chunked SDPA while projections are batched. Measure this family before rewriting it. A 2x speedup in a stage occupying 30% of prefill would give 17.6% total prefill speedup; this is sensitivity, not a measured forecast. |
| P3, profile first | Prefill GEMM tiling, sharded residual/norm continuity, intermediate allocation reuse | Tune with larger M, independently of decode. The 40%-MFU target is not established by decode bandwidth. No numerical prefill-kernel gain is assigned without current stage timings. |
| P4 | Prefill GDN scan/convolution: workspace reuse, layout fusion, chunk/batch parallelism | Preserve recurrent state and chunk chronology. One-token decode kernel is not a prefill substitute. Profile scan/preparation separately; gain unquantified. |
| P5 | Scheduler chunked prefill and mixed-load admission | Reduce decode pauses/TTFT tails under arrivals, not automatically raw isolated throughput. Model-internal chunking already exists. Current BFP8 plugin/state/sampler/cancellation gates remain. |
| P6 | Prefix reuse and recurrent-prefix snapshots | Can avoid repeated work on shared conversation prefixes, but FP32 GDN state/history must correspond exactly to each cached frontier. Workload-dependent gain; fresh-prompt benchmarks must not receive prefix-reuse credit. |
| Longer-term | Mixed prefill/decode rows through shared matmuls and explicit producer/consumer pipelines | Weight sharing can amortize reads. Scheduler, attention, GDN chronology and resource contention remain. No current measurement justifies replacing sum(memory,compute) with max(memory,compute). |

## Deferred or rejected paths

- BFP4 weights: earlier matched HiFi2 B16 results gave roughly 8.6-9.3% full
  decode gain. Corrected-harness BFP4 GPQA is not qualified. It is a separate
  accuracy experiment, not a prerequisite for the BFP8 optimization plan.
- BFP4 KV: prior simulator experiment showed about 16% attention-output RMS
  error versus roughly 1-2% for BFP8. No shipping accuracy qualification.
- BF16 recurrent state: additional long-horizon/eval qualification required;
  FP32 remains selected. No state-precision speedup is counted above.
- Speculative decoding: keep off the main demo path unless accepted-token
  aggregate throughput improves at the same resident/active concurrency.
  Draft/verify/state snapshot and wider projection-row costs are unresolved.
- TP8 or new multi-host layouts: separate topology experiments, with fewer
  replicas and different communication/capacity tradeoffs; not an assumed
  improvement over eight TP4 replicas at B16.
- Blindly increasing readers, barriers or queue depth: previous delivery
  sweeps already found regressions. Evaluate complete delivery+consumer cost.
- Repeating broad decode P0 or targeting the 0.336-ms uncovered gap first:
  insufficient payoff; existing capture already identifies the main costs.

## Acceptance and next actions

Finish current fusion qualification; retain the original baseline if controls,
output comparisons or GPQA fail. Run the existing B16 prefill-budget and
compact-front-end physical queue unchanged. The auxiliary simulator hit an
unsupported SETDVALID/source-format interaction in compact preparation after
an explicitly fenced convolution; no numerical comparison passed. Physical
testing remains necessary, and the simulator/native runtime is unmodified.

Prepare separate follow-ups for compact GDN integration and weak-projection
retuning. Every experiment must record original/candidate/after controls,
source/config hashes, changed-input trace replay, all-rank correctness, memory
peaks, per-stage and full-step timing. Include input throughput, TTFT, decode
TSU and all-in throughput; protect 16K and report 128K/256K tradeoffs explicitly.
Preserve BF16 rounding boundaries when claiming bit equivalence. New arithmetic
requires the appropriate long-horizon/dense reference and unchanged full GPQA
protocol before serving promotion. Run a full Galaxy scaling/HTTP measurement
after selecting the winning TP4 candidate. Packaging/Shield CI and agentic-eval
qualification remain separate release gates.

## What reaching 30 TSU at B16/32K would require

**Current user requirement: 30 native tokens/s/user at B16/32K; no speculative
decoding.** This supersedes the earlier conditional MTP scope. Retain BFP8
weights/KV, BF16 activations and FP32 recurrent state. Historical speculative
estimates remain archived, but speculative work is not an active experiment or
an acceptable way to meet this target.

The measured 60.4222-ms candidate must reach 33.3333 ms: another 27.0889 ms
removed, or 44.83% less step time / 81.27% more output throughput. Native
20 TSU requires 50 ms, still 10.4222 ms below the candidate. Small independent
knob changes do not establish either target.

The current useful-byte model is about 12.883 GB/chip/step: 7.112 GB encoded
weights, 4.563 GB KV reads and 1.208 GB recurrent-state traffic. At an assumed
512 GB/s, 30 TSU requires 75.5% useful-byte bandwidth; 70% yields 27.8 TSU
before accounting for omitted traffic, dependencies and compute. This is a
necessary traffic budget under those assumptions, not a sufficiency proof or
a measured physical utilization target.

The largest fixed-precision native opportunity is a compact decoder pipeline:
packed projection directly into GDN operands/history, compact recurrence and
epilogue, then compatible projection/collective/residual/norm/MLP layouts.
The baseline layout family was 22.58 ms in the instrumented profile; some
of this has already been removed by the current candidate. It cannot all be
counted again. The queued compact-convolution prototype is only the first
piece (combined front-end target 4-6 ms), not the complete fused decoder.
A broader 10-15 ms reduction is an ambitious engineering target for combined
layout/fusion work, not demonstrated savings; alone it gives about 19.8-22.0
TSU from this candidate. Native 30 therefore also needs substantial improvements
to matmul, SDPA delivery and remaining recurrence/collective costs.

At 80% effective useful-byte bandwidth, streaming alone takes about 31.45 ms,
leaving only 1.88 ms of the 33.33 ms target for unhidden work. At 90%, it takes
27.96ms, leaving 5.37 ms. These are necessary traffic budgets, not forecasts:
they omit extra physical traffic and cannot establish actual DRAM utilization.
The implication is to prioritize complete operator boundaries with compatible
L1 layouts, fused producer/consumer stages and pipelined delivery. Merely
placing the existing operators under one launch cannot remove their traffic.

The next implementation milestone is the opt-in `single_step_compact_gdn`
policy. Packed projection feeds convolution and z directly from compact L1;
convolution updates disjoint history in place; preparation consumes compact
Q/K/V; the gated epilogue writes compact L1 output for the existing output
projection and TP reduction. Only the64 scalar-gate channels still expand to
public rows. This is a complete GDN boundary prototype, not a megakernel or a
30-TSU result. Its incremental 4-6 ms target is included in the broader 10-15 ms
layout/fusion target above; never add the two.

The remaining large intervention is to join compatible stages inside the
device program: projection output delivery into gating/preparation, recurrent
output into normalization, and projection/collective/residual boundaries.
First compare the compact block including all native matmuls and collectives,
then use the remaining measured stage cost to select the fused boundary.
Existing baseline family timings alone do not prove another 27 ms recoverable.
If the remaining target budget is not met, report that gap instead of presenting
small tuning gains or a speculative multiplier as completion.

The [projection sweep](../galaxy-evidence/projection-sweep-launch-v1/README.md)
is a persistent follower behind fusion/GPQA and the original B16 priority
queue. The [compact-block experiment](../galaxy-evidence/compact-gdn-launch-v1/README.md)
now precedes the projection follower, preserving its frozen source. The larger
compact-pipeline work remains the primary native opportunity;
projection tuning is a bounded supporting experiment, not a promised path to 30.
