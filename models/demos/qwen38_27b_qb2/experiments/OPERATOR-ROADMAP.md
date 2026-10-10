# Operator optimization scope and priorities

October 10, 2026 UTC. Primary workload: B16/32K on one TP4 replica; 16K is
secondary, with 128K/256K regression checks at feasible concurrency. Retain
BFP8 weights/KV, BF16 activations and FP32 recurrent state. Eight replicas
require a separate physical Galaxy measurement; TP4 results are not multiplied
and labeled measured Galaxy throughput. This continues the Metal path.

## Evidence and boundaries

- Compact GDN: completed B16/32K measurements give **49.913 ms / 20.035 TSU**,
  +20.81% against the fresh 16.584-TSU before-control. At 16K it gives
  43.955 ms / 22.751 TSU (+23.60%). All three repeats per context match the
  baseline's output hashes and the candidate cleaned up successfully.
  The after-control passed with 0.00202% drift at 32K and 0.000702% at 16K.
  Eight-replica G0/API checks passed; compact GPQA passed **177/198 (89.39%)**
  in 59m 53s. Five output-budget cutoffs count incorrect; there were no context
  cutoffs. The completed-answer audit independently matches the score.
  The eval measured 804.2 aggregate output tokens/s across variable requests;
  saturated whole-Galaxy throughput remains unmeasured.
  [Completed qualification and source boundaries](../galaxy-evidence/compact-qualified-v1/README.md).
- Qualified shared-QK baseline: 67.245 ms/token, 14.871 tokens/s/user and
  237.94 aggregate decode tokens/s/TP4. Full GPQA: 178/198 (89.90%); all
  questions remain in the denominator, including five output-budget cutoffs.
- Earlier direct-preparation/epilogue candidate: three completed B16/32K measurements
  give 60.422 ms/token, 16.550 tokens/s/user and 264.80 aggregate decode
  tokens/s. The completed after-control drift is +0.003082% with identical
  output hashes. Full GPQA subsequently passed at 177/198 (89.39%); six output
  cutoffs remain incorrect in the denominator. Serving promotion remains separate.
  [Raw candidate snapshot](../galaxy-evidence/operator-scope-v1/candidate-sweep.json)
  retains source/config hashes and every measured sample.
- Earlier shared-QK full timing profile covers 64 layers, four ranks, three replays and all
  38 operation types / 5001 device-operation rows per rank/replay. Device
  spans reconcile with fenced host steps within 0.403-0.461%. These counts
  are not program counts. Profile overhead is 7.46%; never subtract it
  uniformly from individual operations or add family medians as wall time.
- That earlier full profile is the shared-QK baseline. Subsequent candidate two-layer
  profiles locate opportunities remaining after preparation/epilogue fusion;
  multiplying them by layer count is an extrapolation, explicitly labeled.
- The completed compact profile now covers 33 operation types / 2553 records
  per rank/replay with matching outputs and 4.43% whole-step profiling overhead.
  Its largest kernel sums are matmul 18.987 ms, attention 12.627 ms, custom GDN
  stages 9.184 ms and all-reduce 2.647 ms; these are not additive wall time.
  [Compact profile and interpretation](../galaxy-evidence/compact-profile-v1/README.md).
- No physical DRAM-utilization, active-compute or NoC-congestion counter
  conclusion follows from wait-inclusive RISC intervals. Overlapping firmware
  lifetimes also do not establish useful compute overlap. A matched prefill
  stage profile is still needed before assigning prefill compute-stage savings.
- Nine B16 compact output-projection candidates passed accuracy but all lost to
  the baseline. An oversized down-projection block then hit an L1 allocation
  clash. The failed attempt is preserved, and the remaining sweep/followers
  were relaunched with block-34 variants deferred; no model change was promoted.
  [Failure and recovery](../galaxy-evidence/projection-l1-recovery-v1/README.md).

The [complete inventory and matmul table](../galaxy-evidence/operator-scope-v1/INVENTORY.md)
are generated from the retained CSV and checked against every rank/replay's
independent timing summary. [Raw capture](../galaxy-evidence/p0-priority-v1/README.md).

The [Shield reference audit](../galaxy-evidence/shield-reference-38027117236/README.md)
identifies additional prefill/serving transfers: long-prompt traces with device
chunk offsets, fused all-gather/MLP/SwiGLU, burst admission sized separately from
device activation chunks, and run-coalesced recurrent-slot moves. The reference
is TP8 and mixed BFP4/BFP8; its 16.84 client TSU at C15/32K is not a matched
comparison with our BFP8 TP4 B16 16.55 native TSU. Its partial benchmark run
does not qualify accuracy. These follow-ups have not replaced the active
compact-GDN/projection queue or changed the qualified serving launch.

The [October 10 recovery receipt](../galaxy-evidence/compact-recovery-v2/README.md)
records fusion GPQA, the prefill-budget A/B and the corrected independent-session
trace harness. The [shared compact scratch pool](../galaxy-evidence/compact-pool-recovery-v1/README.md)
resolved the full-model L1 allocation failure. Compact subsequently passed the
full comparison, GPQA, profile and 4096-step B16/B32 boundary checks. These are
completed results, not pending queue items.

The [projection follow-up](../galaxy-evidence/projection-l1-results-v1/README.md)
finished all 62 revised cases with no accepted gain. Prefill batching stopped
at an existing before-arm dense-reference failure; no gain is qualified.
[Resident-state and compact-gate tests](../galaxy-evidence/gdn-followup-results-v1/README.md)
both passed with small measured stage improvements. Their combined full-model
comparison is now running, with G0/API/full GPQA conditional on a matched >=1%
32K gain. The new epilogue unused-row experiment follows that qualification;
see [its frozen queue and limitations](../galaxy-evidence/gdn-epilogue-padding-v2/README.md).

The prefill before-arm failure now has a separate numerical isolation test
queued after that epilogue experiment. It reproduces the original seeded case,
checks its causal reference independently, and varies exponentiation and
accumulation without changing model defaults or loosening accuracy gates.
Frozen CPU preflight passed; physical results remain pending. This diagnostic
must identify a valid baseline before batching performance can be credited.
[Exact plan and persistent launch](../galaxy-evidence/prefill-numerics-v1/README.md).

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
`speedup = old_step / (old_step - saved_ms)` for larger changes. The current
compact baseline is 49.913 ms, so 1 ms now corresponds to about 2% more TSU.

| Order | Change and ownership | Expected benefit | Evidence, effort and risk |
|---|---|---|---|
| 0, completed | Direct preparation + fused GDN output epilogue | Measured 6.82-ms saving; +11.3% decode, reaching 16.55 TSU. | Matched controls and full GPQA 177/198 passed. Serving promotion and full Galaxy client performance remain separate. |
| 1, qualified frozen candidate | Compact GDN convolution/history, compact packed-projection consumption, and gate preparation | Measured B16/32K full-model 60.300 -> 49.913 ms, 16.584 -> 20.035 TSU (+20.81%); B16/16K +23.60%. After-control and full 177/198 GPQA passed. | Opt-in policy. Exact 64-step four-rank checks passed at B16/B32. Shared-pool full-model candidate completed with unchanged tokens; 4096-step B16/B32 check passed. This gain replaces the initial estimate and overlaps broader fusion savings. Later default-off development changes are not the exact evaluated source. |
| 2 | Retune output/down matmuls first, then GDN packed projection; include input/output conversions | Target 1.5-3.5 ms across matmuls, ~3-8% over the current compact baseline. This is a target range, not a demonstrated 90%-bandwidth result. | Vary reader count, bank/worker placement, K blocking and output sharding together. Measure the complete projection boundary. Avoid losing more to padding/resharding than the kernel saves. Existing gate/up is lower priority. Medium effort/risk. |
| 3 | Production SDPA reader/compute delivery: chunk size, worker distribution, tagged lookahead and bank-local delivery | Target 1-2.7 ms, ~2-4% of the baseline. | SDPA is 12.63 ms; modeled KV-only floor is 8.91 ms at peak or 9.90 ms at 90%. Required math/reduction adds cost. Existing remote-delivery prototype loses to production and must not be promoted. Medium/high effort; preserve accurate exponentiation, FP32 accumulation and page ownership. |
| 4 | Recurrence/preparation scheduling with shared Q/K and FP32 state | Target 0.5-1.5 ms after compact, ~1-3%; remeasure the boundary before credit. | Resident-state physical correctness passed, including dense reference. B16 timing 99.041 -> 88.131 us projects 0.524 ms across 48 layers; combined full-model validation is running. Post-compact recurrence and preparation kernel sums are 4.405 and 0.931 ms. Medium/high risk. |
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
or 27.8 tokens/s/user under the existing traffic model. The compact candidate
is around 49.91 ms: about 14.0 ms still separates it from that target. The
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
| Causal convolution | Compute retained row only; read old history once, update disjoint ranges in place, write compact tiles | Native packed-batch convolution already replaces per-user calls at B8/B16/B32. Compact replacement measured in full model and passed full GPQA. |
| Gated RMSNorm | Fuse recurrence output preparation, gating/norm and z multiplication | Current candidate covers this. Do not claim its gain again under general eltwise fusion. |
| RMSNorm, binary/unary, reduce | Fuse residual/gates/SwiGLU where dependency and rounding permit; retain activations in L1 | Some activation fusion already exists. Full-row/head reductions still require completed inputs. |
| All-reduce/all-gather | Persistent buffers, layout alignment, link/worker placement, progress/credit-based tiled overlap | Communication shares NoC and depends on complete partial sums. No free overlap assumption. |
| QKV-head creation, RoPE, paged fused cache update | Align producer output with final consumer/cache; fuse neighboring transformations | KV update arithmetic itself is small; do not risk page mapping for negligible gain. |
| Top-k route prep/finish, top-k, sampling, seed | Avoid unnecessary full-vocabulary materialization; preserve exact sampling semantics | Device sampler already avoids per-step full-logit host readback. Low priority. |
| Copy/indexed-fill, plus-one, embeddings | Remove copies made unnecessary by a fused producer; retain trace-address and history invariants | Trace-owned buffers, slot reuse and independent-user ownership remain mandatory. |

## Prefill and serving: a separate priority lane

The compact B16/32K workload with 128 output tokens spends about 95.84 s in
prefill versus 6.34 s in steady decode. Another 10% decode speedup saves only
about 0.58 s of that request; a 10% prefill speedup saves about 8.7 s. Both matter,
but their effect on short-response end-to-end throughput is very different.

| Priority | Work | Expected benefit and gate |
|---|---|---|
| P1, measured separately | B16 prefill budget 32K -> 64K with matched 32K/64K/32K controls | TTFT 98.66 -> 93.09 s (-5.6%); input 5322 -> 5640 tok/s (+6.0%). Decode remains 14.87 TSU on that older decoder, with identical generated tokens. No combined-fusion/full-GPQA promotion. B32 allocation previously failed. |
| P2, blocked by baseline test failure | Batched full-attention cache fills and prefill attention calls | Default-off prototype reduces B16 cache fills 32 -> 2 and SDPA calls 16 -> 1. Four-rank before/batched/after boundary tests cover shuffled pages, nonzero slots, continued prefixes and dense per-user references. Engineering target 1.25-1.5x boundary, conditional 5-8% TTFT saving if it occupies 25% of prefill; stage share is unmeasured. Existing before-arm reference failed at B3/prefix32/chunk65; candidate was not reached. Earlier exact B2 outputs had 159% timing drift. No qualified gain. |
| P3, profile first | Prefill GEMM tiling, sharded residual/norm continuity, intermediate allocation reuse | Tune with larger M, independently of decode. The 40%-MFU target is not established by decode bandwidth. No numerical prefill-kernel gain is assigned without current stage timings. |
| P4 | Prefill GDN scan/convolution: workspace reuse, layout fusion, chunk/batch parallelism | Preserve recurrent state and chunk chronology. One-token decode kernel is not a prefill substitute. Profile scan/preparation separately; gain unquantified. |
| P5 | Scheduler chunked prefill and mixed-load admission | Reduce decode pauses/TTFT tails under arrivals, not automatically raw isolated throughput. Model-internal chunking already exists. Current BFP8 plugin/state/sampler/cancellation gates remain. |
| P6 | Prefix reuse and recurrent-prefix snapshots | [Scoped requirements](PREFIX-CACHE-OFFLOAD-SCOPE.md): hybrid plugin groups, immutable KV plus exact GDN checkpoints, then TT host transfers and optional SSD. Not a capability-flag change. Workload-dependent prefill gain; no isolated decode credit. |
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

Compact qualification and profiling are complete. Retain its exact qualified
source while the combined resident-state/compact-gate before/candidate/after
comparison runs. A matched >=1% 32K gain triggers eight-replica G0/API and full
198-question GPQA. The default-off epilogue padding screen then runs its poison,
replay, matched-timing and 4096-step checks. Each has a distinct frozen source;
development HEAD itself is not the GPQA-qualified artifact.

Use the remaining compact stage costs to select broader fusion; do not apply
the original layout-family budget a second time. Every experiment must record original/candidate/after controls,
source/config hashes, changed-input trace replay, all-rank correctness, memory
peaks, per-stage and full-step timing. Include input throughput, TTFT, decode
TSU and all-in throughput; protect 16K and report 128K/256K tradeoffs explicitly.
Preserve BF16 rounding boundaries when claiming bit equivalence. New arithmetic
requires the appropriate long-horizon/dense reference and unchanged full GPQA
protocol before serving promotion. Run a full Galaxy scaling/HTTP measurement
after selecting the winning TP4 candidate. Packaging/Shield CI and agentic-eval
qualification remain separate release gates.

## What reaching 30 TSU at B16/32K would require

The [matched roofline audit](../galaxy-evidence/roofline-reconciliation-v1/README.md)
separates the saved specification's BFP4/8K assumptions from today's BFP8/32K
measurement. Its original 86-TSU memory ceiling becomes 39.74 TSU for our
current useful-byte estimate. Applying the plan's unproven 85%-bandwidth and
4.7-ms-overhead assumptions gives 29.15 TSU. The remaining gap from compact's
20.03 TSU is still 1.46x; the original operating point must not be presented as the
current workload's ceiling. The current artifact contents could not be
refreshed; the audit identifies the saved October 6 PDF extraction it used.

**Current user requirement: 30 native tokens/s/user at B16/32K; no speculative
decoding.** This supersedes the earlier conditional MTP scope. Retain BFP8
weights/KV, BF16 activations and FP32 recurrent state. Historical speculative
estimates remain archived, but speculative work is not an active experiment or
an acceptable way to meet this target.

The measured compact candidate's 49.9128 ms must reach 33.3333 ms: another
16.5794 ms removed, or 33.22% less step time / 49.74% more output throughput.
Native 20 TSU is measured and the frozen compact candidate passed full GPQA.
Small independent knob changes do not establish the 30-TSU target.

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
counted again. Compact GDN is only the first piece (measured full-model
savings of 10.39 ms), not the complete fused decoder.
A broader 10-15 ms reduction is an ambitious engineering target for combined
layout/fusion work from the older 60.42-ms fusion path; compact has already
realized about 10.39 ms of that budget. Do not count it again as additional
savings. The original range implied about 19.8-22.0 TSU, not a 30-TSU proof.
Native 30 therefore also needs substantial improvements
to matmul, SDPA delivery and remaining recurrence/collective costs.

At 80% effective useful-byte bandwidth, streaming alone takes about 31.45 ms,
leaving only 1.88 ms of the 33.33 ms target for unhidden work. At 90%, it takes
27.96ms, leaving 5.37 ms. These are necessary traffic budgets, not forecasts:
they omit extra physical traffic and cannot establish actual DRAM utilization.
The implication is to prioritize complete operator boundaries with compatible
L1 layouts, fused producer/consumer stages and pipelined delivery. Merely
placing the existing operators under one launch cannot remove their traffic.

The current full-model validation milestone is the opt-in `single_step_compact_gdn`
policy. Packed projection feeds convolution and z directly from compact L1;
convolution updates disjoint history in place; preparation consumes compact
Q/K/V; the gated epilogue writes compact L1 output for the existing output
projection and TP reduction. Only the64 scalar-gate channels still expand to
public rows. This is a complete GDN boundary prototype, not a megakernel or a
30-TSU result. Its measured 10.39-ms gain is included in the broader 10-15-ms
layout/fusion target above; never add the two.

The remaining large intervention is to join compatible stages inside the
device program: projection output delivery into gating/preparation, recurrent
output into normalization, and projection/collective/residual boundaries.
First compare the compact block including all native matmuls and collectives,
then use the remaining measured stage cost to select the fused boundary.
Existing baseline family timings alone do not prove another 16.58 ms recoverable.
If the remaining target budget is not met, report that gap instead of presenting
small tuning gains or a speculative multiplier as completion.

The [compact projection design](../galaxy-evidence/projection-compact-layout-v1/README.md)
was tested by the revised 62-case sweep. It produced no accepted speedup, so
repeating reader/block knobs without a new mechanism is lower priority than
compatible pipeline changes. Historical launch files are not live status.

## Latest completed knob sweep, 21:47 UTC

The revised projection sweep finished 62 cases / 50 comparisons. All qualified
comparisons were slower than baseline; no knob change is promoted. This does
not rule out kernel redesign, but the prior 1.5-3.5-ms estimated gain has not
been realized. The prefill batching test failed the existing before-arm dense
reference on a ragged chunk with prefix and requires a separate investigation.
The three GDN experiments subsequently ran independently behind successful projection.
[Results](../galaxy-evidence/projection-l1-results-v1/README.md) and
[queue](../galaxy-evidence/gdn-independent-queue-v1/README.md).

## GDN follow-ups completed, 21:52 UTC

Resident FP32 state and compact gates both passed standalone hardware correctness.
At B16 they project 0.524 ms and 0.415 ms savings respectively from matched
before/after stage measurements. Each is about 1% of the 49.913-ms full step;
combined model measurement remains required. The compact-versus-flat 4096-update
B16/B32 test also passed. All tests in that three-unit chain completed. These incremental gains
do not meet 30 TSU; larger attention/GDN dataflow changes remain necessary.
[Exact measurements and limitations](../galaxy-evidence/gdn-followup-results-v1/README.md).

## Combined model validation and exact GDN attribution, 22:03 UTC

The combined resident-state/compact-gate policy passed 4096-step B16/B32 exact
integration checks. Full-model B16/32K and 16K before/candidate/after timing is
running persistently; conditional eight-replica G0/API/GPQA follows only a clean
>=1% primary-workload gain. Estimated 0.94-ms saving / 20.42 TSU is unmeasured.
[Launch and frozen source](../galaxy-evidence/gdn-combined-launch-v1/README.md).

Exact cached-source matching resolves the compact profile's custom stages:
recurrence 4.405 ms, epilogue 2.853 ms, convolution 0.996 ms, preparation 0.931 ms.
Next epilogue hypothesis: avoid clearing unused padding for compact output while
retaining all public-output padding guarantees; estimated 1-2-ms full-step
opportunity, requiring poison/wrap/replay correctness and matched timing.
The default-off implementation and poison/wrap/replay test are now queued after combined qualification. Hardware correctness and speed remain unmeasured.
[Attribution and reproducible analysis](../galaxy-evidence/compact-kernel-map-v1/README.md).

## Epilogue padding experiment queued, 22:23 UTC

The standalone opt-in removes the reader's 48-KiB initialization of unused
input rows while the compact writer preserves output padding. Poison mode
fills both CB slots with NaNs to check isolation. CPU preflight passed 709 tests
and 104 subtests; physical coverage includes 18 layout/placement cases and
four 4096-step real-weight comparisons. Its bounded persistent unit waits behind
the combined candidate's G0/API/GPQA follower. Estimated 1-2-ms full-step saving
is a hypothesis, not an additional measured gain or a path alone to 30 TSU.
[Implementation, source hashes and queue receipts](../galaxy-evidence/gdn-epilogue-padding-v2/README.md).
