# Qwen plan gates: observed status, Oct 8 2026 UTC

Current user priorities: **32K ISL first, 16K second; 128K/256K remain active
secondary optimization targets**. Report gains and losses across all four.
Maximize achieved performance; 2680 output tok/s/Galaxy is a checkpoint, not a
stopping target.

We are partway through P1/P2, not at the projected "after P2" performance point.
The plan's 4.7-ms fixed overhead and 85% peak-DRAM utilization are modeling
assumptions/targets. Neither has been established by the current implementation.
The 14K output tok/s target belongs to the plan's 8K-context operating point;
it is not a 128K/256K throughput forecast.

## Why current timings differ from the targets

There are implementation gaps and uncalibrated assumptions, not a measured
single cause that accounts for the whole difference:

- **One state DRAM read/write does not make GDN a pure streaming kernel.**
  The measured candidate performs FP32 SFPU arithmetic and reductions, with
  tile unpack/pack and circular-buffer synchronization between delta, state
  update and output stages. State is read once from DRAM but revisited in L1.
  Shared Q/K normalization is repeated across three value heads and four
  value-column partitions. These costs are visible in the source; their
  individual shares of latency have not been isolated by a complete profile.
  B64 recurrence is 596.43 us versus the 206.61-us state-bandwidth-plus-10-us
  target, even before the separate output epilogue.
- **P2's assumed launch/traffic savings have not been delivered.** Persistent
  traces do not fuse device programs. Output layout conversions, external
  gating and intermediate buffers remain; B64 projections and serving buckets
  are also unfinished. We cannot substitute the modeled 4.7-ms launch cost for
  a measured cost or claim the <=15-program/layer gate.
- **The numerical contract has a cost.** The tested half-tile attention path
  used approximate exp despite its configuration flag and failed the original
  accuracy gate. Current passing attention uses a full query tile, accurate
  exp and FP32 accumulation. Real-weight GDN also required FP32 normalization
  to meet the unchanged per-head error limit. These are measured correctness
  reasons for the current policy; their isolated end-to-end performance tax
  has not been measured and should not be invented.
- **Measured attention falls below the assumed 85% bandwidth.** Passing
  placement cases deliver roughly 68-75% of assumed peak as useful KV bytes
  divided by call time, not hardware-counter utilization. Raising 70% to 85%
  would improve that call by about 21%, before other layer costs; it cannot
  explain or close a several-fold model-throughput gap by itself. Current KV
  is still interleaved, so worker placement alone does not provide bank-local
  streaming. Larger reader barriers were measured and rejected as slower.
- **Long context changes the traffic bound.** The 14K target is at 8K; the
  128K/256K measurements are different workloads. At near 256K, even KV-only
  full-bandwidth traffic caps ordinary BFP8 single-token decode around 1,796
  output tok/s/Galaxy under this model. The separate 2,680 target needs less
  KV traffic per accepted output token, not just reaching the assumed 85%.

All 12 bounded native/single-step profiles have now completed. They isolate
major operation and stage intervals; full-model critical-path reconciliation,
compute/bandwidth calibration and separation of active work from waits remain open.

| Phase / requirement | Current evidence | Remaining gate |
|---|---|---|
| G0: eight TP4 replicas | Historical physical eight-replica short-context baseline passed output agreement and <3% TPOT degradation; aggregate 288.16 tok/s at B1. | Repeat qualification for the new attention/GDN policy and long-context operating points. An 8x TP4 projection is not this test. |
| P0: stage attribution | All 12 bounded native/single-step two-layer profiles completed with four ranks and applicable RISC intervals; raw CSV reanalysis matches. Raw bank-local read probe reaches 499-508 GB/s/chip. | Reconcile against traced full-model TPOT; calibrate compute/collectives and transfer reader bandwidth into attention with redistribution included. |
| P1: one-token recurrence | Opt-in custom Metal recurrence is integrated into decode, updates FP32 state in place, eliminates the candidate path's out-of-place state copy, and fuses FP32 Q/K normalization. 4,096 changing-input steps and rebinding checks pass. | Gated RMSNorm and SiLU(z) epilogue are still separate; the public/native op and complete fusion boundary are not finished. Every measured standalone latency misses the bandwidth-plus-10-us gate. |
| P2: graph cleanup | Compact decode paths, batched RoPE, packed projections and packed convolution for multiples of eight exist. Full-model B32 fixed-shape traces are measured. | No proof of <=15 programs/layer or L1 retention across the full graph. Small/irregular batches retain a per-user convolution path. Gating/layout conversions remain. |
| P2: B32/B64 serving buckets | Fixed-shape B32 benchmark works. Resident serving bucket selection still uses 1/8/16. | Extend and qualify B32/B64 resident buckets. Full-model B64 is blocked by the DRAM-sharded projection M==1 limit and remains unmeasured. Component B64 recurrence is not full-model support. |
| P2: performance and quality | At 8K, single-step GDN gives 302.89 output tok/s per TP4 at B16 and 434.12 at B32. B1 is 33.71 tok/s, about 29.67-ms TPOT. | B1 <=12 ms, B64 <=36 ms, physical Galaxy >=14K tok/s, unchanged reference evaluations: none achieved. Updated policy has not passed online requalification. |
| BF16 recurrent state | Current policy and recurrence retain FP32 state. | A separate long-horizon and model-eval experiment; not enabled while bandwidth work is prioritized. |
| P3: optional Blaze layer | No Qwen fused Blaze decoder layer implemented or promoted. | Decide using complete post-P2 overhead attribution; the >30% trigger has not been measured. |
| P4: prefill | Current 8K measurements are about 6.8-7.2K input tok/s per TP4; bounded chunking made B32/32K and B16/128K fit. | >=20K input tok/s and >=40% measured MFU; no calibrated MFU pass. |
| Mixed prefill/decode overlap | Prefill and decode still execute as separate programs. | Mixed-row matmuls, scheduler integration and correctness/performance proof. No credit for max(memory, compute) overlap in measured results. |

## P1 numerical and latency evidence

The [qualified recurrence receipt](../galaxy-evidence/gdn-model-integration-v1/gdn-kernel-validation-v1/long-horizon/candidate.json)
passes its original per-head PCC/RMS checks over 4,096 changing-input steps.
Final state/output relative RMS is 3.11e-7 / 4.91e-7. The measured kernel includes
Q/K normalization and recurrence, but excludes the still-external gated output
normalization. Consequently even a latency pass here would not finish all P1.

| Users per TP4 | Measured recurrence call (us) | Current state-bandwidth +10-us target | Gate |
|---:|---:|---:|---|
| 1 | 36.72 | 13.07 | Miss |
| 8 | 108.75 | 34.58 | Miss |
| 16 | 175.24 | 59.15 | Miss |
| 32 | 308.58 | 108.30 | Miss |
| 64 | 596.43 | 206.61 | Miss |

These timings include trace dispatch/synchronization. The target uses the plan's
512-GB/s/chip assumption. Real layer-0 block timing including projection,
convolution, preparation, recurrence and output epilogue/projection improves
29.1% at B16 and 40.6% at B32; B1/B8 regress. The full model at 32K/B16 improves
23.86%; the completed 32K/B32 pair improves 40.09%. The fresh 128K/B16 pair
improves 14.99%; earlier 128K/B8 and
256K/B4 were approximately flat/slightly slower. This favors a batch-dependent
recurrence choice; new model-eval qualification is still pending. See
[integration evidence](../galaxy-evidence/gdn-model-integration-v1/README.md) and
[matched model sweeps](../galaxy-evidence/gdn-matched-comparison-v4/README.md).

The last full GPQA result remains 171/198 (86.36%), below the plan's 89.2% gate.
Numerical component passes and repeatable generated tokens do not replace this
quality gate. New attention/GDN policy reference-eval qualification is pending.

## Current long-context work

The user now prioritizes 32K ISL, followed by 16K. Both 128K/256K remain
active secondary tuning targets. Hardware runs retain BFP8 KV/FP32 state.

- **Fresh 128K/B16 pair completed:** native 142.06 vs single-step 163.36 output
  tok/s per TP4, **14.99% uplift**, three repeats, matched prompt/source/precision
  and clean device close. Prefill is ~4044 input tok/s for both. Output hashes
  repeat within each arm but differ between arms; no eval qualification follows.
- **Placement/chunk diagnostic completed all 78 cases:** passing attention-call
  gains are 2.52% at 32K/B16, 1.94% at 32K/B32, 3.57% at 128K/B16 and 3.60% at
  256K/B8. At 256K/B4 native wins. Accuracy-failing candidates remain excluded.
  These are component measurements, not full-model improvements.
- **Accurate partial-query simulator passed:** all 16 candidate cases and eight
  full-tile controls pass; candidate partial-query outputs are bit-identical to
  the full-query controls. Six original partial-query cases failed. Subsequent
  physical hardware checks passed; model evaluation remains untested.
  The retained first attempt failed a generated-file evidence lookup, fixed in
  v2 without relaxing numerical or source-verification checks.
- **Bounded profile v4 queued at 07:25:25 UTC:** 12 captures prioritizing
  32K/B16/B32, then 16K/B16/B32, then 128K/B16 and 256K/B8. V3 hardware passed
  but its collector incorrectly demanded compute timings from programs with
  no compute kernel. The fix requires empty source/hash lists and three zero
  compute-binary sizes before treating absent timings as not applicable.
  Reanalysis of a copy of the original capture passes all four ranks. Remote
  CPU validation passed 292 tests plus 40 subtests. Single-step stage attribution
  is still queued; this capture is native recurrence with synthetic caches.
- **Physical partial-query sweep completed:** 30 cases, ten qualified
  full/partial/full comparisons, production instructions and unchanged BFP8 KV.
  Every candidate output is bit-identical to its control. Attention-call gains
  at 32K/B8/B16/B32 are 4.28/2.16/0.85%; at 16K, 7.03/3.48/1.38%. Regressions
  at 128K/B16 and 256K/B4 are 0.33/1.18%. Not promoted to the full model.
- **32K/B32 full-model pair completed:** 250.13 native versus 350.42 single-step
  output tok/s per TP4, **40.09% uplift**. Prefill remains ~5185 input tok/s.
  The 2803 output tok/s/Galaxy figure is an eight-replica projection. Persistent
  capacity and corrected profiling jobs remain active with `Linger=yes`.
  Results, failures and recovery are in
  [profile-recovery-and-throughput-v1](../galaxy-evidence/profile-recovery-and-throughput-v1).
- KV is still interleaved. A source audit of Blaze's 92% bandwidth reference
  identifies bank-local streaming and transaction-ID buffering as useful next
  experiments. The cited rate is recorded expert-matmul streaming bandwidth,
  not a measured guarantee for reload or Qwen attention.
- The [BFP4 KV simulator experiment](../galaxy-evidence/bfp4-kv-simulator-v1/README.md)
  completed all 24 cases. Total synthetic long-context output RMS was 15.65-16.64%
  for K4/V4 vs 1.28-1.86% for K8/V8. Keep BFP8 while real activations/logits and
  model evals remain necessary to decide BFP4 suitability.

Raw comparisons, plots, source/launch pins, failure receipts and the priority
change are in [optimization-followup-v1](../galaxy-evidence/optimization-followup-v1).
The existing baseline G0 and failed P0 recovery receipts are respectively
[eight-replicas-v3](../galaxy-evidence/eight-replicas-v3/full-model.json) and
[profile-recovery-v1](../galaxy-evidence/profile-recovery-v1/profile-v5-outcome.json).

## Long-context throughput projection and traffic limits

Completed operating points (256K matched pair is effectively flat at -0.169%):

| Context | Batch per TP4 | Measured output tok/s per TP4 | 8x projection, output tok/s/Galaxy | Ideal traffic-model ceiling at that batch |
|---|---:|---:|---:|---:|
| 32K, single-step GDN | 16 | 259.22 | 2,074 | 6,991 |
| 32K, single-step GDN | 32 | 350.42 | 2,803 | 8,654 |
| 128K, single-step GDN | 16 | 163.36 | 1,307 | 2,841 |
| 262016, native GDN | 8 | 89.98 | 720 | 1,459 |

All output rates exclude prefill. The 8x projection is not physical Galaxy
measurement. The ceiling is an optimistic one-read traffic model at assumed
512 GB/s/chip, excluding collectives, launches, compute stalls, conversion traffic
and additional buffers. It is not a promised optimized result or measured DRAM
utilization, and does not complete P0/P5 calibration.

At 32K/B32, the ideal per-step traffic budget is **7.04 ms weights + 4.72 ms
state read/write + 17.83 ms KV = 29.58 ms**, versus the measured **91.32-ms
TPOT**. Current throughput is 32.4% of this optimistic ceiling. The remaining
61.74 ms cannot all be called dispatch overhead: it includes additional memory
traffic, math, synchronization, collectives, layouts and launch effects that
the simplified model excludes. At 128K/B16 the ratio is 46.0%; at 256K/B8 the
earlier native point is 49.3%. These ratios are not DRAM-counter utilization.

### Stage evidence against the memory lower bound

Compare boundaries carefully: native two-layer profile kernels are eager with
real weights and synthetic activations/cache; attention and single-step GDN
microbenchmarks are separate traced calls. Optimized two-layer profiles are now complete; a calibrated full-model
compute/communication roofline remains open.

| Stage and geometry | Measured time | Optimistic memory time | Interpretation / next lever |
|---|---:|---:|---|
| BFP8 attention, 32K/B32, one attention layer | 1480 us full-query; 1468 us partial-query | 1114 us KV scan | About 75% useful-KV bandwidth. Bank-local reads and pipelining have more potential than partial-query arithmetic alone. |
| Single-step GDN, B32, one recurrent layer | 309 us | 98 us state read/write | About 32% of the state-only ceiling. FP32 SFPU work, repeated Q/K normalization, unpack/pack and synchronization remain; gated output norm remains external. |
| Single-step GDN, B16 | 175 us | 49 us state read/write | About 28% of the state-only ceiling. Additional operands and computation omitted from the lower bound. |
| MLP gate/up + down, B16, per layer | 98-100 us summed matmul kernel intervals | 73.4 us weights | Roughly 74% weight-only equivalent bandwidth; collectives, layouts and other MLP ops are additional. Native-profile evidence. |
| Vocabulary head, B16 | 494-501 us summed matmul kernel intervals | 349 us logical weights | Roughly 70% weight-only equivalent bandwidth; excludes padding/activation traffic, concat, layout and sampling. Native-profile evidence. |
| Packed-convolution preparation, B16 | 110-111 us over 13 device-op rows | Not calibrated | Layout conversion remains material; 3 tilize calls account for about 73 us of kernel intervals on rank 8. |
| RoPE path, 32K/B16 | 63-64 us over 16 device-op rows | Not calibrated | Most rows are layout/data movement; retain activations in compatible layouts and fuse where measured useful. |

The old native recurrence uses 33 device-op rows and 580-584 us of summed kernel
intervals per recurrent layer in this capture. This supports targeting graph
cleanup, but is not a direct estimate of the remaining cost in the single-step
variant. Likewise firmware intervals can include waits for preceding ops; do
not sum them as an end-to-end critical path or call RISC durations active time.

For a TP4 chip, per-step bytes are modeled as:

```text
weights = 3,603,087,360 B  # dominant decoder matrices + LM head, BFP4
GDN read/write per user = 48 * 12 * 128 * 128 * 4 * 2 = 75,497,472 B
KV read per user = 16 * 2 * context * 256 * (1088 / 1024)
traffic = weights + batch * (GDN read/write + KV read)
ideal step seconds = traffic / 512e9
Galaxy output tok/s = 8 * batch / ideal step seconds
```

At 262016 tokens, **KV reads alone** cost 2.2806 GB per chip per user per step.
Even ignoring weights/state, ordinary single-token decode with a full BFP8 KV
scan has a traffic bound of roughly **1,796 output tok/s/Galaxy**. Thus the
user's 2,680 target at this length requires reducing KV traffic per accepted
output token, not solely improving placement. BFP8 bandwidth work still has
considerable headroom versus the current projection; the separate BFP4 KV
experiment does not yet qualify a lower-traffic policy.

## Latest evidence: Oct 8, 09:15 UTC

[Shared-Q/K and bandwidth report](../galaxy-evidence/shared-qk-and-bandwidth-v1/README.md)
contains complete physical and raw-profile evidence. Shared normalization passes
4,096 bit-identical updates and improves the real GDN block by 12.2% at B32 and
9.5% at B16. Its full-model opt-in policy is running a persistent matched sweep;
no new model-level uplift or eval pass is claimed. B1 retains the faster fused
normalization path.

Bank-adjacent bulk reads with four independent slots reach 499-508 GB/s/chip,
while row placement reaches 340-343 GB/s and tile-at-a-time bank readers about
190 GB/s. This raw diagnostic excludes compute-worker redistribution and
attention math. The next bandwidth gate includes those costs; do not compare
its simple interleaved control directly to the production attention reader.

The 256K capacity pair and all 12 profile captures are complete. Earlier
"running/queued" statements above are historical snapshots. Updated near256K/B8
single-step output is 89.831 tok/s/TP4 versus native 89.983, effectively flat.
Reference-eval qualification and physical eight-replica scaling remain open.
