# Qwen plan gates: observed status, Oct 8 2026 UTC

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

The next attribution step is a bounded per-stage profile that separates state
reads/writes, FP32 math, synchronization, layout/epilogue work and collectives.
The incomplete P0 capture cannot justify a precise percentage decomposition.

| Phase / requirement | Current evidence | Remaining gate |
|---|---|---|
| G0: eight TP4 replicas | Historical physical eight-replica short-context baseline passed output agreement and <3% TPOT degradation; aggregate 288.16 tok/s at B1. | Repeat qualification for the new attention/GDN policy and long-context operating points. An 8x TP4 projection is not this test. |
| P0: stage attribution | Device profiling was attempted; export/recovery failed completeness after a 67.5-GB CSV and memory pressure. Only three of six windows were recoverable. | A bounded, complete per-stage capture, launch counts and calibrated bandwidth/collective costs. No qualified stage attribution from the failed capture. |
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
23.86%, but 128K/B8 and 256K/B4 were approximately flat/slightly slower. See
[integration evidence](../galaxy-evidence/gdn-model-integration-v1/README.md) and
[matched model sweeps](../galaxy-evidence/gdn-matched-comparison-v4/README.md).

The last full GPQA result remains 171/198 (86.36%), below the plan's 89.2% gate.
Numerical component passes and repeatable generated tokens do not replace this
quality gate. New attention/GDN policy reference-eval qualification is pending.

## Current long-context work

The user prioritizes 128K/256K placement and bandwidth with BFP8 KV; 32K gains
and regressions remain explicit tradeoffs. BFP4 KV investigation is a separate
CPU simulator experiment and is not a precision change in these hardware runs.

- Fresh native 128K/B16 completed with three repeatable measurements and clean
  device close at 05:10 UTC: **142.06 output tok/s per TP4**, 8.879 tok/s/user.
  The single-step comparison subsequently started; no paired uplift yet.
- A new persistent placement/chunk experiment was launched at 05:16:12 UTC.
  **263 CPU tests and 40 subtests passed**. It waits on the shared device lock
  behind the running full-model case. Its six geometries start at 256K/B8 and
  128K/B16, then smaller long-context cases and 32K tradeoff checks.
- Thirteen variants per geometry isolate equal-core-count location changes,
  full-grid output-sharding overhead, and 256/512-token chunks. Bitwise output
  hashes strengthen location-only comparisons. Numerical failures, repeated
  output changes or excessive timing drift cannot produce a qualified win.
- KV remains interleaved. This is not yet a bank-sharded KV implementation.
  No new placement speedup, full-model gain or online-eval pass is claimed before
  hardware receipts establish it.

Launch/source hashes, CPU results and the completed fresh 128K receipt are in
[the current evidence snapshot](../galaxy-evidence/placement-long-launch-v1).
The existing baseline G0 and failed P0 recovery receipts are respectively
[eight-replicas-v3](../galaxy-evidence/eight-replicas-v3/full-model.json) and
[profile-recovery-v1](../galaxy-evidence/profile-recovery-v1/profile-v5-outcome.json).

## Long-context throughput projection and traffic limits

Best completed operating points as of the native 128K/B16 receipt above:

| Context | Batch per TP4 | Measured output tok/s per TP4 | 8x projection, output tok/s/Galaxy | Ideal traffic-model ceiling at that batch |
|---|---:|---:|---:|---:|
| 32K, single-step GDN | 16 | 259.22 | 2,074 | 6,991 |
| 128K, native GDN | 16 | 142.06 | 1,137 | 2,841 |
| 262016, native GDN | 8 | 89.98 | 720 | 1,459 |

All output rates exclude prefill. The 8x projection is not physical Galaxy
measurement. The ceiling is an optimistic one-read traffic model at assumed
512 GB/s/chip, excluding collectives, launches, compute stalls, conversion traffic
and additional buffers. It is not a promised optimized result or measured DRAM
utilization, and does not complete P0/P5 calibration.

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
