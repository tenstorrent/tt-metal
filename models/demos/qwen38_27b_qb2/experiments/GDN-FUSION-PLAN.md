# GDN fusion implementation plan

October 11, 2026, 02:00 UTC. This is a design, not an implemented fused kernel
or a queued hardware experiment. Primary target: B16, 32K context, TP4; preserve
BFP8 weights/KV, BF16 activations and FP32 recurrent state. No speculative decode.

## Decision and expected benefit

Prototype recurrence-to-epilogue delivery first, retaining the existing
four-way value partition and numerical operations. Then fuse convolution,
normalization and gate delivery if the first boundary wins. Keep packed and
output matmuls and their TP collectives native initially.

The working target for these two boundaries together is **1-3 ms per model
step**, not a measured or guaranteed gain. At the qualified 49.913-ms baseline,
that corresponds to 20.44-21.32 TSU versus 20.035. This alone does not deliver
30 TSU, which requires removing 16.579 ms. Matmul delivery and attention remain
necessary parallel opportunities in the overall roadmap.

The target must be measured against the best compatible unfused implementation,
including the new padding skip. Do not add old padding, compact-gate or resident
state projections to a new fusion result without a matched combined model run.

| Work | Current evidence | Incremental planning target | Main risk |
|---|---|---|---|
| Skip unused epilogue input padding | Hardware passed; B16 packed L1 66.254 -> 39.336 us | 1.292 ms over 48 layers, about +0.53 TSU from baseline, still a projection | Whole-model saving may differ |
| Recurrence + epilogue in one program | Design only | 0.5-1.5 ms beyond padding skip | Core allocation, cross-core gather and numerical modes |
| Conv/normalization/gate delivery into recurrence | Design only | 0.5-1.5 ms beyond the first fused boundary | Preserving rounding and shared-head ownership |
| Packed/output projection joined to pipeline | Deferred design | No credible separate estimate yet | Matmul/CCL scheduling, bandwidth contention, much larger scope |

Targets prioritize experiments; they are not additive performance promises.
One millisecond near 20 TSU is only about 0.41 TSU. Batch small gains before
G0/API/full GPQA, per the user's less-than-1-TSU validation policy.

The matched byte-only roofline is approximately **39.74 TSU** at B16/32K/BFP8,
using 12.883 GB/chip/step and an assumed 512 GB/s peak. The latest measured
combined result, 20.319 TSU, is about **51.1%** of that ceiling; 30 TSU requires
75.5% overall useful-byte efficiency. This is not measured bus utilization.
Matmul and attention in the latest qualified profile already reach about 73%
and 71% of their own byte-only bounds; reducing the intervening work matters
as well as streaming faster. At 85% streaming efficiency, the modeled bytes
alone cost 29.60 ms, leaving only 3.73 ms of unhidden work for a 30-TSU step.

## What the existing profile establishes

[Exact generated-kernel attribution](../galaxy-evidence/compact-kernel-map-v1/README.md)
identifies 48 calls per rank and replay for each stage:

| Stage | Instrumented kernel sum | Per-layer mean |
|---|---:|---:|
| Recurrence | 4.4051 ms | 91.77 us |
| Epilogue | 2.8531 ms | 59.44 us |
| Convolution | 0.9959 ms | 20.75 us |
| Direct preparation | 0.9309 ms | 19.39 us |

These sums include waits, are not additive critical-path time, and are not
physical DRAM/NoC utilization counters. Whole-step profiler overhead was 4.43%.
The profile predates resident-state/compact-gate and padding-skip experiments.
The matched resident-state/compact-gate whole-model gain was only 0.324 TSU;
component projections did not add perfectly.

The [padding experiment](../galaxy-evidence/gdn-epilogue-padding-results-v3/README.md)
now directly establishes that clearing unused input rows was expensive. It
does not establish that all remaining time is DRAM bandwidth or dispatch.

## Current dataflow and irreducible traffic

[`DecoderLayer._delta_compact`](../tt/decoder.py) currently performs:

1. Packed native projection into compact L1 rows `[1,B,4160]`.
2. Native gate arithmetic; compact convolution updates history and emits
   Q512/K512/V1536 per row in L1.
3. Direct preparation normalizes Q/K and produces FP32 Q/K, V and gates in DRAM.
4. In-place FP32 recurrence reads/writes state and writes raw output to DRAM.
5. Epilogue reads raw output and packed z, normalizes/gates it, and writes compact
   BF16 L1 output `[1,B,1536]` for native output projection and TP all-reduce.

At B16 per chip, state is `16*12*128*128*4 = 12,582,912` bytes per GDN layer.
One read plus one write is 25,165,824 bytes. At an assumed 512 GB/s peak, that is
49.152 us/layer, or 2.359 ms across 48 layers, before all other work. The peak is
an ideal modeling assumption, not an achieved bandwidth measurement. State
cannot be retained for all layers in L1; do not count away these bytes.

Raw recurrence output is only 98,304 bytes per layer. Removing its DRAM write
and read saves about 0.018 ms across 48 layers at that ideal peak. Including
prepared-vector writes and repeated partition reads brings the eliminated
intermediate traffic to approximately 75.4 MB/chip/model-step, or 0.147 ms at
peak (0.210 ms at 70%). Multi-millisecond gains must therefore come primarily
from less reformatting, launch/synchronization overhead and better overlap,
not the intermediate byte count. Convolution QKV is already L1.

## F1: recurrence and epilogue

Preserve 12 local value heads and four 32-value partitions per head. At B16
there are 768 partition tasks. Each partition keeps ownership of four FP32
state tiles and updates them in place. Its raw 32-value output is 128 bytes.
An epilogue consumer must gather all four partitions of the same user/head;
RMS normalization over four independent 32-value slices would be incorrect.

Use an explicit schedule assigning all four partitions for a head to a group
of workers in the same wave. The existing strided `work_items` assignment is
not a sufficient producer/consumer schedule: heads may span cores/waves.
Derive groups from the actual compute grid, allow unused cores, and represent
partial final waves explicitly. Never wait for a partition omitted from a wave.

Two implementations should be compared at the same boundary:

- **Dedicated epilogue consumers first:** disjoint recurrence and consumer
  cores in one program, with separate compute configurations. Producers retain
  register-resident FP32 recurrence; consumers run the existing epilogue math
  on complete 128-value vectors. Sweep a small number of consumer-core counts
  and placements. Reserving cores can reduce state bandwidth; that loss is
  part of the timing, not something to hide in a component benchmark.
- **Same-group consumer later:** one recurrence worker also performs epilogue
  after gathering the other partitions. This avoids dedicated consumers but
  serializes that worker and requires an explicit solution for different
  approximation modes and DEST-register usage. Do not assume it is faster.

The recurrence uses `math_approx_mode=False`; the qualified epilogue uses
`True`. Combining them into one compute kernel with one global flag can change
rsqrt/sigmoid behavior. Dedicated consumers preserve both configurations.
The same-core option requires audited per-operation primitives or numerical
qualification of a deliberate change; flipping the global flag is not fusion.

For each destination, preallocate two L1 mailboxes with sequence numbers and
credits. Each producer writes its disjoint 128-byte region, completes the NoC
write barrier, then publishes readiness. The consumer waits for all four
matching sequence numbers, gathers in the existing order, computes, writes
the existing compact output, and returns credit only after consumption.
Assign every CB and mailbox a single lifecycle owner. No overwrite before
acknowledgement and no cyclic ready/credit dependency. Runtime addresses and
semaphore state must be correct on trace replay, cached-program rebind and
independent model instances.

Prefetch z/norm weights while state is in flight when buffers permit. Keep
the existing full-vector RMS reduction, epsilon, norm-weight multiplication,
sigmoid and explicit final BF16 rounding. Preserve output padding ownership.
Do not introduce an associative partial-statistics reduction in the first
version: a different sum order adds an avoidable numerical variable.

State writes retain the canonical DRAM ABI. Current nonresident compute already
allows state writes to overlap final readout; the resident variant publishes
both outputs after readout. Any earlier resident publication must retain state
storage until both compute and NoC finish. This is a variant-specific experiment,
not a newly discovered absence of overlap everywhere.

## F2: frontend and prepared-operand delivery

Keep one writer per convolution-history element. Preserve the ordered four
convolution products, BF16 partial rounding and SiLU behavior. Normalize Q/K
once for each shared QK head, then deliver to its three value heads and four
partitions. Do not normalize independently in every consumer or update shared
convolution history repeatedly.

Reuse the existing FP32 normalization order, epsilon and Q scaling from
`compute_math.hpp`. Gate preparation must preserve the BF16 beta boundary and
FP32 log-decay/softplus/exp path. Deliver normalized operands through bounded
L1 buffers, with the same explicit ownership protocol as F1. The benefit must
include frontend producer cost and lost recurrence cores.

The current recurrence reserves about 152 KiB of CB space per worker with
two-item lookahead. Its resident compute uses all eight FP32 DEST tiles.
Produce an actual per-core L1/CB/DEST allocation before implementation;
mailboxes, expanded epilogue operands and lookahead cannot simply be added
without checking capacity. The existing shared scratch pool is safe only for
serialized CQ0 layers within one replica; do not share it across replicas.

## Experiments and acceptance

| Order | Experiment | Accelerator budget once implementation is ready | Decision |
|---|---|---|---|
| 1 | F1 schedule/credit model and allocation audit | CPU only | No missing producer, alias or cyclic wait; fits actual resources |
| 2 | F1 single-layer changing-input comparison, all TP ranks, B16/B32 and trace/rebind checks | 5-15 min estimate, bounded 30 min | Exact state/history/output versus matching unfused path; 4096 updates |
| 3 | F1 producer/consumer placement sweep with before/fused/after controls | 10-20 min estimate, bounded 45 min | End-to-end boundary improves beyond control drift |
| 4 | F2 equivalent correctness plus combined F1/F2 boundary | Same bounded component process | Preserve numerical and ownership contracts |
| 5 | Combined full-model B16/32K and 16K comparison; long-context regression cells | Plan after component acceptance | Measure TSU and tradeoffs; never infer it from isolated kernels |
| 6 | G0/API/GPQA for the accumulated candidate | Only after batching small gains | Existing accuracy gates and cache/trace integration preserved |

These are planned experiments, **not current systemd jobs**. Budgets describe
execution after implementation; they are not an engineering delivery ETA.

Add sampled device phase markers for small-vector readiness, state DMA
completion, operand-formatting completion, compute start/end, state-write
acknowledgement, mailbox wait, and epilogue completion. Run an uninstrumented
control to quantify profiler overhead. This resolves overlap/placement questions
without repeating the existing broad profile before there is a candidate.
RISC elapsed time must not be reported as active DRAM or compute utilization.

Long-horizon component testing must vary inputs and state, poison unused lanes,
exercise mailbox wrap and changed addresses, check inactive neighbours, and
finish with clean device teardown. Reuse existing reference tests instead of
adding tests that merely restate the new schedule. Keep B1/B8 on the existing
fallback initially. State checkpoints must remain exportable after completion
without replacing canonical buffers or leaving the only valid copy in L1.

AgentX remains blocked until prefix caching and SSD offload work through serving.
A correct standalone checkpoint restore does not remove that prerequisite.
