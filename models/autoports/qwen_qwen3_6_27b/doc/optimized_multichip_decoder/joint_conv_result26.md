# Joint correction + convolution: measured rejection and integration boundary

CPU-only analysis of completed `joint_conv_perf25.json` and
`joint_conv_smoke25_v4.json`, 2026-11-13. No hardware/build/model imports or
production edits. The reproducible stdlib extractor `joint_conv_result26.py`
writes `joint_conv_result26.json` and `.csv`, including raw artifact hashes,
the recorded source hashes, all28 timing rankings and41,220 exact per-rank,
per-user, per-tensor checks across the two runs.

**No integration patch is warranted:** every measured fused candidate is much
slower than the existing complete boundary. The fastest fused choice is
80-worker `rest_only` for every measured batch, at4.23×–5.13× control latency.
The requested condition “if materially faster” is not met. No main or
packed-correction wiring is prepared or applied; defaults remain unchanged.

## Complete-boundary ranking

All values are medians in microseconds from the same uninstrumented TP4 run,
seven100-replay samples with rotating path order and four warmups. The baseline
named `native` is existing `PackedCorrection._combine` + material QKV slice +
`local_conv`, including its untilize and in-place history update. It is not
an unfused host reference and does not omit convolution. The fused boundary
starts from the same joint tensor and includes correction, convolution, SiLU,
corrected side output and history update. Matmul, recurrence and downstream
Z/beta/alpha consumers are outside both boundaries.

| B | Existing boundary | 48 packed | 48 rest | 80 packed | 80 rest | 110 packed | 110 rest |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **28.586** | 213.168 | 213.022 | 120.970 | 120.917 | 121.220 | 121.105 |
| 3 | **30.023** | 229.396 | 229.171 | 129.052 | 129.014 | 129.351 | 129.045 |
| 17 | **39.386** | 339.304 | 338.284 | 189.478 | 188.175 | 190.381 | 189.714 |
| 32 | **48.697** | 456.076 | 455.627 | 249.860 | 249.819 | 252.044 | 250.675 |

Within each output family,80 workers wins at all batches, followed by110,
then48. The best fused regressions are92.331/98.992/148.788/201.122us at
B1/3/17/32. All candidate sample intervals are far from the control interval;
this is not a near-tie or noise-based rejection. Both output families were
measured. `rest_only` receives the benefit of omitting corrected raw QKV;
the probe does **not** immediately restore the old full packed contract.
Even with that advantage it is much slower.

The comparison uses synthetic rank/user-distinct values in the actual
production TP4 shape/layout. It does not load actual projection weights or
establish whole-layer quality. Report these as focused measured rejection
costs, not final optimized decoder performance.

## Dtypes and passed quality

Both artifacts actually measured **FP32 joint input**, width-sharded L1 on
eight cores with `[32,896]` shards and logical `[1,B,7168]`. The helper accepts
BF16 input and DRAM placement in source, but these two reports contain neither
BF16-input nor DRAM-input hardware evidence. Do not claim otherwise.

Both output families preserve correction FP32 addition followed by BF16 RNE
before convolution; history and tap coefficients are BF16 row-major. The
four products and accumulation use the original FP32 arithmetic, followed
by its BF16 rounding and SiLU semantics. The canonical convolved output is
BF16 row-major `[B,1,2560]`; raw history is mutated in place as BF16 row-major
`[B,3,2560]`. The side output is either BF16 tiled `[1,B,4160]` (full corrected
packed) or `[1,B,1600]` (Z1536 + beta32 + alpha32).

Every recorded logical output, history and side-output comparison is finite,
bit-exact and PCC1.0. Coverage includes changed banks, rebound joint/history/
taps/outputs, eager/trace equality,50 queued changed inputs with accumulating
history, and final timed steady-state output. Poisoned padded rows/columns
remain immutable in the actual L1 input; candidate side-output padded users
are zero. The watcher+allocation-tracking smoke covers B1/B3 at80 workers,
both output families; its positive control rejects a fresh allocation before
replay. The uninstrumented performance run extends quality to B17/B32 and
48/110 workers. Both meshes close normally. These successful results follow
the prior repaired failures; this is not rejection after the first API or
arithmetic error. Original failures remain preserved separately.

## Source evidence for the performance hypothesis

`joint_conv_candidate.py` gives one work item all users for one32-channel
column. Its reader manually populates seven full32×32 BF16 tiles using scalar
RISC-V loads/stores (7,168 element stores per QKV column), in addition to
clearing padded primary/correction user rows. The writer manually untilizes
four B×32 blocks and issues four64-byte writes per user per QKV column; the
reader issues three64-byte history reads per user plus coefficient reads.
Those loops/transactions remain in both output families. This is a source
inference for the bottleneck, not a measured phase breakdown.

At80 workers each of the80 QKV columns has its own worker. Increasing to110
does not further split that expensive work; it mainly moves cheap side-output
columns. At48 workers some workers handle two QKV columns, matching the large
observed slowdown. Omitting80 corrected-QKV tiles saves163,840 bytes/rank of
side-output writes, yet improves latency by only0.040–1.303us at80 workers.
This supports, but does not prove, that scalar layout construction and tiny
transactions dominate over full-packed side-output traffic. More workers or
`rest_only` alone is not a supported route to beating the control.

## Admissions and lifetimes for any future renewed candidate

The following is a design boundary, not a proposed integration of this slow
kernel. A future opt-in constructor would need to require local linear
attention, narrow correction, installed packed-correction projection,
flat decode, and packed convolution enabled at every valid batch including
B1. Initial measured scope should retain FP32 joint scratch, Blackhole,
B1..32, S1, the production width-sharded L1 geometry, exact BF16 history/tap
shapes, and canonical output format. Unsupported explicit combinations must
raise at setup; do not silently run another family or mutate policy during
forward. Any BF16-input choice needs its own measurement and exactness gates.

The clean split would expose a raw compact joint projection from
`PackedCorrection` without its old combine, then let `_gdn` pass the current
`state.conv` and taps to the fused operation. Do not retain a layer/state owner
inside a bound callback or capture stale history. The current `_gdn` slices
QKV before convolution and its local beta/decay and local norm/gate helpers
consume full packed offsets4096/4128 and2560, respectively. The full-packed
family is the smaller wiring change: bypass the old combine/QKV slice/conv
exactly once, preserve corrected packed for every other consumer, and keep
`use_batched_conv` true so recurrence receives the packed row-major QKV owner.

The rest-only family requires explicit coordinated downstream offsets:
Z0:1536, beta1536:1548, alpha1568:1580, plus compatible local beta/decay,
norm/gate and affine/gate entrypoints. Calling existing helpers with the short
buffer would be wrong. Reconstructing the full4160-wide tensor immediately
would discard the intended lower-movement contract and would require its own
complete-boundary measurement. The existing focused test does not perform
that restore, so this rejection already respects the coherent rest-only
family's advantage.

Final output slots/history/reset banks must exist before the first trace
capture and remain valid for replay. Owner-specific outputs may be allocated
at state/setup time with a bounded supported-batch contract; do not grow a
forward-time cache keyed by new shapes. Alternatively preserve invocation-
owned temporary output semantics with trace callers copying into preallocated
stable slots, without retaining newly allocated tensors across captures.
All inputs, history, taps and outputs must have distinct legal buffer
bindings; the existing helper rejects aliases and validates same-mesh DRAM
for history/taps/outputs. Never reuse a single owner-global destination that
overwrites a different live state owner's result.

## Conditional whole-layer sweep proposal

Do not spend a whole-layer integration run on this4–5× slower family. If a
new kernel changes the scalar packing/transaction structure and first beats
the existing focused complete boundary with unchanged exactness, then prepare
the smallest opt-in full-packed wiring above, default false, and run paired
current-policy actual-weight layer0 experiments at decode B1/3/17/32. Preserve
the selected recurrence, projection geometry and norm policy in both arms.
Compare full warmed traced step latency and every current output/contribution/
per-user/state gate against the frozen accepted single-chip TTNN reference.

Also cover nonaligned prefill S33/S513 and post-prefill decode continuation:
S>1 convolution remains on its existing path, while any S1 temporal tail must
either obey the explicitly supported new contract or use the unchanged
declared prefill path. No public sequence-alignment requirement may be added.
Use strict allocation tracking with program cache included, changed input and
state traces, rebound owners, queued histories, and watcher separately from
timing. Collect matched whole-layer tt-perf-report to verify the eliminated
combine/slice/untilize operations and absence of new layout conversions or
collectives. Rest-only wiring should follow only if its complete downstream
consumer boundary demonstrates a material improvement; it is not justified
by the present sub-microsecond side-output difference.
