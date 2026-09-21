# E/G grouped compensation: final research result

The frozen `group2_valid/` candidate is a useful **long-context option**, not
an unconditional replacement. It reduces resident time about 5.7% and distinct
normal 256K time about 2.6%, while short/max-transition cases remain slower.
All 60 final records pass the numerical and integrity gates. No production
promotion, dispatch change, or v1/v2 source modification occurred.

## Performance against the best v2 baseline

Fixed Q256/K512/D128, LoFi, BF16 destination, unchanged CB formats/capacities,
two K/V input slots and original readers/writers. These are alternating paired
trace timings, with fresh controls on the same device for every comparison.
TFLOP/core counts useful QK+PV operations, not compensation instructions.

Resident: one core, 16 repeated Q jobs, 512 identical resident K iterations,
10 warmups and 9 measured pairs; preprocessing and recurring input DM excluded.

| Variant | Best v2 ms | Group2 ms | Time reduction | v2 TFLOP/core | Group2 TFLOP/core |
|---|---:|---:|---:|---:|---:|
| E: Q7/BF16, K/V RNE5+BFP8 |278.928321|262.930864|5.735%|1.970957|2.090876|
| G: Q7/BF16, K/V RNE BFP4 |279.063050|263.086214|5.725%|1.970006|2.089641|

Fresh v1 paired controls also passed: E 286.954225→262.941224ms (8.368% less
time), G 287.031147→263.042543ms (8.357%). Sources:
[e-valid-perf-v1.json](e-valid-perf-v1.json),
[g-valid-perf-v1.json](g-valid-perf-v1.json).

Distinct Q/K/V: one core, original recurring-DM reader, 7 warmups and 15 measured
pairs. The 32K cases have 2048 distinct Q rows, 256K has 1024, and 8K transitions
have 2048. Positive time change means slower.

| Case | E v2→group2 ms | E time change | G v2→group2 ms | G time change |
|---|---:|---:|---:|---:|
| Normal 32K |18.083924→18.177766|+0.519%|18.071214→18.170466|+0.549%|
| Normal 256K |71.135830→69.274144|−2.617%|71.051064→69.207729|−2.594%|
| Increasing maxima 32K |18.114364→18.240136|+0.694%|18.102755→18.227607|+0.690%|
| Identity/change transitions 8K |4.473707→4.590780|+2.617%|4.469063→4.587494|+2.650%|

Sources: [e-valid-distinct-v1.json](e-valid-distinct-v1.json),
[g-valid-distinct-v1.json](g-valid-distinct-v1.json). These distinct-input
gains must not be replaced by the more favorable resident gains.

Why the difference: unchanged-max odd chunks can leave PV directly in local
state and avoid a numerator fold. Max changes force a fold; if a pending local
contribution exists, that still uses the more expensive local-aware scalar
helper. The candidate also initializes protected root state per query and
tracks validity. Those costs are less amortized on short contexts. This is a
source-level explanation consistent with the measurements, not a measured
branch-frequency or cycle-attribution model.

## Numerical qualification

**60/60 final records pass**: 48 primary distinct-input cases, 8 second-seed distinct
timing/accuracy cases, and 4 resident control records. The primary seed is
20260923; the distinct timing/accuracy cases use second seed 20260924.
The latter is not an untouched blind holdout: earlier replay timings on that
seed motivated the validity-path optimization, although the grouped numerical
scheme was already fixed.
Coverage includes first K, odd final K, multiple distinct Q jobs, normal,
outliers, common Q/K/V, constant V, uniform attention, uniform+constant V,
zero V, changing maxima and alternating identity/change transitions, with
K lengths 512, 1024, 1536, 4096, 8192, 32768, 262144.

The exact per-case gate is original-BF16-input FP64-reference
`candidate_L2_pct <= 1.05 * baseline_L2_pct + 0.0001 percentage points`.
Zero references use explicit absolute errors; both zero-V cases are exactly
zero. PCC is undefined for constant references and is not fabricated.
Row p95/p99/worst, reference magnitude and baseline–candidate distance are
retained in every record.

- Worst candidate/baseline L2 ratio: **1.0000203489**, only 0.002035% relative
  increase, versus the allowed 5%. This is second-seed E normal 256K:
  3.821618481→3.821696247%.
- Largest increase in worst-row L2: 0.000201762 percentage points,
  E primary normal 256K (4.481654531→4.481856294%).
- Largest candidate–baseline output distance: 0.027512397% of reference norm,
  G uniform attention 256K.
- 17/60 output records differ from baseline bits, as allowed by this numerical
  sprint. Every candidate eager/two-real-trace replay matches raw bytes.
  Preparation, original/prepared input immutability, canonical adapter on
  distinct inputs,
  selected source hashes, fidelity and CB geometry all pass.

Representative primary normal inputs: H1,D128,Q256; each row below uses the
exact final validity candidate, not an earlier grouped implementation.

| Variant | K length | Baseline L2 % | Candidate L2 % | Baseline PCC | Candidate PCC |
|---|---:|---:|---:|---:|---:|
| E |32768|3.146164653|3.146153302|0.999524872|0.999524876|
| E |262144|3.596807271|3.596713375|0.999560800|0.999560798|
| G |32768|17.501226259|17.501253981|0.984840677|0.984840618|
| G |262144|16.819579882|16.819525946|0.986278222|0.986278232|

Sources: [e-valid-long-v1.json](e-valid-long-v1.json),
[g-valid-long-v1.json](g-valid-long-v1.json). The full final set is
`[eg]-valid-{short,stress,long,perf,distinct}-v1.json`, with 5/9/10/2/4 records
per file respectively. The earlier replay candidate's evidence is separate.

These are relative-to-existing-variant acceptance results, not a claim of
universal absolute accuracy. Inherited E common-K32 stress at 8K has about 48.22%
L2; G normal 8K about 17.43% and common-K32 about 82.43%. Grouping does not fix
these low-precision errors. Global common-V L2 also masks small variation
around the offset; see the independent B report's residual-scale diagnostics.

## What changed

The canonical `device_attention.py` recipe remains the authority. E keeps
its Q7/BF16 and RNE5/native-BFP8 K/V preparation; G keeps RNE+saturation BFP4,
not the biased native conversion. Fidelity, exp, maximum/correction precision,
full denominator compensation, reciprocal and normalization are unchanged.

CB8 holds protected BF16 numerator hi/lo. CB9's existing planes hold current PV
and a one-chunk local contribution. Fixed group2 boundaries and every final
chunk fold into protected state. Changed maxima rescale every old live
contribution before adding the new PV. The local-plus-new sum stays FP32;
there is no intermediate BF16 local sum or discarded update.

Odd PV writes directly to the local plane. A four-flag per-query state tracks
which row groups have an unmerged contribution. Empty-local updates reuse the
frozen paired compensated MAD/round replay, including identity updates. Only
an even step following an odd identity step can require a local-aware fold.
The even identity fold uses a reviewed 18-instruction replay with the same
grouped association and BF16 round/store templates. All physical local reads
are guarded by validity; stale slots are never consumed.

Flags reset every query and use global row-group indices despite final-row
popping. Original correction publication, SFPU/PACK drains and CB release/
acquire tokens remain. Final normalization reads fixed protected CB8, with the
original high-only numerator contract. No input data movement changed.

Independent source/order review:
[GROUP2_REVIEW.md](../review/GROUP2_REVIEW.md).
Generated scalar-helper SFPMAD/BF16-round windows and ELF hashes:
[group2-codegen-v1.json](group2-codegen-v1.json). These pins cover selected
dependencies, not the whole compiler/firmware closure.

## Rejected steps and remaining scope

Original generic-SFPI group2 was 4.01% slower in E's short resident screen.
Direct PV→local remained 1.67% slower. Removing dead even-fold clears nearly
reached parity. The 18-op replay produced the resident gain but still regressed
on E distinct 32K by 18%; the empty-local paired helper recovered most of that
loss. All sibling sources and evidence remain isolated and frozen.

Group4 was rejected by 4/7 scalar state cases and was not implemented; that
filter is not proof of a hardware-attention failure. This sprint does not
qualify other chunk sizes, D values, causal/masked/paged/ring/sparse paths,
production head/batch dispatch, all finite distributions, multi-chip behavior,
or model evaluations. E/G qualification here is one-core; the independent B
transfer additionally exercises two cores. No full-chip/end-to-end speedup or
chip-wide FLOP-utilization claim is made. Preprocessing cost is excluded.

Measurements ran on bh-lb-08, logical grid 12×10, firmware 19.13.1/KMD 2.9.0.
Reservation 223862 expired after E's first validity distinct run; 224379 restored
the same machine after a fresh matmul smoke. Every claimed gain uses its own
paired control, never a cross-reservation timing subtraction.

Recommendation: retain this frozen research point for long-context follow-up;
do not replace the best v2 path unconditionally or promote it to production
without dispatch/integration qualification. No more device jobs are pending.

## Reproduction and frozen source

Use the parent-owned exclusive `run_locked.sh` for all device runs.
`qualify.py --variant E|G --candidate group2_valid --cases
short|stress|long|perf --output PATH` and `distinct_valid.py` reproduce the
final suites. The local standard-library audit is:

```sh
python3 experiments/sdpa-l2/compute-sprint-v3/compensated/audit_evidence.py \
  experiments/sdpa-l2/compute-sprint-v3/compensated/[eg]-valid-*.json
```

Frozen validity header SHA256:
`ac29b3fac4584629abb98ddfb76915a0b73739db0225b8a95422e65052781532`.
No macro-initialization-hoist or other subsequent optimization was implemented.
