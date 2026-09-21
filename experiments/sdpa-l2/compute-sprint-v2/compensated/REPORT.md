# Compensated B/E/G compute sprint v2

## Outcome

An **exact-one correction specialization with an early guard** improves the
no-recurring-input-DM E/G resident benchmark by about 2.8% over the frozen v1
winner. It preserves all tested output bits. On distinct normal 256K K/V the
gain is smaller, about 1.36–1.39%; normal 32K and forced-growing maxima are
about 0.23–0.29% slower. Retain it as a conditional long-context candidate,
not an unconditional default speedup. No production/shared/v1 sources changed.

The other deeper scheduling hypotheses were exact but neutral or slower.
This is an incremental improvement, not a change in the attainable LoFi
compute-throughput class. No 80% utilization claim is supported.

## Fixed numerical and dataflow contract

The authority is `flux2-frontier-v1/device_attention.py` recipe and device
preparation. E uses Q RNE7/BF16 and K/V RNE5/native BFP8; G uses Q RNE7/BF16
and native-group BFP4 RNE+saturation K/V. Both retain LoFi matmuls, BF16 DST,
score/P formats, full numerator/denominator compensation, exp, accurate
correction, reciprocal and final scaling. In particular G does not silently
use the older resident driver's different native BFP4 preparation.

Q256/K512/D128 and both input slots remain fixed. CB specifications remain
identical to the baseline: E 1,087,488 bytes and G 956,416 bytes. All numerical
defines match except the implementation-header selection. Readers/writers
are the original frozen resident or distinct-input readers, unchanged.

Baseline is v1 `lowp/combined_fence/compute_streaming.hpp`, SHA256
`11f7cb8f4c508a515687702f8d8ee0916cb604f3bcd8012111e5c2a4537b5f12`.
The final private [header](identity_early/compute_streaming.hpp) is frozen at
`bc8c1b13e455225e230146ef151ee1e840527cc2ba34db5fb92ae9789f59e435`.

## Accepted experimental mechanism

The guard compares all 64 first-column BF16 maxima in a two-tile row group.
Only bit-identical, finite old/new maxima qualify. NaNs/infinities and
opposite signed zeros take the ordinary path. The original accurate exp
correction is then exactly BF16 one. The specialized SFPU replay substitutes
that exact constant for correction loads but preserves every MAD/add,
rounding, high/low store, pack and release.

Both LoFi and HiFi2 [exhaustive](identity-proof-lofi-v1.json)
[device probes](identity-proof-hifi2-v1.json) verify every one of 65,280 finite BF16
encoding, including subnormals and both signed zeros, against the frozen
subtraction/correction path at scale 1/sqrt128. Every result is raw 0x3f80.
This is not a proof for arbitrary scales or nonfinite inputs.

Scanning at the correction site was too expensive. The final candidate scans
each already-published previous row's maxima after issuing the next row's
first QK matmul; the final row is scanned after the first partial PV matmul.
The per-K-step local flags are sent through the original mailboxes at the
original correction site. Correction CB reserve/push/pop, including its prior
PV publication fence, remain unchanged. See the independently reviewed
[guard/replay proof](identity/ORDERING.md) and
[earlier-scan ordering argument](identity_early/ORDERING.md).

## Primary resident measurement

One core; Q256/K512/D128, 16 Q repetitions ×512 repeated K blocks, no recurring
input DM. Five warmups and nine alternating baseline/candidate trace pairs.
TF/core counts useful QK+PV FLOPs only, not extra compensation work.

| Variant | v1 ms | Early-guard ms | Time reduction | v1 TF/core | Candidate TF/core |
|---|---:|---:|---:|---:|---:|
| [E](e-identity-early-resident-v1.json) |287.009083|278.971947|2.8003%|1.915465|1.970649|
| [G](g-identity-early-resident-v1.json) |287.124567|279.061640|2.8082%|1.914694|1.970016|

This input deliberately favors identity corrections. It is a compute
scheduling measurement, not long-distinct-input numerical qualification or a
full-chip throughput prediction. Times are synchronized trace wall time
from `perf_counter`, not profiler device-cycle estimates. The exclusive
device lock, fresh paired controls and unchanged source hashes avoid using
historical cross-hardware timing as a baseline. No fixed nominal-clock roof
has been equated to measured active frequency.

## Distinct K/V timing controls

One core, original distinct reader with recurring DM; preparation excluded.
Seven warmups and 11 alternating pairs per case. Same original-input reference,
exact preparation and equality gates as qualification. The short2–4ms cases
include measurable host/trace overhead; small differences should not be
interpreted as precise engine-only gains. Guard frequency is not instrumented.

| Variant / case | Q / K length | v1 ms | Candidate ms | Time reduction |
|---|---:|---:|---:|---:|
| [E normal](identity-early-distinct-timings/e-normal32k.json) |256 /32,768|2.283791|2.290282|−0.2842%|
| [G normal](identity-early-distinct-timings/g-normal32k.json) |256 /32,768|2.280192|2.286392|−0.2719%|
| [E normal](identity-early-distinct-timings/e-normal256k.json) |256 /262,144|18.068300|17.817586|1.3876%|
| [G normal](identity-early-distinct-timings/g-normal256k.json) |256 /262,144|18.007560|17.762245|1.3623%|
| [E growing maxima](identity-early-distinct-timings/e-growing.json) |2,048 /8,192|4.527202|4.540503|−0.2938%|
| [G growing maxima](identity-early-distinct-timings/g-growing.json) |2,048 /8,192|4.495622|4.505922|−0.2291%|
| [E transitions](identity-early-distinct-timings/e-transitions.json) |2,048 /4,096|2.249462|2.226341|1.0278%|
| [G transitions](identity-early-distinct-timings/g-transitions.json) |2,048 /4,096|2.245901|2.223081|1.0161%|

Growing-max inputs add an increasing projection onto mean Q. Transition
inputs retain distinct V but repeat adjacent K blocks with scales1,1,2,2,
4,4,8,8. These are useful scheduling stress controls, not model captures.
No universal relation between sequence length and guard hit rate is assumed.

Independent B transfers and longer repeated-Q/distinct-KV timing are owned
by [the review agent](../review/REPORT.md); their different timing scopes must not be
silently combined with the table above. [B resident](../review/B-early-paired-01.json)
independently measured 337.134190→331.851244 ms
(1.5670%, 1.630674→1.656633 TF/core). Its longer, repeated-Q/distinct-KV
timing also shows input dependence: about 0.57% less time for normal 256K,
but about 0.77% more for normal 32K. See its report for exact scope/gates.

## Qualification and audit

[Sixteen held-out records](identity-early-qualification/) cover E/G at
Q2048/K8192, one core with eight distinct Q jobs, seed20260924: normal with
growing maxima, common Q/K/V, constant V, uniform, outliers with growing
maxima, and paired-maxima transitions. Six additional held-out
[boundary cases](identity-early-boundary-qualification/) use Q1024 with
K1536 (three chunks, normal/outliers with growing maxima) and K512 (one
chunk, no recurrent correction), each in E/G, seed20260925.
Every case matches both canonical
`device_attention()` and v1 raw output bytes, including signed-zero bits.
Both kernels pass two actual trace replays even with zero timing iterations.
Original host/device inputs and represented prepared inputs are immutable.
Preparation matches the canonical oracle exactly as decoded values (signed
zeros equivalent there only); sources remain unchanged during each run.

The timing cases also enforce those gates, including distinct 256K. The
standard-library [audit](audit_evidence.py) independently checks selected
evidence, the frozen candidate hash, numerical define equality and identical
CB formats/depth. Run:

```sh
python3 experiments/sdpa-l2/compute-sprint-v2/compensated/audit_evidence.py
```

The completed audit passes all 35 selected frozen-candidate records.

Selected source manifests are not a claim to cover the entire compiler and
firmware dependency closure. Historical unsuccessful candidate records may
pin earlier harness versions; the audit's current-source check intentionally
focuses on the frozen final candidate. Private kernels were device-JIT built;
new Python files passed AST parsing. No production C++ files were edited.

## Rejected or non-winning experiments

Each row is a fresh paired E resident screen, not subtraction of unrelated
historical measurements. Positive percentages mean less time.

| Candidate | v1 ms | Candidate ms | Time reduction | Decision / record |
|---|---:|---:|---:|---|
| Retain correction in DST half |287.017422|286.998902|0.0065%|Neutral; [record](e-correction-cache-resident-v1.json)|
| Cross-half pack overlap, split8 |287.124545|287.502062|−0.1315%|Slower; [record](e-pack-overlap-resident-v1.json)|
| Cross-half pack overlap, split2 |286.976372|286.940051|0.0127%|Neutral; [record](e-pack-overlap2-resident-v1.json)|
| Cross-half pack overlap, split4 |287.154665|287.165335|−0.0037%|Neutral; [record](e-pack-overlap4-resident-v1.json)|
| Identity, scalar guard at correction |287.068294|305.447280|−6.4023%|Slower; [record](e-identity-resident-v1.json)|
| Guard only, original arithmetic |287.085764|316.731136|−10.3263%|Diagnostic overhead; [record](e-identity-guardonly-resident-v1.json)|
| Identity, unrolled guard at correction |286.990463|288.513281|−0.5306%|Slower; [record](e-identity-fastguard-resident-v1.json)|

The split16 source is prepared but unmeasured. Cross-half overlap retained
normal clear/release and explicit readiness fences; its lack of gain is not
permission to drop those fences. The first unrolled-guard JIT failed because
`#pragma` inside the UNPACK macro argument was attached to the wrong expanded
statement. `_Pragma` fixed compilation; failed/successful logs remain. The
root reviewed clean closure and cleared the device guard, without a reset.

## Engineering recommendation

Keep the frozen v1 winner as the general baseline. Preserve this candidate
and its correctness proof for a long-context specialization or further
code-generation study. Before enabling it by default, use representative
distinct model inputs and longer-batch timing, measure the actual64-row guard
hit rate, and compare a dispatch policy against the small fallback cost.
The fixed geometry and non-ring prototype is not general production support.

This sprint did not eliminate the core compensation cost: both BF16 state
components still require copies, exact arithmetic, rounding, stores and
packing. The early guard removes only work known to be redundant. Broader
throughput claims need new measured scheduling or codegen evidence, not
relaxation of precision or assumptions that memory traffic hides the cost.
