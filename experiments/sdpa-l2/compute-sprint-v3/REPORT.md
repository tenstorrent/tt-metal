# Numerics-relaxed SDPA compute optimization

Status: private candidates implemented, benchmarked and qualified. This is
an experimental report, not a production promotion.

## Conclusion

The small-numerical-change allowance yielded useful additional compute gains:
**5.4–8.8% less time** versus the prior optimized controls, while the largest
observed relative L2 increase was only **0.01891%**, versus the allowed 5%.
There was no need to change fidelity, exp, formats, preprocessing, block sizes,
input buffering or dataflow. A remains unchanged.

Keep the new C/D state-fusion and B/E/G grouped-compensation implementations
as qualified private specializations. Do not indiscriminately replace every
dispatch: the distinct normal 256K gains are only **1.6–2.6%**, with small
short-context/changed-max regressions on some variants. They should join the
production implementation only with explicit geometry guards and additional
model/shape qualification. No universal context-length threshold is established
by one pair of 32K/256K measurements.

## Contract and scope

The approved relaxation is a per-case upper error budget:
`candidate L2% <= 1.05 * baseline L2% + 0.0001 percentage points`.
The small absolute floor handles near-exact baselines. Exactly zero references
use the separate absolute-error gate documented in [PLAN.md](PLAN.md).
PCC, row-error tails, maximum absolute error, reference magnitude and output
distance are recorded alongside global L2. Lower error is not rejected.

All variants keep their established fidelity, exp, input preparation, DST and
CB formats, input buffering and dataflow kernels. Geometry remains noncausal
Q256/K512/D128. Only private compute implementations and test drivers are new.
The numerical relaxation changes recurrent-sum association, not the variant's
precision recipe. No HiFi3 or cheaper-exp experiment is included.

## What changed

**C/D: fuse recurrent state into existing FP32 accumulation.** When every
query row's running maximum is finite and unchanged, directly accumulate the
current PV result and denominator into the prior FP32 state banks. This removes
separate state-update passes. It changes partial-sum association. An early
guard scans completed maximum rows while the following QK work runs, reducing
the guard's exposed scalar overhead. Changed maxima retain the original path.
See [source review](ROOT_REVIEW.md) and the FP32 evidence directory.

**B/E/G: group two numerator contributions before compensation.** Keep one
BF16 PV chunk as local state, then fold it with the next into protected BF16
high/low state. A changed maximum rescales and folds every live contribution;
the final chunk always flushes. Denominator compensation is unchanged. A
four-flag validity map permits the original fast compensated update whenever
no local contribution is live. This avoids an expensive generic fallback.
See [independent review](review/GROUP2_REVIEW.md).

These are specialized implementations, not permission to enable the same
dispatch for causal, ring, other tile geometry, or arbitrary API settings.

## Evidence interpretation

Resident tests use one compute core with resident, repeated KV and count useful
QK+PV FLOPs. They isolate compute-side throughput but favor unchanged maxima.
They exclude input preparation and are neither measured chip TFLOPs nor model
speedups. Distinct-input tests retain the original dataflow and are reported
separately. Paired gains compare against the best relevant v2 control.

Fresh unchanged A control on replacement reservation 224379:
137.965306 ms, **1.992370 TFLOP/s/core**; 14 raw-bit-identical replays,
unchanged input tensors and selected sources. Its repeated-KV numerical
result must not be represented as a distinct random long-context result.

Completed intermediate work establishes why fallback cost matters: the first
replay-based grouped candidate reduced E resident time by 5.91% and B by
5.74%, but regressed distinct-input time. It is not a general replacement.
The revised validity-map candidate recovers most of this loss and gives
measured 2.62%/2.59% reductions on E/G distinct normal 256K.

### Final-candidate resident measurements

Useful TFLOP/s per core; lower time is better. Fresh controls run alongside
each candidate. C/D use nine alternating rounds at eight Q repeats and 512
K chunks; E/G use nine paired replay samples at sixteen Q repeats and 512
K chunks. Repeats lengthen the measurement, not the fixed Q256/K512 geometry.

| Variant | Best control TFLOP/s/core | Candidate TFLOP/s/core | Time reduction |
|---|---:|---:|---:|
| D |0.838884|0.894862|6.26%|
| C |1.125333|1.234116|8.81%|
| B |1.655196|1.748918|5.36%|
| A |1.992370|Unchanged|—|
| E |1.970957|2.090876|5.74%|
| G |1.970006|2.089641|5.73%|

Throughput increase and time reduction are different quantities: for example,
C's 8.81% time reduction is a 9.67% throughput increase. Neither is a model
speedup. Dividing by nominal fidelity-specific ceilings at 1.35 GHz gives
about 64.7%/67.0% for D/C and 72.1% for unchanged A. These are derived useful
FLOP ratios, not measured FPU-active counters or a fresh clock measurement.

B uses twelve warmups and ten alternating v1/v2/candidate triplets at eight
Q repeats and 512 K chunks. Its resident result is 166.069653→157.170295 ms
versus v2. E/G's paired v1 controls are also recorded, but the table compares
against the faster v2 resident controls.

### Revised E/G distinct-input results

Positive numbers below mean slower. Same original reader/writer; a second seed.
That seed is distinct from the primary accuracy suite but not a blind holdout:
its earlier replay timings helped identify the expensive fallback.

| Input | E time change | G time change |
|---|---:|---:|
| Normal 32K |+0.52%|+0.55%|
| Normal 256K |−2.62%|−2.59%|
| Increasing maxima 32K |+0.69%|+0.69%|
| Identity/change transitions 8K |+2.62%|+2.65%|

These favor retaining a long-context specialization, not replacing every
short-context invocation. The valid-state check, grouped-state bookkeeping,
and live-local changed-max fold still cost time when few updates can be skipped.

E/G exact final sources pass **60/60 records**: 48 primary accuracy cases,
eight second-seed distinct timing cases and four resident control comparisons.
Worst relative L2 growth is about **0.002035%**, versus the allowed 5%.
Maximum observed worst-row L2 increase is 0.000202 percentage points. This
does not establish a universal error bound or repair baseline stress failures.

| Variant/input | Baseline L2 % | Candidate L2 % | Candidate PCC |
|---|---:|---:|---:|
| E normal 32K |3.146165|3.146153|0.999524876|
| E normal 256K |3.596807|3.596713|0.999560798|
| G normal 32K |17.501226|17.501254|0.984840618|
| G normal 256K |16.819580|16.819526|0.986278232|

Sources: `compensated/{e,g}-valid-{short,stress,long,perf,distinct}-v1.json`;
see [E/G report](compensated/REPORT.md) for per-case provenance and diagnostics.

### B distinct-input and numerical qualification

One repeated Q job over genuinely distinct KV, unchanged original dataflow.
This is not a multi-Q full-model measurement. Positive percentages mean slower.

| Input | v1 ms | v2 ms | Candidate ms | Time change vs v2 |
|---|---:|---:|---:|---:|
| Normal 32K |84.307857|84.974230|85.469194|+0.58%|
| Increasing maxima 32K |84.270613|84.715495|85.316992|+0.71%|
| Normal 256K |84.417530|83.931141|81.924534|−2.39%|

The v1 general control is faster than v2 on the two 32K cases; versus v1 the
candidate regresses 1.38%/1.24%, respectively. This reinforces the conditional
recommendation instead of hiding the faster short-input control.

Final B qualification passes **62/62 cases** (43 matched cases plus 19 cases
across two new seeds), plus four resident/distinct timing comparisons. Worst
relative L2 growth is 0.012322%, and worst-row L2 rises at most 0.010319
percentage points. Twenty-one qualification cases change baseline output bits;
all actual trace replays remain bit-exact. Original matched group-two and the
optimized validity-map implementation reproduce the same output hashes on
all 43 matched cases. See [B report](review/REPORT.md).

### C/D distinct-input measurements

Same Q2048, two cores, three alternating baseline/candidate rounds, nine timed
replays per round, independent seed 1244. Positive percentages mean slower.

| Variant/input | Control ms | Candidate ms | Time change |
|---|---:|---:|---:|
| D normal 32K |21.795763|21.928185|+0.61%|
| D normal 256K |167.519197|164.874338|−1.58%|
| D increasing maxima 32K |22.125269|22.264042|+0.63%|
| D increasing maxima 256K |176.393981|177.456971|+0.60%|
| C normal 32K |16.628947|16.593357|−0.21%|
| C normal 256K |126.133132|122.914583|−2.55%|
| C increasing maxima 32K |16.935053|16.857892|−0.46%|
| C increasing maxima 256K |134.909364|134.298773|−0.45%|

Sub-percent effects should not be generalized beyond these paired runs. The
whole-Q identity requirement reduces opportunities on distinct KV: an earlier
instrumented D normal 256K run accepted 1,441/4,088 chunks (35.25%), versus all
noninitial chunks on repeated coherent KV and none on increasing maxima.
That profiling used the same predicate but is separate from unprofiled timing.

Source: `fp32/integrity-early-distinct-{C,D}-{32768,262144}-v2.json`.

Modern integrity qualification additionally covers K1/K2/K3 boundaries,
15 distributions at 32K, and seven distributions at 256K on a new seed. All
gates pass, including original device-input hashes, selected source stability,
and at least two actual raw-bit-exact trace replays. The late-to-early guard
move preserves output hashes in all directly compared records. Baseline versus
fused candidate is deliberately not required to be bit-exact.

Largest observed relative L2 growth for the final FP32 candidate is about
**0.01891%** (C, coherent repeated KV at 256K), well below the allowed 5%.
Normal 32K L2 is unchanged at 0.178024% D / 0.376902% C on seed 1243. On held-out
normal 256K, D changes 0.180500% to 0.180507%. Stress failures inherited from
the baseline remain failures in an absolute-accuracy sense: deliberately
increasing extreme logits can have very large error, and common-mode global
L2 can hide poor representation of small residual variations.

See [FP32 report](fp32/REPORT.md) for exact labels, per-case L2/PCC, retained
rounding decisions, ordering review, and rejected optimization attempts.

## Safety, build and reproducibility

Tests use Blackhole bh-lb-08, logical device 0, firmware 19.13.1, KMD 2.9.0,
12×10 exposed grid. Reservation 223862 expired; work paused and resumed on
224379 after a fresh locked matmul smoke passed September 19, 00:19:09 UTC.
The same existing host libraries are reused. Changed compute kernels are
compiled by device JIT and executed; no full host rebuild is claimed.

Device jobs serialize through the frozen v1 `run_locked.sh`, with timeout and
dirty-marker protection. Eager and replay outputs must agree in raw bits even
though baseline and candidate may differ. Qualification records pin selected
sources and verify original/prepared inputs are unchanged. This is not a full
compiler/firmware provenance closure or a proof for untested distributions.

Root verified all v1/v2 source hashes and the entire preexisting tracked diff
remain identical to the start-of-sprint snapshot after recovery. No production
kernel, public interface, canonical numerical recipe, or prior result changed.
Local Python compile checks and `git diff --check` pass. Root and subagents
independently audited final evidence and reviewed CB ownership, max-change and
final-flush handling. Earlier compile/implementation rejects remain documented;
their results are not represented as final-candidate evidence.
