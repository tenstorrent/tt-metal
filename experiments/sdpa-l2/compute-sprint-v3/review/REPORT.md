# B: independent grouped-compensation review and transfer

Complete: the final empty-local-validity candidate passes all62 B qualification
cases and reduces resident time5.36% and distinct256K time2.39% versus best v2.
It is0.58–0.71% slower at distinct32K, so retain it as a conditional research
point, not an unconditional replacement. Original group-two and replay-only
candidates are not recommended: their scheduling overhead is too high on
distinct inputs. All v1/v2 and production sources remain unchanged.

## Final B performance

Fresh interleaved controls on reservation224379, fixed Q256/K512/D128 and
unchanged input buffering/data movement;12 warmups followed by10 alternating
v1/v2/candidate triplets. Times are host-blocking trace medians, not instrumented
phase times. TFLOP/core counts only useful QK+PV FLOPs.

| Workload | v1 general ms | v2 early ms | Final valid ms | Final TFLOP/core | Time vs v2 |
|---|---:|---:|---:|---:|---:|
| Resident512 K chunks,8 Q repeats |168.693622|166.069653|157.170295|1.748918|−5.36%|
| Distinct32K normal,32 Q repeats |84.307857|84.974230|85.469194|1.608053|+0.58%|
| Distinct32K increasing max,32 Q repeats |84.270613|84.715495|85.316992|1.610921|+0.71%|
| Distinct256K normal,4 Q repeats |84.417530|83.931141|81.924534|1.677629|−2.39%|

Versus v1, the same candidate changes time by−6.83%,+1.38%,+1.24%,−2.95%
respectively. Resident v1/v2 throughput is1.629451/1.655196 TFLOP/core.
Each row has its own paired controls; no cross-reservation subtraction is used.
All four timers passed numerical, raw eager/trace and input/source integrity
checks.

Resident reuses identical KV and favors unchanged maxima. Distinct timers
repeat the **same Q job** over distinct KV with recurring input DM; they are
not distinct-Q fullchip model timings. Distinct multi-Q inputs are covered in
the separate qualification suite. This is a useful long-context scheduling
improvement, not evidence of general model or chip-wide speedup. Increasing-max
and shorter-context behavior still exposes the extra bookkeeping/branch cost.

## Original group-two result

43/43 input cases passed the per-case v3 error budget, with 129 variant records
covering matched v1, v2 early-guard, and group-two outputs. All controls agreed
bitwise. Candidate eager/two-trace replays matched raw BF16 bits, inputs were
unchanged, and recorded source pins remained unchanged during each job.

Coverage: K lengths512,1024,1536,4096,32768,262144; short cases include multiple
distinct Q jobs per core. Distributions include normal, uniform attention,
constant V=1 and3.25, zero V, common Q/K/Q+K/V, scaled QK, outliers, increasing
maxima, forced identity-to-changed-max transitions and repeated identical KV.
This is a single-seed scheme screen, not a completed held-out qualification.

| Case | v1/v2 L2 (%) | Group-two L2 (%) |
|---|---:|---:|
| Normal32K | 2.623707 | 2.623677 |
| Common V32K | 1.100822 | 1.100900 |
| Scaled QK32K | 5.029177 | 5.029351 |
| Normal256K | 3.312712 | 3.312464 |
| Constant V256K | 0.548638 | 0.548638 |
| Common V256K | 1.127972 | 1.128033 |

Eleven accepted cases have different candidate output bits. Worst relative
global L2 increase across the43 cases is0.007011%, far below the5% allowance.
At256K normal, worst-row L2 rises from4.512527% to4.522846% even as global L2
slightly improves; p95/p99 differences are much smaller. No material new
row/coherent-input regression was identified, but full diagnostic metrics are
retained rather than hidden behind the global gate.

### Common-mode caveat

Small original-reference L2 does not imply preservation of small variation
around a large offset. After subtracting32 from common-V32K outputs/reference,
baseline L2 is3884.79% and group-two L2 is3885.06%. Candidate–baseline distance
is24.08% of that tiny residual-reference norm. At256K this differential is175.04%
of the residual-reference norm, while both implementations already have roughly
11000% residual-normalized error. These are diagnostic normalizations, not new
acceptance gates. They show inherited BF16-path loss of small common-mode
variation, and why tiny output-quantization changes can be substantial relative
to that variation. Grouping does not solve that existing issue.

### Performance screen: reject original implementation

Fixed Q256/K512/D128, HiFi2, BF16 destination, same CB geometry, two input slots,
unchanged reader/writer. Four repeated Q jobs ×64 resident K chunks; six warmups
then eight alternating triplets, useful QK+PV FLOPs only:

| Implementation | Median ms | TFLOP/core |
|---|---:|---:|
| Frozen v1 general B | 10.5923 | 1.62192 |
| Frozen v2 early-guard B | 10.4270 | 1.64763 |
| Original group-two | 10.7421 | 1.59930 |

Group-two is1.41% slower than v1 and3.02% slower than v2. Numerical acceptance
alone does not justify adoption. No sustained original-group-two timer or model
speedup claim is made.

## Direct-PV-local screen

Independent B Q1024/K1536, two-core testing passed seven distributions and two
raw-bit trace replays per variant. Every candidate output hash matched the same
input's original group-two output. The short paired timer still loses to the
best control: v1 10.5890ms, v2 early10.4240ms, direct10.49995ms
(1.63619TFLOP/core). This is0.728% slower than v2, despite being0.841% faster than
v1. No wider direct qualification is justified by that result alone.

Reviewed the next `group2_noclear` source change: even folds omit dead local-plane
zeroing because the following odd PV overwrites it before any read. Odd clears
remain. Also reviewed the concrete18-instruction `group2_replay` even-identity
fold: data dependencies, DST offsets, delayed macro stores, final drain, and
denominator replay restoration are consistent. These reviews authorize bounded
tests, not promotion or an unmeasured speedup claim.

## Replay: resident win does not generalize

The 18-instruction replay version passed B K1 and K3 numerical/replay checks,
and matched the original group-two output hashes on the seven K3 cases. Its
fresh paired resident timer improves substantially, but distinct-input timing
exposes the expensive scalar changed-max fallback:

| Workload | v1 ms | v2 early ms | Replay ms | Versus v2 |
|---|---:|---:|---:|---:|
| Resident, Q repeats8/K chunks512 | 168.6484 | 166.0212 | 156.4964 | −5.74% |
| Distinct32K normal, Q repeats32 | 84.2978 | 84.9355 | 97.2953 | +14.55% |
| Distinct32K growing maxima, Q repeats32 | 84.3227 | 84.7239 | 101.1058 | +19.34% |
| Distinct256K normal, Q repeats4 | 84.4608 | 83.9554 | 85.5373 | +1.88% |

All these numerical and raw replay gates passed. The resident result reaches
1.75645 TFLOP/core versus1.65568 for the v2 control. Distinct timers repeat the
same Q job over distinct KV, including recurring input data movement; they are
not full multi-Q model benchmarks. Replay is **not** a general B replacement.
Broader replay qualification was stopped once its real-input regression was
established.

## Empty-local fast fallback: final numerical qualification

`group2_valid` tracks four per-Q/per-row-group validity flags and uses frozen
paired compensation whenever local is empty. This includes even identity
steps following odd changed-max steps. Only genuinely retained local data takes
the grouped fold. Source review and the 9,216-row-group host state-machine
probe passed; details and the stale-local negative control are in
`GROUP2_REVIEW.md` and `validity-probe.json`.

The expired-reservation `B-valid-k3-01` command produced no recoverable result;
it is not counted. The replacement `B-valid-k3-02` and final suite ran on the
new authorized reservation224379. All62 cases /186 variant records passed:

- The43 matched cases reproduce **every original group-two output hash**.
- Held-out seed20260926: Q2048/K8192, two cores,13 distributions; this exercises
  multiple distinct Q jobs per core, odd/even grouping and forced transitions.
- Held-out seed20260927: Q512/K32768, one core, six distributions (normal,
  growing maxima, common V, scaled QK, outliers and common Q+K).
- Original-seed coverage includes zero V, constant V=1 and3.25, uniform
  attention, Q/K/V common modes,1/2/3 K chunks, long256K context and stress.

Twenty-one cases change output bits from both identical controls. Worst global
L2 growth is0.0123221% relative, versus the allowed5%; its absolute increase is
0.000215092 percentage points (held-out32K outliers,1.745583%→1.745798%).
Largest candidate–baseline distance is0.0259801% of reference norm. Largest
worst-row increase remains0.0103186 percentage points (normal256K).
Largest p95/p99 increases are0.00211158/0.0000670525 percentage points.

Common-V256K PCC changes0.0248113→0.0245094, from an already very poor inherited
baseline. The common-mode variation caveat above therefore remains important;
the small whole-output L2 and tiny allowed numerical changes are not evidence
that this BF16 family preserves weak signals on a large V offset. No material
new recurrence drift or row-error regression was found in the held-out set.

Every variant passed two raw-uint16 trace replays, finite-output checks and
source/input immutability. Gate calculations are per case, against original
input FP64 reference; zero-reference cases use the specified absolute-error
gate. This is fixed-shape, noncausal experimental qualification, not model-level
or arbitrary-shape production acceptance.

## Independent E/G cross-review

`audit-eg-final.json` independently passes all10 final E/G evidence files and
60 records: numerical gate arithmetic, raw-bit replay, original/prepared input
integrity, exact preparation, source hashes, numerical defines, CB geometry,
two input slots and timing/TFLOP arithmetic. Full distinct-input records verify
the frozen canonical adapter. Resident records correctly mark that fullchip
adapter comparison as not applicable, rather than claiming it was run.

The shared header remains frozen at SHA256
`ac29b3fac4584629abb98ddfb76915a0b73739db0225b8a95422e65052781532`.
E/G retain their original native prepared BFP8/BFP4 KV formats; these are not
ordinary unprepared casts. Their result shows the same conditional behavior:
long-context normal benefits, but shorter/changing-max cases can regress.

## Evidence

- `GROUP2_REVIEW.md`: recurrence and CB/packing lifetime review, including
  direct-PV-local and fixed-parity proof.
- `state_probe.py`, `state-probe.json`:8,192 independent scalar recurrence trials;
  not a hardware SDPA emulator or a device acceptance substitute.
- `B-group2-k1-01.json`, `B-group2-k2-01.json`, `B-group2-k3-01.json`,
  `B-group2-k8-01.json`, `B-group2-32k-01.json`, `B-group2-256k-01.json`:
  all original numerical results, row/PCC/absolute diagnostics and source pins.
- `B-group2-perf-screen-01.json`: paired original performance screen.
- `B-valid-k3-02.json`, `B-valid-final-{k1,k2,k8,32k,256k,heldout8k,heldout32k}.json`:
  the final62-case qualification,186 variant records.
- `B-valid-perf-final.json` and `B-valid-distinct{32k-normal,32k-grow,256k-normal}-final.json`:
  the four final paired timing controls and12 variant records.
- `audit-b-final.json`:27 completed B artifact files /396 records including
  rejected intermediate implementations; all recorded source pins match.
- `audit-eg-final.json`: independent closing review of10 E/G files /60 records.
- `audit_final_results.py`: repeatable standard-library audit; the older
  `audit_results.py` is preserved unchanged because its source was pinned by
  earlier tests (its E/G path predates resident adapter=N/A records).

Reproduce through the parent-owned exclusive wrapper with
`qualify_valid_suite.py --suite short|wide|heldout --label-prefix NEW_PREFIX`,
plus the seven-distribution K3 call recorded in `B-valid-k3-02.json`.
Timing flags are preserved exactly in each final JSON's `arguments` object.
Do not reuse existing labels, which are immutable evidence.

Private kernels were JIT compiled and executed on IRD223862 and then224379,
bh-lb-08 device0, through the shared exclusive-lock wrapper. No production build
or edit was needed; no reset or dirty-marker mutation was performed by this
agent. The reservation interruption was observed, not bypassed.
