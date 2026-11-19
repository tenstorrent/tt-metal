# Native seed AutoFix result: BF16 repaired but slower; FP32 update still fails

CPU-only analysis of completed parent-owned hardware results, 2026-11-13.
The original failures, probes, helpers and production sources are unchanged.
`native_seed_autofix26_result.json` contains source-artifact SHA256s, all nine
checkpoint summaries, same-dtype custom comparisons, timings, trace evidence,
and an independent CPU reconstruction from the saved final tensors. Reproduce
with `python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/native_seed_autofix26_analyze.py`.
This script imports CPU PyTorch only; it imports no TTNN/model code.

## Decision

Reject the measured **seed-preserving BF16** candidate for cost, not quality:
it passes every collected required numerical comparison in B1/normal,
including all nine fixed-input checkpoints, 50 queued changed-input updates,
changed state, same-dtype custom comparisons, and the separately timed
702-step trajectory. Its complete-boundary median is 98.74123 us versus
52.70483 us for current `cumulative_exp_scalar` custom BF16, a 1.87348× cost.
This is a meaningful coherent BF16 persistence repair, not an immediate
FP32-to-BF16 restore or first API-error rejection. Wider batches/modes and
actual-weight layer quality are untested for this seed repair; do not promote
the result to a universal quality pass.

Reject the measured **seed-preserving FP32** candidate for both quality and
cost. It fails the unchanged .995 per-head output gate first at sampled
checkpoint64, and reaches only .225967298 at702. Its median is 99.30520 us
versus custom FP32 52.53952 us (1.89010×). The complete probe's `fail` status
is correct: FP32 remains a required path. It must not mask BF16's improvement.
These rejections apply to the measured families, not every future native or
FPU/SFPU recurrence implementation. Production should retain the passing,
faster custom recurrence; no additional native precision rewrite is needed
to justify rejecting these particular slower candidates.

## Matched before/after evidence

`native_seed_fixed702_26.json` uses actual 1×4 TP4, watcher enabled,
allocation tracking enabled and program-cache tracking retained. The separate
`native_seed_perf26.json` has watcher/profiler/tracking disabled and seven
samples of100 traced replays after two warmups. All paths measure the same
raw grouped packed QKV + g/beta → preparation → recurrence → compact output
and persistent state boundary. Timed state resets and 702-step final checks
remain intact. All rows below are from that completed performance run;
quality-failing timings are rejection evidence, not accepted optimized numbers.

| Path | Median us | Output min head PCC at702 | State min head PCC at702 |
|---|---:|---:|---:|
| Custom BF16 | 52.70483 | .999997257 | .999999776 |
| Custom FP32 | 52.53952 | .999999984 | 1.000000000 |
| Prepared native FP32 | 89.08166 | .193892391 | .999869095 |
| Prepared native BF16 | 88.40376 | .992188569 | .999984962 |
| Precise readout FP32 | 97.89250 | .177455495 | .999869095 |
| Precise readout BF16 | 97.26510 | .992535243 | .999984962 |
| Precise seed FP32 | 99.30520 | .225967298 | .999885022 |
| Precise seed BF16 | 98.74123 | **.997482984** | .999993769 |

The same-dtype custom output comparisons at702 independently score
.225949447 for seed FP32 and .997476951 for seed BF16. Every original versus
readout-only native persisted-state equality check passes for both dtypes;
the readout change did not corrupt the update. Seed states intentionally
differ and improve both state metrics. The short watcher smoke passed, but
four updates alone were insufficient to reveal the FP32 problem.

Selected fixed-input checkpoint output minima (all nine checkpoints, including
1/2/4, are retained in JSON):

| Updates | Readout FP32 | Seed FP32 | Readout BF16 | Seed BF16 |
|---:|---:|---:|---:|---:|
| 16 | .999918313 | .999952496 | .999950967 | .999976565 |
| 64 | .986733010 | **.992585955** | .995717604 | .998549483 |
| 128 | .334254002 | .393555472 | .992517332 | .997482130 |
| 256 | .177467926 | .225981544 | .992535243 | .997482984 |
| 512 | .177455495 | .225967298 | .992535243 | .997482984 |
| 702 | .177455495 | .225967298 | .992535243 | .997482984 |

“First at64” means first sampled failure; no assertion is made about the exact
transition between16 and64. The late stabilization supports a perturbed
fixed-point explanation rather than unbounded corruption.

## Quantitative localization: update dominates remaining FP32 error

The final raw artifact was loaded on CPU with `map_location='cpu'` and
`weights_only=True`. The independent calculation contracts each saved live
FP32 state with its **actual device-prepared Q**, in FP64. Across all48 heads,
the readout-only hardware agrees with this same-state contraction at minimum
PCC .9999999999778593, and the seed hardware at .9999999999743162. Both still
disagree with the trajectory oracle. This directly locates the dominant
remaining FP32 error before readout, without using a relaxed gate.

The strongest cancellation example is rank1/user0/head8, where actual prepared
`q·k = -1.8695008083e-5` and the centered oracle-output norm is only
`2.2306200693e-5`:

| Error at702, rank1/head8 | Original native | Precise readout | Precise seed |
|---|---:|---:|---:|
| Norm of projected state error, `qᵀ(S−Sref)` | 1.1346423e-4 | 1.1346423e-4 | 9.6892642e-5 |
| CPU saved-state contraction minus hardware output norm | 5.3060266e-6 | 5.7478715e-10 | 6.4932757e-10 |
| Total output-error norm | 1.1730369e-4 | 1.1346475e-4 | 9.6893113e-5 |
| State relative Frobenius error | .0112025 | .0112025 | .0100904 |

Preserving the seed reduces this projected-state error by14.60%, but it remains
approximately4.34× the centered target signal and149,220× the residual
readout error. High whole-state PCC therefore cannot certify projected output
quality. The readout correction removes its own numerical error successfully;
its slightly worse oracle PCC than the original native output is not evidence
that it broke state or readout mapping.

Preparation is also quantitatively excluded as the dominant explanation for
this fixture. All six prepared buffers are bit-identical. Across ranks, Q
normalization differs from FP64 by at most3.126e-9, K by2.220e-8, decay exp by
3.031e-8; beta is exact. A new independent CPU trajectory uses the actual
device-prepared Q/K/V/decay/beta, FP64 update arithmetic and explicit FP32
persistence after **each** of702 steps. Its minimum per-head output PCC versus
the original persistence-matched oracle is **.9999999999016225**. Thus the
observed .22597 seed score is not explained by this preparation difference.
This is a host precision discriminator, not an emulator of FPU rounding.

BF16 saved-state contraction is intentionally not used for localization:
hardware readout sees live FP32 state before BF16 persistence rounding.
The measured BF16 improvement is established by its original oracle and
custom gates, not an invalid contraction of its rounded stored state.

## Ownership, trace and memory evidence

All eight paths pass exact eager/trace comparisons and changed-input plus
changed-state replay. All six native-owner rebinding checks pass. The watcher
quality report records zero unsafe buffers after all captures and after
releasing the allocation positive control; that control was rejected before
replay. There are no acknowledgement/corruptible-scope exemptions. The case
exception reports required trace=true, rebind=true, quality=false; the error
is numerical, not a trace/ownership admission failure. This is evidence for
the executed B1 case, not wider unmeasured worker reuse.

The result's `memory_estimate` describes the base prepared owner and omits the
seed-only CB17. Add65,536 bytes per active worker: seed recurrence CB totals
are606,208 bytes/core for FP32 and573,440 for BF16. These are source accounting
figures, excluding kernels/allocator overhead, not measured physical peaks.
The seed adds no persistent DRAM state, no collective or extra global launch.
Its extra L1 and ~1.4–1.5us over readout-only are nevertheless costs.

## If a further repair is separately justified

No readout-only or preparation-only change is the next useful experiment.
The unresolved subphase is within native decay/prediction/delta/outer-product
update; these data do not identify one unique instruction. Seed preservation
isolates one narrowing and proves it was contributory, not sufficient for
FP32. There was no same-incoming-state hardware replay at the first failure,
so do not claim that finer attribution has already been measured.

The next bounded discriminator from `native_update_precision_audit25.md` is
direct-FP32 state and decay with exact SFPU scalar multiplication, preserving
CB10 for remaining native prediction and mirrored direct CB17 for the seed.
If pursuing a coherent repair rather than subphase diagnosis, also replace
prediction/residual with the proven precise contraction/direct SFPU operations,
then replace the outer product only if necessary. Preserve the original
separate product/add rounding, BF16 persistence packing, per-head gates and
all ownership contracts. Never mark an existing FPU-consumed CB direct while
retaining its FPU consumer. Such an implementation remains unmeasured and
must beat the complete custom boundary before selection. The demonstrated
1.87–1.89× cost already justifies keeping it off the production path.
