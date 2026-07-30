# AutoFix Report

## Starting Evidence

- Fresh independent Stage 02 review: `more-work-needed`.
- Finding: two-way MLP gate/up packing had been dismissed without an adapted
  PCC/performance trial.
- Diagnosis: `AUTODEBUG.md`.

## Hypothesis Experiment

- Hypothesis: setup-time packing of the equal-width gate/up RHS removes one
  runtime linear and may improve all decoder modes.
- Experiment: one 34,816-wide linear, exact 17,408-wide gate/up slices, and the
  existing SiLU-on-multiply ordering.
- Result: static tests and full-attention traced PCC passed unchanged.
- Performance: full-attention batch-32 `tt-perf-report` regressed from
  2386.759 to 2388.755 us/replay; host median regressed from 2.559 to 2.572 ms.
- Verdict: refuted as a final optimization.
- Evidence: `tracy/candidate_mlp_pack_full_b32/perf_report.csv`.
- Fix: candidate reverted; measured rejection recorded.

## Determinism Finding

- The trace test now restores identical mutable cache/state and proves two
  replays are bit-exact for both representative layer kinds at batch 32.

## Final Status

- The review finding is resolved by an earned performance rejection.
- Final graph is the faster pre-existing fused winner.
