# Final29 completion-policy audit

CPU-only, not a formal stage review. No source or gate changes, device work,
optional measurements, or completion claim. This audit covers the B1 statistical
finding and the actual multigoal check dispatch; the68-job validation and six
profiles must still finish and pass their applicable requirements.

## What the evidence establishes

The fixed five independent alternating B1/S512 process pairs per kind give:

| Layer | Control mean ms | Final mean ms | Median paired difference ms | Exact one-sided upper bound ms |
|---|---:|---:|---:|---:|
| GDN0 |2.395564|2.194060|−0.216594|+0.037668|
| Attention3 |1.761020|1.701235|−0.062273|+0.057086|

The retained method is `exact_paired_median_upper_bound`, confidence0.95,
margin0. With n5 it uses the maximum paired difference and actual96.875%
coverage. Both upper bounds are positive. Therefore **statistical nonregression
is inconclusive under the predeclared test**, despite lower sample means and
four favorable pairs out of five for each kind. Neither a confidence pass nor
a population regression follows from these data.

Independently read all20 process JSON/command bundles in `final29_plan.json`.
For every arm, invoked the existing checker's `Gate.decoder(...,
performance=True)` with the exact current source/native inventory and effective
policies. All passed its numerical/finite/contribution/per-user/state/trace and
performance-provenance checks. Separately checked all10 paired fixture
identities, current harness hash, native inventory/mapped libraries, immutable
control pin and empty policy, B1/S512 shapes, three-sample estimator, exact
comparable argv, alternating arm order and strictly increasing timestamps.
Recomputed both upper bounds with the unchanged checker function; they equal
`final29_b1_uncertainty.json`. This avoids overlooking attention because the
single original gate method raises on the first GDN bound before reaching it.
It is not a substitute for the other outstanding final gates.

## Existing checker has no manifest-only carry mechanism

`tests/check_optimized_multichip_evidence.py:629–731` unconditionally requires
each bound<=0. `Gate.run` at1121 onward returns `pass: false` for any error.
`review.verdict=pass-with-carried`, `review.carried`, and an explicit user wrap-up
stop rule do **not** override this assertion. Unknown extra manifest fields are
not an acceptance mechanism. The original checker result must stay false if
this remains its sole error. Do not delete pairs, replace the estimator,
invent a tolerance, borrow earlier source timings, suppress the exception, or
mark the JSON pass field true.

The original goal requires before/after warmed timing; `$optimize` says
“Batch 1 stays a regression row ... do not regress it.” Neither specifies a
95% zero-margin median-confidence gate. The project nevertheless declared
that stricter decision rule before observing these final data, so its
inconclusive result remains part of the final record. `user_wrapup28.json`
explicitly permits nonblocking carry but says to keep the best version passing
required gates; it does **not** explicitly waive a failed numerical gate or
specifically declare this statistical requirement nonblocking.

The independent reviewer must therefore decide whether this is permissible
performance uncertainty rather than an unsatisfied required gate. This audit
does not make that decision. The installed stage-review skill allows
`pass-with-carried` only in round3, with Required Work empty and no remaining
correctness/accuracy or broken required-gate finding. A reviewer who treats
this as required missing evidence must retain `more-work-needed`; a cap alone
does not transmute it into a pass.

## Honest record if the independent review accepts carry

Leave the existing manifest's full `b1_prefill_regression` specification and
all raw samples intact, including the failing original-checker result. Record
`review.verdict="pass-with-carried"`, actual three review rounds/reports, and an
explicit `review.carried` item such as:

```json
{
  "id": "B1_PREFILL_CONFIDENCE_INCONCLUSIVE",
  "status": "carried-performance-uncertainty",
  "nonregression_95pct_established": false,
  "quality_gate_waiver": false,
  "evidence": ["final29_b1_uncertainty.json", "final29_b1_measurement_review.md"],
  "followup": "Prespecified confirmation in a later authorized stage; retain all original pairs."
}
```

Use repo-relative paths in the real manifest. Record any overall accepted-stage
status in a **separate explicit acceptance report**, not by relabeling the
artifact check as passed. That report should say `status: pass-with-carried`,
`artifact_checker_pass: false`, identify the sole preserved error by group and
bounds, cite the exact user instruction and independent review authorization,
and list all other checker groups as actually passed. State the threshold was
not established. Do not create such an accepted report before the complete
checker run and independent review support it. The final response/docs must
likewise distinguish carried confidence uncertainty from passed correctness.

This is a transparent exception record contingent on reviewer classification,
not an implementation workaround or a substitute for an explicit policy
change if the reviewer requires one. No checker/validation source modification
is needed to retain the honest failed diagnostic; changing those pinned files
now would also stale the in-flight validation source inventory.

## Actual runner check dispatch

The live runner process3230692 executes installed package0.1.24
`scripts/multigoal`, with stage5 prompt
`/home/ttuser/qwen36-27b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/prompts/model_bringup_multigoal/05-optimized-multichip-decoder.txt`.
The log manifest retains the older0.1.23 prompt list from original startup;
actual process argv identifies the resumed0.1.24 runner.

`find_check_script` at1537–1539 derives the sibling
`05-optimized-multichip-decoder.check.sh`. **That file does not exist in either
installed0.1.23 or0.1.24.** Both packages supply checks only for stages6–11.
`run_stage_checks` at1996–2000 records `stage_5_check=none` and returns when no
sibling exists. It does not discover or run repo-local
`check_optimized_multichip_evidence.py` automatically. Thus there is currently
no external stage5 script imposing additional hidden requirements; the generic
resume text claiming one will run is not specific to this stage installation.
If a check file is installed before completion, dispatch will discover it then.

This finding is **not permission to skip** the repo-local final evidence checker
or stage review. The stage owns those substantive requirements independent of
whether the multigoal wrapper has a shell hook. Leave runner/plugin files
unchanged. No massive JSONL scan was used: process argv, short manifest,
installed filenames and runner source establish the dispatch behavior.
