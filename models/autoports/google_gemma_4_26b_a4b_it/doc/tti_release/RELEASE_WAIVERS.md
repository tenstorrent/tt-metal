# Stage 11 release waivers

## Accuracy reference status

The TTI report marks `meta_ifeval` and `meta_gpqa_cot` as `NA`, not failed, because this new custom autoport spec has neither a published-score mapping nor a GPU reference result. This handoff waives only that missing comparison baseline.

The underlying mandatory evaluations are not waived:

- `meta_ifeval` ran on the CI-nightly 5% sample and scored 82.62 (instruction-level strict accuracy).
- `meta_gpqa_cot` ran on the CI-nightly 5% sample and scored 40.0 (flexible-extract exact match).
- Both result rows are present in the copied JSON artifacts and the final report.
- TTI acceptance is `PASS` with zero blockers.

No benchmark, API, conformance, context-length, request-length, or model-correctness failure is waived.
