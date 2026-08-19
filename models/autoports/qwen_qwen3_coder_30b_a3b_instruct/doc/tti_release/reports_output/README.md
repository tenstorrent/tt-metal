# Read this before quoting the report in `release/`

`release/report_Qwen__Qwen3-Coder-30B-A3B-Instruct_2026-08-19T002852+0000.md`
is TTI's own output for the `--workflow release` run of 2026-08-18 22:35:11 UTC.
It is copied **byte-identical** and is deliberately not edited — including its
`Acceptance status: ✅ PASS`.

**`PASS` does not mean the evals passed. Zero of six eval rows passed.**

| | |
|---|---|
| Rows registered | 6 |
| Rows that passed | **0** |
| Rows that failed | 3 — `meta_ifeval`, `meta_gpqa_cot`, `gpqa_diamond_cot_zeroshot` |
| Rows ungraded (`NA`) | 3 — no `published_score` and no `gpu_reference_score` exists for this model, so TTI reports them for information and does not grade them |

The report's own line says so if you read it closely —
`Evals: ✅ PASS (0/6 passed, 3 waived, 3 NA)` — but the headline is a green
tick, and the three failures are green only because this model is registered
`EXPERIMENTAL`, a tier whose policy demotes every eval failure to
"informational" (`report_module/acceptance_criteria.py:434-435`). The tick is a
statement about the tier, not about the model. The three failing rows still
print `❌ FAIL` in the report's own accuracy table.

**None of the three failures is a model result:**

- `meta_ifeval` and `meta_gpqa_cot` never executed. TTI's Meta-eval path builds
  a `<repo>-evals` dataset name and accepts only the Llama-3.1/3.2 Evals
  collection, so these two tasks cannot run for **any** non-Llama model.
  Structural and permanent here.
- `gpqa_diamond_cot_zeroshot` died at dataset download — `Idavidrein/gpqa` is
  gated and the account lacked access at the time. Access was later granted and
  the task re-ran clean against the same live server: **56.1 %**, 198/198,
  exit 0. The report predates that re-run and has not been edited.

**What the model actually scored**, from rows that did execute: `mbpp_instruct`
77.2 %, `humaneval_instruct` 92.7 %, `ifeval` 81.1 % / 87.1 % strict,
`gpqa_diamond_cot_zeroshot` 56.1 % (re-run). All ungraded for want of a
reference score, not because anything went wrong.

**One model-side defect exists and is not in the report as a failure**: the
`isl=131072` benchmark point completed, but with a 94.4-minute TTFT. Long-context
prefill on this port is severely superlinear.

Everything above is set out with its evidence, its mechanism and its source
lines in **`../RUN_NOTES.md`** — the result summary table at the top, then
Findings 1-5. Read that before quoting any number from `release/`.
