# Stage 11 blocker audit: 2026-08-14

This is a current-source check for the remaining Stage 11 readiness blocker.
It does not replace `stage_review_final.md` and is not a clean-pass verdict.

## Verdict

No valid closure was found for the mandatory `meta_ifeval` and
`meta_gpqa_cot` gates for `google/gemma-4-31B`.

The existing release handoff remains:

- `release-workflow-pass/readiness-fail`.
- Evaluated implementation: `models/autoports/google_gemma_4_31b`.
- Blocking rows:
  - `meta_ifeval`: 25.181850822484343.
  - `meta_gpqa_cot`: 26.339285714285715 after the corrected `[A-D]` answer
    filter.

## Local Evidence Checked

- `RELEASE_REPORT.md` still marks acceptance `FAIL` because both mandatory
  Meta rows lack a published or GPU reference.
- `release_report_data.json` records null published/GPU references for both
  rows and an empty waiver map.
- `release_model_spec.json` records base checkpoint
  `google/gemma-4-31B`, autoport code path
  `models/autoports/google_gemma_4_31b`, and `known_issues: []`.
- `runtime_model_spec.json` records
  `impl.code_path=models/autoports/google_gemma_4_31b`,
  `docker_server=false`, `local_server=false`, `service_port=8000`, and
  `max_context=113280`.
- `autofix/meta_accuracy_AUTOFIX_RESULT.md` remains applicable: one real GPQA
  parser defect was fixed, but the full exact canonical reference could not be
  produced locally.

## Current Source Audit

- Hugging Face exact revision API for
  `google/gemma-4-31B@d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3` still reports
  `model-index: null`; no exact eval artifact is available.
- Hugging Face current `google/gemma-4-31B` API also reports
  `model-index: null`.
- The exact base checkpoint tokenizer/config has no chat template. This
  supports the raw completion prompt contract and does not permit borrowing
  instruction-tuned chat results.
- Google's Gemma 4 model card still labels its benchmark table as
  instruction-tuned model results. Those GPQA numbers do not grade the raw
  base checkpoint or the reconstructed `meta_gpqa_cot` task.
- Current public `tt-inference-server` `v0.20.0` release specs and support docs
  include `google/gemma-4-31B-it` on `p300x2` through stock
  `models/tt_transformers`; they do not include a release spec, GPU reference,
  or waiver for exact base `google/gemma-4-31B`.
- Existing public Tenstorrent issue references for Meta eval waivers apply to
  other checkpoints/tasks, such as Llama 3.1 or `google/gemma-4-31B-it`, and do
  not satisfy the row-specific waiver rule for this autoport base model.

## Current Host Notes

The current reservation host is `qb2-120-p02t03` with four visible Blackhole
`p300c` UMD devices. `tt-smi -ls --local` passed on 2026-08-14.

`STAGE11_PREREQUISITES_p300x2.md` records that a fresh no-Docker p300x2 rerun
would first need the vLLM autoport selector/install and TTI harness fixes
rebuilt. That is separate from the mandatory Meta gate: a valid rerun on this
host is still expected to remain readiness-fail until an exact canonical
reference, valid row-specific waiver, or product-owned replacement gate exists.

## Required Closure

One of these is still required before Stage 11 can become `clean-pass`:

1. Exact canonical control for both tasks on
   `google/gemma-4-31B@d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3`, BF16
   weights, exact tokenizer/BOS behavior, raw prompts, unchanged greedy caps,
   and corrected GPQA scorer.
2. A current row-specific issue waiver proving the correct canonical
   implementation fails the same exact row for a non-autoport reason.
3. A product-owned contract change that makes these raw-base Meta rows invalid
   or non-mandatory and provides an approved replacement gate.
