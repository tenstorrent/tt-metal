# Stage 11 independent rereview — final

Date: 2026-07-17 UTC

Verdict: **MORE-WORK-NEEDED**

## Required work

- **P1: Close both mandatory Meta accuracy gates.**
  `meta_ifeval` scores 25.181850822484343 and corrected
  `meta_gpqa_cot` scores 26.339285714285715, but neither row has an
  exact-checkpoint canonical reference or a valid row-specific waiver. The
  regenerated release report correctly marks readiness `FAIL`. `$tti-release`
  and `$stage-review` prohibit clean-pass while these mandatory rows remain
  ungraded and unwaived.
- **P1: Classify the document-111 HF/TT output divergence within that gate.**
  Exact BF16 HF produces coherent cyclotron reasoning on the same prompt for
  which the saved TT response mechanically repeats an answer phrase. One
  reduced-precision autoregressive trajectory need not token-match BF16, but
  this qualitative divergence is not evidence-backed as harmless. The exact
  canonical control must determine whether it is isolated or a systematic TT
  quality regression; a confirmed regression requires model-path diagnosis and
  repair before the rows can pass.

## Findings that reconcile cleanly

The rereviewer found no remaining fixable artifact inconsistency outside the
mandatory accuracy gate. The following evidence is mutually consistent:

- the exact generated autoport path and tt-metal implementation commit;
- the external no-Docker server mode and copied runtime-spec wiring;
- the 113280-token serving context and its physical-capacity evidence;
- the valid 149-token non-aligned request;
- all 17 benchmark points, including 65,535 actual input tokens plus 128 output
  tokens, with no failed requests;
- 21/21 OpenAI-compatible parameter-conformance cases;
- the dynamic irregular-batch repair and its live release coverage;
- the copied successful workflow log, cleanup, device reset recovery, final
  four-device health, and absence of Stage 11 servers, containers, or port-8000
  listeners.

## Primary-source reference and waiver audit

No valid exact-base reference, issue waiver, or release-note waiver was found.

- The exact Hugging Face revision API
  (`https://huggingface.co/api/models/google/gemma-4-31B/revision/d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3`)
  lists only README, configuration, tokenizer, and weight files. It has no
  `model-index` or eval-result artifact.
- The exact-revision model card
  (`https://huggingface.co/google/gemma-4-31B/blob/d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3/README.md`)
  and Google's Gemma 4 model card
  (`https://ai.google.dev/gemma/docs/core/model_card_4`) state that the
  published benchmark table is for instruction-tuned models. The reported
  GPQA and IFEval numbers therefore cannot grade this raw base checkpoint.
- The exact tokenizer has `chat_template=null`, which supports the chosen raw
  completion contract but supplies no accuracy reference.
- The Gemma 4 technical report (`https://arxiv.org/html/2607.02770`) says its
  final-model evaluation covers IT models. Its 84.3 GPQA and 98.9 IFEval rows
  are not base-checkpoint controls.
- Tenstorrent issue 4090
  (`https://github.com/tenstorrent/tt-inference-server/issues/4090`) documents
  that even `gemma-4-31b-it` standard IFEval lacked a usable reference. It does
  not show this exact base model failing `meta_ifeval` under the same contract.
- Tenstorrent issue 4176 and PR 4331
  (`https://github.com/tenstorrent/tt-inference-server/issues/4176#issuecomment-4715337652`
  and `https://github.com/tenstorrent/tt-inference-server/pull/4331`) report
  83.33% for `google/gemma-4-31B-it` on thinking-enabled, chat-templated
  `r1_gpqa_diamond` with 198 samples. That differs from the 448-row raw-base
  `meta_gpqa_cot` prompts, scorer, endpoint, sampling, and checkpoint.
- TTI v0.18.0 and current public eval configuration likewise target only
  `google/gemma-4-31B-it`, not the exact base revision and reconstructed Meta
  tasks.

These sources reinforce the blocker; none satisfies the waiver rule. A newly
filed issue that merely records the absent chat template, IT-only published
scores, missing canonical references, and current TT scores would still be
disclosure rather than a waiver.

## `$autofix` evidence and exhaustion

`$autofix` found and fixed one real GPQA harness defect: the scorer accepted
the prompt placeholder `X` through `[A-Z]`. Restricting the answer alphabet to
`[A-D]` changes the saved-output result from 94/448 (20.9821428571%) to 118/448
(26.3392857143%). The repair and regression tests are committed in TTI commit
`b803374e04c2460ea3bfabec4bfed832f2af532a`.

The remaining canonical-control path was investigated rather than waived:

- The authoritative TT samples contain 497,845 IFEval and 604,704 GPQA
  completion tokens: 1,102,549 completion tokens plus 146,022 prompt tokens.
- One exact BF16 HF GPQA row completed in 223.36 seconds. HF and TT both chose
  C for row 0, whose gold answer is B. This is useful path evidence but not a
  full accuracy reference.
- A batch-4, 2,048-token control returned no rows before a clean 904.546-second
  timeout and reached 61.529 GiB sampled RSS.
- An exact static batch-32 probe used the first 32 canonical GPQA prompts
  (8,327 prompt tokens) and requested 128 output tokens per row. It did not
  return within 900 seconds and reached approximately 70 GiB RSS. Even
  optimistically crediting all requested tokens bounds this shape below 4.551
  output tokens/second and projects above 67.3 hours for the saved completion
  workload. This is diagnostic rather than a strict long-generation lower
  bound, but it demonstrates no required order-of-magnitude local gain.
- The official `google/gemma-4-31B-it-assistant` MTP drafter preserved the exact
  row-0 output (229 tokens including EOS) but took 227.121 seconds, versus
  223.36 seconds without MTP. It does not accelerate the raw base checkpoint.
- Transformers prompt lookup also preserved exact output. Row 0 improved from
  223.36 to 138.803 seconds (1.61x), still far from a tractable full control.
  On deliberately capped, highly repetitive TT document 111, exact ordinary HF
  produced 256 tokens in 237.512 seconds and prompt lookup produced identical
  tokens in 193.240 seconds (1.229x).
- Document 111 is also a material model-output warning: the HF continuation is
  coherent cyclotron reasoning, while the saved TT continuation repeats an
  answer phrase. That divergence prevents treating TT repetition as evidence
  for high HF lookup acceptance or teacher-forcing saved TT tokens as a
  canonical continuation.
- Installed prompt lookup is batch-1-only, so its measured gain cannot be
  multiplied by the batch-32 probe. The host has 16 physical CPU cores, no CUDA
  or ROCm device, no canonical CPU vLLM path, and no installed
  exact-output-equivalent llama.cpp path. Changing precision/backend would no
  longer be the required exact Transformers BF16 reference.

The local batching, MTP, prompt-lookup, and available-engine hypotheses are
therefore refuted. `$autofix` legitimately exhausted this environment, but
exhaustion does not convert a readiness failure into clean-pass.

## Minimum valid closure package

One of the following is required:

1. **Exact canonical control:** run all 541 IFEval and 448 GPQA rows on
   `google/gemma-4-31B` revision
   `d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3` with BF16 weights, the exact
   tokenizer/BOS behavior, raw prompts, unchanged greedy generation caps, and
   the corrected scorer. Preserve per-sample output, task/config hashes,
   revision, scores, hardware, and runtime metadata. A 32-row pilot per task
   should project at most eight hours; otherwise add independent replicas. One
   141 GB H200, or enough H100-class workers to hold independent exact replicas,
   is the preferred starting point. Compare TT to the resulting approved
   references within tolerance.
2. **Evidence-backed row waiver:** a current linked issue or release note must
   name the exact row and checkpoint, include a full canonical control showing
   the correct implementation fails the same exact evaluation in the same way
   for a non-autoport reason, and carry explicit release-owner approval.
3. **Product-owned contract change:** an authoritative release note must
   declare these exact raw-base tasks invalid or non-mandatory for this
   checkpoint and provide an approved replacement gate, or supply
   product-approved thresholds tied to these exact prompts. The measured TT
   rows must then pass the replacement threshold.

Generic approval, `-it` scores, base-versus-instruct prose, or an issue that
only documents missing references is insufficient.

## Hard-check gaps

- No full exact-revision HF/GPU result exists for either reconstructed Meta
  task, so neither TT score nor the document-111 divergence can be graded
  against a canonical distribution.
- The bounded local experiments prove that this host has no demonstrated
  tractable exact-control path. They do not substitute for the external control
  or a product-owned exact-contract threshold.

## Anomaly ledger

- **Observed anomaly:** The original release acceptance passed through two
  missing-reference waivers.
  **Evidence:** The successful final6 log records acceptance PASS while neither
  Meta row had a published or GPU reference.
  **Affected path:** Release-report acceptance.
  **Control or comparison:** Mandatory-row rules in `$tti-release` and
  `$stage-review`.
  **Likely subsystem:** TTI known-issue/acceptance aggregation.
  **Investigation performed:** Removed the unsupported waivers and regenerated
  the copied report/data.
  **Resolution:** Fixed; both rows are explicit readiness blockers.
- **Observed anomaly:** GPQA accepted the literal prompt placeholder `X` as an
  answer.
  **Evidence:** `[A-Z]` gives 94/448 with 99 final `X` rows; `[A-D]` gives
  118/448 and 26.3392857143%.
  **Affected path:** `meta_gpqa_cot` scoring.
  **Control or comparison:** Offline rescore of the authoritative 448 samples.
  **Likely subsystem:** Generated TTI task parser.
  **Investigation performed:** Corrected parser, cache validation, and
  regression tests.
  **Resolution:** Fixed in TTI commit `b803374e0`.
- **Observed anomaly:** TT document 111 repeats an answer phrase while exact
  BF16 HF gives coherent cyclotron reasoning.
  **Evidence:** Ordinary and prompt-lookup HF generation agree exactly for 256
  tokens; TT and HF diverge qualitatively.
  **Affected path:** End-to-end raw-base GPQA generation and accuracy.
  **Control or comparison:** Exact-revision BF16 HF on the same canonical
  prompt.
  **Likely subsystem:** Unresolved; reduced-precision trajectory, serving
  feedback, or another model-path quality source.
  **Investigation performed:** Confirmed the HF output under both ordinary and
  exact prompt-lookup generation and rejected TT-token teacher forcing as a
  canonical shortcut.
  **Resolution:** More-work-needed pending the full canonical comparison and,
  if systematic, model-path repair.
- **Observed anomaly:** The first post-release mesh smoke hit a device-0
  Ethernet heartbeat failure.
  **Evidence:** Recorded core 31-25 heartbeat timeout after shutdown.
  **Affected path:** Reservation hardware health, after release evidence was
  complete.
  **Control or comparison:** Bounded list/reset/list followed by a repeated
  P150x4 mesh open/close smoke.
  **Likely subsystem:** Recoverable ARC/ERISC/remote-Ethernet infrastructure.
  **Investigation performed:** Verified no holders, reset all four boards, and
  repeated health checks.
  **Resolution:** Fixed infrastructure recovery; not a model result.

## Scope inspected

- `$stage-review`, `$tti-release`, and `$tt-device-usage` requirements;
- release report/data, runtime and release specs, RUN_NOTES, successful final6
  log, smoke/eval/benchmark artifacts, context contract, health evidence, and
  dynamic-grid code/tests;
- both Meta `$autofix` reports and exact local-control artifacts;
- exact-revision Google/Hugging Face sources and current Tenstorrent public
  issues, pull requests, release notes, and eval configuration;
- cleanup state and the clean TTI checkout.

## Residual risk

- Without the exact canonical distributions, the current evidence cannot
  distinguish inherited raw-base behavior from a remaining TT quality
  regression across both mandatory suites.
- The release workflow, API, context, benchmark, provenance, and cleanup gates
  otherwise have direct passing evidence; they do not offset the mandatory
  quality blocker.

This verdict satisfies the user's stop-after-`$autofix`-fails condition, but
Stage 11 remains `release-workflow-pass/readiness-fail`, not customer-ready.
