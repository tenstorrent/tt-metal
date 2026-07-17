# Stage 06 Remediation Review

Verdict: clean-pass

## Required Work

- None. The Stage 06 remediation is ready to commit.

## Evidence-Ranked Conclusions

- P0/P1: no finding. The corrected full-model capacity plan counts 2,789,212,160 logical KV values at the physical tiled-BFP8 ratio of 17/16, yielding 2,963,537,920 bytes/device. Independent arithmetic reproduces the recorded batch-1 total of 27,847,140,744 bytes, 6,378,379,896-byte margin, batch-3 margin of 451,304,056 bytes, and batch-4 shortfall of 2,512,233,864 bytes. The advertised 262,144-token Stage 06 context still fits.
- P0/P1: no finding. The non-greedy defect and repair match the source contract. `Sampling1D` captures `manual_seed` before `sampling`; `_initialize_non_greedy_rng()` now copies real active-slot seeds, executes `ttnn.manual_seed()` once at the request boundary, synchronizes, and replaces the persistent seed tensor with `UINT32_MAX`. Fresh and reused non-greedy trace paths invoke that boundary helper, while `decode_next_token_traced()` contains no seed copy, seed initialization, synchronization, or host RNG work.
- P0/P1: no finding. The focused TP4 test uses the Gemma physical contract (1x4, vocab 262,144, local top-k 32, FP32 candidate gather), first proves fixed-real-seed replay repeats one draw, then proves seed-once/sentinel replay advances and same-seed request reset reproduces all 12 draws. It also checks the sentinel before and after replay and the expected boundary-copy counters. The exact command/result is recorded in `AUTOFIX_NON_GREEDY_SAMPLING.md`; the durable test is in `tests/test_full_model.py`.
- P0/P1: no finding. Greedy behavior was not weakened. `Gemma4GreedyTP4Sampler` is byte-for-byte unchanged from current `HEAD`; semantic greedy still routes to it, force-argmax remains separate, and only the non-greedy branch touches sampler seeds. Existing hardware evidence covers TP-shard boundaries, lower-token tie-breaking, batch two, repeated trace replay, and equality with host argmax.
- P0/P1: no finding. The broader Stage 06 contract remains supported: all 60 production `MultichipDecoder` layers preserve the TP4 policy and cache split; non-aligned/mixed prompts, fixed/inactive rows, external cache/page/position state, traced split sampling, device token/position feedback, changed-only page-table refresh, host-sampling compatibility, and reset/teardown are present in source and focused tests. The static contract suite passes 23/23 locally.
- P0/P1: no finding. Accuracy and quality evidence remains valid: the exact cached HF revision has `GemmaTokenizer.chat_template=None`, so completion rendering is the required `$qualitative-check` fallback rather than an invented chat wrapper. The fresh AIME24 reference contains 149 prompt tokens and 100 generated/top-100 positions; prefill and traced teacher forcing both record 91% top-1, 100% top-5, and 100% top-100. The directly inspected 100-token HF/TT autoregressive outputs are coherent, and the six-prompt mechanical check exits zero.
- P0/P1: no finding. Performance evidence is internally consistent: the compact profiler summary attributes 9.68% of reduced device-op time to the custom sampler and 56.25% to the LM head. Full-stack steady token-out is 33.87 t/s/u versus the 35.26 t/s/u decoder-stack ceiling. The corrected readback wording now matches `token_out_no_readback.json`: the single sampled-token read is the prefill-to-decode request-boundary seed token, not a final-token read; steady decode has zero token host refreshes and zero full-logit readbacks.

## Other Concerns

- P3, historical-document drift: the older `full_model/stage_review.md` still records the pre-remediation KV arithmetic and 19-test count. Evidence: its batch-3/batch-4 figures disagree with the corrected `full_model_plan`, README, work log, and the now-passing 23-test suite. Requirement impact: none for the live Stage 06 surface because this dated review is superseded by the remediation work log and this fresh report. Optional next step: treat the old file as historical or add an explicit superseded marker in a later documentation-only change.
- P3, later-stage drift (out of Stage 06 scope): `context_contract.json` entries for `optimized_full_model_plan` and `datatype_sweep_plan` still label the 2,789,212,160 logical-value count as KV bytes. Requirement impact: this does not invalidate the corrected `full_model_plan` or Stage 06's 262,144-token result, and the review contract explicitly excludes judging Stage 07+ work as Stage 06 regression. Concrete later-stage fix: recompute those downstream plan totals with 17/16 physical BFP8 storage when those stages are next reviewed.

## Hard-Check Gaps

- Non-blocking: the new focused RNG run is recorded as an exact command/result in the AutoFix report rather than a retained JUnit/log artifact. Source inspection, the durable A/B test, the documented TTNN `UINT32_MAX` contract, the device kernel branch, and the passing static integration checks are sufficient to verify this narrowly isolated repair; retaining machine-readable output would only strengthen provenance.
- Non-blocking: the reduced trace test proves shared token-buffer identity, zero host token reconstruction, on-device position advance, changed/unchanged page-table copy policy, and coherent free-running output. A future test could use distinct page-table contents and value-check sampled token N against decode N+1, as suggested by AutoDebug, but the current source wiring plus hardware replay evidence shows no concrete defect.

## Anomaly Ledger

- Observed anomaly: BFP8 KV capacity was undercounted as one byte per logical value.
  Evidence: old arithmetic versus 1,088-byte/1,024-value BFP8 tiles and the corrected `full_model_plan`.
  Affected path: Stage 06 context/capacity documentation.
  Control or comparison: HF cache geometry (50 sliding layers at 1,024; 10 full layers at 262,144) independently reproduces 2,789,212,160 values and 2,963,537,920 physical bytes.
  Likely subsystem: capacity accounting.
  Investigation performed: recomputed KV geometry, weights+KV, reserves, usable descriptor DRAM, and batch envelope.
  Resolution: fixed.
- Observed anomaly: a real seed captured by non-greedy `Sampling1D` reset the PRNG every replay.
  Evidence: AutoDebug call-chain/kernel analysis and the constant-logit fixed-seed versus sentinel A/B.
  Affected path: traced top-k/top-p sampling only.
  Control or comparison: seed-once plus `UINT32_MAX` advances the stream and exact seed 17 reproduces it after request reset.
  Likely subsystem: request/trace RNG-state ownership.
  Investigation performed: isolated Gemma-physical TP4 replay and source verification of fresh/reused trace paths.
  Resolution: fixed.
- Observed anomaly: one TT qualitative story repeats a corpus-style prompt sentence absent from HF.
  Evidence: prompt 2 in `qualitative/vllm_qualitative_outputs.json`.
  Affected path: one base-model completion.
  Control or comparison: the separate 100-token TT autoregressive completion is coherent and non-repetitive; trace feedback counters are clean; the mechanical checker has no findings.
  Likely subsystem: free-running sensitivity/base-model continuation style, not a systematic trace fault.
  Investigation performed: direct HF/TT output review and degeneracy-artifact comparison.
  Resolution: controlled.
- Observed anomaly: TT Metal warns when registering the second split-trace region.
  Evidence: `AUTOTRIAGE.md`, source-current watcher XML, and compact profiler evidence.
  Affected path: sampler trace capture setup, not steady replay.
  Control or comparison: all sampler tensors and outputs are preallocated; repeated replay, reset/recapture, changed tables, teardown, watcher, and profiling pass.
  Likely subsystem: conservative trace-region registration warning.
  Investigation performed: allocation-lifetime audit and post-fix hardware controls.
  Resolution: controlled.

## Scope Inspected

- Goal/skills: the supplied Stage 06 contract; `stage-review`, `full-model`, `tt-device-usage`, `tt-enable-tracing`, and `qualitative-check` skill contracts.
- Code/tests: `tt/model.py`, `tt/generator.py`, `tests/test_full_model.py`, `tests/test_full_model_contract.py`, common `Sampling1D`, TTNN manual-seed documentation/kernel, sampling kernel, TP4 decoder policy, and cache geometry.
- Artifacts: `context_contract.json`; full-model README/work log/old review; AutoDebug/AutoFix reports; reference metadata/refpt; accuracy logs; autoregressive and qualitative JSON/text; trace/performance JSON; sampler/profile summaries; compact CSV reports; watcher/JUnit and triage summaries. vLLM files and raw Tracy trees were not inspected.
- Read-only commands: source/diff/history searches; JSON/XML/hash/arithmetic checks; tokenizer/reference inspection; `git diff --check`; `py_compile`; static contract pytest (23 passed). No server, device open, reset, watcher, profiler, or hardware command was run.

## Residual Risk

- The RNG repair's focused hardware probe isolates the exact sampler/trace lifecycle rather than loading all 60 layers. That is proportionate to the proven defect; an all-layer sampled qualitative run is optional hardening, not a Stage 06 completion blocker.
- Later-stage context/accounting fields and vLLM capability limits require their own stage reviews and are not part of this verdict.

Final verdict: clean-pass
