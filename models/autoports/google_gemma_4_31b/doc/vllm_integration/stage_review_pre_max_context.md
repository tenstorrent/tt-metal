# Stage Review

Stage 09, vLLM integration for `google/gemma-4-31B`, was rereviewed independently against the original Stage 09 goal and the complete `vllm-integration`, `stage-review`, `tt-device-usage`, and `qualitative-check` skill contracts. This was a source-and-artifact review of the live uncommitted main and nested-vLLM worktrees. No server, TT device, profiler, reset, or hardware test was started.

## Verdict

More work is needed. The P2 logit-determinism gap from the initial review is closed, and the broader sampling, qualitative, benchmark, cache-ownership, trace, and integration evidence is coherent. P1 is not closed: the new artifact proves that a `123392`-token KV pool can be allocated and that the resulting server can answer a 149-token request, but it does not prove that the server can execute a `123392`-token prefill.

## Required Work

- P1: Prove a genuinely runnable maximum vLLM context, including the complete peak live set for a maximum-length prefill.
  Evidence: `doc/vllm_integration/evidence/context_capacity_audit.json` correctly derives `10609` pool blocks for `123392`, leaves `332097216` contiguous bytes per bank after all twenty physical KV buffers, charges `331677696` bytes per bank for two BF16 full-context residuals, and therefore reports only `419520` bytes per bank of margin. `full_123392_capacity_passing_server.log` proves all twenty buffers allocate and the app reaches ready state. However, `full_123392_non_aligned_prompt_check.json` exercises only 149 prompt tokens, not the advertised maximum.
  Source contradiction: `tt/model.py:634-635` allocates the prompt-length residual and then calls `_prefill_rope`; `tt/model.py:476-485` materializes four tiled prompt-length RoPE tensors, and `tt/model.py:637-652` retains all four through the layer stack. For this model's sliding/full head dimensions of 256/512, those BF16 tiled tensors alone account for `123392 * (2*256 + 2*512) * 2 / 8 = 47382528` bytes per bank, over one hundred times the claimed post-two-residual margin. A stronger mandatory peak occurs in full attention: `_prefill_attention_tp` retains Q/K/V until its inherited chunk helper returns (`tt/multichip_decoder.py:1112-1176`), while that helper retains all BF16 SDPA chunk outputs and allocates their same-size concatenation (`tt/functional_decoder.py:368-402`). Together with the residual, normalized input, and four RoPE tensors, that is at least `6400` bytes per prompt token per bank. The current allocator model therefore rules out `123392`; even this incomplete lower bound admits at most the 64-aligned `101952` candidate (`8767` blocks, `652752576` post-KV bytes versus `652492800` mandatory bytes), while `102016` is already short by `1020224` bytes per bank. Chunk-local temporaries can require a still lower runtime value. Thus “two residuals fit” is a necessary lower bound, not a sufficient maximum-prefill capacity proof.
  Why this matters: the context contract promises that requests up to `max_model_len=123392` are serviceable. Server startup plus a short request establishes neither that promise nor that `123392` is the largest runnable value.
  Required next step: either execute and retain a successful near-maximum/max-length prefill through the OpenAI API at the advertised value, or derive a conservative peak-live allocation bound that includes RoPE and all other simultaneously live full/chunk tensors, select the largest value that satisfies it, and then hardware-validate that value with a near-maximum/max-length request. Retain the immediately larger candidate's physical failure/shortfall, update `doc/context_contract.json`, the capacity audit, README/work log, and rerun the final shared serving profile at the corrected context.

## Other Concerns

- The final `readiness_vllm/server.log` still emits the allocator warning that allocating buffers while an active trace exists may corrupt them. The anomaly ledger's inherited sampler-trace classification is plausible and the full sampling/qualitative/benchmark outputs show no visible corruption, but the final log does not contain the claimed `decode trace prepare` / `decode traces ready` diagnostics that would bracket this allocation. Preserve those diagnostics in the required context rerun.
- Shutdown reaches `Application shutdown complete`, but the log also contains nanobind leaked-instance/type/function warnings. The work log records no remaining process holders and healthy devices; retain a command log for that cleanup check in the rerun if practical.

## Hard-Check Gaps

- There is no retained maximum-length prefill artifact. This is the required P1 gap, not merely an optional stress test, because the stage explicitly lowers and advertises the serving context.
- The shared sampling suite and 32-request benchmark provide concurrency coverage, while static adapter contracts cover stale token/current-position/page-table state. A focused staggered device-greedy overlap artifact would still improve trace-state evidence, but the existing async runs make this residual rather than independently blocking.

## Anomaly Ledger

- Observed anomaly: vLLM context is reduced from the standalone/HF `262144` contract to `123392`.
  Evidence: `doc/context_contract.json`, `evidence/context_capacity_audit.json`, the `262144` and `157696` failed allocation logs, the `123392` startup log, and the 149-token non-aligned request artifact.
  Affected path: full-depth vLLM-owned hybrid KV-cache serving on four P150b devices.
  Control or comparison: `262144` and `157696` fail full-depth KV allocation; `123392` allocates the KV pool and serves a short request.
  Likely subsystem: per-bank DRAM capacity and the full-prefill peak live set after hybrid KV allocation.
  Investigation performed: exact artifact-arithmetic verification and source-level lifetime inspection of residual, RoPE, normalization, attention, and MLP tensors.
  Resolution: unresolved. The selected value is bounded only by two residual tensors and was not exercised with a maximum-length prompt.

- Observed anomaly: final server emits an active-trace allocation-corruption warning.
  Evidence: `readiness_vllm/server.log:2297` and `anomaly_ledger.md`.
  Affected path: cooperating model/sampler trace setup.
  Control or comparison: retained Stage 06/07 controls and final sampling, qualitative, primary benchmark, and CI burst all complete without visible corruption.
  Likely subsystem: sampler trace registration while the model trace exists.
  Investigation performed: source/log inspection and comparison with the documented inherited warning.
  Resolution: controlled with residual diagnostic risk; not a separate blocker.

- Observed anomaly: several greedy and sampled qualitative continuations use request-list framing or phrase loops.
  Evidence: direct inspection of all twelve outputs in `readiness_vllm/vllm_qualitative_outputs.json`, `qualitative_verdict.md`, and `degenerate_output_check.json`.
  Affected path: user-visible base-model raw continuation.
  Control or comparison: the tokenizer has no chat template, and Stage 08 HF controls reproduce the important supervised-learning, thermodynamics, and translation trajectories.
  Likely subsystem: base-checkpoint corpus behavior and raw continuation format.
  Investigation performed: direct prompt/output reading against the Stage 08 HF/TT controls.
  Resolution: controlled checkpoint behavior, accurately documented rather than presented as strong instruction following.

## Scope Inspected

- Contracts and documentation: original Stage 09 goal; all four required skill files; `doc/vllm_integration/{README.md,work_log.md,anomaly_ledger.md,runtime_fallback_audit.md}`; `doc/context_contract.json`; initial and current stage reviews.
- Evidence: final `readiness_vllm` server/sampling/qualitative/benchmark/logit artifacts; context-capacity arithmetic; failed and passing capacity logs; non-aligned request; Stage 08 qualitative and standalone-token controls.
- Code: `tt/generator_vllm.py`; relevant Stage 09 changes in `tt/generator.py`, `tt/model.py`, `tt/multichip_decoder.py`, and `models/common/readiness_check/run_vllm_server.py`; adapter contract tests; nested vLLM plugin source and sampling-test diffs.
- Worktree scope: stage-owned main/nested changes were distinguished from unrelated pre-existing/untracked workspace content. Commits correctly remain pending until a clean review.

## Residual Risk

- P2 is resolved: `readiness_vllm/logit_determinism.json` retains exact top-20 equality across two runs and target batch positions 0/1/2, with identical selected token/logprob and standalone selected TT token 108 agreement.
- The retained full shared sampling result is `72 passed, 1 skipped`; the final primary and CI benchmarks complete with the documented metrics; qualitative evidence is prompt-format-correct and honestly classified.
- Selected precision, thin adapter delegation, vLLM cache ownership, hybrid geometry, plugin selection, canonical traced device-greedy sampling, explicit optional host compatibility, async serving, non-aligned short-prompt handling, and shutdown are otherwise supported by source and artifacts. They do not compensate for the missing maximum-context execution proof.

more-work-needed
