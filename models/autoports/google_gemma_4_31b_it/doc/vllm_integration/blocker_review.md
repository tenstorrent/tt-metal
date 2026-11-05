# Stage Review

Verdict: more-work-needed

This is an independent early blocker review of stage 09 (vLLM), not a stage-closure review. PC-01 merits operator authorization for the exact proposed Boolean capability `supports_masked_prefill_sampling` (default false) and `prefill_forward(..., prefill_sample_mask=...)` contract. This recommendation is not authorization and does not establish that the eventual implementation works.

## Required Work

- P1: Required mixed-chunk device sampling cannot be expressed by the pinned plugin interface.
  Evidence: At plugin commit `ca7a18246e4111629ed6cb4a7d1192002749fccf`, `src/vllm_tt_plugin/model_runner.py:2665` unconditionally clears `perform_device_sampling` if any prefill row is intermediate. `submit_prefill` at line 4084 passes sampling parameters only when that flag is true and passes no emission mask. Output extraction at lines 4311–4332 bypasses sampling only when every row in the rank is intermediate; mixed rows require host logits. `docs/MODEL_EXECUTION.md:57` explicitly documents this behavior. I reran the supplied host-only probe successfully and obtained the recorded results: mixed mask `[true, false]`, device sampling false, sampling parameters absent. The probe forces initial eligibility true, so it isolates the later unconditional override rather than establishing real request eligibility or device execution.
  Why this matters: The task requires canonical device split sampling without full-logits readback or host feedback on actual mixed intermediate/final chunk traffic. This is missing shared behavior, not a documented-plugin-contract violation. Even a correct model implementation cannot select the device output contract or recover withheld stochastic parameters. `prompt_lens` is the scheduled exclusive end, not total request length; `empty_slots` identifies state destinations. Neither conveys emission eligibility. Device-sampling, async, prefix, selective-readback, grammar, runtime registration, and `sample_on_device_mode=all` mechanisms do not bypass the unconditional rule. Disabling scheduler chunking or returning fabricated logits would evade the acceptance contract.
  Required next step: human authorization for PC-01 in `plugin_changes.md`, with exactly its named capability and mask semantics. Preserve all other host-only eligibility restrictions; opt-in must not force unsupported requests onto the device sampler. Implement and validate only after authorization.

- P1: Ordinary stage implementation and acceptance work is unfinished, independently of the authorization decision.
  Evidence: The README explicitly reports no adapter implementation, server launch, sampling/qualitative gates, or serving measurements. It also reports the generator's stochastic top-k ceiling of 32 against the pinned default top-k 64.
  Why this matters: Approving PC-01 cannot prove cache ownership, continuation, sampling-state isolation, async token feedback, tracing, or output correctness. It does not authorize a top-k default override or waive any model-owned gap.
  Required next step: implement the adapter and model-owned continuation/state work; validate the default sampler recipe or obtain a separate operator decision for any proposed served-default override. Complete registration/import provenance, prefix and batched-prefill checks, actual mixed offsets/concurrent decode/partial tails, tracked repeated trace replay, full sampling and qualitative checks, and serving benchmarks. This is ordinary stage work, not another plugin-authorization request.

## Other Concerns

- PC-01 is appropriately generic and defaults legacy adapters to their existing behavior. Its planned regression matrix covers legacy/opt-in routing, row order, suppression, and seeded state. Include assertion that unsupported controls still use the existing compatibility route, and distinguish the proposed emission mask (`true` means emitting) from the current intermediate mask (`true` means non-emitting).
- `work_log.md` was added during this live review and inspected before finalization. It explicitly preserves missing model-owned contract rows and records the default-sampler issue separately from PC-01.

## Hard-Check Gaps

- The probe is host control-flow evidence only. It has no concurrent decode, nonzero chunk starts, partial final chunk, output extraction execution, real RNG state, or device trace coverage. These are required after implementation; their absence does not invalidate proof of the unconditional routing rule.
- No PC-01 implementation or regression results exist yet, as correctly declared. The plugin worktree was clean at the pinned base.

## Anomaly Ledger

- Observed anomaly: Otherwise device-eligible mixed intermediate/final prefill is forced to host sampling.
  Evidence: Source locations above and rerun of `probe_chunk_sampling.py`, matching `readiness_vllm/chunk_sampling_probe.log`.
  Affected path: Scheduler chunked prefill with emitting and non-emitting rows in one submitted batch.
  Control or comparison: Probe forces initial eligibility true; both all-intermediate and mixed cases end false. Source establishes why mixed extraction subsequently requires logits, while all-intermediate extraction can suppress sampling.
  Likely subsystem: Shared plugin routing and prefill interface.
  Investigation performed: Read documented capabilities/execution contract and relevant routing, submission, and output extraction code; reran host-only reproducer.
  Resolution: more-work-needed; proposed PC-01 needs human authorization.

## Scope Inspected

- Goal/skill paths: `bringup/run_contract.json`, `bringup/agentic_evals/inputs.json`; installed model-bringup startup, stage-review and vllm-integration skills, and `vllm-integration/references/plugin-changes.md` at version 0.1.23.
- Artifact paths: this directory's `plugin_changes.md`, `README.md`, `work_log.md`, `probe_chunk_sampling.py`; `readiness_vllm/chunk_sampling_probe.log`.
- Code paths: pinned standalone plugin `src/vllm_tt_plugin/model_runner.py`; `docs/MODEL_EXECUTION.md`, `docs/MODEL_CAPABILITIES.md`, and README configuration/registration references. No removed Gemma source or Gemma-specific framework branches were inspected or copied.
- Commands run: read-only `cat`, `sed`, `rg`, `git status --short`, `git rev-parse HEAD`; host-only probe with the plugin host stubs and `PYTHONDONTWRITEBYTECODE=1`. No hardware, server, or device reset. Only this report was written.
- Reviewed live tt-metal base: `9df04b7df6909d3c5e5e8e9b286b539855df7af5`. Plugin base: `ca7a18246e4111629ed6cb4a7d1192002749fccf`.

## Residual Risk

- Authorization would permit an implementation attempt, not certify safe masked sampler RNG/penalty/token state or trace behavior. Those require model and shared-plugin tests plus served evidence.
- This review does not assert that every remaining stage finding requires human authorization: substantial ordinary stage work remains. It therefore supplies no clean-pass or claim of stage completion.
