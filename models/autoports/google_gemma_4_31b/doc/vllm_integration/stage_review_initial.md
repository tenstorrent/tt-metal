# Stage Review

Stage 09, vLLM integration for `google/gemma-4-31B`, was reviewed independently against `.agents/skills/vllm-integration/SKILL.md`, `.agents/skills/stage-review/SKILL.md`, `.agents/skills/tt-device-usage/SKILL.md`, and `.agents/skills/qualitative-check/SKILL.md`. This was a read-only review of the live uncommitted main and nested-vLLM worktrees; no server or TT device was opened.

## Required Work

- P1: Prove and serve the largest physically feasible vLLM context.
  Evidence: `doc/context_contract.json` advertises 119552, but also records an allocator upper-bound context of 148224 and selects 119552 by imposing an extra 64 MiB per-bank runtime-margin rule. `evidence/context_capacity_audit.json` shows only 262144 and 157696 as failed hardware attempts. It shows that the immediately larger 119616 candidate is rejected solely because its calculated margin is 137536 bytes below the chosen 64 MiB reserve; there is no artifact showing that 64 MiB is a required allocation or that 119616 fails. The failed logs do establish real OOMs at 262144 (`full_262144_capacity_failed_server.log:754`) and 157696 (`full_157696_capacity_failed_server.log:761`), while the final server log establishes successful allocation at 10279 blocks for 119552 (`readiness_vllm/server.log:738-748`).
  Why this matters: the stage lowers the HF-advertised 262144-token serving capability. The vLLM and stage-review contracts allow that only when hard physical evidence proves the largest feasible value. “Largest value satisfying a conservative safety rule” is not the same claim, especially when the contract's own upper bound is materially higher.
  Required next step: derive the maximum from actual required live allocations, test the resulting largest 64-token-aligned candidate through full-depth server allocation and a request, demonstrate that the next aligned candidate is physically impossible (by exact allocator math tied to required buffers or a bounded failed run), update `context_contract.json` and the capacity audit, then rerun the final shared serving profile at that value.

- P2: Leave the required model-specific vLLM logit-determinism evidence.
  Evidence: `readiness_vllm/sampling_tests.log` is a valid full shared sampling result (`72 passed, 1 skipped`) and includes output-level seeded/greedy reproducibility tests. However, the evidence tree contains no vLLM logit artifact or focused test comparing identical-prompt logits across runs and batch positions with a standalone baseline. The model's `greedy_only` device policy explicitly routes stochastic/mixed sampling to host compatibility, so those shared output tests do not substitute for the required device-model logit comparison. The adapter contract suite tests stale host token/current-position routing with a fake generator, not logit reproducibility.
  Why this matters: the vLLM skill explicitly requires run-to-run and cross-batch-position logit determinism through vLLM plus a standalone comparison, particularly because cache positions, hybrid page tables, dynamic batching, and traced token feedback are stage-critical here.
  Required next step: add and retain a focused model-specific artifact/test that compares logits for the same prompt across vLLM runs and batch positions and against the standalone generator baseline; classify any difference before closure.

## Other Concerns

- The final server log emits `Allocating device buffers is unsafe due to the existence of an active trace` at line 2291. The anomaly ledger classifies this as the inherited cooperating sampler-trace registration and cites prior replay/reset/changed-page-table controls. That classification is plausible and no corruption is visible in the final outputs, but the final log does not contain the claimed `decode trace prepare` / `decode traces ready` brackets, so the precise placement is not independently re-derived from this run. Preserve or repair those diagnostics in the context rerun so the warning can be tied directly to second-trace setup.
- The 16 adapter contracts and 26 plugin CPU contracts were independently rerun without opening hardware and pass. They establish static cache ownership, canonical split-sampling delegation, synchronized trace teardown, basic stale-token/current-position routing, hybrid geometry, and registration. They do not close either required item above.

## Hard-Check Gaps

- No retained command log was provided for the post-run process/device-health audit; `work_log.md` records the result. This is non-blocking because cleanup is corroborated by the final server's clean application shutdown and is not the source of either finding.
- The full sampling suite exercises 32-request concurrency and the final CI benchmark completes 32 requests, but there is no purpose-built staggered device-greedy overlap artifact that changes active batch and page-table state. The full async-scheduled qualitative and benchmark runs plus the static stale-input contract provide meaningful coverage; retain a focused overlap test when addressing P2 rather than treating this as a separate blocker.

## Anomaly Ledger

- Observed anomaly: advertised vLLM context is reduced from 262144 to 119552.
  Evidence: `doc/context_contract.json`, `evidence/context_capacity_audit.json`, both failed capacity logs, and the passing final server log.
  Affected path: full-depth vLLM-owned hybrid KV-cache serving on four P150b devices.
  Control or comparison: standalone context remains 262144; 262144 and 157696 vLLM pools OOM; 119552 serves successfully.
  Likely subsystem: per-bank DRAM capacity and fragmentation after hybrid KV allocation.
  Investigation performed: artifact arithmetic and failed/passing log comparison.
  Resolution: unresolved because 119552 is selected by an unproven 64 MiB reserve rather than shown to be the largest feasible value.

- Observed anomaly: final server emits an active-trace allocation corruption warning.
  Evidence: `readiness_vllm/server.log:2291` and `anomaly_ledger.md`.
  Affected path: first canonical split model/sampler trace setup.
  Control or comparison: prior Stage 06/07 controls, final full sampling, qualitative outputs, primary benchmark, and CI burst all complete without visible corruption.
  Likely subsystem: sampler trace registration while the cooperating model trace exists.
  Investigation performed: source inspection, log placement review, and comparison with the recorded inherited anomaly.
  Resolution: controlled with residual diagnostic risk; not an independent blocker in this review.

- Observed anomaly: greedy and sampled qualitative outputs contain request-list continuation and phrase loops.
  Evidence: all twelve entries in `readiness_vllm/vllm_qualitative_outputs.json`, `qualitative_verdict.md`, and `degenerate_output_check.json`.
  Affected path: user-visible base-model raw continuation.
  Control or comparison: the exact tokenizer has no chat template; Stage 08 HF controls reproduce the supervised-learning and thermodynamics loops and the translation-exercise trajectory.
  Likely subsystem: base-checkpoint corpus behavior and raw continuation prompt format.
  Investigation performed: direct output inspection against prompt-correct HF/TT controls.
  Resolution: controlled checkpoint behavior; not a serving regression.

## Scope Inspected

- Goal/skill paths: the complete Stage 09 contract in the review request and the four skill files named above.
- Artifact paths: `doc/vllm_integration/{README.md,work_log.md,anomaly_ledger.md,runtime_fallback_audit.md}`, `doc/context_contract.json`, all final `readiness_vllm` JSON/log/text artifacts, context-capacity audit and failed logs, and Stage 08 qualitative controls.
- Code paths: `tt/generator_vllm.py`, Stage 09 diffs in `tt/generator.py`, `tt/multichip_decoder.py`, `models/common/readiness_check/run_vllm_server.py`, `tests/test_vllm_adapter_contract.py`, and all nested `vllm/plugins/vllm-tt-plugin` source/test diffs.
- Commands run: read-only `rg`, `sed`, `nl`, `find`, `jq`, Git status/diff inspection, plus CPU-only adapter/plugin pytest checks (16 and 26 passing). No vLLM server, TT device, profiler, reset, or hardware test was started.

## Residual Risk

- The selected precision, vLLM cache ownership, hybrid cache registration, plugin selection, canonical traced greedy sampling, explicit optional host compatibility, non-aligned 149-token request, full shared sampling profile, prompt-correct qualitative controls, primary/CI benchmark metrics, async server configuration, and cleanup are otherwise supported by inspectable source and artifacts.
- Stage-owned local commits correctly remain pending until a future clean review.

more-work-needed
