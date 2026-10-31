# Stage Review

Verdict: more-work-needed

Independent initial review of stage 06, 2026-10-31. This is a live-worktree review on base `bdb236e49bdd887c9719e7a13b35af8bdc659964`; the owner is still completing gates. Findings below distinguish inspected repairs from runtime evidence that has not landed. This report is not stage closure.

## Required Work

- P1: Validate the repaired sampling parameters across every active batch slot.
  Evidence: At review start, `tt/generator.py:set_sampling` passed scalar temperature/top-k/top-p directly to `models/common/sampling/generator.py:708`. That helper interprets scalar temperature as one active lane. A host-only extraction of the actual helper returned `k=[32,1,1,...]`, `p=[0.9,0,0,...]`, and inverse temperature `[1.25,1,1,...]` for `(0.8,32,0.9)`. Consequently only slot 0 sampled. The owner has repaired broadcasting at `tt/generator.py:204`; the reviewer ran `tests.check_pool_admission` successfully against the repaired source. Existing `generator_state_b*.json` predates the repair and only checked greedy recovery after discarding sampled outputs.
  Why this matters: Whole-batch inequality can pass while every active slot except slot 0 remains greedy. Sampling is a full-model completion requirement.
  Required next step: Run the revised tracked reduced state gate for B1/8/16/32. Preserve per-slot sampled outputs, repeat-seed equality, changed-seed behavior, and greedy recovery on the same trace IDs. The revised code at `tests/check_generator_state.py:74` targets these checks. Unaffected greedy full-model quality evidence need not be regenerated solely for this parameter-broadcast repair.

- P2: Complete the focused post-chunk tail and logit determinism evidence.
  Evidence: Earlier mixed prompts cover lengths 31 through 541; the 4096-token benchmark is aligned. A 262143-token prompt rounds its final physical chunk to 1024. These cases do not cover a small physical tail after an earlier chunk through the newly prepared input path. The new 1023/1024/1025 checks at `tests/check_generator_state.py:88` are not yet reflected in JSON. `tests/measure_full_clean.py:31` adds direct full-vocabulary logit comparisons across 32 slots and repeats; `logit_determinism.json` was absent at inspection.
  Why this matters: The full-model skill explicitly requires logit reproducibility and targeted boundary coverage for changed chunking and padding. Stable token IDs alone can hide a logit difference.
  Required next step: Run these focused checks. Restore internally owned reservation after the external remap check, or provide sufficiently large external maps before longer tests: the earlier remapped maps reserve only the short prompts plus four tokens, so retaining `owns_cache=False` would make the later 16-token and 1025-token cases use repeated placeholder pages. Do not count repeat equality over that invalid mapping as boundary correctness.

- P2: Replace instrumented headline timing with separate clean timing.
  Evidence: `work_log.md:25` and `work_log.md:27` identify allocation-tracking-enabled all-layer invocations that also generated the current headline `performance_b*.json`. `tests/full_generation_evidence.py:34` times the same generator. The tracing skill requires performance collection with tracking and diagnostics disabled. The owner has prepared `tests/measure_full_clean.py`, but its final measurement artifacts are pending.
  Why this matters: The current values include validation overhead and do not satisfy the declared timing regime. Synthetic sampler timing already follows the separate-run requirement.
  Required next step: Retain instrumented rows as acceptance evidence; collect B1/8/16/32 4096/128 rows with diagnostics disabled and record the environment. Update both READMEs and `doc/agentic_serving.json` from those rows. Keep teacher-forced timing explicitly separate.

- P2: Investigate the factual divergence in shared qualitative prompt 3.
  Evidence: `qualitative_tt.json`, entry `shared_3`, says the total energy in a "closed system" remains constant; the matching HF control says "isolated system". `qualitative_verdict.md:9` acknowledges the error as a wording limitation without a discriminating control. The stage-review anomaly rule does not accept prose acknowledgement as resolution of visible wrongness.
  Why this matters: A closed system can exchange energy. The existing top-k gate and coherent prose do not explain this particular difference.
  Required next step: A focused pinned-HF continuation from the actual TT prefix immediately before this phrase can distinguish expected model behavior after earlier autoregressive branching from a local numerical divergence. Record the control and outcome; investigate a remaining TT-specific discrepancy rather than merely renaming it a limitation.

- P2: Finish the already-owned context, profiling, memory, and final documentation gates.
  Evidence: `doc/context_contract.json:full_model_plan` still says full-model validation is pending; `memory_plan.json:reserves.transient_peak` is null. The reduced profile and `tt-perf-report` output were absent at inspection. B8 full-stack capability has now landed. Final B1 accuracy has also landed and reproduces prefill 94/100/100% and decode 93/100/100% top-1/5/100; its host compatibility equality check is recorded. Maximum-context completion was not yet recorded in the B1 report. README and runtime audit still contain stale pending statements.
  Why this matters: These are explicit stage requirements and cannot be replaced by prior decoder-only maximum-context evidence or analytical memory figures.
  Required next step: Finish full-stack 262143/262144 context execution and last legal decode position; reduced real-layer profiling only; final persistent/transient/trace accounting; layer-stack lower-bound comparison; final fallback and sampling-contract documentation. Run the stage checker after reconciliation, obtain a clean rereview, then make the stage-owned local commit. Do not push or start vLLM for this stage.

## Other Concerns

- The repaired public sampling boundary rejects unsupported stochastic `top_k` values outside 1 through 32, instead of silently turning 64 or unrestricted sampling into 32 candidates. Host admission checks pass. Document this supported range and that top-p operates on the selected candidate set; do not claim parity with the pinned HF stochastic default `top_k=64`. The required default greedy path is unaffected.
- The common greedy candidate path may choose a different maximum-valued token when more than 32 equal maxima occupy a shard. `sampler_acceptance.json` records the discrepancy and confirms maximum-valued selection on all devices. This is a documented common-sampler limitation, not evidence of non-greedy output.
- `prefill_forward` samples every fixed slot, including zero-length slots. A future continuously batched adapter must explicitly preserve or restore live decode inputs when prefilling a subset of slots; current evidence establishes fixed-batch generation and explicit-input low-level decode, not overlapping serving requests.
- Worker watcher evidence records the inherited Ethernet instrumentation buffer limit. Ethernet traffic still executes. No new unclassified device-health failure was found in the inspected final acceptance logs.

## Hard-Check Gaps

- The stage checker can verify presence/schema and mechanical degeneration, but it does not detect scalar sampling parameters applying to only one batch slot. Preserve the focused regression check.
- Final source provenance and artifact-to-command mapping must be reconciled after the owner finishes edits. This is a live worktree; pre-repair sampled-mode results are stale for the repaired behavior.

## Anomaly Ledger

- Observed anomaly: scalar stochastic settings applied only to slot 0.
  Evidence: Actual shared formatter host reproduction; original `set_sampling` call; prior state tests discarded sampled outputs.
  Affected path: Batched stochastic generation.
  Control or comparison: Scalar settings formatted for 32 slots yield one stochastic lane and 31 greedy lanes.
  Likely subsystem: Wrapper parameter ownership, not the sampling kernel.
  Investigation performed: Read formatter and caller; executed its host-only normalization; inspected owner repair and ran host admission regression.
  Resolution: Code repaired; tracked per-slot device validation remains required.

- Observed anomaly: TT thermodynamics wording differs factually from HF.
  Evidence: `shared_3` raw completions and `qualitative_verdict.md:9`.
  Affected path: Greedy qualitative generation.
  Control or comparison: Same pinned prompt and original HF completion are available; continuation from the TT branch is pending.
  Likely subsystem: Not yet localized; may be expected model behavior after earlier branching.
  Investigation performed: Read both complete retained texts; identified the exact differing sentence.
  Resolution: more-work-needed.

- Observed anomaly: Initial text-only HF construction used an incomplete stop configuration.
  Evidence: `qualitative_prompt_format.json`, `qualitative_hf_unstopped.json`, `tests/normalize_qualitative_stops.py`, pinned `GenerationConfig` loads.
  Affected path: Qualitative reference stopping.
  Control or comparison: Retained deterministic prefixes terminate at the first pinned stop ID; original outputs remain available. AIME100 contains no affected stop.
  Investigation performed: Inspected reference loader, metadata, normalized outputs, and TT stops.
  Resolution: controlled.

- Observed anomaly: Late prefill/reset allocations initially conflicted with traces.
  Evidence: `AUTODEBUG.md`, `AUTOFIX.md`, prior `probe_trace3.log`, and subsequent tracked state/full-model runs.
  Affected path: Request preparation between live split traces.
  Control or comparison: Final code prepares all physical tail buckets and terminal modes before first capture, uses runtime offsets and persistent buffers, and introduces no corruptible allocation exemption.
  Investigation performed: Inspected the whole model/generator/prefill path, decoder diff, raw state tests and capture/replay ordering.
  Resolution: Existing tested paths controlled; new boundary/parameter checks above remain pending.

## Scope Inspected

- Goal/skill paths: `bringup/run_contract.json`, `bringup/agentic_evals/inputs.json`, installed stage06 goal, `stage-review`, `full-model`, `agentic-serving`, `tt-enable-tracing`, `qualitative-check`, model-bringup startup, plus review core/router with model-bringup and trace domains. Read the checkout's `TraceCorrectness.md`.
- Artifact paths: `doc/full_model` READMEs/work log, AIME reference metadata and actual HF/TT text, all six shared HF/TT text outputs, accuracy, generator state, sampler acceptance/timing, batch/performance/memory rows, watcher/fallback records, AutoDebug/AutoFix reports; `doc/context_contract.json`, `doc/agentic_serving.json`, inherited optimized multichip contract and timings.
- Code paths: Full `tt/model.py`, `tt/generator.py`, `tt/prefill.py`, `tt/weights.py`; changed optimized decoder prefill path and its callers; multichip construction/policy/workspaces; common sampler/formatter contracts; reference, readiness, batch, state, sampler, profiling, and clean-timing scripts. No removed Gemma implementation was accessed.
- Commands run: Read-only `rg`, `cat`, `sed`, `nl`, git status/diff; small JSON/text analyses; AST extraction of shared formatter; host-only `python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.check_pool_admission` (exit 0). No device opening, reset, server, or long test was performed by this reviewer. Only this report was written.

## Residual Risk

- B1 readiness and inspected autoregressive outputs support short-context full-model correctness. Long-context full-stack execution and final reduced profiling remain independent completion gates.
- All six shared texts and both AIME completions were read. No mechanical repetition, doubled tokens, language drift, or cross-request text leakage was observed. Long answers are capped and do not establish complete task correctness.
- This initial verdict must be superseded by rereview of the final artifacts and repaired-source validation before stage closure.
