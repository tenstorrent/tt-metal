# Stage Review

Verdict: clean-pass

Independent diff-only follow-up, round 2 of at most 3, for stage 10 optimized-vLLM, Qwen/Qwen3.8-Flash-Next. Reviewed live main at `ed9fd9c4f76adf08e7c69d7cb00edff079c6f5fd` plus staged evidence and the supplied follow-up changes. `stage_review_round1.md` is authoritative for the unchanged stage evidence, findings, and limitations. This review closes its single Required Work item; it does not repeat the full stage review. Paths below are relative to `models/autoports/qwen_qwen3_8_flash_next` unless stated otherwise.

## Required Work

None remaining.

- **Closed — round-1 P2: The shipped server launcher still silently selects budget 256 instead of the accepted budget 9.**
  Evidence: `doc/optimized_vllm/run_server.sh:24` now uses `${TOKEN_BUDGET:-9}`. A byte comparison against HEAD confirms that replacing the fallback 256 with 9 is the entire launcher change. Independently executing the actual launcher through Bash with only environment sourcing and final `exec` stubbed resolves an unset `TOKEN_BUDGET` to 9 and an explicit `TOKEN_BUDGET=17` to 17. Both complete argument lists exactly match `doc/optimized_vllm/launcher_default_check.json`; both preserve context 262144 and resident capacity 2 and return zero. `bash -n` passes.
  Why this closes the finding: direct launch now selects the accepted budget, while explicit overrides still work. Round 1 already established that accepted measurements and gates explicitly used 9; this configuration-only correction does not invalidate those runs.
  Historical evidence: the old launcher SHA256 remains `c1c995359db5317bddc9ce1d48756f8effa5892f2f3c0cd515218585622738af`, matching the unchanged recorded hashes in final acceptance and tracked sync/async execution manifests. Those hashes were not replaced with the corrected launcher's hash.
  Required next step: none for this finding; no hardware rerun or third review is needed.

## Other Concerns

- **Closed as a review condition; sampling limitation retained.** Round 1's reproducibility-only disposition remains scoped to the original four raw failures and the current 400 exact served/native selected-logprob comparisons. The appended work-log text preserves the raw 63 passed / 75 skipped / 4 failed result, the separate three-test recall pass, and the absence of matched original stochastic-prefix/HF controls. The refreshed serving-contract supplement links `doc/optimized_vllm/final_sampling_summary.json` rather than claiming an unconditional sampling-suite pass. The unchanged underlying checks were not re-adjudicated.
- **Closed as a review condition; eval limitations retained.** Round 1's critical-execution pass, three P4 advisories, 120.104851-hour wall-capacity forecast, and absent resident-B2 theoretical floor remain in force. The follow-up changes make no release-readiness or new accuracy claim and do not substitute the owner's B1-floor acceptance for B2 evidence.
- **Closed as a review condition; shutdown limitation retained.** The serving-contract supplement explicitly links `doc/optimized_vllm/owner_decision_shutdown_2026_11_17.json`. No plugin change, orderly-teardown claim, or causal claim about the disabled profiler helper was added. Round 1's required empty owned-PID/device-handle inventories and its accepted incomplete trace-release limitation remain the evidence boundary.
- **Closed as a review condition; performance hand-off retained.** The follow-up does not change measured values or claim a meaningful speedup. Round 1's approximately 0.0435% primary throughput difference remains noise, with untried optional work retained for hand-off under the owner cap.
- **Open — nonblocking post-review documentation closure.** As round 1 explicitly requested, the README title, final manifest, checklist, and serving-contract supplement still need their pending-review wording updated after this passing verdict, followed by local checkpoint receipts. This is ordinary stage-owner closure work, not a new finding or a reason for another hardware run or review. The current supplement correctly says `required_evidence_complete; independent_review_pending` at review time.
- No new issue was found in the changed metadata. `doc/full_model/serving_contract.json:1399` identifies the same measured source commit and plugin commit as the final acceptance manifest, preserves C262144/B2, and records budget 9. All three new evidence references exist. The current `tt/generator.py`, `tt/generator_vllm.py`, and `tt/slot_state.py` bytes match the recorded model-source commit `ff80569624d60dfdce236aee2cb432e849b3515e`.

## Hard-Check Gaps

- No new gap is introduced by the fallback correction. The CPU check deliberately stubs environment sourcing and process execution; it establishes shell argument resolution without starting a server. Accepted explicit-budget-9 device evidence remains covered by round 1.
- Round 1's disclosed advisory stage-check exit, raw sampling failures/skips, single measured request per benchmark arm, missing native-retry exit sidecar, and serving-specific no-profiler requirement remain unchanged. This verdict does not upgrade any of them to stronger evidence.
- `doc/optimized_vllm/review_fix_precommit.log` records passed applicable hooks and skipped inapplicable hooks. Independent working-tree and staged `git diff --check` checks for the follow-up files pass.

## Anomaly Ledger

- Observed anomaly: a direct launcher invocation previously selected budget 256 instead of selected budget 9.
  Evidence: HEAD-to-current one-line diff in `doc/optimized_vllm/run_server.sh`, `launcher_default_check.json`, and independent Bash expansion.
  Affected path: direct launch without `TOKEN_BUDGET`.
  Control or comparison: explicit override 17 remains 17; both runs preserve C262144/B2 and exactly match the saved argument arrays. Accepted hardware runs already used explicit 9.
  Likely subsystem: launcher configuration.
  Investigation performed: exact byte-diff check, syntax check, actual shell expansion with source/exec stubs, and verification that historical source hashes remain historical.
  Resolution: fixed.

- No new anomaly was found in the follow-up changes. The unchanged controlled/fixed anomalies and their scope limits remain recorded in the round-1 ledger.

## Scope Inspected

- Goal/skill paths: `/home/ttuser/qwen38-flash/multigoal_logs/10-10-optimized-vllm.prompt.txt`; installed `stage-review/SKILL.md`, `model-bringup/SKILL.md` startup, and relevant installed `vllm-integration/SKILL.md` sampling/profiling rules. Plugin inventory confirms enabled tt-model-bringup 0.1.24 and tt-autodebug 0.1.8; the packaged environment validator succeeds.
- Changed artifacts: `doc/optimized_vllm/launcher_default_check.json`; final appended section of `doc/optimized_vllm/work_log.md`; `doc/full_model/serving_contract.json` stage-10 supplement; `doc/optimized_vllm/review_fix_precommit.log`.
- Changed executable: `doc/optimized_vllm/run_server.sh`, fallback only.
- Reference artifacts: `doc/optimized_vllm/stage_review_round1.md` including its complete Required Work and Other Concerns; `chunk_selection.json`; final acceptance manifest; existence of referenced sampling and shutdown records; historical acceptance/tracked execution launcher hashes. Production model files were compared byte-for-byte only to validate the refreshed source-commit metadata.
- Commands run: read-only `cat`, bounded `sed`/`tail`/`rg`/`nl`, `git status/rev-parse/log/show/diff`; standard-library Python JSON/hash/byte comparisons and Bash subprocess argument capture; `bash -n`; `git diff --check` and staged equivalent; `codex plugin list --json`; packaged `scripts/environment.py` validation. The shell capture used `source() { :; }; exec() { printf "%s\\0" "$@"; }; builtin source "$1" /unused/evidence`, with the launcher passed as a positional argument and `TOKEN_BUDGET` unset or set to 17.
- No model runtime import, server, TTNN/device operation, reset, new hardware experiment, implementation edit, or sub-review occurred. Only this report was written.
- Push state: the supplied run plan permits local stage-owned checkpoint commits only; never push. Final closure commits and logged SHAs remain stage-owner work after this verdict.

## Residual Risk

- Round 1's sampling-admission, eval-accuracy/long-answer, topology-coverage, statistical, shutdown, and persistent-local-evidence limitations remain unchanged. The narrow launcher fix provides no additional evidence for those broader claims.
- Complete the already-planned closure wording and local checkpoint receipts, preserving raw failures, historical hashes, owner decisions, and the scoped acceptance language. No further hardware rerun or review round is requested.
