# Bug Escapes — Findings Log

Running log of learnings, observations, and decisions from each work session.

---

## 2026-04-24 — Agent-based pruned YAML generation

*Root cause of brittle pruning:* `build_pruned_yaml()` in `verify_common.sh` used a hardcoded Python extractor that only preserved `name, cmd, skus, owner_id, team` — silently dropping any extra fields like `model-name`, `docker_image`, or custom runner config. As workflows evolve, this can cause pruned YAMLs to be missing required runner hints.

*Fix: agent-based generation with Python fallback:* Added `_build_pruned_yaml_agent()` that calls the LLM API directly (same curl approach as `check_failure_is_real`, branches on `LLM_BACKEND`). The agent receives the full entry JSON + test name + job name and produces a correct pruned YAML preserving ALL fields. If the LLM is unavailable, the response is malformed, or YAML validation fails, it falls back to the Python extractor. Python fallback also updated to preserve extra fields via `for k,v in entry.items(): if k not in pruned: pruned[k] = v`. Commit: `ba58e54f` in tt-auto-triage.

*Validation run 24901845331:* Full phase-4 detection run on `t3000-unit-tests.yaml` (copilot backend, mock-verify=true, auto-verify=true, lookback=7d). Completed in ~4 min. 47 candidates → 0 confirmed escapes → `verify-all` skipped (no qualifying escapes). The new agent-based pruning path was NOT exercised (expected — no escapes to verify). Code is deployed; path will activate on first real escape with a valid fix commit.

*workflow-layers.json gap found:* Scanning `models-post-commit.yaml` produced 0 workflows in Phase 1 — the workflow wasn't in the static config. Root cause: 12 workflows added to tt-metal since the config was last updated weren't registered. Added `models-post-commit`, `ttnn-post-commit`, `cpp-post-commit`, `ops-post-commit`, `tt-train-post-commit`, `blackhole-e2e-tests`, `models-t1/t2/t3-e2e-tests`, `runtime-perf-tests`, `galaxy-e2e-tests`, `galaxy-perf-tests`. Commit: `6e8b8520` in tt-auto-triage. Validated: run 24902236618 confirms `models-post-commit.yaml` now found (Phase 1: 1 workflow, 1 'other').

---

## 2026-04-24 — Initial sessions

*Smoke test fix:* `run.sh` Phase 0 unconditionally checked `agent --version` even when `LLM_BACKEND=copilot`. Fixed to branch on `LLM_BACKEND` and check `copilot --version` instead. Commit: `f7b5e5a7` in tt-auto-triage.

*Phase 2 timeout:* Copilot backend times out at 300s when sent 80-100+ candidates in a single LLM call. Cursor handles large batches faster. Batches of ~7 candidates work fine. Fix needed: chunk Phase 2 into ~20 candidates per call.

*Deduplication confirmed working:* Run 24855762867 correctly grouped 3 "board reset" vLLM jobs into 1 group, suppressing 2 duplicates.

*Log tail reading:* Errors in long logs appear near the end, not the beginning. Fixed `detect_failures.py` to read the last 100k chars instead of the first. Confirmed this fixed T3K Llama signature extraction.

*Verification pruning confirmed:* `build_pruned_yaml()` correctly builds a single-entry test YAML and pushes it to before/after branches. Only the failing job runs during verification.

*triage-ci.yaml (tt-metal):* Copilot credentials wired up — `llm-backend` input and `COPILOT_PAT: ${{ secrets.AUTO_TRIAGE_TOKEN }}` added to both detect and verify jobs. Committed directly (was pending from prior session).

*End-to-end copilot run:* Has not completed successfully yet. Blocked on Phase 2 batch size issue.

---

## 2026-04-24 — Phase 2 pre-extraction fix + Phase 1 filter fix

*Phase 1 basename filter:* `TEST_WORKFLOWS=blackhole-post-commit.yaml` was matching against full paths like `.github/workflows/blackhole-post-commit.yaml`. Fixed Phase 1 filter to use `endswith("/"+wf)` so bare filenames work. Commit: `4b05fb95` in tt-auto-triage.

*Phase 2 pre-extraction:* Root cause of copilot 300s timeouts: LLM was doing 60+ file I/O operations (list dir, grep, read) per 20-candidate chunk. Fixed by pre-extracting error lines in bash before the LLM call. Each candidate now includes grep output directly in the prompt instead of a log directory path. Commits: `70513c45` (phase2), `1c5ac90f` (prompt) in tt-auto-triage.

*Validation run 24898787242:* 88 candidates, 5 chunks of 20, all completed without timeout. Chunk times: 32s, 23s, 24s, 20s, 17s. Total Phase 2: ~2 min. 0 confirmed escapes (blackhole-post-commit may be healthy, or all 88 were infra noise — need broader scan).

*Backend log labels:* `cursor_agent_query:` messages now say `copilot_agent:` / `cursor_agent:` based on actual backend. Commit: `a9d10cdd` in tt-auto-triage.

---

## 2026-04-24 — First full phase-4 copilot run + false-negative root cause

*run.sh API key check:* `run.sh` was unconditionally requiring `CURSOR_API_KEY` even on copilot backend. Fixed to guard with `[ "${LLM_BACKEND:-cursor}" != "copilot" ]`. Commit: `352ce0f6` in tt-auto-triage.

*End-to-end copilot run succeeded:* Run 24899258967 completed full Phase 0→4 on `t3000-unit-tests.yaml` with copilot backend in ~4 minutes. Phase 2: 47 candidates, 17 likely_flaky, 3 chunks, 1 CONFIRMED.

*Confirmed ongoing failure:* `AccessorTests/AccessorBenchmarks.PagesIteratorInterleaved/3` (job: `download-artifacts`, `t3000-unit-tests.yaml`) confirmed real test failure with high confidence. Error: `TT_THROW @ system_memory_manager.cpp:717`. Phase 3 found no fix point — still active on main.

*Phase 4 correctly produced 0 bug escapes:* Ongoing failures with no fix commit are not bug escapes. Written to `ongoing-failures.json` instead. `bug-escapes-output.json` correctly shows empty array.

*Phase 2 false-negative root cause found:* Phase 2 downloaded only the FIRST failing run's logs for pre-extraction, then broke out of the loop. When that run's log tail was dominated by post-test noise (AI summary tool errors, Docker cleanup output), the grep found nothing and the job was classified as infra noise — even though later runs had real FAILED lines. Affected: `t3k_ttnn_tests [wh_llmbox]` and multiple other jobs. The same AccessorBenchmarks failures that were caught in `download-artifacts` were also present in `t3k_ttnn_tests [wh_llmbox]` but missed.

*Phase 2 multi-run extraction fix:* Changed download loop to try all 3 failing runs' logs for extraction, breaking only when error lines are found. If all runs' tails are empty, falls back to "no errors." Commit: `6b2ff7a7` in tt-auto-triage. Validation run 24899825890 in progress (max-phase=2, t3000-unit-tests.yaml).

*Artifact naming:* Both `bug-escapes-output` (26MB, full detection output) and `bug-escapes-final-report` (1KB, final report) are uploaded with different names — no conflict. Earlier diagnosis of artifact overwrite was incorrect.

---

---
## 2026-04-24 18:08 UTC — blackhole-post-commit.yaml

- **Run**: https://github.com/tenstorrent/tt-metal/actions/runs/24904276503
- **Branch**: ebanerjee/bug-escapes
- **Workflow scanned**: `.github/workflows/blackhole-post-commit.yaml`
- **mock-verify**: true (business hours)
- **Lookback**: 7 days, 3 consecutive runs threshold
- **Candidates**: 88 (2 marked likely_flaky)
- **Confirmed consistent failures**: 0
- **Bug escapes**: 0 (horizontal=0, vertical=0, cross_layer=0, unknown=0)
- **Verdict**: Clean — no bug escapes detected in blackhole post-commit over last 7 days.
