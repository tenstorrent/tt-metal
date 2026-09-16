# Stage Review

Verdict: clean-pass

Scope: independent read-only checkpoint review within Stage 11, Qwen/Qwen3.8-27B QB2. This verdict accepts only the adapter full-table recovery repair. Its host semantics and the serial post-rebind device path are supported by the evidence below. The larger reduced serial-versus-grouped B2 control still needs work and failed its unchanged gate. This is not a B2-control, full-model, serving, APC, performance, or Stage 11 pass. Reviewed live dirty checkout HEAD `76663d3f41bb15d7587ac709d6616af4a4234d28`; no implementation, historical receipt, or gate was modified by this reviewer.

## Required Work

- None for the adapter repair checkpoint. Full/narrow/permuted host replacement, preservation of the known-shadow method, unsafe-partial refusal, and actual serial post-rebind recovery establish its bounded contract.

## Other Concerns

- P1 for subsequent B2 acceptance, not a repair blocker: resolve the grouped first-decode token mismatch before accepting the reduced B2 control.
  Evidence: `../batched-prefill-candidate/probes_revision4/actual.json` has `status="failed"`, `numerics_pass=false`; the sole started group is greedy slots `[1,7]`. Its serial arm completes both prefills and all four decodes, including `decode_3` after device-only same-cache rebind. Its grouped arm completes both prefills, then fails at `device_control.py:271` before any `decode_0` numerical comparisons. Recorded token differences are:

  | Physical slot | Serial token | Grouped token |
  | --- | ---: | ---: |
  | 0 | 98744 | 1076 |
  | 4 | 12442 | 423 |
  | 5 | 0 | 9 |
  | 6 | 265 | 92049 |

  Active slots 1 and 7 both produce token 220 in both arms. Seeds, positions, RoPE, page-table hashes, history, cursor, and history count are equal. All recorded `decode_0` logits, selected recurrence, and selected KV summary hashes match across all four shards; inactive KV hashes also match. These hash matches do not turn the aborted comparison into completed numerical gates. Both prefills completed 32 comparisons each, all exact with PCC 1.0. Only 64 of 768 required numerical gates completed; reversed order and sampled groups did not execute. Actual child PID 1119741 exits 1. Independently running the artifact verifier exits 1 with `Incomplete numerical control`.
  Why this matters: the unchanged contract explicitly requires exact token/seed metadata. An inactive-row discrepancy is not resolved by labeling those rows inactive, and the abort prevents the remaining controls from running. Current evidence does not establish whether the cause is grouped execution, inactive-row sampling, or a control/initialization issue.
  Repair-specific assessment: grouped `prepare()` restores the trusted shadow through `gen._refresh_table(full_host_table)`. The failure occurs at grouped `decode_0`, before any grouped rebind; `_table()` therefore runs its byte-unchanged known-shadow branch. The inserted missing-shadow recovery branch is exercised by the successful serial `decode_3`. This new B2 failure does not invalidate the adapter repair. The grouped helper is loaded from the isolated candidate model; preflight proves the live model remains the baseline with that helper absent. This review supplies no separate APC or serving acceptance.
  Required next step: retain this failed receipt; isolate the inactive-token discrepancy with a serial control and inspect the inactive sampling/output initialization path. Fix the responsible implementation or demonstrate the exact baseline behavior with a control. Rerun the affected reduced control on fresh paths with the existing gates; do not mask the differing rows or claim a completed B2 pass from the matching active rows.

- No defect found in the narrow adapter repair. `generator_vllm.py::_table()` checks for the deliberately invalidated host shadow, derives the existing bound shape, rejects invalid dimensions/width or any row mapping that is not a complete integer permutation, and allocates/fills a fresh zeroed host table only after validation. It does not read device data or replace the bound device table. A partial missing-shadow update cannot invent preserved rows. Removing the inserted branch restores the baseline adapter byte-for-byte and by AST, preserving known-shadow behavior.
- The source transition is explicit: adapter SHA `3fe0a120a030d7dca258002b7bef24aa07d0ca494d15ea4f7e9e66b3efb6d3cd` becomes `7b01605e8791912f92c91baa0858fc8dbffbe5697e28a345a2e56c8ca19c48f1`; the other 68 inventoried sources are unchanged. Applied-source preflight passed during the independent artifact-verifier invocation. The production adapter matches the candidate, and no old observation was relabeled.
- Actual serial recovery is exercised, not merely inferred from unit tests. `run_arm()` binds the same cache and exact device page table, which sets `page_host=None` in `QwenGenerator.bind_cache()`, releases all five recorded handles, and checks unchanged cache identities and exact recurrent/KV data. Subsequent adapter `decode_3` must construct the complete host table through the new branch. That decode completes with normal metadata checks, device table/cache/history identity guards, canaries, and trace recapture. Serial final trace allocation is 1,572,864 bytes/device within the 134,217,728-byte region. The grouped failure occurs at `decode_0`, before its rebind, and does not reproduce the original missing-shadow crash.
- All prior comparison and ownership gates remain intact. Independent AST comparison finds only `preflight` and `main` changed in the probe. `device_control.py`, `native_binding.py`, and `run_actual.py` are byte-identical to revision3. Removing only the new source-transition helper and requirement restores the entire previous verifier AST. PCC remains 0.995; expected groups, stages, exact metadata/canaries, 768 comparisons, 24 captures/48 guards, actual exit, native bindings, and cleanup requirements are unchanged.
- Native bindings before and after execution agree on `_ttnn.so` SHA `8f5b58e74dd5cdbaeb2355ea216b9c9db6316cac40e47e9cc4a9ad97d8f2258c` and `_ttnncpp.so` SHA `d9d8d553dd3889f0c3e1f31b6b384a2bacd485974b11ef1dc5b36519917839eb`. The source guard checks the default-false native option and absence of model opt-in. The wrapper, program, commands, log, result, and ownership-to-execution receipt hashes independently match their current files.
- Ownership and cleanup are supported even though the run failed correctness. The observer records 428 samples, no ownership errors, and maximum sample gap 0.10209581255912781 seconds. All four devices show only the child PID while owned; before/after owner sets are empty. `cleanup_errors=[]`, `device_close_returned=true`, and the raw log shows device closure. The observer's status is correctly `fail` because its wrapped child exited 1; it must not be relabeled an overall ownership-wrapper pass.
- `AUTOFIX.md`, the probe README, and `root_applied.json` describe a pending retry. These are preparation-time claims, not the final outcome. A later summary should link this failed run and distinguish successful serial adapter recovery from incomplete B2 validation without rewriting historical receipts.

## Hard-Check Gaps

- Host tests use extracted real methods with stdlib matrix substitutes, not real Torch operations. This is disclosed by their source. The current actual serial run covers the natural full-table replacement with real Torch/TTNN; explicit full-row permutations and unsafe partial refusal remain host-only coverage. No additional device run is required solely to strengthen the narrow adapter evidence.
- The grouped arm did not reach rebind, so this receipt proves serial post-rebind recovery only. The full reduced comparison remains subject to the separately scoped B2 follow-up above.
- Independent reviewer reran the four adapter host tests: all passed. The 21 harness host tests were inspected through their source, log, and exit receipt and were not rerun by the reviewer. The source/gate equivalence claims were independently recomputed.

## Anomaly Ledger

- Observed anomaly: revision3 raises `AttributeError` on `None.clone()` after a valid device-table rebind.
  Evidence: diagnosis `AUTODEBUG.md`, preserved `ACTUAL_BOUNDARY.json`, generator `bind_cache()`/`_refresh_table()` source, baseline/candidate method tests.
  Affected path: adapter host-table construction after lower-level device-only binding.
  Control or comparison: unchanged known-shadow method; full/narrow/permuted replacement and malformed/partial refusal host cases; revision4 serial `decode_3`.
  Likely subsystem: adapter assumption that a trusted CPU shadow always exists.
  Investigation performed: inspected exact branch, all-source bridge and actual serial progression; reran four host tests.
  Resolution: fixed for full-row host replacement; unsafe partial recovery explicitly refuses.

- Observed anomaly: four inactive token rows differ after grouped `decode_0`.
  Evidence: exact table in Other Concerns; raw summary hashes and failure traceback.
  Affected path: reduced B2 equivalence through native decode sampling.
  Control or comparison: serial arm from the same run; no serial-versus-serial inactive-output control is present in this receipt.
  Likely subsystem: not yet established; inactive sampling/output or control initialization requires investigation.
  Investigation performed: compared every saved metadata field, selected tensor hash, inactive KV hash and prior prefill numerical row; verified failure remains fatal.
  Resolution: more-work-needed for B2 acceptance; not a blocker for this repair checkpoint because the failed call uses the unchanged known-shadow path.

- Observed anomaly: raw log warns about unknown motherboard tray-ID fallback and L1 semaphore allocation/fragmentation.
  Evidence: `actual.log`; discovery reports local chips 0–3, TP4 fabric initialization succeeds, model loads and both prefill comparisons complete, then the exact-token assertion terminates execution.
  Affected path: topology labeling and allocator headroom.
  Control or comparison: matching requested mesh and native bindings, clean ownership and closure, no allocation or device-health failure recorded.
  Likely subsystem: runtime discovery/allocator diagnostics.
  Investigation performed: inspected the complete short raw log and failure/cleanup records.
  Resolution: controlled for this bounded failed correctness run; this does not establish full-model memory headroom or performance.

## Scope Inspected

- Goal/skill paths: parent-provided repair/B2 checkpoint contract; installed `tt-model-bringup:stage-review/SKILL.md` and `model-bringup/SKILL.md` startup. Both required plugins are enabled in the host config; installed dependency/environment script validates successfully.
- Artifact paths: `../batched-prefill-rebind-diagnosis/{AUTODEBUG.md,ACTUAL_BOUNDARY.json}`; this candidate's `AUTOFIX.md`, patch, source bridge/inventories, baseline/candidate adapter, tests, host logs, and root application receipt; `../batched-prefill-candidate/probes_revision4/` README, source/native bridges, command pins, host test/receipt files, actual result/log/execution/ownership records; revision3 code comparison; original owner observer.
- Code paths: live `models/autoports/qwen_qwen3_8_27b/tt/generator_vllm.py::_table/decode_forward`, generator `_refresh_table/bind_cache`; revision4 `check_equal_span.py`, `device_control.py`, `native_binding.py`, `run_actual.py`, `verify_execution.py`; source bridge validator and host tests. Existing unrelated dirty implementation changes were not reviewed or modified.
- Commands run: read-only `cat`, `sed`, `rg`, `git status`/scoped `git diff`; short stdlib AST, JSON, hash, and metadata-analysis scripts; `/usr/bin/python3 -B .../batched-prefill-rebind-candidate/test_host.py -v` (4 passed); `/usr/bin/python3 -B .../probes_revision4/verify_execution.py` (expected exit 1, incomplete control); installed `scripts/environment.py` (dependency validated). No devices, server, hardware tests, or compilation were started by the reviewer. This narrow source repair is Python-only.

## Residual Risk

- The reduced model uses layers 0 and 3, one currently completed slot order and greedy mode. It does not prove 64-layer, serving, APC, prompt-quality, capability-wide, or performance behavior. Context remained 262144; testing short spans does not demonstrate maximum-context execution.
- Ownership is sampled at 100 ms and cannot exclude events between samples.
- Cache safety evidence comprises selected KV pages, future-zero tails, six inactive recurrent rows and explicit inactive/null/spare KV sentinels across four shards. It does not assert every physical cache page was read back.
- A checkpoint may accurately record the adapter repair and its serial evidence. It must preserve this unresolved B2 finding and cannot advertise the whole reduced control or Stage 11 as passed.
