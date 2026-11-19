# Stage Review

Verdict: clean-pass

Independent follow-up round 2 of 3, stage 7 optimized-full-model,
ornith-ai/Ornith-1.5-9B. Reviewed the live worktree on `main` over
`ebd521f744bb61f787a164f6b207c6defb8dd306`. Scope is closure of round-one
findings and changes named in `review_round2_brief.md`, not a second review of
the whole stage. No devices, servers, tests, resets, or implementation edits
were run by this reviewer. Only this report was written.

## Required Work

- None. The resident causal-page admission finding is closed; no new blocking
  correctness, contract, or evidence defect was found in the changed files.

## Previous Findings

- **P1 resident replay across an unmapped causal page: closed.**
  `tt/generator.py` now computes the final submitted position as
  `host_position + position_advances + steps - 1` and checks both configured
  context and every page in the causal prefix through that position. Inactive
  rows remain exempt. `decode_resident` calls this before its submission loop;
  ordinary resident `_decode_preflight` calls the same helper before remap
  writes, sampling changes, capture, or execution. The original position-31,
  table-`[0,-1]`, one-advance counterexample now rejects before position 32 is
  submitted. Checking before a slot permutation is valid because the existing
  remap permutes each position together with its page-table row.
  The helper uses existing CPU metadata and adds no device operation.

  `tests/full_model_resident_test.py` covers the last mapped position, first
  unmapped position, entire multi-step windows crossing a hole, inactive rows
  with unmapped tables, changed mappings, context exhaustion, the ordinary
  resident sibling, and unchanged execution/advance counts on rejection.
  `logs/resident_guard_host_final.log.gz` records all 49 host tests passing.

  `resident_page_guard.json`, its raw final log, and the probe source establish
  tracked real-layer 0/3 B3 execution across the mapped page boundary and after
  explicit table continuation. Both final active token vectors equal explicit
  token/position feedback. Sparse requests fail with unchanged replay counters
  and position advances. No invalid-address trace is intentionally submitted.
  Resident submission patches host transfer/synchronization APIs to raise;
  trace dictionaries are compared directly and program count remains 412.
  The six captures are retained without release. Allocation tracking is enabled;
  the program-cache exclusion variable is unset, which the checked runtime
  parser treats as false.

  `serving_guard_final.json` reruns the existing tracked scheduler/ownership
  probe, including resident remapping, inactive state, pending output ownership,
  and table reload commands. Current full-stack B1 and masked B32 performance
  reruns retain zero resident token/position/page/readback/capture/release
  counter increments and unchanged prepared programs. No broader arithmetic
  or accuracy rerun is required for the verified host-only guard diff.

- **TTFT reporting concern: closed as a reporting requirement; variability remains.**
  Recomputed primary medians are 90.017093 ms baseline, 92.049307 ms earlier
  selected run, and 89.839262 ms after the guard. Current samples are
  87.299565/89.839262/91.796728 ms. Both READMEs, the work log, and
  `perf_summary.json` preserve the earlier number and explicitly avoid claiming
  a material TTFT gain. Recomputed current B1 resident median is 8.707633 ms,
  or 114.841770 tokens/s/user; caller-readback median is 114.599219 tokens/s/user
  and traced teacher forcing is 113.956140 tokens/s/user.
  Masked B32 is 11.473304 ms against the matched 18.017388 ms control,
  a 36.320940% latency reduction, with equal final token vectors. The retained
  all-active timing is explicitly identified as predating the admission repair.

- **Cap-limited qualitative output concern: controlled, unchanged.**
  No new qualitative comparison or completion claim is introduced by this fix.
  The unchanged compute bodies retain the round-one accepted evidence and its
  scope. `doc/eval_readiness/limitations.json` still records both named limits
  with `original_model: "same"` and an existing matched-control evidence path.
  This follow-up does not re-judge the unchanged samples or treat truncated
  reasoning as a completed answer.

- **Local commit/SHA bookkeeping: remains the authorized post-review step.**
  The baseline commit and live worktree are identified above. Per the run plan,
  the stage owner creates local stage checkpoints and records their SHAs after
  a passing review. No push or vLLM work is authorized; neither was performed
  by this reviewer. This sequencing is not an unresolved review finding.

## Other Concerns

- No new required remediation. Keep the measured TTFT spread, the existing
  HF-controlled output limits, and the all-active timing's historical scope in
  the handoff.
- `runtime_audit.md` still uses the phrase "changed-only page tables" in its
  persistent-buffer inventory. Its boundary table and repair discussion
  correctly explain that an explicit reload uploads all fixed rows. The
  code, probe, and performance claims are unambiguous; this wording does not
  reopen the contract or require another review.

## Hard-Check Gaps

- The prior sparse-table coverage gap is closed by the host cases and tracked
  boundary probe. The mechanical stage and degenerate-output checks pass, but
  closure relies on these targeted controls and source inspection as well.
- Earlier compute evidence is deliberately reused. I verified all 28 entries
  in `final_audit.json` against journal exit codes and actual production hashes:
  the 21 historical rows differ only in `tt/generator.py`; the seven post-repair
  validation rows match current production source. Reversing precisely the
  supplied guard diff reproduces the earlier measured generator hash.
- The measured page probe differs from its current source only by the location
  of one import. Independently moving `trace_allocation_tracker` back below the
  `SamplingParams` import reproduces measured SHA256
  `53c47f6f215d9a59fe8bd95b5aacbfd10207b8c2adb482a29b893992efc2e7ad` exactly.
  Current probe SHA256 is
  `311323b7fde18dca6750107bf161ff1c3f49c519eb6bf5e30aa1fe88b1b5ded3`.
  `probe_format_provenance.json` records this formatting-only distinction.
  It does not invalidate the executed probe logic.

## Anomaly Ledger

- Observed anomaly: An admitted resident request could advance into an unmapped page.
  Evidence: Round-one position-31 counterexample; new shared admission helper,
  host cases, `resident_page_guard.json`, and tracked scheduler rerun.
  Affected path: Bulk and ordinary one-step resident decode admission.
  Control or comparison: Explicit feedback across a mapped boundary and after
  page-table continuation; sparse windows reject before execution.
  Likely subsystem: Host causal-window validation.
  Investigation performed: Followed host metadata, remapping, admission ordering,
  and replay advancement; verified the exact production diff and run provenance.
  Resolution: fixed.

- Observed anomaly: The first boundary probe failed its page-refresh assertion.
  Evidence: `resident_page_guard_probe_assertion.json` reports three refreshed
  rows; `logs/resident_page_guard.log.gz` retains `changed_only_page_refresh`.
  Affected path: Diagnostic expectation after explicit table reload.
  Control or comparison: Existing `full_model_serving_decode_probe.py` already
  asserts all three fixed rows refresh; production reload loops over every row.
  Investigation performed: Compared the failed artifact with the existing
  contract, corrected assertion, passing final log, and source hashes.
  Resolution: fixed test expectation; no invalid-address execution and no
  production reload change.

- Observed anomaly: Tracking-disabled timing logs emit the generic live-trace
  allocation warning.
  Evidence: `optimized_guard_final.log.gz` and `batch32_guard_final.log.gz`;
  current tracked page/scheduler probes complete without an allocation failure.
  Affected path: Existing warmup/request preparation diagnostics.
  Control or comparison: Round-one accepted program-cache-inclusive full-stack
  tracking, plus current tracked probes, with unchanged trace/allocation bodies.
  Investigation performed: Verified production changes are exclusively CPU
  admission, tracker settings, passing replay controls, and fixed program counts.
  Resolution: controlled within the previously reviewed allocation scope.

## Scope Inspected

- Goal/skill paths: Supplied round-two brief and verbatim prior findings;
  installed 0.1.23 stage-review and model-bringup skills, stage-7 goal template,
  optimize requirements, and relevant trace admission/allocation requirements.
- Artifact paths: Round-one report and guard diff; changed model/stage READMEs,
  work log, runtime audit, performance summary, final audit and archive manifest;
  command journal; four new device-result JSONs and corresponding logs; retained
  failed probe; host, stage, degenerate, health and pre-commit logs; import-order
  provenance; named baseline controls and unchanged limitations record.
- Code paths: Changed portions of `tt/generator.py`, all of the resident host
  test and page probe; relevant unchanged serving, B1/B32 measurement and command
  wrapper code; runtime parsing of program-cache exclusion.
- Commands run: Read-only `git status`, `git branch`, `git rev-parse`, `rg`,
  `cat`, `sed`, and small standard-library Python analyses of source hashes,
  JSON, gzip/XZ, logs and measurement arithmetic. No TTNN import or device use.
- Independently verified: All 107 archive byte lengths and SHA256 hashes match;
  new tracked probe JSON equals its raw log; all three B1 printed timing rows
  and ten B32 printed case rows equal their result JSON entries; all 41
  production source hashes match each of the four new device evidence commands.
  Current generator SHA256 is
  `18b1e33905a1b279d1654fee4256ba8175306a770159263cd80bf903bf03afa5`.
  Logged final host suite, gates and final pre-commit pass complete successfully.

## Residual Risk

- This is an inspection review. The boundary-specific tracked control uses two
  real layer kinds; full-stack B1 and B32 reruns cover the affected public paths
  without replacing the prior full-stack allocation, accuracy or capacity gates.
- Short warmed timing windows do not establish a stable TTFT improvement. The
  unchanged maximum-context and qualitative limits from round one remain.
- Post-review local commits and SHA logging still belong to the stage owner;
  no checkpoint or push was created by the reviewer.
