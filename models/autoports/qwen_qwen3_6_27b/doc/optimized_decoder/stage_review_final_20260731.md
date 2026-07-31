# Stage Review

Verdict: more-work-needed

Fresh independent final rereview completed 2026-07-31 without opening TT
devices.

## Required Work

- P2: The fresh functional batch-32 prefill controls are still excerpts, not
  raw runner artifacts.
  Evidence:
  `candidates/functional_prefill_b32_fresh.log` is a 26-line, 1,198-byte
  hand-assembled record containing two commands, selected `Console result`
  lines, and explanatory prose. It does not contain the raw console streams
  from either functional run. No other functional B32 prefill artifact exists
  under the Qwen autoport. The README calls these values saved baselines, and
  the prior review explicitly required “raw console output.”
  Why this matters:
  The commands and selected values are now clear, but an independent reviewer
  still cannot distinguish a faithfully captured runner result from a copied
  markdown claim. This is the one prior required-work item that was not closed
  in the requested evidence form.
  Required next step:
  Rerun the two exact commands already recorded in
  `candidates/functional_prefill_b32_fresh.log`, redirect each complete
  stdout/stderr stream to its own stage-owned log, and link those raw logs from
  the README/work log. The logs must retain the fallback setting, full command,
  shape, timing, exit status, and normal device closure. Then rerun independent
  stage review.

## Other Concerns

- None. `doc/context_contract.json`, the README, and the work log consistently
  leave the status at rereview-pending. The work-log checklist maps the
  applicable optimize requirements; multi-device collectives, MoE, LM head,
  sampling, full-model, and serving rows are out of scope for this single-chip
  decoder module.

## Hard-Check Gaps

- The large-prefill B32 candidates cannot produce latency/PCC because their
  adapted legal-grid attempts exceed the Blackhole L1 circular-buffer limit.
  This is an earned rejection: the artifact records multiple reduced K-block
  retries, exact requested bytes, the 1,572,864-byte limit, and separate
  full/linear failures rather than stopping at the first API error.
- Static pytest coverage is source/contract-oriented. Standalone saved hardware
  runners provide numerical optimized-path coverage with fallback exceptions
  enabled. This remains acceptable for the module stage.

## Anomaly Ledger

- Observed anomaly: explicit large-prefill 2D configs initially failed.
  Evidence: `candidates/large_prefill_2d_autofix.log`.
  Affected path: prefill QKV, O, packed gate/up, and down matmuls.
  Control or comparison: legal reduced B1 full/linear candidates measured
  3.879834/11.737458 ms versus retained 3.110356/11.005 ms; B32 reduced
  candidates require 2,431,744/1,917,696 bytes versus 1,572,864 available.
  Likely subsystem: program-config circular-buffer L1 capacity.
  Investigation performed: grid, K-block, per-core M, and device-grid
  adaptations at both B1 and B32.
  Resolution: controlled; default automatic prefill selection is justified.

- Observed anomaly: the original decode residual path had avoidable DRAM
  boundaries.
  Evidence: `candidates/residual_sharded_chain_autofix.log`,
  `tt/optimized_decoder.py`, and
  `tracy/final_residual_chain_b1/decode_perf_report.csv`.
  Affected path: full-attention decode input norm, first residual add, and
  post-attention norm.
  Control or comparison: prior B1/B32 1.218025/1.518185 ms; promoted default
  1.066906/1.368461 ms with PCC 0.999008/0.999583 and
  0.999593/0.999824.
  Likely subsystem: residual layout and conversion placement.
  Investigation performed: measured the width-sharded chain as a unit at both
  batches, promoted it, reran the final default, and captured a filtered Tracy
  profile. The profile shows both norms and the first residual add on the
  8-core width-sharded layout, followed by the explicit packed-gate/up
  L1-interleaved boundary.
  Resolution: fixed.

- Observed anomaly: initial official-weight full-attention PCC was about 0.69.
  Evidence: corrected real-weight candidate logs and README.
  Affected path: packed Q projection splitting.
  Control or comparison: corrected BFP8 PCC 0.998368503; BFP4 attention PCC
  0.987799141.
  Likely subsystem: Qwen per-head Q/gate channel ordering.
  Investigation performed: replaced contiguous splitting with per-head
  splitting and reran the official-weight oracle.
  Resolution: fixed.

- Observed anomaly: fresh watcher runs disable Ethernet watcher inspection.
  Evidence: `fresh_verification_20260731.md`.
  Affected path: fresh single-device watcher reruns.
  Control or comparison: historical
  `watcher/{full,linear}_b{1,32}.log` runs have no disabled features and close
  cleanly; fresh runs retain compute, NoC, CB/L1, and assertion checks.
  Likely subsystem: unrelated active-Ethernet watcher-buffer noise.
  Investigation performed: compared fresh scoped runs with the saved
  all-features controls.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: user optimized-decoder contract;
  `.agents/skills/optimize/SKILL.md`;
  `.agents/skills/tt-device-usage/SKILL.md`;
  `.agents/skills/stage-review/SKILL.md`.
- Artifact paths: `doc/context_contract.json`;
  `doc/optimized_decoder/README.md`; `work_log.md`;
  `fresh_verification_20260731.md`; prior review; all candidate, watcher,
  cache/static, shard-advisor, and filtered Tracy evidence.
- Code paths: `tt/optimized_decoder.py`;
  `tests/test_optimized_decoder.py`; standalone optimized correctness runners.
- Commits/worktree: functional base `c3cc345a10b`; optimized checkpoint
  `c55a8c067c8`; live staged remediation worktree.
- Commands run: read-only `git status`, `git diff`, `git log`, `find`, `grep`,
  `sed`, and CSV inspection; no hardware, server, profiler, or vLLM commands.

## Residual Risk

- The optimized implementation, optimized-path tests, correctness/PCC,
  non-aligned prefill, paged cache, deterministic state-mutating traces,
  representative layer kinds, B1/B32 decode performance, runtime
  dtype/fidelity rows, promoted residual chain, prefill-config rejection,
  watcher cleanliness, context capacity, and stage scope are supported.
- Stage closure is blocked only on retaining the two fresh functional B32
  control runs as raw artifacts. A later independent rereview is required.
