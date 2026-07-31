# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- None. The README and work log intentionally retain rereview-pending status
  until the stage owner records this independent verdict and creates the final
  stage-owned checkpoint.

## Hard-Check Gaps

- None beyond the controlled limitations already classified in
  `stage_review_final_20260731.md`. The adapted batch-32 large-prefill
  candidates remain blocked by exact Blackhole L1 circular-buffer limits, and
  the retained automatic prefill path is supported by correct, faster batch-1
  controls and non-regressing batch-32 measurements.

## Anomaly Ledger

- Observed anomaly: the previous rereview could not independently authenticate
  the functional batch-32 prefill controls from complete runner streams.
  Evidence:
  `candidates/functional_full_prefill_b32_raw.log` and
  `candidates/functional_linear_prefill_b32_raw.log`.
  Affected path: functional full- and linear-attention batch-32 prefill
  baselines.
  Control or comparison: full seq33 reports 72.456282 ms versus optimized
  49.799153 ms; linear seq5 reports 316.626799 ms versus optimized
  294.746846 ms.
  Likely subsystem: evidence capture, not decoder behavior.
  Investigation performed: inspected both complete PTY streams. Each retains
  the exact command header, batch/sequence/full-layer arguments,
  `throw_exception_on_fallback=true` in both the command and loaded TTNN
  configuration, the `*_PREFILL_SMOKE_OK` shape and warmed result, complete
  normal device and cluster closure, and `COMMAND_EXIT_CODE="0"`. Neither log
  contains a traceback or fatal/error result. README, fresh verification, and
  work log link the exact artifacts and reproduce their values.
  Resolution: fixed.

- Observed anomaly: explicit large-prefill 2D configs originally failed or
  were slower.
  Evidence: `candidates/large_prefill_2d_autofix.log`.
  Affected path: prefill QKV, O, packed gate/up, and down matmuls.
  Control or comparison: legal batch-1 candidates did not beat the retained
  path; adapted batch-32 candidates record exact 2,431,744-byte and
  1,917,696-byte requests against a 1,572,864-byte L1 limit.
  Likely subsystem: program-config circular-buffer capacity.
  Investigation performed: grid, K-block, per-core M, and legal-shape
  adaptations were retried for both layer kinds and required batches.
  Resolution: controlled.

- Observed anomaly: the original decode residual path had avoidable DRAM
  boundaries.
  Evidence: `candidates/residual_sharded_chain_autofix.log`,
  `tt/optimized_decoder.py`, `tests/test_optimized_decoder.py`, and
  `tracy/final_residual_chain_b1/decode_perf_report.csv`.
  Affected path: full-attention decode input norm, first residual add, and
  post-attention norm.
  Control or comparison: batch-1/batch-32 full decode improved from
  1.218025/1.518185 ms to 1.066906/1.368461 ms while preserving the recorded
  PCC bar.
  Likely subsystem: residual layout and conversion placement.
  Investigation performed: measured the compatible width-sharded chain at both
  batches, promoted it to the default, asserted the optimized policy in tests,
  and captured the final runtime profile.
  Resolution: fixed.

- Observed anomaly: initial official-weight full-attention PCC was about 0.69.
  Evidence: corrected official-weight candidate logs and README.
  Affected path: packed Q projection splitting.
  Control or comparison: corrected BFP8 PCC is 0.998368503; BFP4 attention PCC
  is 0.987799141.
  Likely subsystem: Qwen per-head Q/gate channel ordering.
  Investigation performed: changed contiguous splitting to per-head splitting
  and reran the official-weight oracle.
  Resolution: fixed.

- Observed anomaly: fresh watcher runs disabled Ethernet watcher inspection.
  Evidence: `fresh_verification_20260731.md` and
  `watcher/{full,linear}_b{1,32}.log`.
  Affected path: fresh single-device watcher reruns.
  Control or comparison: saved all-feature watcher controls close cleanly;
  fresh scoped runs retain compute, NoC, CB/L1, and assertion checks.
  Likely subsystem: unrelated active-Ethernet watcher-buffer noise.
  Investigation performed: compared the scoped reruns with the saved
  all-feature controls.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: user optimized-decoder contract;
  `.agents/skills/optimize/SKILL.md`;
  `.agents/skills/tt-device-usage/SKILL.md`;
  `.agents/skills/stage-review/SKILL.md`.
- Artifact paths:
  `doc/optimized_decoder/stage_review_20260731.md`;
  `stage_review_final_20260731.md`;
  `candidates/functional_full_prefill_b32_raw.log`;
  `candidates/functional_linear_prefill_b32_raw.log`;
  `candidates/large_prefill_2d_autofix.log`;
  `candidates/residual_sharded_chain_autofix.log`;
  README, fresh verification, work log, candidate matrix, watcher logs, and
  filtered Tracy evidence.
- Code paths: `tt/optimized_decoder.py` and
  `tests/test_optimized_decoder.py`.
- Commands run: read-only `git status`, `git diff`, `grep`, `sed`, `tail`,
  `wc`, and staged-stat inspection. No device, profiler, server, or vLLM
  command was run.

## Residual Risk

- The optimized implementation, optimized-path tests, prefill/decode
  correctness and semantics, non-aligned logical sequence lengths, paged
  cache, deterministic state-mutating traces, both meaningful layer kinds,
  batch-1 and batch-32 performance, runtime precision/fidelity policy,
  topology/config sweeps, watcher cleanliness, context capacity, and strict
  optimized-decoder scope are supported by the reviewed artifacts.
- All findings from both prior independent reviews are closed. The full and
  linear functional batch-32 control streams now independently authenticate
  the last missing before/after rows. The optimized-decoder stage satisfies
  the user contract and is ready for its local stage-owned checkpoint.
