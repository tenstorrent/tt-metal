# Stage Review

Verdict: more-work-needed

Fresh independent rereview completed 2026-07-31 without opening TT devices.

## Required Work

- P1: Large-prefill matmul program/config tuning is missing.
  Evidence: `tt/optimized_decoder.py` passes `program_config=None` for the
  prefill QKV, output, packed gate/up, and down projections. The final full
  prefill report marks the packed gate/up and down rows `SLOW` at 10.99% and
  11.39% of the window, respectively, and explicitly recommends moving input
  0 to L1; the same report recommends a DRAM-sharded program config for QKV.
  The linear-prefill report likewise marks the packed gate/up and down rows
  `SLOW`. Neither `candidates/` nor the topology/config tables contain a
  large-prefill 2D program-config or sharded-input candidate.
  Why this matters: the user explicitly required large-prefill program configs
  to be addressed with evidence, and the optimize checklist requires 2D
  configs for large prefill matmuls plus an attempted fix for applicable
  current `tt-perf-report` advice. The saved reports show a material MLP op
  class still using defaults without an earned rejection.
  Required next step: derive and measure legal 2D/core-grid/block/subblock and
  input-memory candidates for the material full- and linear-prefill
  projections at batch 1 and batch 32. Keep the faster correct path, or retain
  exact adapted TTNN/L1 blockers and before/after PCC/latency for rejected
  candidates. Re-profile the final default.

- P1: The decode residual-layout/reshard rejection is unsupported.
  Evidence: the README says whole-residual norm sharding was rejected because
  composite token mixers require interleaved boundaries and that the partial
  chain wins, but there is no corresponding candidate log, measured
  chain-vs-chain row, or minimal op-contract repro. The final full-decode
  profile contains interleaved-to-sharded and sharded-to-interleaved
  conversions around QKV, per-head normalization, output projection, and MLP
  boundaries. The candidate matrix measures packed-interleaved and individual
  DRAM-sharded matmul geometries, not a feasible norm-to-attention-to-residual-
  to-MLP layout chain.
  Why this matters: the optimize contract requires a feasible L1-sharded
  residual chain to be measured as a unit, or an exact blocker, before
  concluding that remaining reshards are necessary. A prose statement that a
  consumer requires interleaved storage does not earn that rejection.
  Required next step: implement and measure the longest feasible compatible
  sharded chain at batch 1 and batch 32, with output conversion outside the
  measured layer when needed for comparison, or save a minimal adapted repro
  proving the next material consumer cannot accept the layout. Update the
  topology table with current conversions, candidate boundaries, and measured
  action.

- P2: Fresh functional batch-32 prefill baselines are not backed by saved
  runner artifacts.
  Evidence: `fresh_verification_20260731.md` reports 72.458432 ms and
  316.611302 ms and describes five warmed iterations, but no exact executable
  commands or functional-run logs for those two results are present under the
  optimized-decoder evidence root. The staged
  `candidates/default_*_prefill_b32.log` files are optimized runs, not the
  functional controls.
  Why this matters: the previous review specifically required comparable B32
  before/after evidence. Agent-written markdown is a claim; the rereview must
  be able to re-derive the control from a runner artifact.
  Required next step: save the exact functional B32 full/linear prefill
  commands and their raw console output, including shape, warmup/iteration
  count, fallback setting, and latency, then link them from the README/work
  log.

## Other Concerns

- `doc/context_contract.json` says
  `optimized_decoder_complete_clean_pass` while the current README and work log
  correctly say rereview is pending. Keep stage-status metadata consistent
  until a later clean pass.
- The abbreviated work-log checklist does not demonstrate completion of the
  full optimize checklist. In particular, its checked sharding/config item
  obscures the two P1 gaps above. Map the final evidence explicitly to the
  applicable optimize checklist rows after remediation.

## Hard-Check Gaps

- The repository `pytest` coverage is intentionally source/contract-oriented;
  numerical optimized-path coverage is supplied by standalone hardware
  runners. This is acceptable because the saved runner logs instantiate
  `OptimizedDecoder` directly and set
  `throw_exception_on_fallback=True`, but future CI integration would reduce
  stale-artifact risk.
- The traced regression performs two state-mutating replays. Together with the
  four historical watcher runs and the fresh reruns, this is reasonable
  repeated-run coverage for this module stage; no additional stress finding is
  raised.

## Anomaly Ledger

- Observed anomaly: initial official-weight full-attention PCC was about 0.69.
  Evidence: README/work log and the corrected real-weight candidate logs.
  Affected path: packed full-attention Q projection split.
  Control or comparison: corrected default PCC 0.998368503; BFP4-attention
  control PCC 0.987799141.
  Likely subsystem: Qwen per-head Q/gate channel ordering.
  Investigation performed: changed contiguous splitting to per-head splitting
  and reran the official-weight oracle.
  Resolution: fixed.

- Observed anomaly: fresh watcher runs disable Ethernet watcher inspection.
  Evidence: `fresh_verification_20260731.md`.
  Affected path: fresh single-device watcher reruns.
  Control or comparison: historical `watcher/{full,linear}_b{1,32}.log` runs
  show `disabled features: None`, pass PCC, and close cleanly.
  Likely subsystem: unrelated active-Ethernet watcher-buffer noise.
  Investigation performed: compared fresh scoped-disable description with the
  saved all-features watcher controls.
  Resolution: controlled.

- Observed anomaly: the final BFP4-down candidate is slightly faster than the
  selected default in the synthetic trace matrix.
  Evidence: B1/B32 BFP4-down 1.215069/1.512133 ms versus default
  1.218025/1.518185 ms.
  Affected path: MLP down projection precision.
  Control or comparison: differences are 0.003/0.006 ms and documented as no
  material win; final BFP8 down is verified in profiler rows.
  Likely subsystem: measurement-scale precision tradeoff.
  Investigation performed: candidate was measured at both required batches.
  Resolution: controlled; no precision-veto finding is raised because the
  candidate was rejected for immaterial performance, not synthetic
  correctness.

## Scope Inspected

- Goal/skill paths: user optimized-decoder contract;
  `.agents/skills/optimize/SKILL.md`;
  `.agents/skills/tt-device-usage/SKILL.md`;
  `.agents/skills/stage-review/SKILL.md`.
- Artifact paths: `doc/context_contract.json`;
  `doc/functional_decoder/`; `doc/optimized_decoder/README.md`;
  `work_log.md`; `fresh_verification_20260731.md`; all candidate, watcher,
  shard-advisor, cache/static, and Tracy/perf-report evidence.
- Code paths: `tt/optimized_decoder.py`; all
  `tests/optimized_*`; `tests/test_optimized_decoder.py`.
- Commits/worktree: functional base `c3cc345a10b`; optimized implementation
  `c55a8c067c8`; live staged evidence-remediation worktree.
- Commands run: read-only `git show`, `git diff`, `git status`, `find`,
  `grep`, and `sed`; no hardware, server, or vLLM commands.

## Residual Risk

- Correctness, paged-cache routing, non-aligned sequence handling, representative
  layer-kind coverage, B1/B32 traced decode performance, runtime dtype/fidelity,
  watcher cleanliness, and scope isolation are well supported.
- Stage closure remains blocked only by the required optimization/evidence work
  above. A later independent rereview is required after remediation.
