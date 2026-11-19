# Stage Review

Verdict: clean-pass

Round 2 of 3. Independent, diff-only follow-up for stage 4, multichip-decoder,
Qwen/Qwen3.6-27B. The round-one boundary finding is closed. All three prior
Other Concerns are addressed for this stage, and the changed files introduce
no new required work.

Reviewed the live staged worktree on `main`, HEAD
`0915d93157bd320c7875533fba09bfbffeb2ebeb`. Production decoder, optimized
baseline and core numerical runner hashes still match the round-one source.
The review covers the supplied follow-up changes and evidence; it does not
repeat the historical whole-stage review.

## Required Work

- None.
- **Closed — prior P2: neighboring logical lengths at final row-projection
  dispatch boundaries.** Both layer kinds now have matched single-chip and
  TP4 runs at B2/S256, B2/S257, B32/S256 and B32/S257. The observer wraps the
  first eager row-projection calls, forwards to the original implementation,
  and records actual calls into native MMRS. It adds no device operation.
  Both attention output and MLP down projection were observed in every case.

  | Layer | B/S | Observed rows | Both projection paths | Native pieces per projection | Minimum initial PCC |
  | --- | --- | --- | --- | --- | --- |
  | GDN, 0 | 2/256 | 512 | separate matmul | 0 | 0.999174291 |
  | GDN, 0 | 2/257 | 514 | native MMRS | 5 | 0.999159352 |
  | GDN, 0 | 32/256 | 8192 | native MMRS | 64 | 0.996169386 |
  | GDN, 0 | 32/257 | 8224 | separate matmul | 0 | 0.998257820 |
  | Full attention, 3 | 2/256 | 512 | separate matmul | 0 | 0.999644911 |
  | Full attention, 3 | 2/257 | 514 | native MMRS | 5 | 0.999665039 |
  | Full attention, 3 | 32/256 | 8192 | native MMRS | 64 | 0.999473009 |
  | Full attention, 3 | 32/257 | 8224 | separate matmul | 0 | 0.999210837 |

  Evidence: `delivery_boundary_{baseline,sharded}_l{0,3}_b{2,32}_s{256,257}.json`,
  their command and run/watcher logs, `delivery_boundary_commands.jsonl`,
  `tests/run_multichip_boundaries.py`, and the unchanged
  `tt/multichip_decoder.py::_linear` predicate. The 514-row cases execute four
  128-row pieces and one 32-row padded tail; the 8192-row cases execute 64
  128-row pieces. Input/output logical B/S is preserved. Native outputs have
  local width 1280; separate projection outputs have width 5120 before the
  existing reduction consumer. This matches the actual dispatch predicate.

  Independently checked all 336 initial numerical scalars: output,
  contribution, every user's prefill/decode PCC, and both prefill/decode state
  tensors. All are finite and at least 0.995; minimum is
  `0.9961693860968743`. The 144 baseline comparison scalars in trace cases
  are also finite and passing, minimum `0.9991364218881805`. Each TP4 artifact
  has metadata identical to its corresponding baseline. All 16 command
  records exit zero and identify the delivered source and core runner.

  Each GDN run records one capture and 25 successful replays; each
  full-attention run records one capture and 29 successful replays. Original
  and changed-input outputs and states are bit-equal to eager on both repeated
  checks. Full attention additionally checks changed positions/RoPE and
  physical-page remapping. All runs enable watcher 10 and allocation tracking
  before startup, include program-cache allocations, reject the deliberate
  unsafe-allocation control, and use no acknowledged buffers. The eight TP4
  watcher logs contain completed dumps and four-device detach records, with
  no watcher failure. Baseline watcher logs are also clean.

  `collect_multichip_evidence.py` now requires all eight cases, both consumer
  keys, actual flattened rows, native piece coverage and expected output
  widths. `run_delivery_gates.sh` invokes the focused suite. The gate summary
  contains 54 artifacts, including the eight additions; its boundary rows
  agree exactly with their individual reports. The previously accepted
  source and other gate families are retained.

## Other Concerns

- **Closed — stale fusion and stack wording.** `kernel_contracts.md` now
  consistently gives `512 < B*S <= 8192` for native MMRS and identifies the
  delivered stack as checkpoint layers 0..4, GDN/GDN/GDN/full/GDN. These
  statements agree with the unchanged production predicate and prior accepted
  stack contract.
- **Closed — trace attribution.** The current-contract text and
  `work_log.md` now attribute only original-input final-output equality to the
  five-layer trace test. Changed-input and complete-state equality are
  correctly attributed to the separate single-layer and 64-owner tests. No
  additional stack experiment was necessary or requested.
- **Closed for this stage — remaining measured optimization candidates are
  disclosed.** The work log retains prefill reshape cost and the replicated
  B1 decode advantage as next-stage opportunities. Persistent flat prefill
  and a phase-specific residual interface remain optional optimization work;
  the follow-up does not claim these costs were eliminated.
- **Packaging fix verified.** The additional packaging-only change puts
  private compressed launch-log exclusions after the `!*.log.gz` inclusion.
  `git check-ignore -v` selects the final exclusion for
  `debug_b32/autodebug_launch.log.gz`; that file is absent from the index.
  The 1865 staged paths remain under the target autoport and contain no PT,
  Tracy-directory or private launch-transcript paths. This fixes the reported
  staging issue without changing measurements.

## Hard-Check Gaps

- The prior missing-boundary aggregate gap is closed by the explicit eight
  cases and observations from both consumers.
- The prior optional scalar hardening is implemented: ordinary numerical
  artifacts now require every recorded output/contribution/per-user/state
  scalar to be finite and at least 0.995. The prior checks remain present.
- Archive reconstruction remains the documented prerequisite for scripts
  reading original log/large-JSON paths in a fresh checkout. This is unchanged,
  not a new acceptance gap. The manifest contains 684 entries. Independently
  verified all 35 boundary/review-two archive entries against stored hash,
  reconstructed hash, original length and local original bytes. No private
  launch transcript appears in the manifest. Historical entries were not
  re-reviewed in this diff-only round.
- No new hard-check gap was found. `precommit_review2.log` records passing
  applicable hooks. The later ignore-order change was inspected directly;
  this review did not rerun hooks or any test.

## Anomaly Ledger

- Observed anomaly: the prior review lacked neighboring-length evidence at
  the two production dispatch crossovers.
  Evidence: the eight new `delivery_boundary_sharded_*` artifacts and their
  matched baseline/command/watcher records.
  Affected path: attention output and MLP down row projections.
  Control or comparison: unchanged optimized single-chip baseline with
  identical input/reference metadata; direct observation of actual native
  calls in the TP4 execution.
  Likely subsystem: evidence coverage, with no established implementation
  defect.
  Investigation performed: recomputed all relevant minima, shapes, piece
  totals and source hashes; checked both consumers, trace records and logs.
  Resolution: fixed; every requested boundary is covered and passes.

- Observed anomaly: native-MMRS boundary logs warn that L1 semaphore
  allocation can fragment L1.
  Evidence: the native-path B2/S257 and B32/S256 TP4 run logs.
  Affected path: persistent native collective resources.
  Control or comparison: these focused runs finish with clean watcher and
  allocation checks; the unchanged implementation retains the previously
  accepted 64-owner resource evidence.
  Likely subsystem: runtime resource-allocation advisory.
  Investigation performed: classified the warning against source identity,
  successful native-piece execution and the existing resource contract.
  Resolution: controlled within the accepted target/resource envelope; no
  new resource failure appears in this evidence.

- Observed anomaly: boundary startup logs contain motherboard-ID fallback
  and single-chip subset warnings; one baseline run reports AICLK 1343 MHz
  versus requested 1350 MHz, within the runtime's 5% tolerance.
  Evidence: boundary run logs, specifically
  `delivery_boundary_baseline_l3_b32_s257.log` for the clock message.
  Affected path: topology labels/single-chip reference startup and diagnostic
  timing.
  Control or comparison: all baseline/TP4 numerical comparisons pass; these
  watcher/tracking runs are excluded from reported performance evidence.
  Likely subsystem: runtime discovery and startup clock settling.
  Investigation performed: checked command environments, mesh metadata,
  completed runs and that this follow-up makes no new speed claim.
  Resolution: controlled; no numerical or watcher failure accompanies these
  startup advisories.

## Scope Inspected

- Goal/skill paths: original
  `/home/ttuser/qwen36-27b/multigoal_logs/04-04-multichip-decoder.prompt.txt`;
  installed `stage-review`, `model-bringup`, `multichip`, `optimize`,
  `tt-enable-tracing` and `tt-device-usage` requirements under
  `/home/ttuser/qwen36-27b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/`.
  The exact prior Required Work and Other Concerns were read from
  `stage_review_round1.md`.
- Artifact paths: all 16 new boundary reports, command sidecars, run logs and
  watcher logs; `delivery_boundary_commands.jsonl`; boundary console archive;
  `gate_summary.json`; relevant README, kernel-contract and work-log changes;
  `precommit_review2.log`; `evidence_archive.json` and the 35 relevant new
  archived logs; `.gitignore`.
- Code paths: `tests/run_multichip_boundaries.py`,
  `tests/multichip_jobs.py`, `tests/collect_multichip_evidence.py`,
  `tests/pack_multichip_evidence.py`, `doc/multichip_decoder/run_delivery_gates.sh`;
  unchanged row-dispatch and numerical/trace runner sections needed to
  validate the new observer and results.
- Commands run: read-only `cat`, `sed`, `rg`, `git status`,
  `git diff --cached --name-only`, `git show :<path>` and `git check-ignore -v`;
  small Python standard-library artifact analyses for hashes, paired metadata,
  finite PCC minima, actual dispatch, replay assertions, log diagnostics,
  manifest reconstruction in memory and staged scope. Fourteen relevant staged
  files were byte-identical to the reviewed worktree versions. No TTNN import,
  test, device operation, server or reset was performed. Only this report was
  written.
- Source hashes independently rechecked: production
  `082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44`;
  optimized baseline
  `315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22`;
  core runner
  `0095c650047bdb2b37b420a23f5a3a957510fd2c6f85471ef017319bb7df9ab1`;
  boundary observer
  `166a4416b3f6f5b05d14371e8548e30701e1264b8054e4b27bf16381f0301b0a`.
- Push state: the supplied run plan prohibits the stage agent from pushing;
  supervisor-owned push handling is unchanged. The stage-local checkpoint
  commit is pending this passing review, as expected.

## Residual Risk

- The prior review's accepted scope and residual risks remain: this is
  decoder-layer evidence on the fixed 1x4 target and serialized queue, not
  full-model generation or vLLM validation.
- The observer records the first eager prefill dispatch. It is sufficient
  for these requested boundaries, all of which fit within the unchanged
  internal 512-token prefill chunk; it is not presented as a general trace
  instrumentation or performance tool.
- Future full-model allocator fragmentation, generated-text quality and
  optimization of the disclosed layout costs remain later-stage work.
