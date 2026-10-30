# Stage Review

Verdict: clean-pass

Independent review of stage 01, functional-decoder, on 2026-10-30. Target:
`google/gemma-4-31B-it`, revision `842da3794eaa0b77d5f08bae87a17459d91ff475`.
Reviewed the live worktree on `mvasiljevic/gemma4-31b-fresh-pipeline-v0811`, base
`db97382f276a5aecc1cd98a7eb5ba5def783f832`. The execution checkout has no remotes.
Decoder SHA256: `a9d4f4d8ba3c710fb7508e0f0804e499a1312a847484f4d4e6668bd6b4334731`.

## Required Work

- None. No material correctness defect or unmet stage-01 evidence requirement
  remains in the reviewed implementation and final artifacts.

## Other Concerns

- This pass covers representative sliding/full layers 0 and 5, with actual model
  geometry and weights. It does not establish full-model accumulated accuracy,
  autoregressive token feedback, serving registration, or production trace preparation.
- The public layer interface explicitly gives the caller ownership of page mappings,
  input/RoPE tensors and positions. Prefix hits have 32-token granularity; the suffix
  may have an arbitrary logical length. The downstream generator must preserve these
  ownership rules and prepare every signature it supports before recording traces.
- Maximum-context evidence is batch 1. The report correctly distinguishes that result
  from B8/16/32 validation and 4096-token profiling. It does not claim simultaneous
  maximum-length requests or a full-model memory budget.

## Hard-Check Gaps

- No unmet stage-01 hard check found. The long-context reference deliberately samples
  output rows while the TTNN path executes the complete prompt. This is explicit in
  the context contract and is supported by ordinary-HF short controls and an independent
  long-context eager/FP32 attention control; it is not an all-row long-context comparison.
- Local checkpoint creation and SHA logging follow this independent clean-pass under
  the stage contract. The owner must finish that bookkeeping before closing the goal.

## Evidence Re-derived

- Inspected the complete decoder, callers and HF reference. Q/K normalization and
  RoPE, unrotated V normalization, full-attention shared K/V projection, scale 1,
  GQA geometry, four layer norms, gated tanh-GELU MLP, residual ordering and layer
  scalar agree with the installed Transformers 5.12.1 implementation. Checkpoint and
  config loads pin the requested revision. Runtime imports are TTNN and the generic
  LightweightModule only; no removed Gemma implementation or Gemma-specific framework
  branch is referenced by the autoport.
- Raw logs substantiate all 12 final real-weight jobs, with allocation tracking enabled
  and program-cache allocations included. Both actual layer kinds pass PCC >= 0.995.
  At 262144 tokens, exact-maximum prefill PCC is 0.9999331202 sliding and 0.9981801950
  full; traced decode PCC is 0.9999138672 and 0.9999798988 respectively. Non-aligned
  262143-token prefill also passes. No capability reduction is used.
- Boundary tests cover 1, 31, 32, 33, 1023, 1024, 1025, 2049 and short reuse on the same
  instance, with changed input data and page ownership. Resuming at 1024 matches
  continuous output with PCC 1.0 for both kinds. Repeated prefill and decode outputs
  are checked bitwise. B8/16/32 tests use permuted nonzero pages, untouched sentinel
  pages, changed positions and reversed request rows.
- Capture validates replay output, not just eager warmup. The six interleaved cases
  preserve one decode trace ID, including unseen logical length 34 after preparing 33.
  Persistent inputs/caches are allocated before capture; eager outputs are consumed
  and released before replay. There are no corruptible-buffer acknowledgments or
  allocation scopes. Conversion guards run during forward. Final watcher logs are
  free of fault reports, and the real-shape synthetic pytest suite records four passes.
- Recomputed all eight profile summaries from the raw CSVs without running TT code.
  Each profile has exactly 128 complete decode sessions, with 42 device operations per
  sliding replay or 40 per full replay. Filtered report CSV counts equal raw-window
  device-op counts, and rendered text files contain actual performance tables.
  Source/CSV hashes match, and measured matmul rows use BFLOAT16 inputs/weights and
  HiFi4. Whole-layer firmware spans agree with reported times within timestamp rounding.

| Layer kind | Batch | Prefill device ms | Mean decode device ms |
|---|---:|---:|---:|
| sliding_attention | 1 | 97.224 | 3.039 |
| sliding_attention | 8 | 768.890 | 3.249 |
| sliding_attention | 16 | 1537.799 | 3.665 |
| sliding_attention | 32 | 3060.013 | 4.518 |
| full_attention | 1 | 136.684 | 3.344 |
| full_attention | 8 | 1089.629 | 4.029 |
| full_attention | 16 | 2176.647 | 4.870 |
| full_attention | 32 | 4338.214 | 6.602 |

The 4096-input/128-decode workload and batch-1/concurrency-1 headline are explicit.
Host teacher-forcing refresh and profiler drains lie outside complete layer replay
windows. These numbers support layer timing, not end-to-end throughput. The whole-ASIC
roofline uses 120 cores and 1.35 GHz from the [p300c specification](https://docs.tenstorrent.com/aibs/blackhole/p300.html),
and 512 GB/s per ASIC from the [QuietBox 2 specification](https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html).
The installed tt-perf-report ArchitectureSpec supplies the 4096-operations/cycle and
HiFi4 phase-divisor convention. Useful-work and DRAM numerators are documented
estimates; the smaller 110-worker runtime budget is not substituted for the ASIC peak.
All eight final report/agentic JSON rows and all 34 evidence-manifest entries were checked.

## Anomaly Ledger

- Observed anomaly: default full-attention decode failed at maximum context.
  Evidence: `full_context.log`, `long_attention_probe.json`, `full_context_post_view.json`.
  Affected path: paged full-attention decode.
  Control or comparison: exact HF K/V and HF/TT Q; separate exponential and core-split variants.
  Likely subsystem: precision loss in the long serial BF16 attention recurrence.
  Investigation performed: reviewed AutoDebug/AutoFix, probe source, long reference control
  and final end-to-end cache-producing run. K128/core-cap16/precise-exp passes the gate.
  Resolution: fixed.

- Observed anomaly: an unseen logical 34-token prefill allocated FillPad state after capture.
  Evidence: `new_bucket_case.log`, `new_bucket_fixed.json`, both final boundary/watcher logs.
  Affected path: prefill tail handling interleaved with a live decode trace.
  Control or comparison: logical 33 and 34 share the same physical 64-row bucket.
  Likely subsystem: padding conversion and late program-cache allocation.
  Investigation performed: inspected metadata-only sequence views and repeated replay
  with all tracker categories enabled, without exemptions.
  Resolution: fixed.

- Observed anomaly: earlier FP32 PCC computation rounded a high-correlation result to 1.
  Evidence: `AUTOFIX.md`, `tests/run_decoder.py`, final batch JSON/logs.
  Affected path: reported correctness metric.
  Control or comparison: final harness correlates in FP64.
  Likely subsystem: host metric accumulation precision.
  Investigation performed: checked helper and final rerun results rather than using old summaries.
  Resolution: fixed.

- Observed anomaly: the initial profile lost device markers.
  Evidence: `tracy/sliding_attention/batch1_overflow`, final `profile_provenance.json` and CSVs.
  Affected path: profiling collection, not decoder execution.
  Control or comparison: rejected incomplete capture versus 128 complete sessions in every accepted profile.
  Likely subsystem: finite profiler marker storage.
  Investigation performed: verified profiler drains occur outside complete layer windows;
  independently checked all session/op counts, report counts and timings.
  Resolution: fixed; rejected capture is excluded from claims.

- Observed anomaly: accepted raw rows report PROGRAM CACHE HIT=False after warmup/replay.
  Evidence: `ttnn/cpp/tools/profiler/op_profiler_json.cpp:175`,
  `ttnn/api/tools/profiler/op_profiler_serialize.hpp:385`, `tools/tracy/process_ops_logs.py:475`.
  Affected path: profiler metadata.
  Control or comparison: the first emission stores the hit bit inside cached short text;
  later emissions reuse that text unchanged, and the parser reads its stale bit.
  Likely subsystem: profiler serialization cache.
  Investigation performed: traced producer/cache/parser source and checked actual trace session data.
  Resolution: controlled; this field is not used to invalidate established warmup.

- Observed anomaly: motherboard-identification, optional WASM-copy and report-tool warnings.
  Evidence: final correctness/profile logs and `work_log.md`.
  Affected path: platform naming and optional report presentation.
  Control or comparison: successful 1x1 runs, clean watcher evidence, existing local trace/CSV
  files and complete raw/filtered report rows. The reported B850M-C matches the official system spec.
  Likely subsystem: platform lookup and reporting tools.
  Investigation performed: inspected warning context and verified required artifacts directly;
  histogram overflow-bin counts are not dropped profiler records.
  Resolution: controlled for this single-device stage.

## Scope Inspected

- Goal/skill paths: root `AGENTS.md`, `bringup/run_contract.json`,
  `bringup/agentic_evals/inputs.json`, expanded
  `bringup/artifacts/multigoal-runs/20261030T163027Z/01-01-functional-decoder.prompt.txt`;
  installed tt-model-bringup 0.1.22 `stage-review`, `functional-decoder`,
  `agentic-serving`, `tt-device-usage`, `tt-enable-tracing` and startup instructions.
  Supplementary tt-review-core/router, model-bringup-review and trace-review checks
  were scoped to this functional stage, without importing optimization-stage gates.
- Artifact paths: this directory's README/work log, final/post-view JSON and raw logs,
  both final watcher directories, pytest/context logs, weight statistics, runtime audit
  and identity, AutoDebug/AutoFix/probe/control evidence, all eight accepted Tracy
  collections/reports/provenance, `evidence_manifest.json`, `../context_contract.json`,
  `../agentic_serving.json` and `../functional_decoder.md`.
- Code paths: `../../tt/functional_decoder.py`, all `../../tests/*.py`, installed
  HF Gemma4 modeling source, local TraceCorrectness guide and profiler metadata sources.
- Commands run: read-only `cat`, `sed`, `rg`, git branch/HEAD/remote/status queries,
  and small standard-library Python analyses of JSON/CSV, timestamps and hashes;
  opened the two official hardware specification pages. No TTNN import, device command,
  server, reset, test rerun, or implementation modification was performed by this reviewer.

## Residual Risk

- Long-context comparison samples the final 32 prefill rows and final decode row;
  exact-maximum prefill compares its last row. It does not prove every output row.
- Inputs are synthetic layer activations with real checkpoint weights. Full-model
  accuracy and free-running quality remain subsequent-stage obligations.
- Sliding K/V retains the whole prefix in this functional layout. Memory efficiency,
  broader prepared-signature management and production serving behavior remain future work.
- Performance estimates exclude inter-replay host work, extra physical traffic and
  profiling perturbation; they must not be presented as model serving throughput.

## Post-hook Addendum

Verdict remains: clean-pass.

Reviewed the unstaged delta against the previously reviewed index after the commit
hook sorted imports in six harness files. The changes are import ordering, imported
name ordering and blank lines only. Independent AST comparison confirmed identical
normalized top-level imports and identical non-import nodes in all six files.
The decoder is byte-for-byte unchanged. The reordered imports do not move any
environment assignment, device operation or measured-work statement.

Verified each pre-format index hash and post-format worktree hash against
`formatting_provenance.json`, checked all 36 entries in the revised
`evidence_manifest.json`, and confirmed that the eight original collection manifests
retain the measured source hashes. The manifest now maps those measured hashes to
the formatted source explicitly. Existing correctness and profiling evidence remains
applicable; this cosmetic delta requires no device or profiling rerun.

Only read-only diff/hash/AST checks and this addendum were performed for the follow-up.
