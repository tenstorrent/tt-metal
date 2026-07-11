# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Recollect final profiler and standard-suite evidence from the final source/test snapshot.
  Evidence: all four files named as final raw profiler evidence under `tracy/{sliding,full}/{prefill,decode}/final/*_ops.csv` have mtimes between 2026-07-11 10:58:33 and 10:58:40 UTC, while `tt/optimized_decoder.py` has mtime 11:15:49 UTC. `evidence/standard_suite.xml` completed at 11:50:19 UTC, while `tests/test_optimized_decoder.py` has mtime 11:50:58 UTC. The README/work log record the current hashes (`a0228c...` and `7bacd5...`), but no runner artifact records source/test hashes, so the older runs cannot be proven to represent those hashes. The final warmed timing, watcher, and long-context runs do postdate the current files, but they do not replace the required final advice-backed profiler or complete standard suite.
  Why this matters: the stage contract requires final warmed prefill/decode profiler evidence and a final optimized correctness/stress suite. The measured dtype/topology rows, device-window totals, advice disposition, and suite result are stale or unbound to the delivered snapshot.
  Required next step: rerun the complete standard suite and all four final Tracy/`tt-perf-report` nodes after freezing the stage files. Record SHA-256 values in the command output or a run manifest generated before each run, then regenerate the filtered CSVs and advice-backed tables from those raw CSVs.

- P1: Make the material candidate matrix reproducible and evidence-backed before claiming the final default beats every correct candidate.
  Evidence: `candidates/` contains only eight logs (one failed BFP4-attention PCC run and seven warmed sliding-decode timing runs). The timing logs print the unchanged policy name `p150_bfp8attn_bfp4mlp_lofi_dram_sharded_v1`; they do not print the effective dtype, fidelity, core count, or block-width fields, nor the command/environment that selected the named candidate. `test_optimized_warmed_latency` only times execution and performs no PCC assertion. There are no runner artifacts for the claimed tuned split-Q/K/V result, BF16/BFP8 packed-gate/up L1 failures, four-core family failures, QKV block-7/gate-up block-21/down block-28 blockers, or the adapted four-M=32 prefill result. A search of all non-prose optimized-stage logs/XML/CSV/tables finds none of the reported blocker byte counts or the claimed 1.212 ms split-Q/K/V and 3.324 ms prefill measurements.
  Why this matters: filenames and agent-written prose do not prove which policy reached the runtime. The optimize contract requires per-candidate settings, correctness, latency, and exact blockers, and the final default must beat the strongest correct material candidate. The current artifacts cannot independently establish either the search matrix or the stated winner.
  Required next step: have every candidate runner print/serialize the complete effective `DecoderOptimizationPolicy`, implementation type, source/test hashes, node id, and command environment. Preserve logs/XML for the missing topology, packed/split, geometry, L1-blocker, and prefill candidates. Pair material timing candidates with real-weight PCC/cache/trace correctness (or a precise semantics-preserving justification tied to the exact config), then rerun the selected default through the same harness.

- P2: Close the full-context correctness claim with accurately scoped evidence.
  Evidence: the README labels the 262,144 and 262,113 rows as full `prefill` PCC and traced `decode` PCC. The inherited test actually checks only the first `min(seq_len, 2049)` full-attention prefill tokens against HF (`test_functional_decoder.py:851-855`) and compares full-context decode output with the TT prefill last-token output (`:890-898`), not with an independent HF full-context decode control. Thus the recorded full-layer values 0.998695 and 0.995178 are prefix PCC and TT prefill/decode self-consistency, respectively. This is useful capacity/cache-path evidence, but it is not the full-output HF PCC described in the result table.
  Why this matters: BFP8 cache fill/update and streaming full attention are optimized-stage changes. A shared late-context cache/page-table error could evade a 2,049-token prefix check and TT-to-TT self-consistency. The original contract explicitly requires full 262,144 context and >=0.995 prefill/decode PCC.
  Required next step: add a bounded independent reference check at representative late positions (including the last token and a page/chunk boundary) for full attention, plus an HF/reference decode check consuming the populated 262,144-token cache if feasible. Otherwise narrow the claims and document the exact control and why stronger comparison is infeasible; do not label prefix/self-consistency PCC as full prefill/decode HF PCC.

## Other Concerns

- The final profiler reports mark all three BFP4 MLP matmuls `SLOW` and expose no output-subblock fields. The prose records block-width sweeps and HiFi2 timing, but the missing candidate binding/artifacts prevent independent confirmation that the required precision-locked geometry closure was earned.
- The runtime source audit is a string-presence check. It forbids `super()._forward_device` while the implementation directly calls `FusedDecoder._forward_device`; profiler rows do show the optimized matmuls, so this is not by itself a fallback finding, but the audit should assert runtime type/policy and effective op binding rather than rely on spelling.
- The README roofline arithmetic is reproducible under the reporter's nominal one-byte BFP8 / half-byte BFP4 model, and the profiler device totals reconcile with the rendered tables. The report should label this as a modeled nominal-format roofline because block-float tile metadata is not included in the stated byte count.

## Hard-Check Gaps

- No artifact manifest binds each test/profile/candidate run to git HEAD, dirty-tree stage-file hashes, exact command, and environment overrides.
- Existing candidate timing logs do not print PCC or effective policy fields; the failed attention-BFP4 XML is the only candidate artifact with a real-weight correctness result.
- Full and sliding final profiler tables exist for both phases and their totals/dtypes rederive correctly, but they predate the delivered implementation.
- The exact/non-aligned context XMLs and logs postdate the current files and pass, but their assertions are narrower than the README table labels.

## Anomaly Ledger

- Observed anomaly: initial combined exact-context run timed out the full-attention node at the default 300-second pytest limit.
  Evidence: `evidence/rejected_harness/context_262144_300s_timeout.{log,xml}`; the later unchanged full node passes in `evidence/context_262144_full_bfp8.{log,xml}` in about 269 seconds with a 900-second limit.
  Affected path: full-attention 262,144-token capacity run.
  Control or comparison: separate sliding run passed; rerun of the full node passed; later non-aligned combined run also passed.
  Likely subsystem: harness timeout budget, not a device/model fault.
  Investigation performed: inspected rejected and acceptance XML/logs and completion timings.
  Resolution: controlled.

- Observed anomaly: stale pre-tail-fix paged-cache failure is retained.
  Evidence: `evidence/rejected_harness/stale_real_weight_pcc_before_tail_fix.xml` shows BFP8 input rejected by `paged_update_cache`; current code keeps tail/decode updates BF16 and later standard/long-context logs pass.
  Affected path: non-aligned BFP8 sliding-cache tail update.
  Control or comparison: current `optimized_decoder.py:611-638`; passing seq-33, 1025/1057, 262113, and 262144 evidence.
  Likely subsystem: paged-cache fill/update dtype contract.
  Investigation performed: compared failure traceback, current code, and superseding logs.
  Resolution: fixed.

- Observed anomaly: claimed final profiler and suite artifacts predate delivered files.
  Evidence: raw profiler CSV mtimes are 10:58 UTC versus implementation 11:15 UTC; standard-suite completion is 11:50:19 UTC versus test-file mtime 11:50:58 UTC; artifacts contain no source hashes.
  Affected path: final dtype/topology/device-time proof and complete optimized suite.
  Control or comparison: current hashes match the prose, and later timing/context/watcher evidence exists, but no later final profiler/complete-suite run exists.
  Likely subsystem: evidence provenance/finalization.
  Investigation performed: compared filesystem mtimes, current SHA-256 values, XML timestamps, and artifact contents.
  Resolution: more-work-needed.

- Observed anomaly: watcher log ends with normal dumps/detach and contains no stage-critical watcher fatal/assert/NOC/overflow/sanitizer finding.
  Evidence: `watcher_final/generated/watcher/watcher.log` and `evidence/watcher_mutable_trace.{log,xml}` (four passing nodes).
  Affected path: mutable traced decode for both layer kinds and position cases.
  Control or comparison: non-watcher standard-suite equivalents also pass.
  Likely subsystem: none; expected clean watcher output.
  Investigation performed: searched the watcher log for fatal/assert/error/NOC/overflow/sanitizer/hang signatures and inspected its tail.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: original Stage 03 contract supplied by the orchestrator; `.agents/skills/stage-review/SKILL.md`; `.agents/skills/optimize/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`.
- Artifact paths: `doc/optimized_decoder/{README.md,work_log.md,evidence/,candidates/,tracy/,watcher_final/}`; `doc/context_contract.json`; fused baseline documentation and evidence under `doc/fused_decoder/`.
- Code paths: `tt/optimized_decoder.py`; `tests/test_optimized_decoder.py`; inherited `tt/fused_decoder.py` and relevant helpers in `tests/test_functional_decoder.py`.
- Commands run: read-only `git status`/`git rev-parse`; `find`, `wc`, `sed`, `nl`, `rg`, `stat`, `sha256sum`, `jq`, and `awk` over source and artifacts. No server, TT device, hardware test, reset, vLLM, or implementation mutation was performed.

## Residual Risk

- No hardware was used during this independent review, by design. Findings about stale/missing runtime evidence require stage-owner reruns.
- Current passing logs support both layer kinds, BFP8 cache operation, non-aligned lengths, batch 2/32, mutable trace replay, full advertised allocation, and a clean watcher run. The verdict remains `more-work-needed` because the final profiler/suite snapshot and material candidate closure are not independently reproducible, and the full-context PCC claims exceed what the test asserts.
