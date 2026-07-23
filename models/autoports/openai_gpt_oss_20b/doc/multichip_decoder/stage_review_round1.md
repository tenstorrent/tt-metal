# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Restore or qualify the optimized decoder's precision-sensitive real-weight prefill paths.
  Evidence: The completed single-chip baseline records that full-attention S=128 needs exact FP32 manual attention plus a decoder-local BF16 dense expert path, and that longer full-attention prefill uses BF16/HiFi4 sparse chunks; the rejected native/sparse controls were below PCC 0.99 (`doc/optimized_decoder/README.md`, selected-path bullets and the S=128 control discussion). The multichip constructor instead sets `use_manual_prefill_attention=False` and `use_dense_long_prefill=False` (`tt/multichip_decoder.py:157-168`), `_prefill_attention` always uses native TTNN SDPA (`:695-719`), and all multichip prefill experts use the same BFP8/LoFi policy (`:320-333`, `:502-512`, `:1019-1077`). The current profiler directly confirms `LoFi BF16 x BFP8 => BF16` sparse prefill rows. The only real-weight prefill comparisons are S=17 and S=127 (`tests/test_multichip_decoder.py:729-923`); S=129 is synthetic (`:530-587`). The real-weight S=128 long-trace test uses its prefill only to seed eager and traced copies and never compares that prefill, cache, routing, or following decode with the current optimized baseline (`:1064-1174`). Nevertheless, `doc/context_contract.json` claims real-weight validated prefill context 128.
  Why this matters: The multichip stage silently drops correctness policies that the current optimized baseline established from real target weights. Synthetic S=129 and trace-versus-self agreement cannot prove that the resulting S=128 or longer full-attention path is correct. The reported S=128 latency may therefore time a path already known to be precision-sensitive and potentially below the functional bar.
  Required next step: Use `$autofix` to either preserve the layer-aware optimized policies in TP4/EP4 form or prove a different multichip policy correct. Add current real-weight comparisons against `OptimizedDecoder` for both layer kinds at S=128 and a non-aligned length above it (at least S=129; include a larger bounded point such as S=2048 if supported by the stage harness), checking output, logical paged K/V, routing/top-4, and following decode. Correct `context_contract.json` to contain only actually validated contexts.

- P1: Make long-position trace replay reusable for sequential decode and measure that production path.
  Evidence: At positions 127 and above, `_decode_attention` selects the manual path using the Python `cache_position` (`tt/multichip_decoder.py:866-940`). `_manual_paged_decode_attention` bakes Python-derived `window_start`, `valid_length`, page slices, and tensor shapes into the graph (`:787-833`). The long-position trace test captures only position 128 and repeatedly executes the identical graph without changing hidden input or position (`tests/test_multichip_decoder.py:1064-1174`). The S=128 timing similarly captures one fixed position and replays it 500 times (`:1303-1377`). `README.md` explicitly admits that callers must warm/recapture when the logical range changes; during actual generation that range changes on every token.
  Why this matters: The evidence proves deterministic replay of a fixed graph, not a warmed decode trace that advances through arbitrary current positions. Above position 127, the advertised 131072-context production path would pay warm/capture work per token, while the reported traced latency and speedup omit that work. This is not a production-suitable decode trace contract.
  Required next step: Implement a trace-stable long-position path with device-mutable positions and fixed/persistent buffers or an equivalent bounded/masked range. Validate sequential replays across 127-131 and a later page transition for both layer kinds and arbitrary page mappings, including eager equivalence and per-step K/V updates. Re-measure sequential warmed trace latency. If an exact TTNN limitation prevents reusable capture, record the minimal repro and report the honest eager/recapture-inclusive latency; do not present fixed-position replay as production traced decode.

- P1: Recreate current, source-provenanced timing, profiler, context, and watcher evidence.
  Evidence: The current `tt/multichip_decoder.py` and test file have mtimes `2026-07-23 05:23:47`, after the context JUnit (`05:04`), accepted timing runs (`05:06-05:16`), current Tracy profile (`05:13`), and watcher run (`05:19`). The later `final_consolidated_suite.junit.xml` starts after the source write and covers the ordinary correctness/trace suite, but explicitly skips the full-context and all performance tests. The exact timing artifact named by the reports, `logs/multichip_perf_layer12_seq17.json`, was overwritten by the profiler run and currently records only one prefill repeat and three trace replays at 6.620425/0.697414 ms. `README.md`, `work_log.md`, and `perf/perf_report.md` instead claim an accepted 20/500 result of 5.0788/0.6315 ms; no surviving JSON or JUnit artifact contains that run.
  Why this matters: The final implementation is not tied to the hardware evidence used for closure, and one of the headline timing rows is directly contradicted by its named JSON. A passing profiler smoke is not a substitute for the claimed warmed wall measurement.
  Required next step: Freeze the remediated source state, record its commit or content hashes in the artifacts, rerun all accepted 1x1/1x4 timings with the documented repeat counts, preserve profiler output separately so it cannot overwrite wall-timing JSON, and rerun full-context and watcher gates on that same state. Regenerate all tables from the surviving JSON/CSV rather than copying values into prose.

- P1: Earn the rejected collective, residual-layout, and expert-parallel alternatives on the current implementation.
  Evidence: `README.md` says the July 18 historical results are strategy priors only and “will be rerun against the current optimized baseline,” but its candidate table still rejects reduce-scatter/delayed gather, carried width-sharded residuals, fused gather/matmul, and TP4 active experts from historical results. `work_log.md` contains no current candidate run. The tests and current source contain only the selected all-reduce/replicated-residual/EP4 path, and the fresh July 23 profiler measures only that default. The alternative CSVs (`ops_perf_results_ccl_rs_ag_pad64.csv`, `ops_perf_results_ep4_qkv10*.csv`) are dated July 18, before this stage's July 23 starting state. Despite that, `perf/perf_report.md` turns those unrerun statements into the final decision. The selected path regresses every measured prefill row and S=128 sliding decode, making the unearned rejections material rather than academic.
  Why this matters: The multichip skill requires topology candidates to be adapted through the next consuming norm/residual/MLP/attention boundary and compared as coherent traced families on the current code. Historical measurements from a different baseline do not establish that the final replicated all-reduce path is the fastest practical implementation.
  Required next step: On the remediated current source, measure the material candidate families with their complete boundary contracts: ordinary all-reduce; padded RS+AG control; reduce-scatter with the sharded residual carried into the next norm/consumer (not immediately regathered); supported fused matmul/CCL variants; and current TP4 versus EP4 gate-selected experts. Record residual layouts, collective bytes/dtype, persistent buffers, correctness, and traced whole-layer latency. Use `$autofix` for topology/API failures before rejecting a family.

- P2: Qualify the advertised context on the full-attention path and fix the context reproduction gate.
  Evidence: `test_full_context_cache_allocation_and_last_page_update` hardcodes only `SLIDING_LAYER` (`tests/test_multichip_decoder.py:673-724`). It initializes an otherwise zero prefix, checks only output shape/replication and a nonzero last-page K/V write, and never asserts finite output or tests the full-attention manual gather over the 131072-token logical prefix. The artifact predates the final source state. The README's exact node id includes `[blackhole-1x4-12]`, but the actual JUnit testcase is `[blackhole-1x4]`, so the documented command does not select an existing test.
  Why this matters: Sliding attention gathers only a bounded 128-token range, while full attention exercises the materially larger long-context path. Allocation plus one write on sliding attention does not establish usable advertised context for both layer kinds, and the promised reproduction command is invalid.
  Required next step: Parameterize the endpoint gate over sliding and full layers, exercise the final source with arbitrary page mapping, assert finite output and correct endpoint K/V placement on every rank, and add the strongest feasible current-baseline control. Preserve separate passing JUnit artifacts and correct the documented node ids.

- P2: Produce watcher evidence that covers the Ethernet/CCL subsystem or justify and compensate for disabling it.
  Evidence: Both the documented command and `logs/watcher_final.log` use `TT_METAL_WATCHER_DISABLE_ETH=1`; the log states “Watcher server initialized, disabled features: ETH.” The reviewed layer executes two ring all-reduces and communication is the largest decode category. No evidence explains why ETH checking is disabled or supplies a complementary fabric/ERISC check. The watcher run also predates the final source write.
  Why this matters: A worker-only watcher pass does not fully cover the most risk-sensitive subsystem in a CCL-heavy multichip stage. The multichip contract asks for a watcher-clean run, not an unexplained exclusion of the fabric cores.
  Required next step: Rerun `TT_METAL_WATCHER=10` without disabling ETH on the final source. If the platform requires disabling ETH for a known watcher limitation, document the exact limitation and add an equivalent current fabric/ERISC health or triage check that closes the omitted coverage.

## Other Concerns

- The full-layer single-chip trace is about 6.75 ms at both S=17 and S=128, versus about 0.8 ms for the sliding layer. Source inspection of the prior `OptimizedDecoder` shows its full-layer `use_dense_long_prefill` condition also includes `seq_len == 1`, explaining the anomalous baseline. The reports correctly avoid interpreting the resulting above-100% efficiency, but the 10.6x/6.5x “speedups” should not be treated as multichip scaling evidence.
- The final watcher console reports nanobind leaked instances/types/functions at interpreter shutdown. Devices subsequently detach and close normally and both tests pass, so this does not by itself identify a model-path failure, but it should remain classified in future watcher logs.
- The evidence directory mixes July 18 historical profiler/watcher files and July 23 current files. The final report names the intended current CSVs, but a manifest with source hashes and run timestamps is needed to prevent accidental reuse.
- `README.md` says the independent review status is “recorded below,” but no review status was present before this report.

## Hard-Check Gaps

- Most JUnit XML files retain only pass/fail and not captured PCC stdout. The assertions establish PCC >= 0.99, but the exact decimal values copied into README/work log are directly auditable only in `watcher_final.log` for the boundary run.
- Hardware discovery and mesh smoke are recorded only in agent prose. The current watcher/profiler artifacts do independently show four devices and a 1x4 fabric run, so this is an evidence-format gap rather than a separate correctness blocker.
- There is no run manifest tying source/test hashes, environment variables, JUnit, timing JSON, and profiler CSVs together. This gap contributes directly to the stale/overwritten performance finding above.

## Anomaly Ledger

- Observed anomaly: The multichip full-prefill precision policy differs from the correctness-qualified optimized baseline.
  Evidence: `tt/multichip_decoder.py:157-168`, `:320-333`, `:695-719`, and `:1019-1077`; `doc/optimized_decoder/README.md`; LoFi BF16×BFP8 rows in `perf/prefill_seq17_ops.csv`.
  Affected path: Full-attention S=128 and longer prefill; potentially sliding prefill precision as well.
  Control or comparison: The single-chip optimized real-weight S=128 manual-attention/dense-expert path and longer BF16/HiFi4 path.
  Likely subsystem: Attention accumulation and routed-expert precision/top-4 sensitivity.
  Investigation performed: Compared source branch conditions, dtype/fidelity setup, tests, context claims, and profiler rows.
  Resolution: more-work-needed

- Observed anomaly: `context_contract.json` calls S=128 a real-weight validated multichip prefill context, but no test compares that prefill with a reference.
  Evidence: `tests/test_multichip_decoder.py:1064-1174` seeds both eager and trace from the same unchecked S=128 prefill; real comparison tests use S=17 and S=127.
  Affected path: Context contract and S=128 performance acceptance.
  Control or comparison: Current single-chip `OptimizedDecoder` S=128 output/cache/routing.
  Likely subsystem: Evidence bookkeeping plus prefill correctness.
  Investigation performed: Traced every test that constructs real weights and enumerated its compared outputs.
  Resolution: more-work-needed

- Observed anomaly: Long-position trace replay fixes the Python cache position and logical gather range.
  Evidence: `tt/multichip_decoder.py:787-833`, `:866-948`; `tests/test_multichip_decoder.py:1064-1174`.
  Affected path: Sequential decode at every position >=127, including the advertised long-context path.
  Control or comparison: The short native trace mutates positions 17-19 in one capture (`tests/test_multichip_decoder.py:925-1062`).
  Likely subsystem: Trace lifecycle, paged logical gather, and dynamic-position masking.
  Investigation performed: Followed host integers versus mutable device tensors through both trace tests and the timing harness.
  Resolution: more-work-needed

- Observed anomaly: The layer-12 S=17 timing artifact contradicts the headline accepted timing.
  Evidence: `logs/multichip_perf_layer12_seq17.json` records repeats 1/3 and 6.620425/0.697414 ms; reports claim repeats 20/500 and 5.0788/0.6315 ms.
  Affected path: Warmed latency, speedup, efficiency, and report reproducibility.
  Control or comparison: The other three timing pairs retain their stated repeat counts and values.
  Likely subsystem: Artifact naming/provenance; profiler overwrote wall-timing JSON.
  Investigation performed: Read every timing JSON and compared values/repeat counts with all three markdown tables and JUnit inventory.
  Resolution: more-work-needed

- Observed anomaly: Full-context, timing, profiler, and watcher artifacts predate the final implementation/test write.
  Evidence: file mtimes and `final_consolidated_suite.junit.xml`; the latter is current but skips context/perf, and it is not a watcher run.
  Affected path: Final-code evidence provenance.
  Control or comparison: The consolidated correctness/boundary/trace cases ran after the final source write and passed.
  Likely subsystem: Stage execution ordering and artifact manifesting.
  Investigation performed: Compared source/test/doc mtimes with every evidence artifact timestamp and JUnit testcase list.
  Resolution: more-work-needed

- Observed anomaly: Material topology alternatives are rejected only by historical July 18 evidence after the stage says historical evidence will not be accepted.
  Evidence: `README.md` strategy-prior paragraph and candidate table; July 18 alternative CSV mtimes; absence of candidate code/tests/current run records.
  Affected path: O-projection collective, residual handoff/norm, and expert topology selection.
  Control or comparison: Fresh July 23 profile covers only selected replicated all-reduce/EP4.
  Likely subsystem: Multichip optimization methodology.
  Investigation performed: Searched source, tests, work log, README, profiler report, and artifact inventory for current candidate runs.
  Resolution: more-work-needed

- Observed anomaly: Advertised-context endpoint coverage exercises only sliding attention and does not check finite output.
  Evidence: `tests/test_multichip_decoder.py:673-724`; `logs/full_context_131072.junit.xml`.
  Affected path: Full-attention 131072 decode and public context contract.
  Control or comparison: The earlier single-chip optimized stage records a real-weight finite-output 131072 capacity run.
  Likely subsystem: Long-context qualification.
  Investigation performed: Read the endpoint test and compared its layer/range behavior with `_manual_paged_decode_attention`.
  Resolution: more-work-needed

- Observed anomaly: The documented full-context pytest node id does not exist.
  Evidence: README uses `[blackhole-1x4-12]`; JUnit and test parameterization use `[blackhole-1x4]`.
  Affected path: Reproduction instructions.
  Control or comparison: `full_context_131072.junit.xml` gives the actual collected node id.
  Likely subsystem: Documentation drift.
  Investigation performed: Compared README command, test decorators/signature, and JUnit testcase name.
  Resolution: more-work-needed

- Observed anomaly: Watcher runs with ETH checks disabled on a CCL-heavy path.
  Evidence: README watcher command and `watcher_final.log` initialization line.
  Affected path: Ring all-reduce/fabric health coverage.
  Control or comparison: Worker watcher checks attach to all four devices and both layer tests pass.
  Likely subsystem: Watcher configuration and Ethernet fabric observability.
  Investigation performed: Inspected console/device watcher logs for attachment, disabled features, errors, completion, and device close.
  Resolution: more-work-needed

- Observed anomaly: The full-layer baseline produces superlinear apparent speedups.
  Evidence: timing JSONs and tables; current baseline is ~6.75 ms while multichip is ~0.63-1.03 ms.
  Affected path: Performance interpretation.
  Control or comparison: Sliding baseline is ~0.80-0.87 ms; prior optimized source includes full-layer decode in a dense-control condition.
  Likely subsystem: Previous-stage single-chip baseline dispatch, not TP4 hardware scaling.
  Investigation performed: Compared layer timing rows and inspected `OptimizedDecoder._optimized_moe_forward`.
  Resolution: controlled — reports disclose the anomaly and do not claim >100% efficiency, but the ratios remain non-scaling evidence.

- Observed anomaly: Profiler reports a merged four-device prefill roofline of 108.1%.
  Evidence: `perf/perf_report.md` and grouped CSVs.
  Affected path: DRAM interpretation only.
  Control or comparison: Report explicitly states this is a merged model and not one device exceeding peak.
  Likely subsystem: Aggregated roofline accounting.
  Investigation performed: Cross-checked grouped operation/category CSVs and the report wording.
  Resolution: controlled

- Observed anomaly: Watcher console reports nanobind leaks at process shutdown.
  Evidence: tail of `logs/watcher_final.log`.
  Affected path: Python binding teardown.
  Control or comparison: Tests pass, watcher checks complete, and all four devices detach and close normally.
  Likely subsystem: Generic binding lifetime/teardown rather than decoder runtime.
  Investigation performed: Read the surrounding shutdown sequence and device log tail.
  Resolution: controlled

- Observed anomaly: `/dev/shm` and unknown-motherboard warnings are disclosed as non-fatal.
  Evidence: README/work log; motherboard warnings are present in `watcher_final.log`.
  Affected path: Host discovery and MPI shared-memory setup.
  Control or comparison: Four-device fabric opens, both watcher cases pass, and devices close normally.
  Likely subsystem: Host metadata/shared-memory environment.
  Investigation performed: Compared disclosed warnings with runtime completion and device count.
  Resolution: controlled

- Observed anomaly: The final consolidated suite reports five skips.
  Evidence: `final_consolidated_suite.junit.xml` has 26 tests, 0 failures/errors, and 5 skips.
  Affected path: Full context and performance gates.
  Control or comparison: Separate artifacts exist for the skipped gates, but the context/timing provenance problems above remain.
  Likely subsystem: Opt-in test orchestration.
  Investigation performed: Read every testcase and skip message in the XML.
  Resolution: controlled for ordinary suite orchestration; more-work-needed for the stale/contradictory opt-in artifacts already classified above.

## Scope Inspected

- Goal/skill paths:
  - Original repo-local multichip-decoder stage contract supplied by the stage owner.
  - `.agents/skills/stage-review/SKILL.md`
  - `.agents/skills/multichip/SKILL.md`
  - `.agents/skills/tt-device-usage/SKILL.md`
  - `tech_reports/LLMs/llms.md`, section 3.3 Multi-Device
- Artifact paths:
  - `models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/README.md`
  - `models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/work_log.md`
  - `models/autoports/openai_gpt_oss_20b/doc/context_contract.json`
  - All JUnit XML, timing JSON, watcher logs, and PT reference artifacts under `doc/multichip_decoder/logs/`
  - `doc/multichip_decoder/perf/perf_report.md`
  - Current/historical profiler CSV inventory under `doc/multichip_decoder/perf/`, with direct inspection of the current raw/checkpointed and grouped CSVs
  - `doc/functional_decoder/multichip_provenance.json`
  - Relevant optimized-decoder README/work-log evidence used as the current single-chip contract
- Code paths:
  - `models/autoports/openai_gpt_oss_20b/tt/multichip_decoder.py`
  - `models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py`
  - Relevant branches in `tt/optimized_decoder.py`
- Commands run:
  - Read-only `sed`, `nl`, `grep`, `find`, `wc`, `ls`, `stat`, and `head` inspections.
  - Read-only `git status`, `git diff --stat`, and `git diff --check`.
  - No server, TT device, reset, reservation, test, profiler, watcher, or vLLM command was run by the reviewer.

## Residual Risk

- The direct implementation review found coherent TP4 Q/K/V/O ownership, rank-selective O bias, local paged KV shapes, arbitrary full-page permutations, EP4 gate-selected sparse execution, replicated layer boundaries, and clean runtime fallback structure. Existing current-suite JUnit evidence passes both layer kinds at S=17, eager boundary positions 127-131, synthetic non-aligned S=129, two-layer device handoff, and fixed-position/short trace tests. Those positives do not close the required work above.
- No implementation or test files were modified during review. Hardware-only remediation outcomes cannot be predicted from artifact inspection; the next stage owner should use `$autofix` for the precision and trace issues, rerun the current candidate/performance matrix, and request a fresh independent stage review.
