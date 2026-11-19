# Stage Review

Verdict: clean-pass

Fresh independent follow-up, round 2 of 3, Stage 4 multichip decoder for
Qwen/Qwen3.8-Flash-Next. The round 1 required performance-evidence repair is
closed. No new required work was found in the changed artifacts. This review
checks the repair and updated documentation; it does not repeat the whole-stage
review or claim an independent hardware reproduction.

## Required Work

None.

- **Closed — round 1 P2: Report latency from separate runs with allocation
  tracking and diagnostics disabled.**
  Evidence: all six `clean_timing_{baseline,mesh}_l{0,1,3}.json` reports and
  adjacent provenance/logs record completed, exit-zero executions. Arguments
  select logical length 33, capacity 128, `--trace --batch 32 --repeats 32
  --clean-timing`, and the default collective/reduction. Provenance sets
  `TT_METAL_TRACE_ALLOC_TRACKING=0`, leaves watcher and profiler settings unset,
  and uses separate `multichip_l{0,1,3}_clean_timing.pt` reference files. Reports
  independently record tracking and runtime audit disabled. The unchanged
  harness selects `nullcontext` for the audit in clean mode.
  The six source maps are identical, contain 217 files each, match current files,
  and match the respective final tracked mesh runs. Selected implementation
  SHA256 remains `af4699739ad869466bbc825909daf868e8a2f01ed64e0af149602fe6128349c1`;
  harness SHA256 remains
  `c6a0def66126689743ddf3b9f35212768f5e726e64e078da4fdfc65ebbd19518`.
  Run timestamps show serial, nonoverlapping measurements. All six log hashes
  match provenance, all logs end with `RUN_RECORDED_FINISHED exit_code=0`, and
  no error/fatal/traceback/assertion/exception line was found.

  I reconstructed all 12 decode medians from the individual samples: discard
  the first two of 32, convert the remaining values to float32, and select the
  lower median. They equal the report values exactly. All 384 B1/B32 replay
  records have exact output and state parity; B1 cycles through input indices
  0, 1, 2. The source places state restoration, input copies and result checks
  outside the timer; timed decode includes trace execution and synchronization.
  Prefill is the source's median of five warmed public calls, also with state
  restoration outside timing. The raw prefill samples are not persisted, so
  its aggregation is verified through the unchanged runner and reported scalar.

  All nine latency pairs, speedups and efficiencies recompute exactly from the
  clean reports. `performance_summary.json`, the README and performance analysis
  use these clean values. Separate `sources.correctness` entries retain the
  tracked baseline/final-mesh records; their report/provenance hashes and the
  copied mesh PCC values match. Historical instrumented values are explicitly
  diagnostic. The manifest's six clean timing records agree with their reports.

  | Layer | Prefill baseline / mesh ms | B1 baseline / mesh ms | B32 baseline / mesh ms | B1 speedup / efficiency | B32 speedup / efficiency |
  |---|---:|---:|---:|---:|---:|
  | 0 GDN | 61.863331 / 21.415272 | 3.077050 / 2.264450 | 96.715439 / 57.873066 | 1.358851x / 33.971% | 1.671165x / 41.779% |
  | 1 PLE/GDN | 67.549210 / 26.830248 | 4.245894 / 3.431540 | 135.496033 / 95.144455 | 1.237314x / 30.933% | 1.424109x / 35.603% |
  | 3 QSA | 103.345619 / 80.129524 | 4.955306 / 4.747707 | 152.576401 / 131.113815 | 1.043726x / 26.093% | 1.163694x / 29.092% |

## Other Concerns

- **Open as a nonblocking hand-off — performance candidates.** The previous
  concern remains correctly bounded. `performance_analysis.md` and
  `work_log.md` under `Carried to next stage` retain QSA indexer partitioning,
  packed-only indexed experts, fused projection/collective families, injection
  K partitioning, PLE lookup parallelism and precision-locked geometry as
  unmeasured work. The new timing evidence does not claim global optimality or
  reject those candidates. This is not unfinished round 1 Required Work.
- **Open as a nonblocking hand-off — full-stack capacity assumptions.** The
  changed README and work log preserve shared `MeshWorkspace`/immutable tables,
  bounded maximum-context IO and planned embedding/head/final-mixer placement.
  They still disclose that reservation tests do not instantiate 48 layers, and
  distinguish B32/capacity 128 from B1 maximum context. The clean timing update
  does not expand the validated capacity or concurrency contract.

## Hard-Check Gaps

- **Closed — inherited allocation positive-control indexing.**
  `evidence_manifest.json` now links
  `../functional_decoder/trace_alloc_tracking/positive_control.json`, its log,
  and the prior optimized-decoder acceptance index. The linked JSON and log
  show tracking enabled, rejection of the deliberately live buffer with the
  unsafe-buffer error, and successful replay after deallocation. No new control
  run was required for this indexing repair.
- **Open, unchanged and nonblocking — historical per-request B32 gate.** No
  harness change was made. The work log explicitly carries the request to gate
  whole-output PCC and retain per-request PCC as diagnostics when the harness
  is next changed. The updated performance analysis correctly describes the
  native-AR candidate's two per-request diagnostic misses and the absence of a
  historical whole-B32 aggregate; it does not call them a demonstrated failure
  of the user's aggregate gate. The original accepted aggregate results remain
  separately indexed.
- **Open, unchanged and outside this decoder stage — future generator trace
  contracts.** The work log carries coexistence of live traces, changing
  scheduler/page mappings and advancing token feedback to later work. The
  README and summary correctly describe restored-state decoder signatures:
  three prepared B1 inputs and one diverse prepared B32 batch. The additional
  clean replays do not claim broader trace-lifecycle or serving coverage.

## Anomaly Ledger

- Observed anomaly: Final latency/speedup/efficiency previously came from
  tracked and audited validation runs.
  Evidence: round 1 Required Work; the six new clean timing reports, provenance
  and logs; updated `performance_summary.json` and documentation.
  Affected path: Published Stage 4 performance measurements.
  Control or comparison: Paired one-chip/four-chip runs of unchanged source,
  matching logical workload and separate timing references, with tracking,
  runtime audit, watcher and profiler disabled.
  Likely subsystem: Benchmark configuration and evidence selection.
  Investigation performed: Source/settings/hash verification, reconstruction
  of every decode median and all nine ratios, replay-exactness inspection, and
  verification of clean-versus-correctness source mapping.
  Resolution: fixed. The original tracked reports remain correctness evidence.

No new anomaly was found in the changed artifacts. The other round 1 anomaly
classifications were not reopened because this repair changes no implementation
or test source and does not replace their evidence.

## Scope Inspected

- Goal/skill paths: original goal
  `/home/ttuser/qwen38-flash/multigoal_logs/04-04-multichip-decoder.prompt.txt`;
  installed v0.1.24 `stage-review` follow-up contract, `model-bringup` startup,
  relevant `multichip` acceptance requirements and `tt-enable-tracing`
  allocation/performance separation requirement. Enabled dependency entries
  and the package environment helper were checked. No full-model/vLLM gate was
  imported.
- Workspace: live `main`, HEAD
  `cdb5e8f161972ca9d2958e81d809c29f686b817a`, with staged Stage 4 changes.
  There is no distinct round 1 commit; unchanged code is established by the
  recorded source maps, not by an empty unstaged diff.
- Artifact paths: `stage_review_round1.md`; six new clean report/provenance/log
  triplets; updated `performance_summary.json`, `README.md`,
  `performance_analysis.md`, `evidence_manifest.json`, `work_log.md`;
  `precommit_round2_evidence.log` (applicable hooks passed); linked inherited
  positive control. Original tracked reports/provenance were read only to check
  preserved source mapping and source identity.
- Code paths: relevant unchanged sections of
  `tests/probe_multichip_decoder.py` for option validation, prepared-reference
  selection, audit disabling, warmup, timers, trace comparisons and aggregation.
  Production/test source hashes were checked without re-reviewing their
  implementation.
- Commands run: read-only `cat`, `sed`, `rg`, `git status/diff/branch/rev-parse`,
  `sha256sum`; small Python JSON, float32, hash and log analyses. Checks included
  1,302 current-file/source-hash comparisons, 12 sample-derived decode medians,
  384 replay records, nine speedup/efficiency pairs, six console hashes, serial
  run intervals and updated summary/manifest mappings. No TT device, server,
  reset, hardware test or target-model execution was used. The only written
  file is this report.
- Commit/push contract: local acceptance commit after passing review; never
  push. The pending acceptance commit and final status updates are expected
  owner completion steps, not evidence defects or reasons for another review.

## Residual Risk

- These are short-context representative-layer host-wall timings of prepared
  workloads. They do not establish end-to-end generation throughput, serving
  concurrency, long-context latency or full-stack memory use.
- There is one clean baseline/mesh run per representative kind. The QSA B1
  improvement is approximately 4.37% for this recorded workload; the evidence
  does not establish its variability across independent runs.
- Round 1 remains the review of the unchanged full-stage correctness, long
  context, watcher and profiler evidence. This follow-up closes its required
  repair and preserves its documented, nonblocking hand-off limits.
