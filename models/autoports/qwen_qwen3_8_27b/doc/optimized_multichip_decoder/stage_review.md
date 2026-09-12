# Stage Review

Verdict: clean-pass

Independent review and rereview of Stage 5, optimized-multichip-decoder,
Qwen/Qwen3.8-27B, on the live `mvasiljevic/qwen38-full-bringup` worktree
starting at `1623f9cb595dd1e87cb17abc6f3449838c1762ee`.

Final reviewed model SHA-256:
`211b70db973294f7660ba9cd3c4ef0601ffbac32dab4866e3511b4804ff76213`.
Final measured runner SHA-256:
`22b17036b15d2c85626b4113f397d9f2ddc2274517ba5c9f116ae0b9196fa294`.
The initial more-work-needed report is preserved as `stage_review_initial.md`.

## Required Work

None. Both initial findings are closed:

- The short-prefill branch guards activation dtype equality. Independent
  comparison of actual old/repaired profiler CSVs shows four identity BF16
  casts removed for each layer kind on every rank. Current-source correctness,
  trace, stress, watcher, capacity and performance checks passed.
- Final metrics, accounting, context evidence, anomaly classifications and the
  evidence index are refreshed. All 66 indexed JSON hashes match their files.
  Final reports use the current source, empty model policy overrides, true
  `MeshShape(1,4)`, strict PCC, successful exits and
  `TT_MESH_PASS_THROUGH_THREAD_POOL=1`. Diagnostic or earlier candidate
  timings are not substituted for the final default.

Final unprofiled B1/S128 prefill/decode medians are 1.248232/.421990 ms for
linear attention, 1.177266/.306304 ms for full attention, and
2.347569/.704825 ms for their stack. These match `after_review_*` and README.
Decode uses warmed trace replay with one replay and synchronization per primary
sample. The reported 10.9–12.8% improvement uses this stage's true TP4 baseline.

## Other Concerns

- The native factory's internal C++ signature changes its symbol. Python's
  legacy fourth argument remains accepted, with a new optional fifth mesh
  coordinate. Both native components were rebuilt/installed together; final
  timing, watcher and profile provenance matches the installed-library hashes.
  Old binary consumers require a rebuild.
- Remote multi-reader coordinates are explicitly unsupported. Different
  harvesting maps and other architectures are source-reviewed, not validated
  by these local four-chip Blackhole tests. The limit is disclosed.
- Ordinary paired runs support the selected host pool setting: eager prefill
  falls 6.498%, 7.259% and 8.583% for full, linear and stack, while decode
  medians remain essentially unchanged. This is a local workload result.
- BFP4 KV diagnostics retain output/per-user/replay gates and reveal real
  raw-cache precision loss without a material short-context win. Final BFP8
  cache evidence retains strict state PCC. No synthetic-only precision veto
  was found.

## Hard-Check Gaps

- None blocks the stated stage. The tests do not certify heterogeneous
  harvesting, remote meshes or concurrent shared-CCL stacks; these are not
  claimed as validated.
- The index checks model hashes; separate source and installed-library
  provenance identify dependencies. These agree with final runs. Formatting
  preserved model, runner and native C++ hashes.
- Baseline artifacts precede the wrapper's exit-marker addition. Completed
  logs and repeated one-reader controls support comparison. Final acceptance
  and watcher runs have explicit exit markers.
- Watcher launch code removes all `TT_METAL_WATCHER_DISABLE*` variables.
  Saved environments and successful driver close support the full-watcher
  claim. Historical passing JSON followed by failed teardown is excluded.

## Anomaly Ledger

- Observed anomaly: four identity BF16 casts per short-prefill layer.
  Evidence: initial source/rows, repaired guard, `review_cast_audit.json`.
  Affected path: selected short prefill.
  Control or comparison: four-to-zero counts on all eight kind/rank pairs;
  refreshed strict correctness.
  Likely subsystem: model dispatch.
  Investigation performed: independent source and actual device-row review.
  Resolution: fixed.
- Observed anomaly: historical 2.009 ms traced-prefill median and .643/.910 ms
  signposted decode intervals despite stable device spans.
  Evidence: `AUTODEBUG_profile_gap.md`, `AUTOFIX_profile_gap.md`, saved
  host zones, untouched paired controls and diagnostic counters.
  Affected path: host completion wait.
  Control or comparison: untouched default control already gives .739 ms
  traced prefill; pool bypass removes sampled diagnostic tails; six final
  untouched profiles have .066389–.080633 ms host-minus-device spans.
  Likely subsystem: transient completion delay with pool/scheduling
  sensitivity; exact historical causation remains unproven.
  Investigation performed: C++ zone and source inspection, ordinary paired
  metrics and counter interval analysis. Broad scheduler snapshots are not
  treated as exact causal decomposition.
  Resolution: controlled with evidence and a measured local optimization.
- Observed anomaly: native mesh-coordinate and padding-only storage asserts.
  Evidence: AutoDebug/AutoFix reports, native diffs and final 18-case tests.
  Affected path: DRAM-sharded matmul.
  Control or comparison: readers 1/2/3, TP4/offset TP2, regular/tail/narrow
  outputs, distinct rank data and three retained allocation generations.
  Likely subsystem: physical placement and writer assignment.
  Investigation performed: coordinate adapter/bindings, physical lookup,
  full-shard allocation bounds and zero-write worker drain reviewed.
  Resolution: fixed for the supported local mesh.
- Observed anomaly: full-watcher model passes followed by teardown abort 134.
  Evidence: AutoTriage, ERISC assertion/PC captures, router diff and passing
  repaired traffic/close cases.
  Affected path: fabric handback.
  Control or comparison: empty mesh closes; workload traffic sets sticky
  tags; owned-NoC cleanup closes with all watcher checks enabled.
  Likely subsystem: packet-tag cleanup.
  Investigation performed: barriers, ownership, rendezvous and exit checks.
  Resolution: fixed for the tested fabric; assertions remain enabled.
- Observed anomaly: approximately 101% matmul compute utilization.
  Evidence: original tables, runtime reader attributes, supplemental CSV.
  Affected path: profiler interpretation.
  Control or comparison: eight assumed workers versus sixteen/twenty-four
  actual workers; all 112 final matmul rows show BF16/BFP4 and LoFi.
  Likely subsystem: profiler denominator.
  Investigation performed: runtime rows, native placement and accounting.
  Resolution: controlled; original rows and labeled correction retained.
- Observed anomaly: narrow conversion rejection, stale installed libraries,
  wrapper exit 2, shared-memory warnings and nanobind shutdown diagnostics.
  Evidence: stage anomaly ledger and respective failure/control artifacts.
  Affected path: rejected candidate and experiment environment.
  Control or comparison: padded/sliced candidate passes but loses; both native
  components installed; frozen wrapper exits 0; baseline timing reproduces;
  existing unit-mesh tests show the same shutdown diagnostics.
  Likely subsystem: the documented isolated mechanisms.
  Investigation performed: adaptations and recorded controls inspected.
  Resolution: controlled; failed artifacts do not establish final acceptance.
- Observed anomaly: boundary reporter counted Q/K norms as residual norms;
  text hooks exposed CSV line endings and patch whitespace.
  Evidence: final boundary CSVs, static logs and archive manifests.
  Affected path: reporting/packaging.
  Control or comparison: four 40-core residual norms identify the boundary
  on every rank; CSV normalization preserves parsed cells and exact originals;
  patches retain byte-exact gzip copies.
  Likely subsystem: reporting filters and formatting.
  Investigation performed: independent boundary reconstruction and final logs.
  Resolution: fixed/controlled; measured model semantics unchanged.

## Scope Inspected

- Goal/skills: original Stage 5 contract; local stage-review, optimize,
  tt-enable-tracing and tt-device-usage skills; `tech_reports/LLMs/llms.md`
  section 4.
- Code: model/runner changes and exact repair difference; inherited projection
  and logical-tail paths; trace, capacity, index, accounting, summary, profile
  and watcher helpers; GDN probe; native factory/header/nanobind/router repairs
  and regression tests; descriptor cache binding, completion queue and pool
  selection.
- Evidence: README/work log, anomaly/advice/cumulative/collective/inter-layer
  contracts; 66 final artifacts/hashes/exits; 31-case correctness suite; two
  100-step eager/queued stress cases; six prefill-trace cases; four full-watcher
  cases plus a traced-prefill watcher stack; five maximum-context probes;
  eight tail timing reports; six final per-device profile families; accounting
  and 112 runtime matmul rows; 212-row precision-locked geometry search;
  60 coherent topology results; precision/prefill/SDPA matrices; native
  build/install/test logs; AutoDebug/AutoFix/AutoTriage reports and archives.
- Final recorded checks: the native test's repository `expect_error`
  migration preserves the exception/message and passes 18 cases in 13.32 s.
  Applicable pre-commit hooks pass. `final_host_checks` exits 0 after
  gzip-aware accounting, the 66-artifact verifier and working/staged diff
  checks. These were run by the stage owner, not this reviewer.
- Reviewer commands: read-only Git status/diffs/checks, rg/sed/cat, and Python
  3 standard-library JSON/CSV/hash/statistical analysis. No tests, builds,
  TTNN imports, linked probes or device commands ran here. Only review reports
  were written; the initial report was preserved.

## Residual Risk

- This verdict covers the local TP4 decoder stage. It does not certify a
  complete model, serving integration, arbitrary concurrency or untested
  hardware/topologies.
- Non-aligned lengths, batches through 32 and B1 context 262,144 retain the
  stated coverage. Capacity evidence is decoder execution plus conservative
  resource arithmetic, not a claim of full-model construction.
- Actual final stack rows connect residual add directly to the next input
  norm on every device. No collective or reshard is inserted at that boundary.
  Coherent hidden-1280 alternatives carry sharded ownership through downstream
  consumers, with comparison-only gathers outside timing; their measured
  rejections satisfy the topology contract.
- Historical host variability is preserved and conservatively classified.
  Final performance claims use untouched default runs, excluding diagnostic
  counter instrumentation.
- Per the stage-review workflow, the owner must now create local stage-owned
  checkpoint commit(s), record branch/SHA in the work log and avoid pushing.
  This administrative checkpoint follows clean review and does not change
  the technical verdict above.
