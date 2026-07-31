# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- None that block Stage-02. The stage-owned files are still untracked in the
  live worktree; the stage-review workflow requires the owner to make the
  isolated local checkpoint commit after this clean pass.

## Hard-Check Gaps

- The normal/watcher XML and profiler artifacts do not embed source hashes.
  This is not a concrete stale-artifact finding: the current implementation
  hash is
  `99426b46058b9d08b6ad9757c455c0cd786f24b944c9779e3a28baf10dde36b7`,
  the current test hash is
  `cc7259751827ecb77358015cc73e1e300802d05913285970c3eabb02d4038d59`,
  the current test expands to the same 19 cases in both XML files, and the
  current exact-32 fused-reduction gate versus non-32 fallback is reflected in
  the measured batch-32 and sequence-128 profiler graphs.
- The final `moe_compute` and sequence-128 remediation records are summary
  JSON rather than retained raw stdout. Both contain exact commands, shapes,
  policies, thresholds, and results; the review contract accepts summary JSON,
  and neither summary contradicts source, tests, or the retained profiler
  evidence.

## Anomaly Ledger

- Observed anomaly: The added all-128-token active-expert diagnostic reports
  PCC `0.9876476411`, below the normal `0.995` stage gate.
  Evidence: `candidate_seq128_remediation_matrix.json`.
  Affected path: sequence-128 packed-all-expert fallback and experimental
  fused weighted reduction.
  Control or comparison: The unchanged `FunctionalDecoder` and selected fused
  fallback both produce exactly `0.9876476411`. Established Stage-02 MoE
  prefill cases remain layer 1 / sequence 1025 at `0.9995727457` and layer 4 /
  sequence 33 at `0.9997632031`.
  Likely subsystem: synthetic active-expert diagnostic sensitivity, not a
  fused-decoder regression.
  Investigation performed: Compared the functional control, selected
  fallback, whole-shape fused call, tiled calls, sparse variants, and
  higher-fidelity controls.
  Resolution: controlled.

- Observed anomaly: A whole-shape sequence-128
  `deepseek_moe_fast_reduce_nc_fused` candidate is faster but produces PCC
  `0.4088181766`.
  Evidence: `candidate_fused_reduce_prefill_seq128_pcc.json`,
  `candidate_fused_reduce_summary.json`, and
  `candidate_seq128_remediation_matrix.json`.
  Affected path: MoE prefill weighted expert reduction.
  Control or comparison: Four exact 32-token calls restore
  fallback-equivalent PCC (`0.9876487399`) but regress wall latency from
  `10.079501` to `10.410248501` ms. The selected source therefore uses the
  fused reduction only when `token_count == 32`.
  Likely subsystem: fused-reduction geometry outside its valid one-tile
  regime.
  Investigation performed: Whole-shape, four-tile, sparse-tile, one-token,
  higher-fidelity, and separate-projection variants were recorded.
  Resolution: fixed by reverting the invalid/slower experiments and retaining
  the exact-32 gate.

- Observed anomaly: The first graph audit dismissed `moe_compute` as
  multi-chip-only, although `compute_only=True` supports a 1x1 mesh.
  Evidence: current repository single-card test and
  `candidate_moe_compute_single_card.json`.
  Affected path: dedicated MoE expert-compute fusion.
  Control or comparison: The exact North geometry
  (E=128, tokens=32, top-k=8, H=2048, N=768, SILU) ran on one Blackhole, but
  the operation's internally fixed BFP4 weights produced expert-output PCC
  `0.992762` and `0.990510`, below the Stage-02 `0.995` acceptance bar before
  the still-required external routing-score combine.
  Likely subsystem: fixed low-precision weight contract in `moe_compute`.
  Investigation performed: The exact target geometry was executed rather than
  rejected by source inspection; README, work log, and graph audit were
  corrected.
  Resolution: controlled; the candidate is validly rejected on measured
  correctness at the subgraph output boundary.

- Observed anomaly: The first fused paged-cache update attempt faulted because
  K and V used overlapping grids.
  Evidence: `candidate_cache_update.json`, graph audit, current source, watcher
  results, and dense decode profiler rows.
  Affected path: decode KV-cache update.
  Control or comparison: The retained implementation places V on a disjoint
  core set with one required reshard, passes PCC and watcher validation, and
  improves dense batch-1 decode.
  Likely subsystem: fused cache-update NoC/core ownership.
  Investigation performed: Disjoint-grid remediation was tested and retained.
  Resolution: fixed.

- Observed anomaly: Python shutdown reports nanobind leaked instances/types
  after the test suite.
  Evidence: `pytest_full.log`.
  Affected path: process teardown only.
  Control or comparison: The same diagnostic exists in the accepted functional
  stage; all 19 tests finish first, devices and cluster close normally, and
  the watcher log has no device fault signature.
  Likely subsystem: binding teardown.
  Investigation performed: Ordering, functional-stage control, device close,
  XML results, watcher signatures, and final inventory were checked.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths:
  `.agents/prompts/model_bringup_multigoal/01b-fused-decoder.txt`,
  `.agents/skills/stage-review/SKILL.md`,
  `.agents/skills/graph-fusing/SKILL.md`,
  `.agents/skills/functional-decoder/SKILL.md`,
  `.agents/skills/tt-device-usage/SKILL.md`, `.agents/RUN-PLAN.md`, and
  `.agents/EXPERIMENT.md`.
- Prior review paths: `stage_review.md`, `stage_rereview.md`, and
  `stage_final_review.md`. Their findings were treated as leads and
  independently checked against the current tree and new controls.
- Code paths: `tt/fused_decoder.py`, `tt/functional_decoder.py`,
  `tests/test_fused_decoder.py`, `tests/fused_decoder_perf.py`,
  `tests/fused_decoder_capacity.py`, the single-card `moe_compute` unit test,
  and the relevant `moe_compute` implementation/binding sources.
- Documentation/artifact paths: fused-decoder README, work log, graph audit,
  candidate JSON/TXT files, normal and watcher XML/logs, latency JSON, capacity
  JSON, all nine fused and nine functional raw Tracy CSVs, filtered perf CSVs
  and tables, final `tt-smi` inventory, and `doc/context_contract.json`.
- Commands run: read-only `rg`, `sed`, `sha256sum`, `stat`, `jq`, and small
  read-only CSV/XML analyses. This review did not open or use TT hardware.

The normal and watcher suites each contain 19 tests with zero failures, errors,
or skips. The suite covers all representative layer kinds, non-aligned logical
lengths 1/31/32/33/65, MoE prefill at 1025 and 33, batch-1 and batch-32 trace
decode, cache permutation, nonzero positions, determinism, sliding history,
and real layer-1 weights.

The accepted PCC evidence exceeds `0.995` for every meaningful delivered path,
including dense batch-32 decode (`0.9998535227`), MoE batch-32 decode
(`0.9981931682`), established MoE prefill (`0.9995727457` and
`0.9997632031`), and official real-weight layer-1 decode (`0.9997505652`).

Re-derived signpost-filtered device totals show that every final fused row
beats its like-for-like functional row:

| Regime | Fused / functional device time | Gain |
|---|---:|---:|
| dense decode b1 | 301.775 / 338.413 us | 10.83% |
| dense decode b32 | 5662.809 / 6614.357 us | 14.39% |
| dense prefill b1 | 541.863 / 585.710 us | 7.49% |
| full MoE decode b1 | 2088.150 / 9438.668 us | 77.88% |
| full MoE decode b32 | 8233.539 / 11076.755 us | 25.67% |
| full MoE prefill b1 | 9960.917 / 14567.020 us | 31.62% |
| sliding MoE decode b1 | 2103.013 / 9451.795 us | 77.75% |
| sliding MoE decode b32 | 8247.599 / 11083.693 us | 25.59% |
| sliding MoE prefill b1 | 9998.234 / 14643.683 us | 31.72% |

Each raw profiler CSV has one start/end signpost pair; in-window raw global
call IDs equal the filtered report rows, summed kernel times reproduce the
documented totals, and no host operation appears inside the measured windows.
The runtime overrides contain no `torch`, `from_torch`, `to_torch`, or
functional fallback. The retained V reshard and layout transitions are tied to
the fused-cache and routing/operator contracts rather than redundant
user-authored movement.

Capacity evidence preserves the 500000-token contract: dense prefill reaches
500000; both MoE kinds reach the non-aligned 499999 boundary; all three layer
kinds trace-decode at position 499999; and dense serving batch 32 retains its
32.768 GB cache. No public alignment or capacity restriction was introduced.

## Residual Risk

- The isolated `moe_compute` rejection uses synthetic tensors and stops at the
  candidate's per-expert output because that output already misses the stage
  acceptance bar. If the operation later supports a higher-precision weight
  contract, it should be reconsidered; that future opportunity does not leave
  a currently valid Stage-02 fusion untried.
- Historical diagnostic commands reference temporary test selectors that were
  intentionally removed when experiments were reverted. The current permanent
  19-case suite, implementation topology, XML, watcher log, and profiler graph
  are mutually consistent, but a source-hash manifest would make future
  provenance checks cheaper.
