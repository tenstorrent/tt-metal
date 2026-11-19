# Stage Review

Verdict: clean-pass

Stage 5, optimized-multichip-decoder, Qwen/Qwen3.8-Flash-Next. Independent
round 1 of 3, reviewing the live worktree on `main` above accepted checkpoint
`79582226b281a19aba1cc5ee3dd4fb80aa87317c`. This verdict passes the technical
review gate. The stage owner must still record this review and create/log the
authorized local stage-owned checkpoint commits before closing the goal.
The goal contract explicitly prohibits pushing.

## Required Work

- None. No failed, missing, stale, or contradicted mandatory acceptance gate
  was found in the inspected final evidence.

## Other Concerns

- The performance opportunities listed under `Not tried (stop rule)` in
  `work_log.md` remain useful handoffs. They include real GR-only HiFi2,
  shared-down/shared-route geometry and fidelity, weighted-reduction padding,
  PLE key/value packing, and additional prefill scheduling. They are not
  measured rejections. The recorded Stop Rule 2 is supported by the candidate
  JSON: QSA compact-output B32 improves 74.706688 to 74.268517 ms (0.5865%),
  and the final remaining-attention HiFi2 refinement improves 72.503128 to
  71.619102 ms (1.2193%). The goal and optimize skill explicitly allow these
  handoffs after the stop rule; they do not block this review.
- The context arithmetic is appropriately narrower than a full-model capacity
  claim. After the unchanged 2 GiB runtime planning envelope, the selected
  resident budget leaves 321,046,720 bytes per rank before IO. Two complete
  32,769-row residual slabs exceed that margin by 14,825,280 bytes. Stage 6
  must preserve bounded IO and verify actual construction/runtime allocation.
  The README, context contract and memory report disclose this consistently.

## Hard-Check Gaps

- No additional mandatory evidence format or hardware rerun is requested.
- The final collector correctly leaves stage completion, review and local
  commit pending; its `evidence_ready` flag alone is not stage acceptance.
- There is no full-model generation, qualitative text, token-output FUNCTIONAL
  throughput, or serving result. Those paths are outside this decoder-only
  goal. B32 is serial request attention with grouped MLP execution, as both
  source and reports state.

## Anomaly Ledger

- Observed anomaly: fresh before-run source digest differed from accepted
  stage-4 source.
  Evidence: `real_mesh_before_l{0,1,3}_provenance.json`,
  `before_source_delta.patch`, `before_source_explanation.md`.
  Affected path: before/after timing comparability.
  Control or comparison: accepted Git source plus the exact helper extraction.
  Likely subsystem: experiment preparation/refactoring.
  Investigation performed: independently reconstructed the complete before
  source in memory from Git and the patch; its SHA-256 exactly matches
  `e4f63391a796313ae2d1622820476c3002e0e7c65f7fa1fdc0f8c2a94fc54c46` in
  all three real-before provenance records. The expression, weights, device
  operations and collective order are unchanged; eager execution gains one
  Python helper call, while the captured device path is unchanged.
  Resolution: controlled.

- Observed anomaly: historical GDN LoFi run has `passed:false` although its
  whole-output PCC exceeds the goal threshold.
  Evidence: `real_dram_gdn-compact_LoFi_native.json`, `anomaly_ledger.md`, and
  the acceptance expression in `tests/probe_multichip_decoder.py`.
  Affected path: candidate selection/harness acceptance.
  Control or comparison: short minimum PCC 0.99932766, B32 whole PCC
  0.99907094, with the recorded diagnostic row below the old 0.995 row gate.
  Likely subsystem: legacy per-row acceptance policy.
  Investigation performed: inspected the historical metrics and final gate.
  The final gate enforces the explicitly requested whole-output 0.995 bar;
  row metrics remain visible and state, finite, tail, stack and exact replay
  checks remain. Final accepted runs all pass.
  Resolution: fixed; historical nonzero result is preserved and is not used
  as a whole-output precision rejection.

- Observed anomaly: initial DRAM geometry/API and constructor-binding
  experiments failed or failed to reach the intended candidate.
  Evidence: `anomaly_ledger.md`, `work_log.md`, `dram_geometry_v3_*`,
  `parallel_indexer_all_counts_v3.json`, `hidden_qsa_rs.json`, and
  `hidden_qsa_agmm.json`.
  Affected path: optimization-family rejection validity.
  Control or comparison: adapted geometry, corrected actual indexer binding,
  and complete hidden-layout decoder runs.
  Likely subsystem: candidate setup and physical program constraints.
  Investigation performed: checked final source binding, adaptation records,
  component contracts and whole-decoder results. Hidden QSA carries the
  fractured residual through GR and MoE instead of immediately restoring the
  original boundary; its B32 76.548355/76.450859 ms loses to the matched
  stream control at 75.122368 ms. Rejections do not rely on the first error.
  Resolution: fixed/controlled, with further distinct combinations handed off
  under the stop rule.

- Observed anomaly: final Tracy captures warn that the optional WASM-viewer
  copy source is absent.
  Evidence: the three `profile_final_mesh_l*.log` files, profile provenance,
  raw capture paths, and `evidence_manifest.json`.
  Affected path: optional viewer copy, not device execution or report data.
  Control or comparison: existing external raw CSV/capture files and local
  advice-enabled reports.
  Likely subsystem: default-path viewer copy with an explicit capture root.
  Investigation performed: all 122 manifest artifacts exist; raw CSV joins
  and six report sums reproduce the reported values. No warning/error-like
  lines were found in the final correctness, clean-timing or watcher job logs.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: `goal_contract.md`; installed stage-review, optimize,
  multichip, tt-device-usage, tt-enable-tracing and model-bringup skills under
  `/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills`;
  optimize method/evidence references and repository `TraceCorrectness.md`.
- Artifact paths: this directory's README, work log, topology audit,
  optimization checklist, profiler findings, anomaly ledger, evidence
  manifest, acceptance/performance summaries, final plan, measured policy
  rows, residual contract and memory plan; `../context_contract.json`;
  before/final clean measurements, tracked contracts, watcher runs, long
  context runs, positive control, component tests, precision/geometry and
  coherent topology experiments, command/source provenance, and raw/derived
  profiles under `tracy/` and the external persistent profile root.
- Code paths: `tt/multichip_decoder.py`, `dram_projection.py`,
  `parallel_ordered_projection.py` and all five new projection kernels,
  `pipelined_striped_gather.py`, `striped_gather.py`; the decoder/long-context
  probes, runtime-audit helper, component probes, MoE timing, job/provenance
  runners and candidate adapters, particularly the hidden-residual path.
- Commands run: read-only `git status`, `git diff`, `git show`, `rg`, `cat`,
  `sed`, and small Python standard-library analyses of JSON, CSV, patch and
  source data. No TTNN import, device access, server, test, reset, or hardware
  command was run by this reviewer. The only written file is this report.

The important acceptance claims were checked against source and runner
artifacts, not accepted from summary prose:

| Layer | Warmed prefill33, before → final ms | Traced B1, before → final ms | Traced B32, before → final ms |
|---|---:|---:|---:|
| L0 GDN | 21.678122 → 21.417217 | 2.267253 → 2.081643 | 57.348942 → 52.452465 |
| L1 PLE/GDN | 26.659092 → 25.954905 | 3.433150 → 3.136728 | 95.013763 → 87.054184 |
| L3 QSA | 79.820610 → 41.624786 | 4.753835 → 2.858865 | 131.109802 → 71.598373 |

The final timing medians recompute from the individual samples within stored
float32 rounding. The selected final candidates reproduce in the default
constructors, with no material regression. Logical32 and1024 paired prefill
and B1 measurements also pass and improve. Baseline/final commands use the
same reference paths, fixture, shapes, capacities and serialized regime.

Final tracked contracts pass 16 changing-input B1 replays and 16 alternating
B32 replays per kind, with exact own-eager output and mutable-state equality,
program-cache allocations included, all logical tails1..32, and two-layer
boundaries without conversion. Minimum tracked B32 whole PCC is 0.99889207;
minimum tail PCC is 0.99954396; minimum stack PCC is 0.99635768. Separate
watcher10 runs repeat the same tests successfully. Exact-tensor output
acknowledgments have one live decoder trace, are read before the next replay,
and are released at trace teardown; no broad allocation exemption is used.
The positive control rejects its deliberate live allocation and succeeds
after deallocation. Component tests retain all-shape exact indexer and
changed-input pipelined lookup coverage with the final custom source.

The actual final source matches all 240 recorded source files in the first
final run; relevant runtime/probe digests are identical across the 21 final
jobs. All 135 measured-policy records join to their raw operations with exact
dtype/fidelity/attribute matches and four-device siblings. All 156 rank rows
for selected native matmuls independently match the selected role policy;
the three SDPA invocations provide the remaining three four-rank policy
checks. Six phase totals independently sum to the published values. These
are warmed eager attribution totals, not substituted for trace wall latency.

Real B32 routes independently reproduce expert unions103/151/169 with32
distinct route rows. MoE share arithmetic reproduces4.13%/2.81%/3.57%; source
retains token-selected EP execution. Real BFP4 attention fails whole-output
accuracy, whereas BFP8 KV passes but increases matched B1 latency. Their
rejections are backed by target-weight/input results. This stage adds no new
BFP4 default.

Final32769 public-prefill plus continuation evidence passes with whole-output
minima0.99984349 and0.99917812, reverse page-table mapping for QSA, exact
position/page/token-history contracts, and actual resident reservations at
least31,710,200,640 bytes per rank. The506,757,120-byte weight increase accounts
for96 GR injections,36 GDN projections and48 shared experts; KV format and
262144 autoregressive context remain unchanged.

## Residual Risk

- This is inspection of saved evidence, not a new independent hardware run.
- Full-stack accuracy, construction peaks, simultaneous large batch/context,
  generation, serving trace lifetime and throughput remain later-stage gates.
  Later-layer real embedding inputs are routing stress, not recorded
  full-stack residual activations, as disclosed in the evidence.
- Preserve the stream-per-rank boundary and prepare all required execution
  signatures before future long-lived generator traces. The decoder tests
  establish their bounded trace lifetimes; they do not establish a serving
  generator's multi-request lifecycle.
- Large raw captures and reference tensors remain external persistent
  artifacts. Keep their manifest paths available during the local checkpoint
  and subsequent bringup stages. No push is authorized.
