# Stage Review

Verdict: clean-pass

Independent final review of stage 04, `multichip-decoder`, on 2026-10-31.
Reviewed the live worktree on `mvasiljevic/gemma4-31b-fresh-pipeline-v0811`,
based on `87140921cfc65695b74f0981b3c19b4182e46e61`. The target is
`google/gemma-4-31B-it` at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. Paths below are relative to
`models/autoports/google_gemma_4_31b_it` unless stated otherwise.
The initial findings are preserved in `doc/multichip_decoder/stage_review_initial.md`.

## Required Work

None. The six initial findings are closed by the final implementation and
artifacts, including the corrected packed-prefill comparison discovered during
follow-up review.

| Initial finding | Closure evidence |
|---|---|
| Frozen-source acceptance | All 34 `final_layer*.json` cases have adjacent successful child exits. Production, optimized baseline, and acceptance-runner hashes match the reviewed files. Both kinds cover B1/8/16/32, lengths 31/32/33/1023/1024/1025/2049, stress, watcher, and 262143/262144-token context. `stack.json` passes direct sliding-to-full attention with no boundary conversion. |
| Cyclic decode through wrap | `final_layer0_wrap2047.json` starts at 2047 with B32 distinct recorded inputs and advances 128 times through the modulo-2048 boundary. One capture is retained; every replay is asserted above 0.995. Minimum PCC is 0.9999211706, final PCC 0.9999322413. |
| Persistent selected-family collectives | Ring RS/AG controls were measured and persistent buffers retained. Production allocates separate staging/output buffers at construction for both row projections and B1/8/16/32, before capture. Buffers remain decoder-owned and are overwritten before consumption. Final runtime rows show persistent RS/AG; final tracked cross-cache stress passes. |
| TP4 packed versus split gate/up | Corrected `finalpacked8` controls write BFP8 directly with the same BFP4/LoFi and M4/K8/N16 prefill configuration as split. All four B1/B32, sliding/full controls pass. Complete prefill plus 128 decode steps favors packed by only 0.21–0.29 ms at B1 but favors split by about 3.12 ms at B32. The balanced split policy is supported. The earlier conversion-heavy comparison is explicitly superseded. |
| Same-shape reader controls | `candidate_layer*_final{q1,o1,gate1,down1}_cyclic.json` compare one reader with the selected two on identical padded shapes and precision. Two readers win. The legal QK three-reader control loses; per-bank widths of 16 or 22 tiles exclude a third reader for the other selected shapes. Padded-6144 alternatives were measured separately. |
| Final performance and record reconciliation | All eight `tracy_final/<kind>/batch{1,32,16,8}/` profiles and both-phase TXT/CSV reports exist. Raw CSV recomputation reproduces every prefill window and mean of 128 complete decode windows. Source and replay-selection hashes match. Performance, acceptance, memory, context, agentic-workload and specified telemetry records agree. |

The lowest recorded stress PCC across the final suite is 0.9998378276.
Runtime acceptance guards reject host conversions inside prefill/decode and
enable exceptions on fallback. Cache ownership, untouched unused pages, local
heads, shuffled page IDs, non-aligned lengths, and absolute positions are covered.
Trace allocation tracking includes program-cache allocations; no broad allocation
exemptions are installed. Stress changes positions and page-table row ordering and
interleaves a second warmed cache's prefill while the original trace stays live.

The reviewed TP4 mapping is internally consistent: replicated BF16 residuals;
column-parallel Q/K/V and MLP intermediate channels; row-parallel O/down;
eight local Q heads, four sliding KV heads of width 256 or one full-attention KV
head of width 512; per-shard zero padding 5376 to 5632 with matching down rows.
Reductions precede post-projection norms. Decode preserves the eight-core
width-sharded L1 residual interface. The bounded sliding cache uses private
2048-token pools while retaining absolute RoPE/mask positions and the advertised
262144-token logical context.

The required 4096-input/128-decode/B1 whole-layer device results are:

| Kind | TP4 prefill / decode | Speedup prefill / decode | TP4 efficiency prefill / decode |
|---|---:|---:|---:|
| Sliding attention | 11422.5364 / 484.7667 us | 1.9185x / 1.9780x | 47.96% / 49.45% |
| Full attention | 13748.4727 / 534.7718 us | 2.3015x / 2.0098x | 57.54% / 50.25% |

All default-batch comparisons use the selected optimized single-chip baseline,
including `tracy_final_k32` for full attention. Device windows include every
layer operation and internal device gaps, use the maximum per-device span rather
than assuming synchronized chip clocks, and exclude gaps between separate
replays. Each profile has 128 replays with the complete captured op-ID set.
Each decode report marks all five material matmuls Optimized. Runtime rows
verify BFP4 weights, LoFi, two readers, and the selected Ring collective family.
Host candidate timings are separately labeled and are not substituted for final
device measurements.

## Other Concerns

- Prefill remains materially below theoretical peak. The report's SLOW rows and
  normalization, conversion, SDPA and collective costs are disclosed in
  `profile_analysis.md`. Precision-locked K14, K14/N8 smaller-grid and
  padded-6144 controls plus the concrete K21 L1 limit support the selected
  geometry. This is not a claim of peak prefill performance.
- The selected path still performs BFP8-to-BFP8 copies at prefill collective
  boundaries. Final measurements include these operations; the stage claims
  no unmeasured removal benefit. This is a future optimization opportunity,
  not an unsupported final performance claim.
- The context/workload status and packet review note are deliberately gated on
  this independent verdict by `summarize_multichip_stage.py`. Refreshing those
  statuses and creating the required local checkpoint follow this review.
  No additional numerical evidence is missing.

## Hard-Check Gaps

- Acceptance JSON is written before the runner's final assertion. I therefore
  checked the adjacent child exits, all zero, rather than treating JSON
  existence as a pass. The summary script now also checks those exits.
- Manifests do not hash the recorded input fixture. Pinned reference loading,
  fixture paths, and real recorded-activation use are visible in source; no
  evidence indicates a mismatch. Hashing the fixture would improve auditability
  but is not a new stage gate.
- All acceptance manifests still name the pre-follow-up candidate-file hash.
  The only later change is the isolated `packed_direct_bfp8` experiment branch.
  Final acceptance has `candidate: null`, and its runner imports that file only
  when a candidate is selected. Production and acceptance-runner hashes match,
  so this candidate-only edit does not invalidate final acceptance.
- The stress runner stores minimum and final PCC, not all 128 PCC values.
  Its explicit assertion checks every replay against 0.995, and successful
  exits plus the exact minimum establish the requested repeated-replay gate.

## Anomaly Ledger

- Observed anomaly: Mapped upload of 4096-row activations stalled before layer work.
  Evidence: `AUTODEBUG.md`, `AUTOFIX.md`, `sliding_rep4096_hosttilize.json`.
  Affected path: setup transfer.
  Control or comparison: host tilization followed by device transfer succeeds for both baseline and TP4.
  Likely subsystem: mapped upload/device tilization.
  Investigation performed: staged upload isolation and owner-checked recovery.
  Resolution: controlled by the setup transfer path; no kernel repair is claimed.

- Observed anomaly: BFP8-CCL prefill produced nonfinite values; the diagnostic also retained tensors until L1 exhaustion.
  Evidence: `AUTODEBUG_CCL8.md`, `sliding_nan_diag2.json`, `sliding_ccl8_barrier.json`, final acceptance.
  Affected path: prefill synchronization and diagnostic ownership.
  Control or comparison: BF16 CCL passed; a plain callback removed Mock retention; barrier-enabled ordinary BFP8 passed.
  Likely subsystem: asynchronous collective ordering plus a separate diagnostic lifetime bug.
  Investigation performed: finite-component controls, lifetime inspection, RS barrier adaptation, tracked stress and worker watcher.
  Resolution: fixed/controlled on the final barrier-enabled path; historical diagnosis is linked to its passing controls in the work log.

- Observed anomaly: Bounded sliding storage exceeded fused-update logical/physical page-count validation.
  Evidence: cache discussion in `work_log.md`, modulo-update production branch, wrap and maximum-context final artifacts.
  Affected path: cyclic cache updates.
  Control or comparison: two generic modulo-aware updates and modulo-aware SDPA pass against the unbounded optimized baseline.
  Likely subsystem: fused-update API contract.
  Investigation performed: adapted page-table/update path with private shuffled pools and absolute positions.
  Resolution: controlled; bounded sliding uses supported generic operations without reducing logical context.

- Observed anomaly: Fused matmul/RS stalled on Linear fabric.
  Evidence: `triage_fused.txt`, `AUTOTRIAGE.md`, `sliding_fused_ring.json`.
  Affected path: rejected fused row-projection family.
  Control or comparison: matching Ring fabric/topology passes at about 870 us host traced decode.
  Likely subsystem: fused collective/fabric configuration.
  Investigation performed: live triage, padded sharded-residual adaptation through consuming distributed norms, matching-fabric retry.
  Resolution: controlled; the coherent passing family is rejected for measured latency, not its original hang.

- Observed anomaly: Adapted fused AG/matmul BFP8 projection output missed prefill PCC.
  Evidence: `sliding_fused_ag_adapted2.json` (0.9849548692), `sliding_fused_ag_bf16.json` (0.9999389716).
  Affected path: rejected fused-AG/sharded-residual family.
  Control or comparison: BF16 projection output repairs PCC; both versions take about 1095 us host traced decode.
  Likely subsystem: projection-output precision within that family.
  Investigation performed: real-weight precision control with compatible sharded consumers and adapted sender geometry.
  Resolution: controlled; the legal BF16 family loses the measured workload.

- Observed anomaly: Original packed-prefill comparison included an avoidable BF16-to-BFP8 output conversion.
  Evidence: candidate implementation and `candidate_layer{0,5}_finalpacked8{,_batch32}_cyclic.json`.
  Affected path: optimization-selection evidence, not production correctness.
  Control or comparison: direct-BFP8 packed output with the same configuration as split passes all four cases.
  Likely subsystem: inherited projection-name dtype selection.
  Investigation performed: source audit, explicit candidate-only output dtype, complete-layer real-weight reruns.
  Resolution: fixed; corrected evidence supports the retained split policy and supersedes the original rejection.

- Observed anomaly: Fully instrumented ACTIVE_ETH kernel needs 29264 bytes against a 26624-byte configuration buffer.
  Evidence: `watcher_configuration.json`, `watcher_eth_limit.exit.json`, `recovery_watcher_eth.json`, final watcher exits.
  Affected path: Ethernet watcher instrumentation build.
  Control or comparison: both worker-core watcher cases pass with `TT_METAL_WATCHER_DISABLE_ETH=1`.
  Likely subsystem: instrumentation configuration-buffer capacity.
  Investigation performed: exact build-limit capture, owner-checked recovery, prescribed scoped retry, ordinary numerical/stress validation.
  Resolution: controlled using the skill-prescribed mitigation; Ethernet watcher coverage remains explicitly unavailable.

- Observed anomaly: Prefill advice says “No program_config specified” and marks MinimalMatmul rows SLOW.
  Evidence: final prefill tables, raw runtime attributes, `profile_analysis.md`, precision-locked geometry controls.
  Affected path: performance diagnosis.
  Control or comparison: runtime attributes contain M4/K8/N16, subblock 1x8 and 11x10 grid; measured alternatives do not improve the layer.
  Likely subsystem: perf-report MinimalMatmulConfig parser, alongside genuine non-peak utilization.
  Investigation performed: runtime/config comparison, candidate inspection and complete-window cost accounting.
  Resolution: controlled; the missing-config advice is inaccurate, while actual low utilization is retained and explained.

- Observed anomaly: A completed stress run initially failed serializing MeshTraceId; profile logs also warn that an optional Wasm-viewer trace copy was missing.
  Evidence: `recovery_stress_json.json`, successful final stress exits, final `profile.log` and profile report directories.
  Affected path: artifact serialization and optional viewer export.
  Control or comparison: trace ID is now string-serialized; raw Tracy files, device CSVs, host capture metadata and selected-replay provenance exist for all final profiles.
  Likely subsystem: result writer and optional export location.
  Investigation performed: result-writer repair and rerun; independent raw-CSV latency and SHA256 checks.
  Resolution: fixed/controlled; neither warning represents lost final numerical or device-window evidence.

## Scope Inspected

- Contracts and skills: repository `AGENTS.md`, `bringup/run_contract.json`,
  `bringup/agentic_evals/inputs.json`; installed stage-review, multichip,
  optimize, agentic-serving, tt-enable-tracing, tt-device-usage, model-bringup,
  and tt-review-core instructions. Bringup skill root:
  `/home/mvasiljevic/.codex-personal-gemma4/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.22/skills`.
- Source: complete production multichip decoder; inherited optimized forward,
  normalization, cache, allocation and projection paths; candidate, acceptance,
  validation, stack, profiling, replay-selection, summary, memory-plan and
  recovery scripts; pinned weight loading and trace helpers; generic collective
  persistent-output implementations and local multi-device/trace guidance.
- Artifacts: final 34 acceptance JSONs/manifests/exits; stack result; all eight
  final profile manifests, raw CSVs, replay-selection provenance, runtime rows
  and TXT/CSV reports; selected single-chip baselines; topology, precision,
  geometry, packed and reader controls; diagnosis/recovery evidence; README,
  mesh plan, work log, memory/context/workload records and available hardware.
- Exact telemetry packet:
  `bringup/artifacts/multigoal-runs/20261030T163027Z/continuation-02-20261030T182350Z/telemetry/packets/9630fa21-6741-4688-bae0-7db3f63850d7.json`.
  Both separate-kind B1 headline metrics and roofline values match the final
  profiles; workload is 4096/128/B1 and missing-reason fields are empty.
- Frozen SHA256s:
  `tt/multichip_decoder.py` =
  `e798abcf508367ba7f526135e99bf167f475ce6ad5c17cbc52c2399e68afb73c`;
  `tt/optimized_decoder.py` =
  `5267f1ee57d0f077e71357445e96e15b3851b022c5846eeb10b77566eb4b6ff6`;
  `tests/run_multichip_decoder.py` =
  `f872f4c7f8141f0ff546148e1262391ac07f2f2e7b2c312055fb2cb5d648a982`;
  `tests/profile_multichip_decoder.py` =
  `618685ca0878302dc0a0ad4c28bf3f53cdc193d798a4cfe0b654e640c9828cb1`.
- Review actions: read-only file/Git inspection and small standard-library
  artifact analyses. No devices, resets, servers, TTNN imports, long tests,
  publication or external messaging. Only this review report was written.

## Residual Risk

- Ethernet watcher instrumentation is unavailable for this build. Worker-core
  watcher, numerical/cache checks and tracked stress passed; these do not claim
  equivalent Ethernet instrumentation coverage.
- Maximum-context tests execute the complete prefill, compare its final 32
  output rows, and decode at position 262143. They do not compare every
  long-prefill output row. All four 262143/262144 cases pass without a context cap.
- Full-stack capacity remains an analytical plan reconciled to measured layer
  allocations. It includes 471091200 bytes/device of setup overhead, including
  persistent CCL buffers, and estimates 11514775552 bytes/device with cyclic
  sliding KV and normal reserves. B32/32768 streamed working-set capacity is
  not a measured full-model or serving result.
- Roofline percentages use explicitly documented useful-work and physical-I/O
  estimates against whole-device theoretical peaks; they are not hardware
  utilization counters. Final latency claims use measured complete windows.
- This stage contains a dense decoder, not a generator or serving adapter.
  End-to-end token feedback, full-model quality, serving registration, and
  cross-request serving behavior are later-stage obligations.
- The verdict applies to the inspected frozen production and harness. Subsequent
  semantic changes require the affected evidence and review to be updated.

Final-source provenance follow-up, 2026-10-31: clean-pass reaffirmed after the
checkpoint hooks and final import repair. I independently checked all ten entries
in `formatting_bridge.json` against the original measured Git blobs retained from
the first comparison and the final files: both byte hashes and the unchanged
non-import AST hashes match. I also inspected the import ordering and routing.
The hooks removed production-module HF model/revision exports that the profile
harness imported. The final repair imports those constants directly from
`optimized_decoder.py`, where the unchanged pinned values are defined, while
continuing to import `MultichipDecoder` from this stage's implementation. This
supersedes the intermediate explicit-re-export attempt and fixes the import issue.
The profile summarizer, replay selector and candidate implementation are unchanged.
The hashes listed above identify the measured sources; the bridge identifies the
final equivalent sources. Final SHA256s are:

- `tt/multichip_decoder.py`:
  `bd32984c751bd0c7b7800e0144207cba6442473f4a15587cd47328e6467fb07c`.
- `tests/run_multichip_decoder.py`:
  `856a955bb0d9d519ef5a76476c51a7bb07fe6fb7d7437a5c0d55ff1193b6af43`.
- `tests/profile_multichip_decoder.py`:
  `fde48e743333cbc3957c73c22c07ba3b04c5d0a44438c7638164cb4a39d9035f`.

These formatting and import-only changes preserve the inspected execution paths
and pinned reference, so existing acceptance and profile evidence remains valid.
No device rerun or new required work is warranted by this final diff.
