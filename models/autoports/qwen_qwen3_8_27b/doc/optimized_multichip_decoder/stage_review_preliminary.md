# Stage Review

Verdict: more-work-needed

Independent preliminary review of stage05, Qwen/Qwen3.8-27B optimized multichip decoder, on the live `exp26-qwen38-bringup` worktree based on `ad9ed5d3c46f649ae4a0fa0804e899e7999d8180`. This review was requested while candidate runs were still executing. Final defaults, acceptance, profiles, context accounting, watcher, mixed traces, documentation and telemetry were explicitly unfinished. This is not final signoff.

## Required Work

- P2: Close the fused column-projection family at the target batch.
  Evidence: At review time every B8/B16/B32 fused result JSON used `fused=rs`; AG and AG+RS measurements were B1. `compact_boundaries32_results.json` shows that retaining compact column inputs reduces native-RS layer latency from 2723.00/1656.87 us to 2549.14/1482.30 us for linear/full attention. The column gather/matmul boundary is therefore material in the new B32 residual contract. B1 AG rejection does not establish rejection under this changed contract.
  Why this matters: The goal and optimize skill require coherent topology comparisons on the target workload. Native DRAM AG has a diagnosed implementation blocker, but the adapted interleaved-weight AG implementation is legal and was only measured at B1.
  Required next step: Compare adapted AG or combined AG+RS with compact hidden-sharded B32 residuals, preserving the winning native RS row projection where possible. Do not let a global weight-layout setting silently replace the row path. The stage owner acknowledged this finding and queued `column_family32` with an RS control, 1D/2D AG, K10/K20 and per-role interleaved AG weights. Inspect completed raw results before closing this item.

- P1: Integrate the selected B32 candidate's actual buffer and layout contract, then validate that implementation.
  Evidence: The latest native-RS candidate results were produced by `tests/multichip_fused_candidate.py`, whereas the intended production implementation is `tt/fused_decode.py`. At inspection, `MultichipDecoder.from_state_dict` created the B32 view with `persistent_ccl=True`; successful bounded-memory candidates used `persistent_ccl=False`. The production native helper always allocated L1 outputs and used conversion-before-compaction, and did not implement the candidate's `fused_output_dram`, `fused_compact_input` or `fused_residual_l1` options. This matters only for whichever options are selected; they need not all become production features.
  Why this matters: Candidate PCC and timing cannot validate a different integrated path. An accidental persistent gather/norm setting would also reintroduce the private L1 family whose 52 MiB reservation failed.
  Required next step: Freeze the selected candidate and implement exactly its used settings. Rerun final default correctness and timing. The stage owner acknowledged the mismatch and confirmed these adaptations and final runs are pending. Keep the final effective decode policy in evidence.

- P1: Complete the acknowledged final gates on the final source revision.
  Evidence: The README is explicitly in progress; constructors still selected stage04 defaults during this review. `doc/context_contract.json` ended at stage04. The retained workspace lifetime result had no source hash and predates moving TT_CCL ownership into the workspace; older guarded-SDPA evidence records workspace hash `b0973635a484da7ce9734edd83bb6046cd96ec1e2e2a9943cf23e3f5b735aab2`, while the inspected workspace hash is `552c6dab9e12bcc3449a7dccb7f18cf62298c276ee3baa1d372df67c1c28b75c`.
  Why this matters: New B32 residual dispatch, compact boundaries, fused buffers, larger prefill chunks and shared semaphore ownership change the applicable correctness, capacity and trace contracts. Earlier passing runs do not validate later ownership changes.
  Required next step: Finish final default B1/8/16/32 output/state PCC, affected short/nonaligned/chunk-boundary and continuation cases, native context, both stack orders without wrapper conversions, tracking-enabled shared-owner/mixed-batch replay including surviving traces, runtime audit, and separate watcher. Record final default warmed prefill/traced decode device timing, advice-backed reports and provenance. Update context/agentic memory accounting and stage documents. Request final independent review before stage closure; commit locally only after clean-pass, never push.

## Other Concerns

- The initial profile audit script only consumed B1 `final_profile_l{0,3}` and required four ordinary DRAM-sharded matmul roles. It cannot validate a distinct B32 fused implementation. The stage owner confirmed final single-layer B32 profiles for both kinds and a matching fused-row audit are planned; inspect these alongside B1 evidence. A full-stack profile is unnecessary.
- Shared semaphore/scratch ownership is explicit dependency injection: each independent constructor creates its own workspace unless `ccl_workspace` is passed. Document the full-stack construction contract accordingly. The stack harness passes the first owner's workspace to the second; do not describe independent constructors as automatically sharing a process-wide pool.
- B32 native L1 RS versus separate compact DRAM RS is a close comparison: 2549.14 versus 2558.22 us (linear), and 1482.30 versus 1488.22 us (full) in `compact_boundaries32_results.json`. Select using the completed integrated result and actual retained-memory cost, and publish reproduced default numbers.
- The report correctly corrected the 12 MiB capacity claim. Two interleaved [1,1,32,1280] outputs reserve 4096 bytes/bank/layer, so 64 layers require 27.5 MiB across 110 banks. The 28 MiB B1 reservation controls pass both kinds; the 32 MiB controls fail. These are allocator-aware feasibility probes, not measured full-model allocations. Native-context and final buffer accounting remain necessary.

## Hard-Check Gaps

- Final runtime tables must expose actual per-role weight dtype/fidelity/readers, norm/residual memory, SDPA config, collective memory/persistence, and inter-op gaps for each selected execution family. B1 ordinary matmul rows alone do not prove B32 fused rows.
- Final report/telemetry must use the final default profile, with tracking disabled for timing and enabled including program-cache allocations for trace acceptance. The profile script already separates these settings and unsets watcher.
- The existing mixed-trace probe is an ownership/exact-replay check with device-restored state, not an accuracy or performance result. Retain that scope, and combine it with final real-input advancing-decode and remapped-page results rather than treating repeated identical replay as all-purpose coverage.
- The final inter-layer contract should state per-batch hidden distribution, on-device memory layout, explicit workspace ownership, setup-before-capture, serialized complete out/down execution and teardown ordering.

## Anomaly Ledger

- Observed anomaly: Native DRAM-sharded fused AG timed out; native fused RS produced NaN correlations.
  Evidence: `fused_sharded_results.json`, preserved triage directories, `AUTODEBUG_fused.md`, `AUTOFIX_fused.md`, `fused_interleaved_control_results.json`, `fused_rs_native_repair_results.json`.
  Affected path: Experimental fused AG/MM and MM/RS.
  Control or comparison: Same-program interleaved weight controls pass; RS with output block width20 passes using native weights.
  Likely subsystem: AG receiver descriptor/K-tile handling; RS descriptor versus output-block width.
  Investigation performed: Source diagnosis, adapted controls and repaired RS measurements.
  Resolution: RS repaired; native AG has an identified source blocker, and adapted AG is legal. B32 adapted-AG performance decision remains required above. A NaN correlation is not claimed to prove literal tensor NaNs.

- Observed anomaly: SDPA chunk512 produced NaN PCC with a 136-entry table.
  Evidence: `sdpa_results.json`, `AUTODEBUG_sdpa.md`, `AUTOFIX_sdpa.md`, `sdpa_pad_aligned512_l3.json`, `sdpa_guard_guard512_l3.json`, `sdpa_guard_owned512_l3.json`, `sdpa_guard_remap_guard512_l3.json`.
  Affected path: Decode reads rounded complete chunks.
  Control or comparison: Explicit 144 mapped pages pass true512 but are slower; original capacity resolves to256 and passes; ragged B8/remapped pages pass.
  Likely subsystem: Page ownership beyond logical table extent.
  Investigation performed: Source diagnosis, explicitly owned padding and static capacity-divisor guard.
  Resolution: Controlled on candidate runs; preserve and revalidate relevant final paths. Native-context HiFi2/FP16 candidate also passes, but is not final default signoff.

- Observed anomaly: Private expanded B32 residual/scratch and fully persistent L1 family exceed capacity.
  Evidence: `AUTODEBUG.md`, `AUTOFIX.md`, `pool32_*`, `movement32_adapted_*`, `l1_scaling_results.json`, `l1_allocator_scaling_results.json`, `l1_exact_outputs_results.json`.
  Affected path: B32 expansion, all-reduce scratch and layer-owned retained buffers.
  Control or comparison: Rectangular expanded layouts, explicit two-slot workspace, compact rows and allocator-aware reservation controls.
  Likely subsystem: Physical tile expansion and per-bank allocation granularity.
  Investigation performed: Source/capacity arithmetic, adapted hardware paths and bounded-memory controls.
  Resolution: Candidate repairs pass. Final integrated ownership, context capacity and async composition still require current-revision evidence.

- Observed anomaly: BFP8 attention candidate has value-cache PCC below0.995 while output PCC passes.
  Evidence: `precision_attn_bf8_l3.json`, `combined_attention_bf8_hifi2_l3.json` and sweep results.
  Affected path: Experimental higher-precision attention weights relative to the BFP4 reference.
  Control or comparison: Full 128-step real-weight decode and lower-latency BFP4/LoFi control.
  Likely subsystem: Intermediate quantization-policy differences.
  Investigation performed: Separate weight/fidelity trials and cache-consuming replay.
  Resolution: Controlled, slower candidate rejected. This is not evidence that BFP8 is generally inaccurate and is not a synthetic veto of a faster lower-precision path.

## Scope Inspected

- Goal: `/home/mvasiljev/agentic-research-data/experiment-26/bringup-logs/qwen38-27b/20261023-astra-medium-exp26-clean/05-05-optimized-multichip-decoder.prompt.txt`.
- Skills: installed stage-review, optimize, agentic-serving, tt-enable-tracing, tt-device-usage and model-bringup entrypoint; repository `TraceCorrectness.md`.
- Artifacts: README, work_log, topology_audit, geometry_contract, AutoDebug/AutoFix reports, baseline advice-backed profiles; candidate policies/results and selected raw run JSONs for geometry, readers, prefill, SDPA, topology, pooling, fused/native repair, B32 compact boundaries and L1 scaling. Parsed all result-family JSONs to enumerate failures and checked successful rows for inconsistent PCC status.
- Code: `tt/multichip_decoder.py`, `tt/decode_workspace.py`, `tt/fused_decode.py`, changed/shared `tt/optimized_decoder.py`; main runner, validation suite, mixed-trace/workspace probes, fused candidate, profile/report/audit scripts and `models/common/modules/tt_ccl.py`.
- Commands: read-only `cat`, `sed`, `nl`, `rg`, `wc`, `git status`, `git rev-parse`, `git diff`, plus host-only Python JSON/hash/arithmetic inspection. No TTNN import, hardware operation, reset, server, full-model execution, source edit or subagent launch. Only this report was written.

## Residual Risk

- This is a moving live worktree. Candidate runs and subsequent implementation changes after the inspected revision are outside this preliminary verdict.
- Source inspection found no additional confirmed model-numerical defect beyond the already diagnosed candidate failures. It cannot establish async hardware ordering or final capacity without the pending acceptance runs.
- Rejection coverage is substantial: real weights/recorded inputs, BFP4/LoFi geometry sweeps, wider-K and reader adaptations, packed/split projections, direct AR, hidden-sharded consumers, precision/activation/CCL combinations and explicit SDPA ownership controls. The target-batch AG gap is a focused remaining comparison, not a demand to restart that search.
