# Stage04 Multichip Decoder Stage Review

Verdict: clean-pass

Reviewer mode: direct independent reviewer using the local `stage-review` skill. I did not spawn a sub-reviewer, open devices, import TTNN, run hardware tests, reset/profile devices, or edit implementation. This report is the only mutation made by the review turn.

## Required Work

None.

The final Stage04 multichip-decoder evidence satisfies the original contract for the scoped runtime, tests, docs and context artifacts. I found no correctness, context, cache ownership, trace, watcher, stack-interface, active-expert, or selected-profile evidence gap that requires more work before the stage can pass.

## Clean-Pass Basis

- Source provenance is consistent. The reviewed runtime is `tt/multichip_decoder.py` at SHA256 `a12a913cf752b765338736dc71f71151ab972af1529a0098455755dd4f499255`, matching `source_provenance.json`, final validation JSONs, stress artifacts, watcher artifacts and profile runs. The optimized baseline hash is recorded as `5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898`.
- The runtime default policy matches the selected policy: 1x4 mesh, TP4, Linear topology, hybrid EP prefill plus indexed TP decode experts, grouped MoE reduction, fused tail, optimized shared MLP, selected LoFi QKV/output fidelity, BF16 sliding attention CCL, BFP8 full attention CCL, sliding BFP8 grouped MoE CCL, and local paged KV caches.
- Source inspection re-derived the important contracts: `(1,4)` mesh enforcement, Blackhole worker-grid check, unrestricted logical lengths over the fixed 1024 internal chunk size, context bound checks against `max_position_embeddings`, replicated `[1,1,S,2816]` interface, current-position tensors owned by the runtime, chunk/page-table slicing for prefill, decode cache position use, grouped shared/routed MoE reduction, and indexed decode expert setup.
- Active expert selection is preserved. Runtime source calls `enable_indexed_decode(...)`; final native `whole_layer.json` files report sparse decode reads with `active: 8`, `indexed: true`, and compact output group dimensions. Dense all-expert decode is not the selected path.
- Baseline correctness passes for both layer kinds. `final_validation_summary.json` reports all required final checks passing on the reviewed runtime hash. The 4096 input / 128 decode traced stress artifacts pass with duplicate replays: sliding min decode PCC `0.9976112404200184`, min cache PCC `0.9999966584468448`; full min decode PCC `0.9994477563467595`, min cache PCC `0.9999715022296233`.
- Context preservation is covered. `context_contract.json` and `memory_capacity_plan.json` preserve advertised context `262144`, `capability_reduction: false`, and `logical_length_alignment_required: false`. Final max-context artifacts cover both `262143+1` nonaligned traced decode and `262144+0` prefill for sliding and full layers with resident full-stack reservation bounds and current runtime hash.
- Cache, positions, page ownership and stack interface are covered. `test_multichip_contracts.py` exercises heterogeneous batch32 logical lengths, prefix preservation, page table/current-position/cache-position rebinding across trace replay, and other-slot preservation. `stack_final_policy.json` passes two-layer sliding/full direct handoff with independent caches, one trace scope, repeat-trace equality and all replicas equal.
- Watcher evidence is clean. `final_watcher_summary.json` records sliding and full watcher runs with exit code 0, watcher checks enabled, noinline enabled, and passing PCCs on runtime hash `a12a913cf752b765338736dc71f71151ab972af1529a0098455755dd4f499255`.
- Final target workload profiling is present for both layer kinds. `profile_final_sliding/whole_layer.json` and `profile_final_full/whole_layer.json` both set workload `{input_tokens: 4096, output_tokens: 128, batch: 1, concurrency: 1}`, `target_workload_measured: true`, and selected mesh `[1,4]`. The final device windows are sliding `325830.533 us` prefill and `705.420 us` decode; full `309683.707 us` prefill and `766.584 us` decode.
- Whole-layer roofline denominators include gaps and all device operations. I inspected `tests/summarize_multichip_perf.py`: it collects rows between `PERF_*` and `PERF_*_END`, groups by replay session/device, computes each device window from first firmware start to last firmware end, then uses the maximum device span per replay. The JSON assumptions explicitly state all padding, inactive prefill-union work, normalization/transcendentals and gaps remain in the time denominator. `capture_integrity.json` passes for both profiles with 128 decode replay sessions per device, stable op sets, and no invalid firmware spans.
- Profile provenance is adequate. Both final profile directories contain native-derived `whole_layer.json`, `whole_layer.windows.csv`, passing `capture_integrity.json`, `op_accounting.json`, `perf_report_commands.json`, compressed `*_perf_report.csv.gz`, compressed `*_table.txt.gz`, and SHA256 provenance. The final `tt-perf-report` commands use explicit `PERF_PREFILL_END` and `PERF_DECODE_END` filters, and `final_signpost_filter_check.json` confirms merged table row counts match native phase counts. The final telemetry packet mirrors the same workload, mesh and whole-layer profile numbers.
- Optimization rejection is sufficient for this stage. `final_policy_alternatives.json` compares matched final-policy controls for sliding sparse gate/down/shared options and sharded-residual/Ring/fused-AGMM controls for both layer kinds; all measured alternatives are slower than the selected default while preserving correctness. Broader historical candidates and advice handling are documented in `optimization_advice_audit.md`.

## Other Concerns

- Final matched optimization controls are representative rather than exhaustive. Given the stage goal, the selected path has current correctness, stress, watcher, memory and native-profile evidence, and slower matched alternatives are recorded. I did not require another exhaustive sweep absent a concrete selected-policy risk.

## Hard-Check Gaps

- I did not run any hardware, device reset, profiling, or TTNN import in this review, by explicit instruction. Hardware-dependent correctness and performance are accepted from the final artifacts only.
- Real full-model allocation order and fragmentation remain deferred to the full-model stage, as stated in `context_contract.json`. Stage04 validates each real layer with anonymous full-stack resident reservations; it does not claim full-model generation or vLLM readiness.
- Generic `tt-perf-report` unclassified-op warnings remain a limitation of those tool subtotals. The accepted denominator path is the native whole-layer summarizer, not the generic tt-perf-report utilization percentages.

## Anomaly Ledger

- Anomaly: Final profile generation was still in progress during review.
  - Evidence: A `profile_final_full` post-processing process was observed, then later exited; both final profile directories subsequently contained whole-layer, integrity, compressed report and provenance files.
  - Affected subsystem: Profile packaging.
  - Investigation performed: Re-polled process state and final profile file trees, then inspected `whole_layer.json`, `capture_integrity.json`, `perf_report_provenance.json` and command logs.
  - Resolution: Controlled; final artifacts are present and stable at report time.
- Anomaly: Generic `tt-perf-report` signpost warnings were present in an earlier profile postprocess pass.
  - Evidence: Review-time `prefill_csv.log` and `decode_csv.log` first showed identical start/end signpost filters; compact prior outputs are preserved under `profile_final_{sliding,full}/prior_signpost_filter/`.
  - Affected subsystem: Supporting tt-perf-report tables.
  - Control or comparison: Current final `perf_report_commands.json` uses `PERF_PREFILL_END` and `PERF_DECODE_END`; `final_signpost_filter_check.json` confirms phase row counts; native `capture_integrity.json` sees both end signposts; `summarize_multichip_perf.py` uses the explicit end signposts for accepted whole-layer windows.
  - Resolution: Fixed before final commit; generic tables remain supporting evidence only.
- Anomaly: Historical router-placement full-layer divergence under an old full-layer BFP8 path.
  - Evidence: README anomaly section and `AUTOFIX_full_router1_bfp8.md`.
  - Affected subsystem: Router placement and sparse MoE decode.
  - Control or comparison: Final runtime isolates router placement at `(10,9)` and passes full-layer stress, watcher, stack and profile capture integrity.
  - Resolution: Controlled in selected runtime.
- Anomaly: Historical sharded RoPE precision/configuration failures.
  - Evidence: README anomaly section and `AUTOFIX_sharded_rope.md`.
  - Affected subsystem: Sliding decode RoPE.
  - Control or comparison: Final default uses the validated selected RoPE policy; final sliding stress, cache checks and watcher pass.
  - Resolution: Controlled; slower/invalid variants are not selected.
- Anomaly: Device-converted sliding BFP4 expert packing failed PCC historically.
  - Evidence: README anomaly section and `AUTOFIX_final_policy_packing.md`.
  - Affected subsystem: Sliding expert GU packing.
  - Control or comparison: Final runtime uses raw host BFP4 packing and passes final sliding stress, max-context, watcher and profile evidence.
  - Resolution: Controlled in selected runtime.
- Anomaly: Native profiler overflowed at larger capture count historically.
  - Evidence: README anomaly section and `AUTOFIX_profile_selected_abort.md`.
  - Affected subsystem: Profile capture capacity.
  - Control or comparison: Final 100k profile captures close cleanly with per-device/replay metadata and passing capture integrity.
  - Resolution: Controlled for the final profile workload.

## Scope Inspected

- Goal and skill instructions: original Stage04 prompt, `stage-review`, `multichip`, `tt-device-usage`, `optimize`, and relevant LLM multi-device best-practice notes.
- Runtime and tests: `tt/multichip_decoder.py`, `tests/run_multichip_decoder.py`, `tests/test_multichip_contracts.py`, `tests/test_multichip_stack.py`, `tests/runtime_audit.py`, and `tests/summarize_multichip_perf.py`.
- Final docs and artifacts: `README.md`, `work_log.md`, `selected_policy_audit.md`, `source_provenance.json`, `mesh_plan.md`, `memory_capacity_plan.json`, `../context_contract.json`, `final_validation_summary.json`, `runtime_fallback_audit.md`, `optimization_advice_audit.md`, `final_policy_alternatives.json`, `final_watcher_summary.json`, final stress/max-context/batch/stack artifacts, final profile directories, and telemetry packet `bff5abf7-6db4-4696-8fc0-a904af9d0396.json`.
- Read-only commands used: `sed`, `grep`, `find`, `jq`, `sha256sum`, `git status`, `git diff --stat`, `git rev-parse`, `ps`, `head`, `tail`, and `wc`-style file inspection. No device or TTNN command was run.

## Residual Risk

Residual risk is limited to hardware-artifact trust, full-model allocation behavior deferred to the full-model stage, and generic tt-perf-report subtotal limitations. Within the Stage04 multichip-decoder scope, the final runtime and evidence are sufficient for clean-pass.
