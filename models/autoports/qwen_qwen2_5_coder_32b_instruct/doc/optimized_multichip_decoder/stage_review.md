# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- None. The rejected topology, dtype, geometry, and buffer families are supported by measured artifacts rather than prose-only deferrals.

## Hard-Check Gaps

- None unresolved. The stage-owned local commit and SHA logging remain the prescribed post-review follow-up; the unrelated dirty `.agents/skills/forge-functional-decoder-from-ir/SKILL.md` is outside this stage and must remain excluded.

## Anomaly Ledger

- Observed anomaly: the first fused-AG, BFP8-CCL, global-BFP8-output, and persistent-off results used the earlier attention-HiFi2/SDPA-8x8 family.
  Evidence: the original JSON `mesh_plan` fields under `results/autofix_*.json`, `results/sweep_activation_bfp8.json`, and `results/sweep_persistent_off.json`.
  Affected path: material multi-device coherent-family rejection evidence.
  Control or comparison: final default `attention_bfp8_lofi_mlp_bfp4_lofi`, SDPA 8x4, BF16 outputs/CCL, persistent buffers.
  Likely subsystem: candidate harness configuration/provenance.
  Investigation performed: required final-policy crosses and checked their JSON mesh plans, PCC, replay counts, and samples.
  Resolution: fixed. Final-policy 7x100 results are fused AG 0.930669 ms, BFP8 CCL 0.787956 ms, persistent-off 0.771160 ms, and MLP-only BFP8 output 0.759567 ms; attention-only BFP8 output is 0.771944 ms. All lose to the current-source default at 0.758020 ms.

- Observed anomaly: optimized reruns initially overwrote six tracked artifacts belonging to the completed `multichip_decoder` stage, and two validation reports retained stale relative result paths.
  Evidence: live worktree comparison against `HEAD` and the paths in `validation/{fused_ag_autofix,bfp8_ccl_autofix}.md`.
  Affected path: stage isolation and stale-artifact provenance.
  Control or comparison: completed multichip commit `ba8a83b14fd` and the new `doc/optimized_multichip_decoder/results/` copies.
  Likely subsystem: result-retention path in the shared test harness.
  Investigation performed: compared prior-stage JSON metrics to `HEAD`, checked all candidate-ledger paths, and checked validation links on the current filesystem.
  Resolution: fixed. The prior-stage result directory is byte-clean against `HEAD`; `_write_result` and Watcher retention now target `doc/optimized_multichip_decoder/results`; validation paths exist.

- Observed anomaly: capacity artifacts labeled attention as HiFi2 after the final policy moved to LoFi.
  Evidence: `results/capacity_seq12224.json`, `results/capacity_seq12225.json`, and the capacity payload in `tests/test_multichip_decoder.py`.
  Affected path: context-contract evidence metadata only; fidelity does not alter allocation size.
  Control or comparison: final source policy and `doc/context_contract.json`.
  Likely subsystem: stale artifact metadata.
  Investigation performed: compared capacity payload text, final mesh-plan precision, and allocation fields.
  Resolution: fixed. Source and both artifacts now say attention BFP8/LoFi; the measured 12224 pass and adjacent 12225 allocation failure remain unchanged.

- Observed anomaly: fused two-link QKV stalled, packed fused gate/up exceeded CB capacity, and dynamic fused outputs were not trace-stable.
  Evidence: `validation/fused_ag_autofix.md`, retained triage files, focused probe, and `results/autofix_fused_ag_full_probe.json`.
  Affected path: fused all-gather plus matmul.
  Control or comparison: standalone two-link AG plus packed projection.
  Likely subsystem: fused CCL/matmul link, CB, and output-lifetime contracts.
  Investigation performed: AutoFix adapted link count, rank/layout, padding/grid, projection decomposition, and four-shard persistent buffers; focused trace and full real-weight tests passed.
  Resolution: controlled. The trace-safe one-link fused gate plus direct-up family is correct but 0.930669 ms at final policy, so the faster separate async-AG family remains selected.

- Observed anomaly: full Ethernet Watcher instrumentation cannot compile the active-Ethernet Ring program because 27,920 bytes exceeds the 25,600-byte kernel-config buffer.
  Evidence: `results/watcher_clean.json`, `results/watcher_clean.log`, and the exact commands in `work_log.md`.
  Affected path: Watcher instrumentation, before model execution.
  Control or comparison: separate final real-layer Watcher run with worker/dispatch monitoring and Ethernet instrumentation disabled only.
  Likely subsystem: Watcher active-Ethernet instrumentation capacity.
  Investigation performed: full instrumentation was attempted first; the scoped retry passed all four devices with no fault-pattern matches. Separate 7x100 trace stress, focused exact AG/RS traces, and direct two-layer shared-buffer tests cover the CCL behavior.
  Resolution: controlled tooling limitation; no model/runtime failure is visible.

- Observed anomaly: Tracy's merged four-device report attributes a large cross-device/signpost gap to a reshape row.
  Evidence: `tracy/layer32/decode_perf_report.txt` reports 595 us device work but an 8,169 us merged op-to-op total.
  Affected path: profiler presentation, not the traced latency harness.
  Control or comparison: current-source final 7x100 benchmark at 0.758020 ms.
  Likely subsystem: multi-device profiler row merge.
  Investigation performed: reconciled device work, per-op rows, roofline estimate, and warmed end-to-end trace separately.
  Resolution: controlled. Device rows are used for topology/dtype advice; the warmed trace benchmark is the end-to-end authority.

## Scope Inspected

- Goal/skill paths: original optimized-multichip-decoder contract; `.agents/skills/stage-review/SKILL.md`; `.agents/skills/optimize/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`.
- Artifact paths: `doc/optimized_multichip_decoder/{README.md,work_log.md,results,validation,shard_advise,tracy}`; `doc/context_contract.json`; prior `doc/multichip_decoder` evidence; independent baseline `/tmp/qwen2_5_coder_32b_optimized_baseline.pt`.
- Code paths: `tt/multichip_decoder.py`; `tests/test_multichip_decoder.py`; focused advisor, fused-AG, and BFP8-CCL probes; live git diff excluding the unrelated skill file.
- Commands run: read-only `git status/diff/show`, JSON/CSV parsing and cross-check scripts, SHA256 verification, artifact-path checks, profiler-row inspection, `git diff --check`, and Python compilation. In accordance with `$stage-review`, the reviewer did not open devices or rerun hardware.

The review independently verified 37 candidate-ledger rows against their JSON artifacts, parsed all 45 result JSON files, matched the compact Tracy hashes to `provenance.json`, matched the independent baseline SHA, and confirmed that current profiler rows show BFP8/LoFi attention, BFP4/LoFi MLP, BF16 outputs/CCL, and BFP8 KV cache. The final current-source default is TP4 on `MeshShape(1,4)`, PCC 0.992527 prefill / 0.993698 decode, 3.541916 ms warmed prefill, and 0.758020 ms warmed traced decode with seven trials, 100 replays per trial, and bitwise stability.

## Residual Risk

- The implementation intentionally targets four Blackhole p300c devices, TP4, and the compiler-derived batch-32 decoder contract; full-model, LM-head, generation, and vLLM behavior were not started or reviewed.
- Active-Ethernet Watcher firmware is not instrumented because of the documented compile-time size limit; worker/dispatch Watcher coverage plus separate CCL, stack, paged, non-aligned, and repeated-trace evidence is the available control.
- Raw Tracy databases were omitted after reduction, as required for safe artifact retention; their hashes and all compact CSV/text report hashes remain in `tracy/layer32/provenance.json`.
