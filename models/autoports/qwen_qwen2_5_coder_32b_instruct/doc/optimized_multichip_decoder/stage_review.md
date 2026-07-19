# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- None. The current default, the completion-audit candidates, and the candidate ledger now agree with their retained artifacts. The expected stage-owned checkpoint after this review is workflow bookkeeping, not missing stage evidence.

## Hard-Check Gaps

- Full active-Ethernet Watcher instrumentation remains unavailable: its 27,920-byte instrumented Ring program exceeds the 25,600-byte runtime kernel-config buffer before model execution. The current-source worker/dispatch Watcher retry passed on all four devices with no fault-pattern matches; separate CCL trace, topology, two-layer, and 7x100 replay evidence controls the excluded Ethernet instrumentation scope.
- No full 64-layer model was run because this stage is decoder-layer scoped. The local Hugging Face config has 64 homogeneous dense Qwen2 layers with identical shapes; real layer 32 covers the only layer kind, and a direct two-layer test covers the inter-layer contract. Full-model assembly remains a later stage.

## Anomaly Ledger

- Observed anomaly: the first completion-audit prefill-L1 screen contained a compile-like timing outlier.
  Evidence: `results/sweep_prefill_l1_inputs.json`.
  Affected path: generic profiler advice to place prefill matmul input 0 in L1.
  Control or comparison: a seven-prefill-trial final-policy rerun in `results/sweep_prefill_l1_inputs_full.json`.
  Likely subsystem: lazy compilation or cache warmup in the first short screen.
  Investigation performed: verified PCC, exact mesh plan, individual timing samples, medians, and the default-off source switch.
  Resolution: controlled and rejected. The full candidate passed PCC but measured 3.579746 ms versus the current DRAM-interleaved default at 3.566172 ms.

- Observed anomaly: the ledger originally called `sweep_geometry_gate64.json` a 64-core gate/up measurement, but its mesh plan records the unchanged 8x4/32-core gate program.
  Evidence: the artifact's `decode_grids.gate_up`, its physical/program shapes, and `_core_grid_for_tiles`; logical K/N are 160/448 tiles with greatest common divisor 32.
  Affected path: profiler-driven gate/up geometry coverage and evidence integrity.
  Control or comparison: the selected 32-core program and a material padded-K retry.
  Likely subsystem: target-core request resolution in the candidate harness.
  Investigation performed: independently compared every geometry label to its JSON mesh plan, required the old row to be relabeled as a diagnostic/no-op, and inspected the adapted source and full artifact.
  Resolution: fixed. `sweep_geometry_gate64_padded_k_full.json` uses physical K=6144, an actual 8x8/64-core gate/up projection, and a legal 8x7/56-core gated-elementwise layout. It passes PCC and bitwise 7x100 replay but loses at 0.792797 ms versus 0.758047 ms.

- Observed anomaly: adding the default-off true-64 candidate after an earlier final run made that final JSON no longer literally producible by the current source because the mesh plan gained `gated_elementwise`.
  Evidence: chronology and the missing field in the superseded live artifact.
  Affected path: the claimed current-source final default and Watcher provenance.
  Control or comparison: exact-default reruns after both completion-audit switches were present.
  Likely subsystem: artifact/source chronology, not selected runtime behavior.
  Investigation performed: required a fresh 7-prefill/7-decode/100-replay default run and a fresh scoped Watcher run, then recomputed hashes and medians.
  Resolution: fixed. Authoritative `final_default.json` SHA256 is `7f862083c1b3b38c0d232a54712183f9c1a2c4e1e1dc0ac15c832d97a0bb40e6`; it records both selected grids at 8x4/32 cores, DRAM prefill input, PCC 0.992527216/0.993698060, 3.566172 ms prefill, 0.758047 ms traced decode, and bitwise stability.

- Observed anomaly: the refreshed raw Watcher log contained trailing spaces, initially failing `git diff --check`.
  Evidence: raw worker-core state lines in `results/watcher_clean.log`.
  Affected path: artifact hygiene and its recorded digest.
  Control or comparison: whitespace-normalized log plus recomputed JSON metadata.
  Likely subsystem: Watcher text formatting.
  Investigation performed: reran whitespace checking and independently matched the normalized file digest to `watcher_clean.json`.
  Resolution: fixed. Normalized log SHA256 is `4f3e73777839c48f2c8a675a35eeada7d27bdc809eeb893bf43e35b4fd3c0f16`; `matches` is empty and repository-wide `git diff --check` passes.

- Observed anomaly: early fused-AG, BFP8-CCL, global-BFP8-output, and persistent-off results used an older precision/SDPA family.
  Evidence: original candidate JSON mesh-plan fields and the completion validation reports.
  Affected path: coherent-family rejection evidence.
  Control or comparison: final-policy, SDPA-8x4, BF16-output/CCL reruns.
  Likely subsystem: candidate harness configuration/provenance.
  Investigation performed: checked final-policy 7x100 crosses for fused AG, BFP8 CCL, persistent-off, and MLP-only BFP8 output, plus the attention-only BFP8 probe.
  Resolution: fixed. The corrected candidates are valid but slower than the current 0.758047 ms default.

- Observed anomaly: the shard-advisor runtime initially disagreed with the live vendored TTNN interface, and its per-op recommendations were not a coherent end-to-end win.
  Evidence: `shard_advise/report.json`, `shard_advise/result/final_ir.mlir`, the advisor comparison artifacts, and the repair notes in `work_log.md`.
  Affected path: compiler-guided sharding coverage.
  Control or comparison: current selected DRAM-sharded QKV/O/gate/down family.
  Likely subsystem: advisor/runtime version boundary and local-versus-coherent optimization.
  Investigation performed: verified the retained no-spill report, matched the vendored interface, and compared all four projection timings together.
  Resolution: controlled. The current coherent family is about 291.6 us versus about 381.2 us for the advisor combination; selected sharding remains measured rather than deferred.

- Observed anomaly: the Hugging Face model advertises 32,768 tokens, but the complete TP4 resident-state contract cannot fit that context on the target devices.
  Evidence: `doc/context_contract.json`, `results/capacity_seq12224.json`, and `results/capacity_seq12225.json`.
  Affected path: supported context length.
  Control or comparison: exact byte accounting at the largest feasible length and the adjacent padded allocation.
  Likely subsystem: device DRAM capacity, principally KV-cache residency.
  Investigation performed: checked physical/logical sequence padding, model-resident allocations, per-device/per-bank bytes, the 12,224 pass, and the 12,225 expected OOM.
  Resolution: controlled capability limit. The stage explicitly supports 12,224 tokens and does not claim the infeasible Hugging Face maximum.

- Observed anomaly: Tracy's merged four-device presentation assigns a large cross-device/signpost interval to a reshape row.
  Evidence: `tracy/layer32/decode_perf_report.txt` versus its device-operation total and the end-to-end trace artifact.
  Affected path: profiler presentation.
  Control or comparison: per-device rows, compact report hashes, and the warmed 7x100 trace harness.
  Likely subsystem: multi-device profiler row merging.
  Investigation performed: reconciled operation dtypes/fidelities, device timings, provenance hashes, advice, and end-to-end latency separately.
  Resolution: controlled. Profiler rows drive topology/dtype diagnosis; `final_default.json` is the latency authority.

## Scope Inspected

- Goal and skill contracts: the original optimized-multichip-decoder task; `$stage-review`; `$optimize`; `$tt-device-usage`; and the referenced graph-rewrite, shard-advisor, and AutoFix evidence requirements.
- Source and tests: `tt/multichip_decoder.py`, `tests/test_multichip_decoder.py`, the current live diff, source defaults, runtime fallback audit, candidate-only padding/L1 switches, syntax compilation, Black check, and test collection.
- Correctness: real layer-32 PCC, independent baseline digest, synthetic non-aligned sequence, paged versus contiguous cache, page-table/position trace refresh, two-layer handoff, K/V-cache PCC, and bitwise replay evidence.
- Performance and topology: all 41 candidate-ledger rows reconciled exactly to JSON at six decimals; 52 stage JSON files parsed; final-policy precision/activation/CCL/KV families; selected versus compiler-provenance topology; fused AG/RS; geometry and K-block families; persistent buffers; prefill configs; and profiler advice closure.
- Profiler/advisor provenance: compact Tracy CSV/text hashes, raw provenance metadata, selected BFP8/LoFi attention, BFP4/LoFi MLP, BF16 activation/CCL, BFP8 KV policy, shard-advisor report/final IR, and coherent timing comparison.
- Capacity, health, and repository hygiene: exact 12,224/12,225 capacity boundary, current-source Watcher hash and empty fault matches, controlled Ethernet instrumentation exception, artifact-path existence, SHA256 checks, JSON/CSV reconciliation, `git diff --check`, and stage-local git scope. The unrelated dirty `.agents/skills/forge-functional-decoder-from-ir/SKILL.md` was excluded.
- Reviewer commands were read-only except for this authorized report replacement. The reviewer did not open or use TT devices. Independently run checks included 11-test collection, the host-free/fallback contract test (1 passed), Python compilation, Black check, and repository-wide whitespace validation.

## Residual Risk

- The authoritative final prefill sample set contains one high first sample (6.803737 ms); the required seven-trial median is 3.566172 ms and the other six samples are tightly grouped enough that the median is not determined by the outlier. Prefill improvement over the starting default is therefore modest (0.33%), while traced decode improvement is 4.28%.
- Active-Ethernet firmware itself is not Watcher-instrumented because of the documented compile-time capacity limit. CCL correctness is instead covered by focused exact AG/RS traces, topology comparison, direct two-layer reuse, current-source 7x100 replay, and worker/dispatch Watcher coverage.
- This verdict covers the live TP4 Blackhole decoder-stage tree and retained evidence. Full-model embeddings, 64-layer execution, LM head, generation, serving, and vLLM integration remain intentionally outside this stage.
