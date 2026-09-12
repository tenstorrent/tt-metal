# Optimize checklist

Status: all runtime checklist items completed with evidence; independent stage-review verdict is recorded separately.

- [x] Decoder path fully traced with no host fallbacks
  Evidence: selected_stress/*.json, final_l{0,3}.json: traced optimized default; Torch/conversion guards.

- [x] Decode activations generally width-sharded in L1 across norm, attention, residual, MLP, and output projection boundaries.
  Evidence: movement_*; batch32_attention_boundary; native head/convolution boundaries documented in README.

- [x] Prefill activations generally DRAM interleaved; use 2D matmul program configs for large prefill matmuls.
  Evidence: prefill_matrix, closure_matrix, attention_matrix, phase_matrix; legal 2D configurations lose to minimal_matmul.

- [x] Operation-topology audit completed: current op sequence, repeated same-input matmuls, collectives, reshard/layout conversions, candidate fused/lower-movement replacements, dtype/fidelity constraints, and action taken are recorded.
  Evidence: work_log.md initial topology audit and subsequent measured decisions.

- [x] Multi-device topology candidates were measured as coherent families when applicable: residual layout, collective placement, fused CCL+matmul use, projection packing or separation, activation/CCL dtype, and persistent-buffer use. A rejection measured only under an incompatible residual/layout contract does not complete this item.
  Evidence: Not applicable: one-device dense decoder; no collectives or mesh parallelism.

- [x] Lower-movement residual candidates were measured without an immediate old-contract restore when applicable. If a reduce-scatter or fused CCL+matmul path only lost after an immediate all-gather or full replication, a stack-compatible sharded/fractured residual path was also measured or a minimal repro proves the next op cannot consume that layout.
  Evidence: movement_carryboth_* and movement_down3_* keep sharded residual/MLP through the next consumer.

- [x] Best-candidate comparison completed: the final path is compared against the strongest available correct baseline, earlier optimized artifact when present, and material candidates from this stage. The final choice wins traced warmed decode or has an explicit target-specific reason for prioritizing another workload. A synthetic-only precision veto does not count as a correctness reason when real-weight evidence passes. A geometry sweep measured only under a different dtype/fidelity does not reject the final dtype/fidelity policy.
  Evidence: candidate_results.csv, verified_benchmark_l{0,3}.json, verified_best_control.json, verified_repeat_l3.json; unsafe rounded-read candidates excluded.

- [x] Final default performance reproduced the selected best candidate under the final code path. If the final default is slower, the report uses the final number and explains why the candidate was not preserved.
  Evidence: verified_benchmark_l{0,3}.json records default implementation and actual policy; selected_matrix harness mismatch was repaired.

- [x] Final dtype/fidelity policy is verified in the measured runtime rows, not only in policy JSON or constructor defaults. For each dominant matmul, the `tt-perf-report` row or an equivalent profiler artifact must show the expected input/weight dtype and math fidelity. If the row shows BF16 or BFP8 where the selected policy claims BFP4, the policy did not reach the measured op and the stage is incomplete.
  Evidence: tracy/final_l0/*_perf_report.csv and tracy/selected_l3/*_perf_report.csv plus raw ops CSV: actual BF16 x BFP4 LoFi rows.

- [x] Used SDPA and other optimized composite ttnn ops instead of hand-built attention primitives where the target model fits their contracts.
  Evidence: Native SDPA/fused cache update, causal convolution/SiLU, delta prep/scan and gated RMS norm.

- [x] Fused or packed repeated same-input projections where legal and beneficial, such as Q/K/V-style projections, paired gate/up projections in 3-matmul MLPs, or other model-specific projection groups. If kept separate, there is measured evidence or a specific unresolved TTNN/runtime blocker after adapting layout, rank, padding, weight packing, and output splitting. If kept packed, it wins against a well-tuned legal separate candidate after counting split, activation, binary elementwise, and layout overhead, or the evidence explains why the separate candidate is invalid.
  Evidence: closure_split_attention_*; closure_packed_rect_* versus closure_rectangular_*; tuned_minimal_l*.

- [x] Explicitly configured `memory_config`, `program_config`, and `compute_kernel_config` for important ops.
  Evidence: final_policy.json and explicit per-op config construction in optimized_decoder.py.

- [x] For any matmul or repeated matmul group that is one of the largest decode-time consumers: swept legal program configs separately for each dominant role, including core grid, larger legal `in0_block_w` values, output subblocks, output blocks, memory configs, and compute kernel config where applicable. The stage is incomplete without a before/after evidence table or an exact TTNN/runtime blocker.
  Evidence: projection_results.csv; projection_sweep_l{0,3}.json; legal role-specific core/block/reader sweep, exact errors retained.

- [x] Decode compute fidelity was swept as a real performance knob for each dominant projection group. Do not assume BFP8 implies HiFi2 is fastest; try legal LoFi and HiFi2 candidates with the same dtype and real traced decode evidence, then keep the fastest policy that passes correctness.
  Evidence: hifi2_attention/output/gate/up/down.json versus LoFi selected geometry.

- [x] Attention projection weight dtype/fidelity was swept separately from MLP weight dtype/fidelity when QKV, Q/K/V, output projection, or fused attention matmul rows are material. If attention projections remain BFP8 or BF16, the report names the BFP4 attention candidate tried on real weights or recorded real activations, plus the precise correctness, latency, or op-contract blocker.
  Evidence: real_dram4_adapt_l3.json and closure_rectangular_l*.json, bfp8_lofi_attention, hifi2_attention/output; real HF layer0/3 validation.

- [x] If dense MLP or expert matmuls are among the largest decode-time consumers: BFP4/LoFi trials for FF1/FF3 or equivalent gate/up projections were run before lower-priority prefill-only advice was pursued to completion. FF2/down BFP4 was also tried or rejected with PCC/runtime evidence.
  Evidence: real_bfp4_mlp_l3.json, real_dram4_adapt_l3.json and closure_rectangular_l*.json, geometry_* and projection_results.csv precede long-prefill tuning.

- [x] Shard specs and core grids that divide tensor dimensions cleanly into tiles where possible, code grids as large as this and the model/hardware allows.
  Evidence: 80-core rectangular working grid; 48-core output; 32-core down; reader bank padding repaired and tested.

- [x] DRAM-sharded decode matmuls.
  Evidence: All dominant decode projection profiler rows show DRAM Sharded=True.

- [x] Collective topology minimized. Avoidable gather, reshard, all-reduce, reduce-scatter, and all-gather operations have been removed, moved to cheaper boundaries, or justified with before/after evidence.
  Evidence: No collectives. movement_* removes output interleave/reshard and repeated B1 residual input conversion.

- [x] Fused matmul-CCL ops used where possible, including fused all-gather-matmul or matmul-reduce-scatter patterns when a collective and matmul are adjacent or can be made adjacent. If rejected, the rejection includes an adapted attempt, not only the first API error.
  Evidence: Not applicable: no adjacent CCL or multiple devices in this stage.

- [x] Repeated decode CCLs use persistent or preallocated intermediate/output buffers where the API supports it. If unavailable or slower, the reason and measurement are recorded.
  Evidence: Not applicable: no CCL buffers.

- [x] For MoE models: optimized the routed active-expert path with `ttnn.sparse_matmul` where the model/hardware fits, correct `nnz` handling, separate gate/up and down tuning, correct sparse-input handling where applicable, routing-score weighting, expert reduction, no dense all-expert runtime path, and no avoidable DRAM round trips through decode intermediates.
  Evidence: Not applicable: dense17408-wide MLP; checkpoint has no routed experts.

- [x] For models with an LM head and sampling: final norm, LM head, logits movement, sampling, and token feedback are included in the optimized token-out path; terminal costs are profiled separately in full-model or reduced non-serving evidence, not in vLLM serving stages; LM-head weights are padded when needed for legal/fast DRAM-sharded or vocab-sharded matmuls; padded vocab IDs are masked in local logits shards before force-argmax or TopK; split-sampling TopK input widths are padded to avoid the slow single-core TopK fallback where possible; avoidable `ArgMaxDeviceOperation`, full-vocab all-gather, generic `TopKDeviceOperation`, host argmax, and full-logits readback have been removed. If a TTNN/runtime limitation blocks removal, the stage remains incomplete until there is a minimal repro or a lower-level fix.
  Evidence: Not applicable: decoder-only goal expressly excludes full-model/generator/serving work.

- [x] LM Head is optimized for DRAM-sharded matmuls if present.
  Evidence: Not applicable: no LM head in a decoder.

- [x] Reduced precision/fidelity experiments appropriate to this module-level optimization stage have been carried out and documented using real weights and input activations. For complete full-model top-k tuning, final datatype frontier selection is deferred to `$datatype-sweep`.
  Evidence: activation_provenance.json + real-HF checks; synthetic BFP4 diagnostic was not a veto.

- [x] Performance accounting reconciled: roofline estimate, device-time decode, and end-to-end decode reported from the same run; avoidable gaps optimized away, and any remaining gap named as a ttnn/runtime/API limitation only after a targeted fix attempt; `perf_summary.json` written when optimizing a complete model or serving path. For vLLM serving stages, use same-harness primary single-user and CI serving-burst serving metrics and set device-time/profile fields to `null` with the no-profiler reason.
  Evidence: Final profiler kernel sum/gaps, same-run traced host median and stored-tile-byte roofline in performance summary.

- [x] Batch capability preserved: batch-1 is the primary optimized latency target, and larger-batch or concurrent-serving correctness was tested up to 32 where hardware and memory allow it.
  Evidence: selected_stress/*.json and watcher_l{0,3}_b32.json; batches1/2/3/8/16/32, exact output shape/per-user PCC.
