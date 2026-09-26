# Final v4 profiler advice audit

Runtime SHA256 recorded by both native profiles: `3d51014f98128dfb21bb484fcece50993524b967754f68f6dfe825ae7f472ba9`.

The final native rows confirm the selected indexed eight-expert topology, fused accurate GELU product, row-major decode RoPE lookup, sharded hidden-width normalization and explicit prefill output program. The final correctness summary reports all default gates passed at PCC 0.995, including actual-text stress, public contracts, sampled maximum/near-maximum context rows and Watcher. The earlier sliding full-context row misses were repaired by BFP8 prefill gate weights; they are not waived by aggregate PCC. See [validation summary](validated_v4_validation_summary.json).

Native group values sum Device Time (microseconds) in one sampled replay; whole-layer spans are 128 sequential-position trace windows and include gaps. Candidate warmed host medians and one-replay operator controls are separate measurements.

| Decode group | Sliding native µs | Full native µs |
| --- | ---: | ---: |
| input norm | 12.358 | 12.374 |
| qkv projection | 64.400 | 76.188 |
| qkv head layout | 17.117 | 16.257 |
| head norms | 105.108 | 45.196 |
| rope lookup and rotation | 106.185 | 155.160 |
| cache update | 15.073 | 16.892 |
| native sdpa | 42.722 | 70.835 |
| output projection and head concat | 40.821 | 70.351 |
| post attention and common norm | 30.894 | 30.949 |
| router | 69.174 | 68.175 |
| routed experts | 157.231 | 133.853 |
| shared experts | 67.634 | 52.284 |
| tail norm and residual | 23.503 | 22.787 |

| Whole-layer measurement | Sliding | Full |
| --- | ---: | ---: |
| Prefill span, µs | 222496.813 | 190201.333 |
| Prefill native operation count | 2173.000 | 2109.000 |
| Decode mean128 span, µs | 881.343 | 899.058 |
| Decode mean summed native kernels, µs | 752.964 | 772.100 |
| Decode native operations per replay | 123 | 126 |

Group values cover every native row in the sampled replay, including movement. They differ slightly from the 128-replay kernel mean. Device spans include gaps. Host candidate medians below are separate controlled measurements; no host/device subtraction is used.

## Actual dtype, fidelity and program policy

| Role | Sliding | Full |
| --- | --- | --- |
| Direct QKV | FP32 × BFP8 → FP32; HiFi2;11×10 configured/86 active;K11/sub1×3 | FP32 × BFP8 → FP32;LoFi;11×10/96 active;K11/sub1×3 |
| Native decode SDPA | BF16 query/BFP8 cache → BF16;HiFi4;8×8 | BF16 query/BFP8 cache → BF16;LoFi;8×8 |
| Decode output | BF16 × BFP8 → FP32 L1;HiFi4;8×8/44 active;K16/sub1×2 | BF16 × BFP8 → FP32 L1;LoFi;11×10/88 active;K16/sub1×1 |
| Router score projection | FP32 × BF16 → FP32;HiFi4;4×1;K22/sub1×1 | FP32 × BF16 → FP32;LoFi;4×1;K44/sub1×1 |
| Indexed expert gate/up | BF16 × BFP8 → BF16;LoFi;44 cores;K44/sub1×1 | BFP8 × BFP4 → BF16;LoFi;44 cores;K44/sub1×1 |
| Indexed expert down | BF16 × BFP4 → BF16;LoFi;44 cores;K22/sub1×2 | Same |
| Compact expert mix | BF16 × BF16 → BF16;HiFi4;88 cores;K1 | Same output/fidelity; destination-accumulation policy differs, retained in raw attributes |
| Shared decode gate/down | BF16 × BFP8 → BF16;LoFi;DRAM-sharded;reader 1;K11 | BF16 × BFP4 → BF16;LoFi;DRAM-sharded;reader 2;K11 |
| Prefill expert gate/down | BF16 × BFP8/BFP4 → BF16;LoFi;32-token active unions | BF16 × BFP4/BFP4 → BF16;LoFi;32-token active unions |
| Prefill QKV | FP32 × BFP8 → FP32;HiFi4;automatic2D | Same |
| Prefill output | BF16 × BFP8 → FP32 DRAM;LoFi;11×8;K16;sub1×4 at 1024 rows | Same |
| Shared prefill | BF16 weights/activations;HiFi2 | Same |

QKV, output and router GEMMs accumulate into FP32. Native SDPA uses FP32 destination plus full synchronization and returns BF16. Sparse experts use BF16 destination. Both routers center FP32 scores before conversion to the native BF16 gate. The JSON retains actual native input/output memory, config, dtype and kernel-source fields; FP32 accumulation never implies FP32 output.

## Expert topology and movement

Indexed sparse inputs carry row-major UINT16 `[1,1,1,8]` IDs, weights retain E128 and outputs have logical E8. All four native sparse rows show `use_indices=true` and `nnz=std::nullopt`. Compact gate output is `[1,8,1,1408]` (physical row 32); compact down output is `[1,8,1,2816]`. Routing values are gathered after the original learned expert scale. The retained 128 weight banks are not mistaken for 128 executed experts.

| Expert family, native sampled kernel sum | Sliding µs | Full µs |
| --- | ---: | ---: |
| Packed gate/up | 92.862 | 68.985 |
| Down | 35.861 | 35.026 |
| Fused GELU product | 8.154 | 8.022 |
| Output-buffer zero fills | 6.478 | 6.514 |
| Compact transpose | 4.049 | 3.813 |
| Weighted matmul | 2.201 | 2.319 |
| Gate/up slicing | 3.382 | 3.305 |

The expanded activation/transpose costs have been reduced by compact E8 storage. The retained compact output-buffer zero fills cost about 6.5 µs per layer; the weighted-reduce alternative was executed successfully on expanded and compact shapes and lost to the selected matmul. Per-head RoPE slice/repeat/tilize remains visible; full-table untilize is gone. Hidden-width normalization still requires explicit sharding transitions, and decode head concat is an actual consumer-layout boundary. Adapted output DRAM-sharded/retained-sharded and direct QKV DRAM-sharded controls were measured, not rejected at the first API error.

## Advice disposition

**decode QKV — measured alternative rejected.** Actual direct FP32/BFP8 packed interleaved versus adapted DRAM readers2/3 K11 completed for both kinds. Legal candidates passed and were slower; readers1 K11/K22 hit L1 capacity. These are the direct backend, not obsolete BF16-lane results. Evidence: `direct_geometry_results.json`, `actual_direct_geometry_commands.json`.

**decode output projection — measured alternatives completed.** Actual64/110-core interleaved, DRAM readers1/2, unpadded retained-sharded reader 1, HiFi2/LoFi, BFP4 weights and L1 output controls exist. Current native rows confirm selected BF16 input/BFP8 weight/FP32 output. Prefill is a distinct path. Evidence: `actual_direct_output_commands.json`, `actual_final_precision_commands.json`, `actual_static_tune_commands.json`.

**decode router — measured alternative rejected.** Legal2-core1x2 and1-core1x4 all pass but lose: sliding989.866/998.113us versus987.932us; full1076.886/1080.291us versus1076.212us. Keep4cores. Advice that calls this BFP8 multiplication is not applicable to its actual BF16 weights. Evidence: `router_subblock_commands.json`.

**decode expert GEMMs — expanded and compact geometry completed.** Expanded packed44-core gateK44/downK22 beats22cores, gateK22/K88, and matched separate. Indexed sparse API exposes8 compact slots. Actual compact controls confirm44-core gateK44/downK22 wins over22cores,down88,gateK22/K88 and separate on both kinds. Fused GELU now wins slightly after compaction. Evidence: `expert_gate_geometry_results.json`, `actual_direct_mlp_geometry_commands.json`, `compact_geometry_commands.json`.

**expert GELU — selected for compact topology.** Fused accurate GELU plus multiply is selected on compact E8 shapes. Matched compact whole-layer medians 830.638/921.665 us versus 832.163/922.738 us unfused, identical 128 PCCs. The earlier expanded-shape fused control passed but was slower; that rejection does not apply after compaction. Evidence: `compact_geometry_commands.json`, `expert_fused_gelu_source.md`.

**expert weighted mix — expanded and compact fused reduction rejected.** Existing Blackhole weighted-reduce op is shape-legal with FP32 accumulation. Expanded control passed but slowed the whole layer. Compact matmul also beats compact weighted reduction; retain compact matmul. Full compact weighted927.842us versus922.738us matmul; expanded weighted1226.202us. Evidence: `probe_optimized_expert_compact.py`, `AUTODEBUG_weighted_expert_reduce.md`.

**decode RoPE producer layout — integrated and contract-validated.** Caller prepares row-major decode tables before tracing. This removes the full-table untilizes and preserves old tiled-table compatibility. Per-head selected-row slice/repeat/tilize operations remain and are not misreported as full-table conversion. Actual row-major controls have identical PCC; final v4 includes both layouts. Evidence: `actual_decode_rope_row_major_layer0.json`, `actual_decode_rope_row_major_layer5.json`, `validated_v4_tile_rope_layer0.json`, `validated_v4_tile_rope_layer5.json`.

**prefill output projection — selected and final-context validated.** All 28 actual-text matrix/cross/L1 controls pass. Select 2D grid11x8 K16 LoFi, DRAM input/output, FP32 accumulation/output. Whole-prefill host medians 222689.914/190090.823 us versus auto225132.786/195118.534 us. K16/32 HiFi2/LoFi and selected K16 LoFi L1-input controls close the cross-fidelity and placement questions. Final max/near-max rowwise gates pass. Evidence: `prefill_output_results.json`, `prefill_output_commands.json`, `prefill_output_cross_commands.json`, `prefill_output_l1_commands.json`, `validated_v4_validation_summary.json`.

**prefill QKV — actual policy controls already present.** This journal contains actual-text BFP8 QKV dense_prefill_2d controls; older headline_dense2d Gaussian/BF16 controls are supplementary, not the sole evidence. Evidence: `actual_static_tune_commands.json`.

**prefill shared MLP — low materiality; decoded shared geometry already measured.** Final prefill shared gate/down totals remain below 0.5% of whole prefill. Generic 88-to110-core/L1 advice is not a material remaining opportunity; widths and legal per-core geometry matter. Decode shared DRAM readers/split/core controls have been measured. Evidence: `tracy/actual_optimized_v4_layer0/prefill_perf_report.csv`, `tracy/actual_optimized_v4_layer5/prefill_perf_report.csv`, `actual_direct_mlp_geometry_commands.json`.

**prefill sparse utilization — report limitation, not optimization failure.** Runtime nnz is the union over 32 tokens and varies by chunk. Top 8 per token does not imply 8 experts per sparse prefill group. Leave utilization unknown absent actual union metadata; do not pass fabricated 8. Evidence: `tt/optimized_decoder.py:OptimizedExperts._active_prefill`.

**full decode input norm — integrated and final-contract validated.** Sharded normalization at all hidden-width sites passes actual headline and 512 controls, then final v4 reuse/batched/context gates. The matched headline control is 832.355 us versus 864.212 us; final native input norm itself is 6.233 us. Evidence: `full_all_norm_headline.json`, `full_all_norm_stress.json`, `validated_v4_validation_summary.json`.

**sliding decode head norms — rejected by complete actual 512-step control.** Native head normalization passes headline but fails actual stress at 1257 PCC .994804004648 and 1459 PCC .994418854936. The complete run executes all 512 checks, audits,cache guard and determinism. Accurate FP32 SFPU head reductions remain selected; native RMSNorm uses different multiply/reduction precision even with FP32 destination. Historical same-fixture precise checks pass these positions; the final v4 cumulative precision policy also passes its 512 stress gate. Evidence: `native_headnorm_headline_layer0.json`, `native_headnorm_stress_layer0.json`, `headnorm_control_summary.json`, `native_headnorm_control.md`, `validated_v4_validation_summary.json`.

**full native rotary — matched topology measured slower.** Existing isolated full-kind boundary control uses the same FP32 head geometry and native HF rotary; it costs approximately 24us more. This shape/fidelity performance evidence remains applicable even though earlier activations were synthetic; no accuracy veto is inferred. Decode table producer-layout work is separate and now measured. Evidence: `doc/fused_decoder/PATTERNS.md`, `doc/fused_decoder/work_log.md:full_boundary`.

**prefill grouped routed FFN — named existing operators do not satisfy the mathematical/data contract.** routed_expert_ffn hardcodes SiLU. Unified activation enum supports Silu/SwiGluOai/SituGlu but no GELU and forces BFP8 dispatched arithmetic/output. H2816/I704/E128/oneASIC are not the blocker. moe_expert_token_remap only copies routing scalars and builds a union mask; moe_routing_remap partitions [1,E] weights across devices. Neither gathers token activations or creates the counts/region offsets required by grouped FFN. A new GELU-capable grouped dispatch/combine implementation is required, outside a drop-in existing-op substitution. Evidence: `prefill_routed_ffn_contract_audit.md`.

## One-replay operator controls

These controls verify the alternative native topology/configs on actual inputs. The selected rows come from the headline profile; controls use a separately hashed 4096/1 fixture. Compare operator costs, not end-to-end throughput or full stress accuracy. The prior matched 4096/128 geometry journals establish the whole-layer winner.

| Policy | Layer | Expert-region kernel µs | Gate/up µs | Down µs | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| selected | 0 | 157.231 | 92.862 | 35.861 | complete |
| indexed22k44 | 0 | 171.356 | 96.730 | 45.435 | complete |
| indexed_separatek44 | 0 | 168.856 | 109.720 | 33.301 | complete |
| expanded44k44 | 0 | 322.746 | 94.534 | 37.184 | complete |
| selected | 5 | 133.853 | 68.985 | 35.026 | complete |
| indexed22k44 | 5 | 151.752 | 75.741 | 45.003 | complete |
| indexed_separatek44 | 5 | 151.575 | 88.991 | 34.421 | complete |
| expanded44k44 | 5 | 303.259 | 74.786 | 35.101 | complete |

Installed `tt-perf-report` does not infer active groups from indexed sparse attributes. Decode reports use `--active-experts 8` only after raw UINT16-index volume and compact E8 output prove the count; the model-local traffic summary derives it from those native metadata and records its basis. Prefill has variable per-chunk expert unions, so no 8-expert override is passed. Native raw CSVs remain unchanged.

## Prefill useful work and scope

The final prefill native sparse gate/down totals are 116.738/41.719 ms sliding and 85.805/32.402 ms full. This is material union-expert work. The bounded [existing routed-FFN/remap contract audit](prefill_routed_ffn_contract_audit.md) checks the actual activation, dtype, layout and dispatch requirements; it does not dismiss them by model name or fabricate a hardware failure. Those named ops cannot express this grouped GELU FFN as a drop-in replacement.

Useful prefill roofline estimates are 0.598%/0.963% and decode estimated DRAM rates 24.486%/23.111% of the stated single-P300 theoretical peaks. These are useful-work/traffic estimates, not hardware utilization counters. Prefill computes extra union experts excluded from the useful 8-per-token numerator; unknown per-chunk nnz is left unknown.

The six scheduled operator audits are complete. Additional installed optimize0.1.14 requirements are now in progress: the parent’s minimal prefill-matmul comparison and the OPT015 reader microbenchmark. Their results are not implied by the existing candidate journals.
