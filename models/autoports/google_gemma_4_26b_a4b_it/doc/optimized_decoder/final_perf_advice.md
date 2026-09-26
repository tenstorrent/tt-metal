# Current native profiler advice audit

Runtime SHA256: `5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898`.

Decode groups sum one sampled replay's native Device Time in microseconds; percentages use that sampled kernel sum. Whole-layer128-replay spans include gaps. Prefill percentages use complete prefill device span. Warm host comparisons remain separate.

| Decode group | Sliding µs | Sliding kernel % | Full µs | Full kernel % |
| --- | ---: | ---: | ---: | ---: |
| input norm | 12.332 | 1.64 | 12.356 | 1.60 |
| qkv projection | 64.718 | 8.61 | 76.109 | 9.83 |
| qkv head layout | 17.097 | 2.27 | 16.215 | 2.10 |
| head norms | 106.298 | 14.14 | 45.252 | 5.85 |
| rope lookup and rotation | 106.144 | 14.12 | 155.002 | 20.03 |
| cache update | 15.044 | 2.00 | 16.945 | 2.19 |
| native sdpa | 42.966 | 5.71 | 71.254 | 9.21 |
| output projection and head concat | 39.227 | 5.22 | 71.744 | 9.27 |
| post attention and common norm | 30.820 | 4.10 | 31.019 | 4.01 |
| router | 68.888 | 9.16 | 68.123 | 8.80 |
| routed experts | 157.802 | 20.99 | 132.842 | 17.16 |
| shared experts | 67.941 | 9.04 | 53.387 | 6.90 |
| tail norm and residual | 22.649 | 3.01 | 23.692 | 3.06 |

| Whole-layer native window | Sliding µs | Full µs |
| --- | ---: | ---: |
| prefill | 221367.855 | 186562.836 |
| decode | 881.411 | 899.336 |

Useful prefill work reaches 0.601% / 0.982% of the stated single-P300 theoretical peak (sliding/full). Estimated decode DRAM rates reach 24.484% / 23.104%. These are model estimates, not hardware utilization counters; extra union-expert work is excluded from the useful FLOPs numerator.

| Prefill sparse group | Sliding native µs (% of span) | Full native µs (% of span) |
| --- | ---: | ---: |
| Gate/up | 116728.500 (52.73%) | 85784.291 (45.98%) |
| Down | 41722.586 (18.85%) | 32404.972 (17.37%) |

The JSON joins every key operation to its raw native attributes, input/output dtypes, memory layout, configured blocks and kernel sources. It verifies all indexed sparse rows use UINT16 eight-element indices, E8 output and E128 resident weights. Decoder traffic accounts for8 executed experts; prefill active unions remain unknown rather than assuming8 per32-token chunk.

The remaining small blocks have explicit bounds and controls. Compact expert mixing has eight logical routes padded to one K tile, so K1 is the only divisor of its actual tiled K; expanded and compact weighted-reduce alternatives were both executed and lost. Expert gate/up's44 output tiles distribute one per44 workers, giving subblock1×1; the22-core/subblock1×2 and separate gate/up families were measured, including the native operator controls below. Router subblocks1×2/1×4 reduce the active worker count from four to two/one and were measured slower.

## Prefill native projection audit

| Layer | Operation | Calls | Native µs | Fidelity / dtype | Explicit program |
| --- | --- | ---: | ---: | --- | --- |
| 0 | MinimalMatmulDeviceOperation 1024 x 2816 x 8192 | 4 | 1939.876 | HiFi4 FP32 x BFP8 => FP32 | `MinimalMatmulConfig(M_block_size=4;K_block_size=8;N_block_size=8;subblock_h=1;subblock_w=4;compute_with_storage_grid_size=11-8)` |
| 0 | MatmulDeviceOperation 1024 x 4096 x 2816 | 4 | 516.803 | LoFi BF16 x BFP8 => FP32 | `MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=11-8;in0_block_w=16;out_subblock_h=1;out_subblock_w=4;out_block_h=4;out_block_w=8;per_core_M=4;per_core_N=8;transpose_mcast=0;fused_activation=std::nullopt;fuse_batch=0;allowed_worker_cores={[0-0 - 10-7]})` |
| 0 | MatmulDeviceOperation 1024 x 2816 x 128 | 4 | 249.768 | HiFi4 FP32 x BF16 => FP32 | `MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=11-10;in0_block_w=8;out_subblock_h=4;out_subblock_w=1;out_block_h=4;out_block_w=1;per_core_M=4;per_core_N=1;transpose_mcast=0;fused_activation=std::nullopt;fuse_batch=0;allowed_worker_cores={[0-0 - 10-9]})` |
| 0 | MatmulDeviceOperation 1024 x 2816 x 4224 | 4 | 586.569 | HiFi2 BF16 x BF16 => BF16 | `MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=11-10;in0_block_w=8;out_subblock_h=4;out_subblock_w=2;out_block_h=4;out_block_w=12;per_core_M=4;per_core_N=12;transpose_mcast=0;fused_activation=std::nullopt;fuse_batch=0;allowed_worker_cores={[0-0 - 10-9]})` |
| 0 | MatmulDeviceOperation 1024 x 2112 x 2816 | 4 | 340.922 | HiFi2 BF16 x BF16 => BF16 | `MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=11-10;in0_block_w=6;out_subblock_h=4;out_subblock_w=2;out_block_h=4;out_block_w=8;per_core_M=4;per_core_N=8;transpose_mcast=0;fused_activation=std::nullopt;fuse_batch=0;allowed_worker_cores={[0-0 - 10-9]})` |
| 5 | MinimalMatmulDeviceOperation 1024 x 2816 x 9216 | 4 | 1505.298 | HiFi2 FP32 x BFP8 => FP32 | `MinimalMatmulConfig(M_block_size=2;K_block_size=16;N_block_size=8;subblock_h=1;subblock_w=4;compute_with_storage_grid_size=11-8)` |
| 5 | MinimalMatmulDeviceOperation 1024 x 8192 x 2816 | 4 | 687.485 | LoFi BF16 x BFP8 => FP32 | `MinimalMatmulConfig(M_block_size=4;K_block_size=8;N_block_size=8;subblock_h=1;subblock_w=4;compute_with_storage_grid_size=11-8)` |
| 5 | MatmulDeviceOperation 1024 x 2816 x 128 | 4 | 252.186 | HiFi4 FP32 x BF16 => FP32 | `MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=11-10;in0_block_w=8;out_subblock_h=4;out_subblock_w=1;out_block_h=4;out_block_w=1;per_core_M=4;per_core_N=1;transpose_mcast=0;fused_activation=std::nullopt;fuse_batch=0;allowed_worker_cores={[0-0 - 10-9]})` |
| 5 | MatmulDeviceOperation 1024 x 2816 x 4224 | 4 | 586.238 | HiFi2 BF16 x BF16 => BF16 | `MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=11-10;in0_block_w=8;out_subblock_h=4;out_subblock_w=2;out_block_h=4;out_block_w=12;per_core_M=4;per_core_N=12;transpose_mcast=0;fused_activation=std::nullopt;fuse_batch=0;allowed_worker_cores={[0-0 - 10-9]})` |
| 5 | MatmulDeviceOperation 1024 x 2112 x 2816 | 4 | 341.355 | HiFi2 BF16 x BF16 => BF16 | `MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=11-10;in0_block_w=6;out_subblock_h=4;out_subblock_w=2;out_block_h=4;out_block_w=8;per_core_M=4;per_core_N=8;transpose_mcast=0;fused_activation=std::nullopt;fuse_batch=0;allowed_worker_cores={[0-0 - 10-9]})` |

`tt-perf-report` does not parse minimal matmul’s `config=MinimalMatmulConfig` fields, so its missing-program warning is false. This does not dismiss independent grid, fidelity or placement recommendations; their controls are recorded below.

The [full QKV L1 controls](qkv_l1_results.md) retain the original M4 capacity failure, both legal M2/N4 adaptations, matched M2 placement and grid comparisons, a65-token boundary, and32 producer/copy pairs. Current native rows additionally verify each full-QKV input comes directly from an L1 gamma-multiply output; the selected path adds no DRAM-to-L1 copy.

## Advice disposition

**decode QKV — completed.** All legal reader1/2/3 isolated controls and all precision-locked whole-layer integrations completed. K11 reader1 exceeds static L1; legal K1 was executed separately. Reader2 wins the K11 microbenchmark, but current whole-layer DRAM alternatives add17.002/6.908us for sliding/full versus selected interleaved QKV. Includes common padding and all movement. Retain current defaults. Evidence: `dram_reader_results.json`, `reader_layer_results.json`.

**decode output projection — completed.** Reader1/2/3 controls use the actual BF16 SDPA/concat producer input, BFP8 weight and FP32 output, not an FP32-input surrogate. Isolated fastest is reader3 sliding/reader1 full; whole-layer alternatives all lose. Full reader1 adds9.544us; sliding reader2/3 add17.712/19.559us. Historical fidelity/subblock trials remain separate. Evidence: `reader_layer_results.json`, `dram_reader_results.json`, `actual_direct_output_commands.json`, `actual_final_precision_commands.json`.

**decode router — completed.** The native weights are BF16, so the report's BFP8 label is inaccurate. Independent actual-input fidelity controls nevertheless exist: sliding K44/HiFi2 passes the headline but fails the 512-step stress at position 1459, PCC 0.9943880922; K22/HiFi4 passes all 512. Full K44/LoFi passes headline, 512-step and recorded reuse-window controls. Larger subblocks on two/one cores all pass but are slower: historical sliding 989.866/998.113us versus987.932us, full1076.886/1080.291us versus1076.212us. Retain four cores and the selected fidelity per kind. Evidence: `router_direct.md`, `direct_router_commands.json`, `router_selected_stress_layer0.json`, `router_repair_hifi4_stress_layer0.json`, `router_selected_stress_layer5.json`, `router_subblock_commands.json`.

**decode expert GEMMs — expanded and compact geometry completed.** Expanded packed44-core gateK44/downK22 beats22cores, gateK22/K88, and matched separate. Indexed sparse API exposes8 compact slots. Actual compact controls confirm44-core gateK44/downK22 wins over22cores,down88,gateK22/K88 and separate on both kinds. Fused GELU now wins slightly after compaction. Evidence: `expert_gate_geometry_results.json`, `actual_direct_mlp_geometry_commands.json`, `compact_geometry_commands.json`.

**expert GELU — selected for compact topology.** Fused accurate GELU plus multiply is selected on compact E8 shapes. Matched compact whole-layer medians 830.638/921.665 us versus 832.163/922.738 us unfused, identical 128 PCCs. The earlier expanded-shape fused control passed but was slower; that rejection does not apply after compaction. Evidence: `compact_geometry_commands.json`, `expert_fused_gelu_source.md`.

**expert weighted mix — expanded and compact fused reduction rejected.** Existing Blackhole weighted-reduce op is shape-legal with FP32 accumulation. Expanded control passed but slowed the whole layer. Compact matmul also beats compact weighted reduction; retain compact matmul. Full compact weighted927.842us versus922.738us matmul; expanded weighted1226.202us. Evidence: `probe_optimized_expert_compact.py`, `AUTODEBUG_weighted_expert_reduce.md`.

**decode RoPE producer layout — completed.** Caller-provided row-major tables avoid full-table untilize. Selected-row slice/repeat/tilize remains. Historical TILE compatibility and the current validation source-delta chain are retained without relabeling old runs. Evidence: `actual_decode_rope_row_major_layer0.json`, `actual_decode_rope_row_major_layer5.json`, `validated_v8_validation_summary.json`.

**prefill output projection — completed.** Sliding retains 2D K16 LoFi. Full selects minimal K8 LoFi after7/8 faster alternating pairs (~204.839us median host-prefill saving in that historical comparison). Actual current minimal-inputL1 passes but screens692.552us slower, so keepDRAMinput/output. This control is separate from the older multicast L1 matrix. Explicit minimal blocks are in native config even when the report says no program_config. Evidence: `minimal_selection.json`, `prefill_output_results.json`, `final_perf_advice_controls.json`.

**prefill QKV — completed.** Select minimalK8 HiFi4 sliding and M2/K16/N8 HiFi2 full, grid11x8. Full QKV input is produced directly in L1 by the input-norm gamma multiply. HiFi2 wins8/8 pairs (-545.573us) and passes512steps/all291maximum-context rows; sliding HiFi2 instead fails actual long-context position32 atPCC0.9949930133358956 versus same-fixture HiFi4 control0.9951303778312409. Original M4 L1 fails a precise CB overlap; legal M2 and N4 adaptations both pass, with M2 stronger. MatchedM2 placement isolates a coupled L1-input/L1-output benefit in8/8 pairs (-664.9255us). Direct producer versus copiedL1 then wins22/32 (-128.295us) with unchanged PCC. M2L1 grid110 gives4/8 faster and+2.8355us median paired delta, no resolved gain. These are whole-prefill host comparisons including movement. Current native rows prove M2/K16/N8, input/outputL1, FP32/BFP8/FP32 and the preceding gamma-multiply L1 producer; output_mem_config=None inherits input placement. Selected integration correctness is separately hash-bound below. Evidence: `minimal_selection.json`, `final_perf_advice_controls.json`, `minimal_pairs_v6_summary.json`, `qkv_l1_results.json`, `prospective_qkv_advice_v7.json`, `validated_v8_validation_summary.json`.

**prefill shared MLP — completed.** Native programs configure11x10 with88active blocks. The legal110-active transpose schedule (M3,N14/9,sub3x2/3x1) passes both realheadline controls but screens766.511/726.980us slower sliding/full. It adds9.375%/5.469% padded gate/down tile work. SharedinputL1 also passes but screens205.697/370.497us slower. Retain existing DRAMinput88active schedule; no API blocker or absolute optimum is claimed. Evidence: `final_perf_advice_controls.json`.

**prefill sparse utilization — report limitation, not optimization failure.** Runtime nnz is the union over32 tokens and varies by chunk. Top8 per token does not imply8 experts per sparse prefill group. Leave utilization unknown absent actual union metadata; do not pass fabricated8. Evidence: `tt/optimized_decoder.py:OptimizedExperts._active_prefill`.

**full decode input norm — completed.** Sharded normalization remains selected at all hidden-width sites. The actual headline/stress controls establish its precision/latency choice; current native rows and current validation establish the integrated path. Evidence: `full_all_norm_headline.json`, `full_all_norm_stress.json`, `validated_v8_validation_summary.json`.

**sliding decode head norms — rejected by complete actual 512-step control.** Native head normalization passes headline but fails actual stress at1257 PCC.994804004648 and1459 PCC.994418854936. The complete run executes all512 checks, audits,cache guard and determinism. Accurate FP32 SFPU head reductions remain selected; native RMSNorm uses different multiply/reduction precision even with FP32 destination. Historical same-fixture precise checks pass these positions; the historical v4 cumulative precision policy also passes its512 stress gate. Evidence: `native_headnorm_headline_layer0.json`, `native_headnorm_stress_layer0.json`, `headnorm_control_summary.json`, `native_headnorm_control.md`, `validated_v4_validation_summary.json`.

**full native rotary — matched topology measured slower.** Existing isolated full-kind boundary control uses the same FP32 head geometry and native HF rotary; it costs approximately24us more. This shape/fidelity performance evidence remains applicable even though earlier activations were synthetic; no accuracy veto is inferred. Decode table producer-layout work is separate and now measured. Evidence: `doc/fused_decoder/PATTERNS.md`, `doc/fused_decoder/work_log.md:full_boundary`.

**prefill grouped routed FFN — named existing operators do not satisfy the mathematical/data contract.** routed_expert_ffn hardcodes SiLU. Unified activation enum supports Silu/SwiGluOai/SituGlu but no GELU and forces BFP8 dispatched arithmetic/output. H2816/I704/E128/oneASIC are not the blocker. moe_expert_token_remap only copies routing scalars and builds a union mask; moe_routing_remap partitions [1,E] weights across devices. Neither gathers token activations or creates the counts/region offsets required by grouped FFN. A new GELU-capable grouped dispatch/combine implementation is required, outside a drop-in existing-op substitution. Evidence: `prefill_routed_ffn_contract_audit.md`.

**prefill router — completed.** Four phase-matched actual4096/128 controls use32 alternating pairs each, exact FP32/BF16/FP32 precision and locked11x10/K8/perM4/perN1/sub4x1 geometry. HiFi2 changes only fidelity; producerL1 changes only the existing scaled-input multiply output, retaining its fused scalar-root operation and explicitDRAMprojection output. All pass HF/cache guards. Sliding HiFi2 has20/32 faster and-67.093us paired median, producerL1 has16/32 and+3.8725us; full HiFi2 has10/32 and+111.637us, producerL1 has20/32 and-32.762us. Paired IQRs cross zero in everycase; apparent savings are at most0.031% of wholeprefill and unresolved against observed variation. Retain HiFi4/DRAM bothkinds. Compute-object strings are opaque; hashed source and selected object identities prove requested fidelity, while current native rows prove the retained config. No historical decode result substitutes for these prefill controls. Evidence: `prefill_router_results.json`, `prefill_router_v8/commands.json`, `validated_v8_validation_summary.json`.

## Historical expert operator controls

These six controls and their selected comparison rows ran on v4 `3d51014f98128dfb21bb484fcece50993524b967754f68f6dfe825ae7f472ba9`. They establish the measured topology/config decisions; they are not current-profile durations. Selected rows used the headline workload; alternatives used separately hashed 4096/1 inputs. End-to-end geometry journals in the archive establish the whole-layer selection.

| Historical policy | Layer | Expert region µs | Gate/up µs | Down µs |
| --- | ---: | ---: | ---: | ---: |
| selected | 0 | 157.231 | 92.862 | 35.861 |
| indexed22k44 | 0 | 171.356 | 96.730 | 45.435 |
| indexed_separatek44 | 0 | 168.856 | 109.720 | 33.301 |
| expanded44k44 | 0 | 322.746 | 94.534 | 37.184 |
| selected | 5 | 133.853 | 68.985 | 35.026 |
| indexed22k44 | 5 | 151.752 | 75.741 | 45.003 |
| indexed_separatek44 | 5 | 151.575 | 88.991 | 34.421 |
| expanded44k44 | 5 | 303.259 | 74.786 | 35.101 |

## Evidence scope

The [v4 archive](final_perf_advice_v4_historical.md) retains the six expert operator controls and historical dtype/topology candidates with their original hashes. Current selected group costs above come only from the current native rows. The [reader audit](reader_layer_results.md) links28 isolated legal cases, two precise K11 reader1 capacity failures and22 complete whole-layer controls; it retains the formatting provenance incident. All22 integrations pass; no meaningful whole-layer reader winner was found. Full shared-down reader1’s0.047us apparent gain is within sample overlap, while reader3 is0.046us slower.

Correctness basis: `validated_v8_validation_summary.json`; 12 fresh primary commands plus5 affected-prefill boundary commands; unchanged sliding broad public/lifecycle coverage explicitly inherited through source proof. Not a full18-command v8 rerun.

Remaining advice evidence: none among the audited recommendations; stage review remains separate.
