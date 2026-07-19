# Mistral-Small-24B optimized multichip decoder

Status: implementation, hardware gates, profiling, watcher stress, and the final independent `$stage-review` pass; the scoped local commit is pending. This stage optimizes the completed `MultichipDecoder` in place on a logical `1x4` Blackhole p300c mesh. It does not start full-model, generator, vLLM, or serving work.

## Final result

The final default is the real TP4 decoder path: DRAM-sharded decode matmuls, packed QKV, separate gate/up projections, BFP4 LoFi weights/kernels, BF16 attention and MLP activations, BFP8 decode collective outputs, explicit persistent-buffer `all_reduce_async`, and a stack-preserved residual contract. Sequential layers share one 1,310,720-byte BF16 L1 CCL workspace and one semaphore set. Prefill uses the general BF16 all-reduce because the fully adapted persistent family was materially slower. Collective phase selection is explicit, so batch-1 prefill cannot be misclassified as decode merely because its flattened token dimension is tile-sized.

The primary measurement is batch 1. “Before” is the completed TP4 public/general-collective contract and “final” is the current internal stack/default collective contract, both with real layer-20 weights on the same `1x4` mesh.

| Primary real-weight TP4 metric | Before, two layers | Final, two layers | Final per layer | Reduction |
| --- | ---: | ---: | ---: | ---: |
| warmed prefill, batch 1 × logical S=18, 100 iterations | 2.101789 ms | 1.896023 ms | 0.948012 ms | 9.79% |
| traced warmed decode, batch 1 × 1, 100 replays | 0.915775 ms | 0.829644 ms | 0.414822 ms | 9.41% |

Prefill before/final are the adjacent public/internal timings in `prefill_sharded_norm_control_batch1_100.log`, captured from the final default after the sharded-norm experiments. Decode before is the public general-BF16 result in `control_batch1_general_ccl_stack_decode_100.log`; final is the current persistent-BFP8 internal result in `final_batch1_stack_decode_100.log`. The final path also reports 0.841827 ms through its public wrapper; the table deliberately includes the collective improvement in the completed-to-final comparison.

Batch 32 remains the secondary stress and context-capacity regime. The exact final-source consolidated gate reproduces 0.403468 ms/layer traced decode and 2.240332 ms/layer warmed prefill. The longer canonical final-default captures are 0.402990 ms/layer over 300 decode replays and 2.224263 ms/layer over 20 prefill iterations. These are current-default results, not superseded candidates.

The model has one meaningful dense decoder layer kind. Final real optimized-TP1 comparisons are:

| Active batch | Prefill PCC | Decode PCC | K / V PCC | Evidence |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.999985284 | 0.999989612 | 1.0 / 1.0 | `final_batch1_real_tp1_{export,compare}.{log,xml}` |
| 32 | 0.999994280 | 0.999988553 | 1.0 / 1.0 | `final_current_real_tp1_{export,compare}.{log,xml}` |

The HF stress gate remains 0.995336–0.995555 prefill for logical S=17/18/32, 0.994984 decode, and at least 0.993675 cache PCC. Reversed paged-cache mapping is 1.0 against the contiguous logical control. Existing acceptance is PCC 0.99 against optimized TP1/HF and 0.9999 for final-path candidate/default comparisons. Mistral-Small-24B is dense, so the MoE active-expert requirement is not applicable.

## Operation-topology audit

The audit preceded local tuning and covered all requested topology families.

| Family / boundary | Starting topology | Action and evidence |
| --- | --- | --- |
| repeated same-input matmuls | separate gate/up consume one hidden tensor; QKV is packed | Retain packed QKV. Adapted packed gate/up is 0.811856 vs 0.807788 ms/two layers, PCC 0.999983; reject. |
| decode inter-layer movement | every layer restored public `[1,B,1,H]` DRAM and the next rebuilt tiled L1 state | Carry `[1,1,B,H]` BF16 in the 11-core L1 block-sharded residual config. No gather, reshard, all-reduce, or public restore occurs between layers. Batch 1 owns a device-side logical row slice so padded physical rows never leak into the next layer or public API. |
| prefill inter-layer movement | every layer restored/permuted public token axes | Carry `[1,1,B·S,H]` BF16 DRAM plus explicit logical `S`; convert only at stack entry/exit. S=17 and S=18 remain valid public inputs. |
| material decode collectives | two general all-reduces per layer exposed DRAM reduce-scatter/all-gather movement | Use explicit Linear/2-link persistent `all_reduce_async`, BFP8 output, shared BF16 L1 workspace/semaphore. Batch-1 two-layer internal latency is 0.829644 vs 0.907943 ms for the adjacent general-BF16 control. |
| collective placement / residual family | replicated DRAM, hidden-fractured, fused interleaved, and L1 residual carries were possible | Hidden-fractured is 0.155569 vs 0.102029 ms for its coherent replicated chain. The final 11-core L1 residual family wins end to end. Exact remaining intra-layer conversions are documented below. |
| persistent buffers | runtime-owned transient collective state | Preallocate and share one workspace/semaphore across layers. Identity, two-layer trace reuse, 40-layer capacity, and watcher tests pass. |
| fused matmul-CCL | WO/down matmul is adjacent to each TP reduction; MRS and AGMM APIs exist | Adapt through subblock, rank-4 weight, semaphore, layout, and padding failures, then measure both exact final boundaries. The coherent fused sum is 0.842464 vs 0.347050 ms control (2.4275× slower); reject. |
| activation sharding | tuned decode sharding; profiler suggested L1 prefill input | Retain decode sharding. Prefill width sharding hits an unsupported kernel constraint; adapted 5×9 block sharding passes but is 4.110194 vs 3.809161 ms DRAM. |
| prefill RMSNorm placement | profiler attributes 24.66% to four DRAM/single-core norms over two layers | Adapt first failures, then feed QKV with an 8×1 block-sharded norm and MLP with a 10×1 width-sharded norm. Isolated QKV/MLP trials are 3.98%/1.69% slower; the coherent combined 100-iteration family is 2.254134 vs 1.896023 ms (18.89% slower), PCC 0.999889. Retain DRAM norms. |
| DRAM-sharded decode matmuls | QKV, WO, gate, up, and down use DRAM-sharded weights and L1 input | Retain. The exact shard-advisor 1D family passes at PCC 0.999984 but regresses 0.579367 to 0.755150 ms. |
| precision/fidelity | BF16 activations/collectives, BFP4 LoFi dense weights | Select BFP8 decode collective output. BF16 activations, BFP4 weights, and LoFi win; BFP8 activations/weights and HiFi2 lose or fail PCC. |
| prefill persistent CCL | general BF16 all-reduce | Adapt DRAM-layout, 40/80-core L1, CB-size, tile-chunking, and per-chunk L1→DRAM failures. The passing PCC-1.0 chunked path is 12.818221 vs 4.448526 ms; reject. |

No collective, gather, or reshard remains between decoder layers. The two collectives and two width-to-block conversions that remain per layer are intra-layer TP work. Every applicable family was tried and closed with evidence.

## Exact final decode topology

The final async CCL does **not** write the residual layout directly. Each WO/down partial is converted to the 40-core L1 width-sharded CCL input, reduced with Linear topology, two links, cluster axis 1, BFP8 output, and the shared BF16 workspace, then converted to the 11-core L1 block-sharded norm/residual family before the BF16 residual add. `AllReduceAsyncDeviceOperation` requires input, buffer, and output memory layouts to be `WIDTH_SHARDED` (`all_reduce_async_device_operation.cpp`, validation lines 45–57); the 11-core norm family is `BLOCK_SHARDED`. Thus a direct async-CCL-to-residual output is rejected by an exact C++ validator constraint, not a first Python/API error. Fused MRS alternatives were separately adapted and measured end to end.

| Decode operation | Final policy / configuration | Primary-profile share |
| --- | --- | ---: |
| QKV | packed, `32×5120×1536`, DRAM-sharded BFP4, `in0_block_w=16`, per-core M/N=1/4 | included in matmul 63.46% |
| SDPA | BF16, HiFi4, grid 8×8; contiguous auto chunks, paged K chunk 128 | 1.90% |
| WO | `32×1024×5120`, DRAM-sharded BFP4, `in0_block_w=4`, per-core M/N=1/16 | included in matmul 63.46% |
| gate/up | separate `32×5120×8192`, DRAM-sharded BFP4, `in0_block_w=16`, per-core M/N=1/8 | included in matmul 63.46% |
| down | `32×8192×5120`, DRAM-sharded BFP4, `in0_block_w=8`, per-core M/N=1/4 | included in matmul 63.46% |
| two RMSNorms | BF16, 11×1 block-sharded L1, block H/W=1/15, subblock W=3 | 4.07% |
| two async all-reduces | Linear, 2 links, axis 1, 40-core width-sharded L1 BFP8 output, persistent BF16 workspace/semaphore | 10.82% |
| two residual adds | BF16 output in the 11-core block-sharded residual config | binary families 7.93% |
| layout conversions | includes 40-core width-sharded → 11-core block-sharded twice plus intra-layer matmul-input preparation | all reshard families 2.65% |

The Python `all_reduce_async` overload exposes topology, links, axis, dtype, memory configs, semaphore, and workspace; worker/chunk/buffer scheduling is internal to this operation and is not an application-settable field. Those fields are available and recorded for the rejected fused MRS/AGMM candidate (`chunks_per_sync=10`, two workers/link, two buffers/channel), but they must not be attributed to the final async all-reduce.

## Coherent family decisions

| Family | Candidate result | Decision |
| --- | --- | --- |
| decode CCL precision | paired 300-replay BF16 0.807669 ms, BFP8 0.806020 ms; PCC 0.999990825 | BFP8 |
| CCL topology | exact-payload Linear 0.114246 ms, Ring 0.117872 ms | Linear |
| packed gate/up | 0.811856 vs separate 0.807788 ms; PCC 0.999983 | separate |
| attention / MLP activation BFP8 | 0.828778 / 0.824331 vs BF16 0.807788 ms | BF16 |
| attention geometry | 0.826607 and 0.864758 vs default 0.806169 ms | `(10,12,16,8,10,4)` |
| MLP geometry | 0.832353 and 0.902445 vs default 0.806169 ms | `(10,32,40,16,16)` |
| attention BFP8 weights | 0.813754 vs BFP4 0.806169 ms; PCC 0.999975 | BFP4 |
| MLP BFP8 weights | block 16 exceeds L1; block-8 retry 0.905056 ms and PCC 0.999398 | BFP4 |
| attention / MLP HiFi2 | 0.845966 / 1.192333 vs LoFi 0.806169 ms | LoFi |
| exact fused boundary A: WO→reduction→residual→norm→gate/up | 0.480580 vs 0.209183 ms control | reject, 2.297× slower |
| exact fused boundary B: down→reduction→residual→next norm→QKV | 0.361885 vs 0.137867 ms control | reject, 2.625× slower |
| fused A+B sum | 0.842464 vs 0.347050 ms; projection rank PCC ranges 0.999603–0.999626 | reject, 2.4275× slower |
| prefill persistent CCL | passing adapted path 12.818221 vs general 4.448526 ms; PCC 1.0 | general BF16 |
| prefill sharded RMSNorms | isolated QKV 2.167228 and MLP 2.119506 vs 2.084208 ms control; combined consumer-ready 2.254134 vs 1.896023 ms over 100 iterations, PCC 0.999889 | DRAM norms |
| prefill trace/op-gap advice | decoder-local trace replay 1.401168 vs warmed eager 1.896023 ms/two layers; eager/trace PCC 1.0 | graph is trace-safe; retain warmed eager as the required decoder default metric |

The DRAM-sharded configs expose no output-subblock knob, so `tt-perf-report`’s subblock advice is not directly applicable. Physical K-block maxima are QKV 16, WO 4, gate/up 16, and down 8 after input-core divisibility. Alternate grids, L1 placement, activation/weight dtype, and fidelity advice were all tried.

## Final profiler evidence

The primary profiler captures are real two-layer batch-1 default paths:

- `tracy/final_batch1_stacked_decode_bfp8`: processed merged and per-device reports over `MULTICHIP_INTERNAL_STACK_DECODE`. Matmuls are 63.46%, async all-reduces 10.82%, residual binary families 7.93%, norms 4.07%, reshards 2.65%, and SDPA 1.90%. Modeled DRAM roofline is 32.4% merged and 33.4% per-device.
- `tracy/final_batch1_profile_stacked_prefill_bf16_pass`: dedicated passing internal-stack capture over `MULTICHIP_INTERNAL_STACK_PREFILL`. Matmuls are 42.95%, norms 24.66%, reduce-scatter/all-gather 11.60%, reshape 5.04%, and SDPA 1.17%. Modeled DRAM roofline is 19.8% merged and 19.9% per-device.

Both directories contain human-readable `perf_report_table.txt`, merged/per-device processed ops CSVs, summary CSV/PNG, and report stdout. Profiler one-replay wall values are diagnostic; the warmed/traced headline comes from the non-profiler 100-iteration/replay runs. The earlier batch-32 reports remain secondary stress provenance in `tracy/final_stacked_{decode,prefill}_bfp8`.

The batch-1 decode three-number accounting is explicit in `evidence/profile_accounting.csv`. The retained profile contains 4,182.453333 µs of device kernels across six executions of two layers, or 348.538 µs/layer. Per rank, one layer stores 78,807,040 bytes of BFP4 tile payload plus BF16 norms and reads 17,408 bytes of physical-tile K/V at position 18. Across four ranks that is 315,297,792 bytes. At the report model's 512 GB/s/chip (2,048 GB/s aggregate), the theoretical bandwidth floor is 153.954 µs/layer. Device time is therefore 2.264× the floor (44.17% theoretical/device), reflecting nonideal small DRAM-sharded matmuls plus norms, SDPA, layout work, and CCLs that this byte-only floor does not model. The canonical 100-replay wall result is 414.822 µs/layer, leaving 66.284 µs (15.98%) beyond device kernels for trace replay submission, runtime synchronization, and host-side measurement. The same profiler capture's noisier one-replay wall value is 456.946 µs/layer; it is recorded separately rather than substituted for the warmed headline.

The prefill report's “high op-to-op gap” advice predicted up to 679 µs, or 29.4%, from tracing. It was exercised in the decoder-local two-layer graph: `prefill_trace_batch1_100.*` is PCC 1.0 and measures 1.401168 ms/two layers (0.700584 ms/layer), 26.10% faster than the adjacent final warmed-eager 1.896023 ms. This validates trace safety and classifies the advice. Prefill trace capture/replay remains a caller-owned orchestration option; it does not replace the required warmed eager default metric or start full-model work.

## Inter-layer and context contract

- Decode stack input/output is logically `[1,1,B,5120]`, BF16, 11-core L1 block-sharded. Supported active batches include 1 and 32. Tile padding is internal; batch 1 is sliced to its logical row count on device before the next layer/public finish.
- Prefill stack input/output is `[1,1,B·logical_S,5120]`, BF16 DRAM interleaved plus explicit `logical_seq_len`. Collective phase is explicit.
- Full-model bringup must call `prepare_*_residual` once, every layer’s `*_forward_stacked`, and `finish_*_residual` once, sharing RoPE and `shared_collective` across layers.
- Restoring `[1,B,S,H]`, gathering, resharding, or reducing between layers violates the optimized contract. The documented intra-layer CCL-output conversions are part of each decoder layer, not the inter-layer handoff.
- Public non-aligned logical sequence lengths remain supported. The decoder owns flattening, padding, masking, cache slicing, logical-length tracking, and final restoration.
- KV cache remains per-rank local-head BFP8 in contiguous or paged layout. The reusable CCL workspace is BF16 L1 and does not reduce DRAM context.
- `context_contract.json` retains 32,768 tokens. Forty layers, 40 K/V cache pairs, TP endpoints, shared RoPE/workspace, and a physical 4 GiB reserve remain resident while paged decode passes at position 32,767. The reserve-adjusted calculated ceiling remains 37,344.

Batch-32 full-context prefill still requires decoder-owned streaming because a fully materialized replicated 32K BF16 residual is 10,737,418,240 bytes per rank. This is an existing prefill limitation, not a context-contract reduction.

## Final gates and artifacts

- Primary batch-1 performance/PCC: `final_batch1_stack_decode_100.*`, `prefill_sharded_norm_control_batch1_100.*`, `control_batch1_general_ccl_stack_decode_100.*`, and `final_batch1_real_tp1_{export,compare}.*`.
- Final-source batch-32 gate: seven tests pass in 147.98 s (`final_current_source_gate_bfp8.{log,xml}`): runtime ownership, S=17/18/32, reversed paged mapping, mutable traced positions, two-layer decode/prefill, and 40-layer/32K capacity.
- Final-source real optimized TP1 comparison: `final_current_real_tp1_{export,compare}.{log,xml}`.
- Exact fused rejection: `fused_exact_boundaries_real_bfp8.{log,xml}`; earlier blockers/adaptations are retained beside it.
- Prefill norm and op-gap closure: `prefill_{qkv,mlp}_sharded_norm_candidate_batch1_20.*`, `prefill_sharded_norm_candidate_batch1_consumer_layout_100.*`, and `prefill_trace_batch1_100.*`; all earlier sharding blockers/adaptations are retained.
- Primary Tracy/tt-perf: `tracy/final_batch1_stacked_decode_bfp8/` and `tracy/final_batch1_profile_stacked_prefill_bf16_pass/`.
- Batch-1 watcher: `watcher_final_batch1_two_layer_shared_bfp8.{log,xml}`. It attached/detached devices 0–3; `watcher_error_audit_batch1_bfp8.txt` is empty and `post_watcher_health_batch1_bfp8.log` reports four visible p300c boards. Ethernet watcher remains disabled for the established platform firmware limitation; worker/fabric execution is covered.
- Shard advisor: `shard_advise/report.json`, `report.txt`, and `final_ir.mlir`.
- Candidate/performance provenance: `candidate_summary.csv`, `performance_summary.csv`, `profile_accounting.csv`, `AUTODEBUG.md`, and `work_log.md`.

Runtime fallback audit is clean. Stress covers non-aligned logical lengths, active batch 1, batch-32 capacity, paged physical mapping, mutable trace inputs, shared persistent buffers, two-layer inter-layer state, maximum logical position, and watcher-clean replay.
