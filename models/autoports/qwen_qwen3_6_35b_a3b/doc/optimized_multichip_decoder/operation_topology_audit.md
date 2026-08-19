# Operation Topology Audit

Model: `Qwen/Qwen3.6-35B-A3B`

Stage: optimized multichip decoder

Pre-stage SHA: `c90c9c4336956c895f7481729f66e4a866b9d678`

This audit was the first optimization step for the measured multichip decoder
path. Inputs were the completed multichip decoder source, the inherited
optimized-decoder contract, and the pre-change one-link multichip Tracy run.

## Final Inter-Layer Contract

The selected decoder layer boundary is a replicated BF16 DRAM/interleaved
hidden tensor with logical shape `[1, batch, seq, 2048]`. There is no gather,
reshard, all-reduce, reduce-scatter, or all-gather between decoder layers in
the final path. Full-model bringup should preserve this residual contract
instead of rediscovering it or inserting layer-boundary collectives.

Public inputs do not require aligned logical sequence lengths. The decoder owns
any internal padding, masks, cache updates, and output slicing.

## Audit Matrix

| Area | Current topology | Candidate or advice | Action and evidence |
| --- | --- | --- | --- |
| Repeated same-input matmuls | Completed multichip decoder already packs full-attention `q/k/v`, linear-attention `qkv/z/b/a`, shared gate/up, and routed gate/up. | Split and retune separate projections only if packing loses. | Kept packed. `logs/perf_report_top_ops.csv` shows the material dense projection rows as packed widths such as `32 x 2048 x 6176` and `32 x 2048 x 4608`; no Q/K/V split cluster remains to remove. |
| Material collectives | Layer-internal row-parallel outputs use all-reduce on TP or EP axes. Layer boundaries are replicated. | Increase link count, async CCL, BF8 payload, collective placement, persistent buffers. | Accepted `num_links=2`. Final CCL time improved in all profiled multichip windows: linear prefill `269.703 -> 228.146 us`, full prefill `1535.556 -> 1318.758 us`, linear decode `89.982 -> 75.970 us`, full decode `93.984 -> 77.437 us`. |
| CCL algorithm | Public `ttnn.all_reduce`, one link in the old default. | Explicit `reduce_scatter_minimal_async` plus `all_gather_async` with CCL helper semaphores. | Retried with real tensor path, not rejected on API shape alone. Correctness matched baseline in `logs/candidate_explicit_rs_ag_correctness.log`; perf in `logs/candidate_explicit_rs_ag_perf.log` was slower or tied (`19.619`, `34.567`, `1.540`, `1.095` ms) than the two-link public all-reduce screen (`19.459`, `31.420`, `1.326`, `1.084` ms). Rejected. |
| CCL dtype | BF16 activation and residual stream. | BF8 CCL payload with cast down before CCL and cast back after CCL. | Rejected for accepted-baseline PCC loss in `logs/candidate_bf8_ccl_correctness.log`: linear `0.9999037412/0.9998964355`, full `0.9998968992/0.9999007654`, non-aligned linear `0.9998986860/0.9998931672`, non-aligned full `0.9998968992/0.9999007654`. |
| Residual layout | Replicated BF16 DRAM/interleaved residual at layer boundaries. | Carry a width-sharded L1 residual through RMSNorm, real mixer, residual add, real MoE/MLP, and final residual. | Not rejected only by immediate restore. First attempts exposed the exact RMSNorm/output-sharding rules in `logs/candidate_width_sharded_residual_rmsnorm_repro.log` and `logs/candidate_width_sharded_residual_rmsnorm_adapted_repro.log`. The stack-compatible probe `logs/candidate_sharded_residual_stack_probe.log` then ran through real linear and full decoder layers. It kept input/post RMSNorm and residual adds WIDTH_SHARDED L1, but real mixer and MoE/MLP outputs restored to DRAM/interleaved. It was slower for both meaningful layer kinds: linear `3.454771 -> 4.485228 ms`, full `2.555675 -> 3.238674 ms`; PCC stayed valid (`0.9998683319`, `0.9993485173`). Rejected. |
| Activation sharding | Decode residual is replicated/interleaved at the public layer boundary; inherited internal optimized decoder uses sparse expert and helper-local layouts. | Phase-specific width-sharded activations for DRAM-sharded decode matmuls. | Measured in DRAM-sharded repros and the stack-compatible sharded residual probe. Full qkgv was faster only when the activation was already sharded (`0.065955 -> 0.050056 ms`), but including the required current-boundary conversion/restore was slower (`0.069743 -> 0.107414 ms`). Linear qkvzba with padding/slice was slower (`0.077718 -> 0.092857 ms`). The stack-compatible residual candidate also lost end-to-end for both layer kinds, so the lower-movement family is rejected under the final residual contract. |
| DRAM-sharded decode matmuls | Dense decode projections are interleaved-weight inherited path. | Use DRAM-sharded program configs and adapted padding/slicing. | Tried with adapted shapes and legal padding. Full qkgv and linear qkvzba results are in `logs/candidate_dram_sharded_full_qkgv_repro.log`, `logs/candidate_dram_sharded_full_qkgv_with_convert_repro.log`, and `logs/candidate_dram_sharded_linear_qkvzba_repro.log`; rejected because whole-contract candidates were slower. |
| Fused matmul-CCL | Row-parallel matmuls followed by collectives are layer-internal. | Fused all-gather plus matmul with adapted output-sharded weights; fused matmul/reduce-scatter source scan. | `logs/fused_ccl_api_source_audit.log` records the source scan. The first `all_gather_minimal_matmul_async` runtime attempt hit the worker-grouping rule; the probe was adapted to legal 2-link TP axis 1 topology with `num_workers_per_link=4`. `logs/candidate_fused_agmm_bf16_nonpersistent_probe.log` records BF16 weights, output-sharded adapted weights, `cluster_axis=1`, `force_transpose=True`, `num_links=2`, and the legal worker setting before hang. Triage in `triage/fused_agmm_bf16_nonpersistent/tt-triage.txt` shows NOC/fabric-router symptoms (`check_noc_status.py:331`, fabric router callstacks from line `350`). Earlier persistent fused retry has the same family of symptoms in `triage/fused_agmm_persistent/tt-triage.txt`. Rejected with adapted runtime evidence after shape/layout/worker changes, not a first API error. |
| Persistent/preallocated CCL buffers | Final selected public `ttnn.all_reduce` path allocates through the public op. | Use persistent output/intermediate buffers for repeated decode CCLs through the experimental RS/AG path. | Public `ttnn.all_reduce` headers/nanobind expose no persistent-buffer arguments. The buffer-bearing explicit RS/AG probe used `[1,1,32,2048]` input, `[1,1,32,1024]` reduced output, preallocated DRAM intermediate/reduced/gathered buffers, and both TP axis 1 and EP axis 0. `logs/candidate_persistent_rsag_probe.log` ran and rejected the path: TP nonpersistent/persistent `0.171927/0.174642 ms`, EP `0.175274/0.172148 ms`, and both failed public all-reduce correctness at about `0.949` PCC. The model-level nonpersistent explicit RS/AG candidate also passed baseline PCC but was slower than public all-reduce, so no persistent CCL path is selected. |
| Collective placement | TP reductions after attention/linear-attention outputs and shared down; EP then TP after routed expert output. | Move collectives later and carry sharded/fractured residuals between layers. | Rejected by residual-layout evidence. Final path has no inter-layer collective. Remaining collectives are layer-internal and measured faster with two links. |
| Packed vs separate projections | Packed projections are inherited. | Reopen split projection families. | Not selected. The optimized-decoder stage already compared packing and precision/fidelity for these groups; this stage's topology audit found no repeated same-input matmul group left to remove in the multichip measured path. |
| Activation/weight precision and fidelity | Inherited optimized-decoder policy: BFP8 dense/shared weights, layer-kind routed MoE dtype, BF16 state/cache/residual outputs, LoFi sparse expert rows where selected. | Multichip-specific activation/CCL dtype and precision reductions. | CCL BF8 was the material multichip precision candidate and was rejected on PCC. Weight precision/fidelity was not changed in this pass; final report rows prove the inherited policy reached measured ops, for example `SparseMatmulDeviceOperation active=4/256 ... LoFi BF16 x BFP8 => BF16` and dense rows `HiFi2 BF16 x BFP8 => BF16`. |
| MoE active experts | Per-token gate-selected active sparse path with EP row routing and TP/EP reductions. | Dense all-expert runtime path. | Dense all-expert execution was not selected. Final `tt/multichip_decoder.py` keeps `moe_routing_remap`, `ttnn.sparse_matmul`, per-row active routing, EP all-reduce, and TP all-reduce. |

## Coherent Family Comparison

| Family | Compared variants | Decision |
| --- | --- | --- |
| Residual layout | replicated boundary; sharded RMSNorm-only with sharded output; real stack-compatible width-sharded residual through mixer/residual/MoE; DRAM-sharded projection under sharded-only and convert/restore variants | replicated boundary selected; lower-movement family lost in the stack-compatible decode probe for both linear and full layer kinds |
| Collective placement | existing layer-internal reductions; explicit RS/AG plus gather; delayed sharded residual family | public two-link all-reduce selected; no inter-layer collective remains |
| Fused CCL plus matmul | source-compatible fused reduce-scatter/matmul check; adapted all-gather-minimal-matmul with output-sharded weights on TP axis 1; persistent and non-persistent fused retries | rejected because legal adapted fused AGMM hangs in fabric/router state and requires reset |
| Packed versus separate projections | inherited packed projections versus no remaining repeated same-input matmul cluster | packed retained |
| Activation and CCL dtype | BF16 CCL versus BF8 CCL under same residual/all-reduce topology | BF16 retained because BF8 missed accepted PCC baseline |
| Persistent buffers | public all-reduce no persistence; explicit RS/AG semaphores without persistent buffers; explicit RS/AG with preallocated intermediate/reduced/gathered buffers on TP and EP axes | public all-reduce retained because persistent RS/AG does not preserve public all-reduce correctness in the microprobe and model-level nonpersistent RS/AG is slower |
| DRAM-sharded decode matmuls | sharded-only, convert/restore, and padded/sliced linear variants | rejected under the final residual contract because whole-contract variants were slower |

## Actioned Report Advice

`tt-perf-report` advice remained enabled in the regenerated tables under
`tracy/baseline_reports/` and `tracy/final_reports/`.

| Advice | Action |
| --- | --- |
| Increase CCL efficiency | accepted two-link Ring CCL default based on Blackhole p300c source guidance and measured CCL/device-time wins |
| Try async CCL | explicit RS/AG candidate measured and rejected |
| Try BF8 activation/CCL payload | BF8 CCL candidate measured and rejected for PCC baseline drop |
| Try DRAM-sharded decode matmul | full qkgv and linear qkvzba adapted candidates measured and rejected |
| Place input 0 in L1 / avoid layout movement | lower-movement residual and sharded activation candidates measured and rejected |
| Use fused CCL+matmul where possible | inspected source/tests, adapted shape/layout/weights/worker grouping, captured persistent and non-persistent fused AGMM hangs with triage/reset evidence |
| Use persistent CCL buffers | inspected public and experimental CCL APIs, ran explicit RS/AG with preallocated buffers on TP and EP axes, rejected on correctness and no model-level perf win |
| Preserve packed projections | verified packed topology remains the measured path |

## Performance Accounting

Final default decode values come from the final Blackhole-normalized
`tt-perf-report` tables and the same final Tracy/screen runs:

| Decode window | Modeled DRAM roofline from report | Device time us | End-to-end traced wall ms |
| --- | ---: | ---: | ---: |
| `MC_LINEAR_DECODE` | 5.2%, 27 GB/s | 1152.415 | 1.346 profiled / 1.281 screen |
| `MC_FULL_DECODE` | 4.4%, 23 GB/s | 968.534 | 1.152 profiled / 1.096 screen |

The gap between modeled DRAM roofline and end-to-end wall is dominated by many
small inherited TTNN elementwise/routing/layout rows and dispatch gaps in a
single decoder layer. This multichip pass removed the material CCL waste it
could remove without changing the inherited optimized-decoder math contract.

## Final Audit Notes

- Final default path uses `num_links=2`, `ccl_mode="all_reduce"`, and
  `ccl_dtype="bf16"` with no `QWEN36_MULTICHIP_*` environment overrides.
- `test_multichip_decoder_graph_summary` asserts those defaults.
- Runtime fallback audit is clean both statically and dynamically.
- Stress coverage matches the risk of this scoped change: synthetic and real
  linear/full layers, traced decode, non-aligned logical lengths, batch 2,
  fallback-on-exception, watcher, and final no-env perf screens all pass.
- No max-context or public-shape contract changed.
- No stage-local multichip optimization from the prompt remains deferred.
