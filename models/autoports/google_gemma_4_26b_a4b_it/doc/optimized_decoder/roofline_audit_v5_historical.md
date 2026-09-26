The final v5 roofline totals and same-run timing reconciliations reproduce from the saved raw profiler records for both attention kinds. No numerical accounting correction is required. This CPU-only audit covers runtime SHA-256 `169c0d97d7d0e9f35d97633f133305f1088b987ef625693d3faa100d25c3e67b`; the auditor ran no hardware and changed no runtime or test acceptance.

[Machine-readable audit](final_roofline_audit.json) records input/source hashes, actual operand metadata, complete operation counts, per-operation estimated traffic, and both reconciliations. Reproduce from the repository root with:

```sh
python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/audit_final_roofline_v5.py
```

The command passed against both completed profiles. It checks the successful profile journal, exact raw/copied CSV hashes, runner runtime and recorded-input hashes, four advice-enabled reports per kind, every saved decode replay, all sparse/native-SDPA records, and host signpost timestamps against raw Tracy `total_ns`. Both reconciliations match regeneration using the final two-command journal, with both reconciliations bound to the completed two-command journal.

| Final v5 quantity | Sliding, layer 0 | Full, layer 5 |
| --- | ---: | ---: |
| Complete prefill device time | 221,240.953 µs | 187,940.530 µs |
| Complete decode device mean, 128 replays | 881.434 µs | 900.013 µs |
| Native operations per decode | 123 | 126 |
| Total decode native operations | 15,744 | 16,128 |
| Prefill native operations | 2,173 | 2,109 |
| Summed decode kernel duration, diagnostic only | 752.984 µs | 772.606 µs |
| Useful prefill FLOPs | 882,489,950,208 | 1,215,408,635,904 |
| Useful prefill work / common LoFi peak | 0.601131% | 0.974601% |
| Estimated DRAM operand bytes per decode | 110,493,448 | 106,384,012 |
| Estimated DRAM rate / stated peak | 24.483690% | 23.086477% |

These are batch-one, concurrency-one, single-ASIC results with a 4,096-token recorded real-text layer-input prefill and 128 teacher-forced decode positions 4,096–4,223. [Sliding summary](tracy/actual_optimized_v5_layer0/whole_layer.json) and [full summary](tracy/actual_optimized_v5_layer5/whole_layer.json) use the complete first-firmware-start to last-firmware-end window for each layer invocation. Every native operation and intra-layer gap remains in that denominator; inter-replay input-refresh and host gaps are outside individual device windows. Decode is the mean of all 128 windows; prefill has one measured window. Summed kernel durations are retained only as diagnostics and are not the denominator. The measured cycle conversion is 0.7407407407 ns/cycle in both phases and kinds. See [summarizer](../../tests/summarize_perf.py), lines 260–340.

The common theoretical compute peak is `120 * 4096 * 1.35e9 = 663.552 TFLOP/s` (LoFi), and the stated DRAM peak is 512 GB/s per ASIC. The [official P300 specification](https://docs.tenstorrent.com/aibs/blackhole/p300.html) lists two 120-core ASICs, 1.35 GHz AI clock, and 1,024 GB/s card bandwidth; 512 GB/s is the per-ASIC share. The fidelity-specific compute convention is recorded in [SDPA performance utilities](../../../../../tests/nightly/sdpa_perf_utils.py), line 26, and the [indexer performance test](../../../../../tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_indexer_score.py), line 807. The implementation uses mixed fidelities and SFPU work. These ratios express useful work and estimated operand traffic against a declared common theoretical peak; they do not measure FPU utilization or memory-controller bandwidth. Values are not clamped.

Useful FLOPs count a multiply-add as two operations, using `S=4096`, hidden width `H=2816`, 16 query heads, shared width 2,112, expert width 704, and eight active experts per token. Q/K/V plus output projections are `2*S*H*(2*16*256 + 2*8*256)` for sliding and `2*S*H*(2*16*512 + 2*512)` for full attention; full attention computes tied K/V once. Shared and active-expert projections contribute `6*S*H*2112` and `6*S*H*704*8`, and router projection contributes `2*S*H*128`. Causal QK/PV contributes `4*16*head_width*pairs`, with `pairs=S*1024-1024*1023/2` for sliding and `S*(S+1)/2` for full attention. All five independently calculated terms match [the summarizer](../../tests/summarize_perf.py), lines 220–236. Padding, masked work, additional expert-union prefill computation, scalar normalization, and transcendental operations are excluded from this useful numerator; their time remains in the complete device denominator.

All 256 sparse rows per kind (two per decode) have `use_indices=true`, absent numeric `nnz`, 128 resident weight groups, a compact output group dimension of eight, and a `UINT16` index tensor of logical shape `[1,1,1,8]`. Weight bytes therefore use the observed ratio `8/128`; missing `nnz` does not imply a dense 128-expert read. The [runtime](../../tt/optimized_decoder.py), lines 172–241, passes router indices and eight-slot outputs. The [sparse device operation](../../../../../ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp), lines 65–69 and 230–289, derives compact output/count from indices. The [sparse program factory](../../../../../ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp), lines 75–94 and 292, uses `num_active` for indexed batch computation. Sparse activations, outputs, routing, and indices in these records are in L1; only the selected weight operand contributes generic DRAM traffic.

| Observed precision/fidelity | Sliding | Full |
| --- | --- | --- |
| Decode expert gate/up weights; down weights | BFP8; BFP4 | BFP4; BFP4 |
| Decode expert gate/up input | BF16 | BFP8 |
| Expert matmul compute; output | LoFi; BF16 | LoFi; BF16 |
| Prefill expert gate/up; down weights | BFP8; BFP4 | BFP4; BFP4 |
| Shared decode weights; compute | BFP8; LoFi | BFP4; LoFi |
| QKV weights; compute; output | BFP8; HiFi2; FP32 | BFP8; LoFi; FP32 |
| Output projection weights; compute | BFP8; HiFi4 | BFP8; LoFi |
| Router weights; compute; output | BF16; HiFi4; FP32 | BF16; LoFi; FP32 |
| Native decode SDPA K/V; output; compute | BFP8; BF16; HiFi4 | BFP8; BF16; LoFi |
| Prefill attention compute | LoFi | HiFi2 |
| Prefill QKV backend; K block; compute | Minimal matmul; K8; HiFi4 | Minimal matmul; K16; HiFi4 |
| Prefill output backend; K block; compute | Regular 2D matmul; K16; LoFi | Minimal matmul; K8; LoFi |

Decode entries above are corroborated by raw native operand metadata; prefill policies are recorded by the matching hashed runner. Both decoders use compact indexed experts and fused gate/up GELU. Physical weight shapes are counted as recorded: shared gate/up is `2816 x 4608`, and shared down is `2112 x 3072`, including padding. Those padded widths do not inflate the useful-FLOP numerator. BFP8 storage is 1,088 bytes per 1,024-element tile and BFP4 is 576 bytes, including exponent storage; BF16/UINT16 are two bytes and FP32/INT32 four. Router weight traffic is therefore `2816*128*2 = 720,896` bytes per operand read, not FP32 weight storage. See [datatype sizes](../../../../../tt_metal/impl/data_format/tile.cpp), line 70, and [traffic accounting](../../tests/summarize_perf.py), lines 35–45 and 144–217.

The final native prefill records contain **four minimal QKV operations** for sliding and **four minimal QKV plus four minimal output operations** for full attention. The independent audit reads the actual `M_block_size=4`, selected `K_block_size`, `N_block_size=8`, and `subblock_h=1` / `subblock_w=4` attributes, checks BFP8 weights / FP32 outputs and measured fidelity, and verifies all 4,096 logical projection rows. Sliding retains regular 2D output projection. Logical native QKV/output shapes reproduce 283,467,841,536 and 401,579,442,176 useful projection FLOPs, including full tied K/V once; internal K-block tail work does not inflate useful FLOPs. Raw attributes, tensor metadata and kernel sources are retained in the [sliding native projection audit](tracy/actual_optimized_v5_layer0/prefill_projection_native_audit.json), [full native projection audit](tracy/actual_optimized_v5_layer5/prefill_projection_native_audit.json), and this audit JSON.

The installed `tt-perf-report` recognizes `Matmul` operation names for tensor/FLOP accounting, but its config-advice parser only understands `program_config` / `in0_block_w` / `out_subblock_h/w`. Minimal matmul uses `config` / `K_block_size` / `subblock_h/w`, so missing-program-config advice is a parser limitation when these raw fields are present. The native audits preserve and validate those fields; raw reports and installed tools were not changed. The model-local summarizer includes all native operations regardless of matmul name. See [the profile command/source audit](profile_v5_commands.md).

Native paged SDPA reads use the causal/sliding range at each original position, rounded to its effective 128-token K chunk. The batch-one, non-MLA path has an unsharded DRAM query, separate BFP8 K/V, and FP32 destination accumulation. The dynamic chunk cap is four 32-token tiles in [the program factory](../../../../../ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp), lines 110 and 387–389; [runtime argument partitioning](../../../../../ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp), lines 35–105, floors the start, ceils the end, and distributes disjoint chunks. The [reader](../../../../../ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp), line 279, reads assigned K/V heads and chunks. The actual metadata agrees with the inferred 128-token chunk at every replay.

For `end=position+1`, sliding `start=max(0,end-1024)` and full `start=0`, the K/V input byte estimate is `2*(ceil(end/128)*128-floor(start/128)*128)*kv_heads*head_width*1088/1024`. Sliding reads 1,152 tokens for the first 127 positions and 1,024 for the last: mean 1,151 tokens and **5,009,152 K/V bytes**. Full reads 4,224 tokens at all positions: **9,191,424 K/V bytes**. These are K/V reads only. The allocated 160-page cache is not substituted for the read span. The full estimate additionally counts native SDPA query/output and metadata, cache updates as one 32-token page read/write per cache, all other recorded DRAM operands, and selected embedding/slice inputs. Extra core rereads, NoC/reduction traffic, and profiler writes are outside this estimate; it is not a controller transaction count. [Native source derivation](roofline_native_basis.md) provides the detailed scope.

| Same profiled process, microseconds | Sliding | Full |
| --- | ---: | ---: |
| Estimated operand bytes / 512 GB/s | 215.808 | 207.781 |
| Complete successive-position device mean | 881.434 | 900.013 |
| Refreshed successive-position host-loop mean | 896.600 | 915.311 |
| Fixed-final-position host median | 879.688 | 907.030 |
| Device minus theoretical transfer | 665.626 | 692.232 |
| Refreshed host minus device | 15.166 | 15.298 |
| Fixed-position host minus device | **−1.745** | 7.018 |
| Refreshed host minus fixed-position host | 16.912 | 8.280 |

[Sliding reconciliation](tracy/actual_optimized_v5_layer0/timing_reconciliation.json) and [full reconciliation](tracy/actual_optimized_v5_layer5/timing_reconciliation.json) use the exact matching profile command, CSV, summary, and runner. The raw Tracy message timestamps equal CSV host signpost timestamps in nanoseconds. The host-loop span includes refreshed inputs/positions, copies, submissions, inter-replay intervals, synchronization, and signpost boundaries; HF reference preparation and output readback lie outside it. The fixed-position observation is the median of five 30-replay batches at position 4,223 with no input refresh, after that loop in the same profiled process. See [runner](../../tests/run_decoder.py), lines 360–395, and [reconciler](../../tests/reconcile_perf.py), lines 23–144. Only elapsed durations are compared across clocks. These arithmetic differences do not isolate dispatch, Python, DRAM, synchronization, or contention costs, and the negative sliding difference is retained. Separate unprofiled observations are recorded separately and excluded from every same-run gap.

Profile correctness checks include prefill, first/final decode outputs, repeat equality, and a clean device-only/program-cache capture. They do not check all 128 HF decode outputs within the timed loop. Complete final-runtime correctness, all 291 sampled long-prefill rows, 512-step stress, and watcher evidence remain in [the v5 validation summary](validated_v5_validation_summary.json) and [watcher summary](validated_v5_watcher_summary.json); this performance audit does not redefine their acceptance criteria.

The earlier runtime `e8101829…` source-only audit is retained as [historical Markdown](roofline_audit_e810_historical.md) and [historical JSON](roofline_audit_e810_historical.json). Its common HiFi4 peak was 165.888 TFLOP/s, four times smaller than the current LoFi denominator. Historical percentages cannot be compared directly without normalizing the peak basis; none of those historical values is presented here as a final v5 measurement.

The completed [v4 audit](roofline_audit_v4_historical.md) and [v4 JSON](roofline_audit_v4_historical.json) remain archived under runtime `3d51014f…`; their values are not v5 measurements. The [boundary timing summary](prefill_boundary_summary.md) separately reports warmed host medians at logical lengths 65, 1023, 1024 and 1025 with actual-input hashes and source-derived physical chunk shapes.

This audit validates saved performance accounting. It does not establish allocation-free trace capture or all tight-cache bounds: the separately reported trace-allocation and K256 tight-cache issues fall outside the recorded profile/validation gates and require their own resolution.
