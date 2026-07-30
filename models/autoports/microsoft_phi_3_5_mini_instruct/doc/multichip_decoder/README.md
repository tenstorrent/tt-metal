# Phi-3.5 Mini Multichip Decoder

Multichip decoder pass for `microsoft/Phi-3.5-mini-instruct` under:

`models/autoports/microsoft_phi_3_5_mini_instruct`

## Scope

This stage starts from the completed optimized decoder and implements the repo-local multichip single-layer decoder path:

- `tt/multichip_decoder.py`
- `tests/test_multichip_decoder.py`
- `doc/multichip_decoder/`

The single-chip baseline is `tt/optimized_decoder.py::OptimizedDecoder`. This model has one meaningful decoder layer kind: dense Phi-3.5-mini self-attention plus dense gated MLP. There is no MoE layer kind, no expert routing, and no full-model or vLLM work in this stage.

## Target Hardware

Hardware inspection on June 15, 2026 reported:

- Architecture: `wormhole_b0`
- Cluster type: `T3K`
- Available devices: 8 Wormhole chips, four N300 boards

The selected target is a `1x8` TTNN mesh with `ttnn.FabricConfig.FABRIC_1D_RING` and `ttnn.Topology.Ring`. The tensor-parallel factor is fixed at 8. Smaller meshes are intentionally out of scope for this autoport stage.

## Mesh Strategy

The runtime residual stream is replicated across all 8 chips. This keeps Phi RMSNorm exact without a norm collective on every sublayer and gives every column-parallel matmul the full hidden vector locally. The large weight matrices are tensor-parallel sharded at load time:

- QKV and MLP gate/up use column/output sharding.
- Attention O and MLP down use row/input sharding.
- Q/K/V heads and KV cache are local-head sharded.
- Row-parallel projection partials are summed with `ttnn.all_reduce` over mesh axis 1.
- Page table, current positions, position IDs, norm weights, and RoPE tables are replicated.

The decoder input and output layout contract is the same: replicated mesh tensor, TILE layout, logical shape `[1, 1, tokens, 3072]`. This is suitable for a stacked decoder because no gather or reshard is needed between layers.

## Per-Device Shapes

| Tensor or weight | Global logical shape | Mesh policy | Per-device shape | Padding |
| --- | ---: | --- | ---: | --- |
| Hidden residual | `[1, 1, T, 3072]` | replicated | `[1, 1, T, 3072]` | none |
| RMSNorm weights | `[1, 1, 1, 3072]` | replicated | `[1, 1, 1, 3072]` | none |
| QKV weight, transposed | `[3072, 9216]` | reordered then shard dim `-1` | `[3072, 1152]` | none |
| QKV output | `[1, 1, T, 9216]` | local Q/K/V chunk | `[1, 1, T, 1152]` | none |
| Local Q heads | `[1, 32, T, 96]` | 4 heads/chip | `[1, 4, T, 96]` | none |
| Local K/V heads | `[1, 32, T, 96]` | 4 heads/chip | `[1, 4, T, 96]` | none |
| Paged K/V cache | `[blocks, 32, 32, 96]` | local-head cache replicated as local tensors | `[blocks, 4, 32, 96]` | none |
| Attention O weight, transposed | `[3072, 3072]` | shard dim `-2` | `[384, 3072]` | DRAM shard local N already tile-aligned |
| Local attention concat | `[1, 1, T, 3072]` | 4 heads/chip | `[1, 1, T, 384]` | none |
| Attention O partial | `[1, 1, T, 3072]` | partial sum | `[1, 1, T, 3072]` | none |
| Gate/up weight, transposed | `[3072, 16384]` | reordered then shard dim `-1` | `[3072, 2048]` | local physical DRAM shard N pads to 2304 |
| Local gate/up output | `[1, 1, T, 16384]` | local gate plus local up | `[1, 1, T, 2048]` | logical none |
| Local gate | `[1, 1, T, 8192]` | 1024 intermediate/chip | `[1, 1, T, 1024]` | none |
| Local up | `[1, 1, T, 8192]` | 1024 intermediate/chip | `[1, 1, T, 1024]` | none |
| Down weight, transposed | `[8192, 3072]` | shard dim `-2` | `[1024, 3072]` | DRAM shard local N already tile-aligned |
| Down partial | `[1, 1, T, 3072]` | partial sum | `[1, 1, T, 3072]` | none |

`T` is prefill sequence length or decode batch rows. Decode currently supports batch size 1, matching the optimized baseline.

## Load-Time Reordering

Two Phi weights require explicit reordering before mesh sharding:

- QKV starts in HuggingFace order `[all Q][all K][all V]`. The multichip loader reorders it to `[Q0 K0 V0][Q1 K1 V1]... [Q7 K7 V7]`, where each local Q/K/V block is 4 heads * 96 = 384 columns.
- Gate/up starts in order `[all gate][all up]`. The loader reorders it to `[gate0 up0][gate1 up1]... [gate7 up7]`, where each local gate/up block is 1024 columns.

Without this reordering, contiguous mesh shards would give some chips only Q or only gate columns and would not match local head/cache ownership.

## Rejected Alternatives

| Alternative | Decision | Reason |
| --- | --- | --- |
| Flatten smaller mesh factors | Rejected | The available target is T3K 8-chip and the stage is allowed to specialize to it. TP=8 cleanly divides heads, KV heads, and MLP intermediate. |
| Mesh-sharded residual stream | Rejected for first path | It would require distributed RMSNorm for exact Phi statistics and an all-gather before every column-parallel QKV/gate-up. Decode communication saved is small versus the added norm complexity. |
| 2D/Galaxy strategy | Rejected | Hardware is T3K, not Galaxy. |
| Dense all-expert execution | Not applicable | Phi-3.5-mini is dense and has no MoE experts. |
| Unreordered QKV or gate/up sharding | Rejected | Contiguous shards would not provide each chip a complete local Q/K/V or gate/up pair. |

## Evidence Status

Completed evidence is recorded in `work_log.md`.

Summary:

- Synthetic layer PCC vs optimized single-chip baseline: prefill `0.9999945714628357`, decode `0.9999955128943894`.
- Real layer-0 PCC vs optimized single-chip baseline: prefill `0.999991788840335`, decode `0.9999935080298819`.
- Decode trace capture/replay passes in the synthetic, real-weight, host-timed, profiler-only, and watcher runs.
- Full-context decode stress validates `current_pos=131071`, page table coverage for 131072 positions, and per-device local KV cache shape `[4096, 4, 32, 96]`.
- Long prefill stress validates a 32768-token page table and local KV cache layout.
- Runtime fallback audit is clean for the hot multichip runtime callables.
- Watcher run passed on the real-weight path with no fatal/error/bounds/overflow/retraining hits in `watcher.log`.

Performance:

| Measurement | Single-chip optimized baseline | 1x8 multichip | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: |
| Warmed decode host trace replay | `983.737 us/token` | `580.549 us/token` | `1.69x` | `21.2%` |
| Decode profiled device + gap | `923.470 us` | `1205.575 us` | `0.77x` | `9.6%` |
| Prefill profiled device + gap, `T=32` | `2612.095 us` | `4585.328 us` | `0.57x` | `7.1%` |

The accepted full-model-layer-stack baseline is optimized for the decode serving path. The profiler reports show that CCL and layout movement dominate this first multichip path: decode has `135.28 us` ReduceScatter and `71.66 us` AllGather device time, while prefill has `137.89 us` ReduceScatter and `78.06 us` AllGather device time. Matmuls are low-utilization for this small layer and sequence length, so scaling efficiency is limited.

Artifacts:

- `perf/host_timing_real_layer0_after_reset.log`
- `tracy/host_only/reports/2026_06_15_14_13_08/ops_perf_results_2026_06_15_14_13_08.csv`
- `perf/prefill_perf_human.txt`
- `perf/decode_perf_human.txt`
- `perf/prefill_perf_report.csv`
- `perf/decode_perf_report.csv`
- `perf/prefill_perf_summary.csv`
- `perf/decode_perf_summary.csv`
- `perf/prefill_perf_summary.png`
- `perf/decode_perf_summary.png`
- `watcher/2026_06_15_1x8_ring_real/pytest.log`
- `watcher/2026_06_15_1x8_ring_real/generated/watcher/watcher.log`

## Limitations

- Specialized to the available T3K `1x8` ring mesh; smaller meshes are intentionally unsupported.
- Decode batch size follows the optimized single-chip baseline path used here.
- Residuals remain replicated to keep exact local RMSNorm and stacked-layer contracts simple. A sharded-residual path was rejected for this stage because it would need distributed RMSNorm and extra all-gathers before column-parallel projections.
- Prefill is slower than the single-chip optimized baseline at `T=32`; this implementation is intended as the full-model decode stack baseline, with prefill correctness and page-table coverage retained.
- Explicit `PHI35_READ_DEVICE_PROFILER=1` runs were not accepted as evidence because they either overflowed device profiler buffers when combined with repeated host trace timing, crashed while profiling a single-chip baseline mesh close, or stalled in profiler reads on the 8-device mesh. The accepted tt-perf-report CSV was produced by `python -m tracy -p` on the multichip-only perf test without manual `ReadDeviceProfiler`.
