# Multichip Decoder - Llama 3.2 1B Instruct

This directory records the multichip decoder state for
`meta-llama/Llama-3.2-1B-Instruct`. The implementation is limited to the
decoder layer stack baseline and does not start full-model or vLLM work.

Implementation:

- `../../tt/multichip_decoder.py`
- `../../tests/test_multichip_decoder.py`
- Baseline: `../../tt/optimized_decoder.py`

## Status

The decoder is specialized for the available 8-device Wormhole T3K mesh. It
uses the optimized decoder as the single-chip baseline, validates prefill and
warmed traced decode against that baseline, and emits the artifacts in this
directory.

## Mesh Plan

Selected hardware strategy:

- Mesh shape: `1x8`
- Topology: `ttnn.Topology.Ring`
- Fabric: `ttnn.FabricConfig.FABRIC_1D_RING`
- Tensor parallel degree: `8`
- Hardware observed by perf tooling: Wormhole, 64 worker cores, 8-device mesh
- Smaller or alternate mesh configurations are intentionally not supported by
  this state.

The residual stream stays replicated at decoder input and output. Internal
attention and MLP work are tensor parallelized, then gathered back to the
replicated residual contract expected by the next stacked decoder layer.

### Tensor Strategy

| Tensor | Global shape | Per-device shape | Strategy | Padding |
| --- | ---: | ---: | --- | --- |
| WQKV | `[2048, 3072]` | `[2048, 384]` | Column/output sharded after Q/K/V chunk reorder | none |
| Q heads | `[32, 64]` | `[4, 64]` | Head axis sharded over TP | none |
| K/V heads | `[8, 64]` | `[1, 64]` | KV head axis sharded over TP | none |
| WO | `[2048, 2048]` | `[2048, 256]` | Column/output sharded after fused all-gather matmul | none |
| W1/W3 | `[2048, 8192]` | `[2048, 1024]` | Column/output sharded | none |
| W2 | `[8192, 2048]` | `[1024, 2048]` | Row/input sharded | none |
| RMSNorm weights | `[2048]` | `[2048]` | Replicated | none |

Shard specs and placement details:

- WQKV, W1, and W3 use mesh sharding on the output dimension
  (`ShardTensorToMesh(..., dim=-1)`), so each device owns a contiguous output
  slice.
- W2 uses mesh sharding on the input/intermediate dimension
  (`ShardTensorToMesh(..., dim=-2)`), so each device consumes its local
  intermediate slice before reduce-scatter.
- RMSNorm weights, page tables, current positions, and layer-boundary
  activations are replicated.
- Decode full-hidden activations use the optimized decoder's width-sharded L1
  residual spec: grid `x=0..7, y=0..3`, shard shape `[32, 64]`.
- Decode row-parallel partial outputs use width-sharded L1 on grid
  `x=0..7, y=0..0`, shard shape `[32, 32]`, then reduce-scatter and
  all-gather.
- Prefill intermediate activations are DRAM interleaved around the Ring CCLs.
- No load-time tensor padding is needed: hidden `2048`, intermediate `8192`,
  query heads `32`, KV heads `8`, and head dim `64` divide cleanly over TP=8.
  Sequence length is page-table rounded to 64-token blocks for paged KV tests.

### Activation And Collective Strategy

- Decoder input and output: replicated full hidden stream.
- RMSNorm: local norm on the replicated stream.
- Attention: local WQKV, local Q/K/V heads, paged attention per device, fused
  all-gather matmul for WO, then all-gather to restore replicated hidden output.
- MLP: local W1/W3, row-parallel W2, `reduce_scatter_minimal_async`, then
  all-gather to restore replicated hidden output.
- Page table and current position tensors: replicated `int32`.
- KV cache dtype: `bfloat8_b`.
- KV cache layout at 8192-token prefill plus decode: each device owns
  `[129, 1, 64, 64]` for keys and values.
- Output topology: replicated across the `1x8` mesh for both prefill and decode.
- MoE/expert strategy: not applicable. This model is a dense Llama decoder.

Rejected alternatives:

- `1x1`: retained only as the optimized baseline; it leaves seven T3K devices
  idle.
- `1x4`: valid divisibility, but uses half the available device bandwidth.
- 2D/Galaxy strategy: rejected because the local hardware is T3K `1x8`.
- Hidden-sharded residual stream: rejected for this state because QKV/W1/W3
  still require gathered inputs and the common 1D norm path does not provide a
  traced distributed RMSNorm decode contract.

The serialized plan is in `mesh_strategy.json`.

## Correctness

All PCC numbers are against the single-chip TTNN optimized decoder baseline.

| Artifact | Weights | Prefill length | Prefill PCC | Decode PCC | Trace replay PCC | Determinism |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `synthetic_correctness.json` | synthetic | 128 | 0.9999958887 | 0.9999932200 | 0.9999932200 | 1.0000000000 |
| `real_weight_correctness_prefill_8192.json` | real HF weights | 8192 | 0.9999963551 | n/a | 0.9999946875 | 0.9999999999999873 |

Additional coverage:

- `real_weight_correctness.json`: real-weight short prefill/decode.
- `real_weight_correctness_prefill_4096.json`: real-weight 4096-token prefill.
- `runtime_fallback_audit.json`: guarded `ttnn.from_torch` and `ttnn.to_torch`
  bridges during measured prefill and traced decode paths.
- `stress_repeated_runs.json`: 3 repeated prefill runs at PCC
  `0.9999964050`.
- `watcher/watcher_summary.json`: watcher-enabled prefill plus traced decode
  replay with clean status.

## Performance

The comparison uses the existing optimized single-chip perf artifact and this
directory's multichip Tracy run at 8192-token prefill plus one warmed trace
decode replay.

| Stage | Single-chip device us | Multichip device us | Speedup | Efficiency vs 8 | Host speedup | Main limitation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prefill 8192 | 28849.668 | 15799.536 | 1.8260x | 0.2282 | 2.2657x | CCL dominates after local compute reduction |
| Traced decode replay | 519.718 | 383.039 | 1.3568x | 0.1696 | 1.2508x | Ring CCL plus Tracy replay gap |

The inclusive decode device-plus-gap view is slower than single chip
(`0.6599x`) because the Tracy range includes a measured replay gap. The
non-Tracy warmed host replay and summed device-op time are faster, so this path
is retained as the full-model multichip layer-stack baseline with the decode
gap limitation documented.

Perf artifacts:

- `perf/perf_provenance.json`
- `perf/ops_perf_results_raw.csv`
- `perf/profile_log_device_raw.csv`
- `perf/prefill_8192_tt_perf_report.txt`
- `perf/prefill_8192_report.csv`
- `perf/prefill_8192_per_device_tt_perf_report.txt`
- `perf/prefill_8192_per_device_report.csv`
- `perf/decode_trace_replay_tt_perf_report.txt`
- `perf/decode_trace_replay_report.csv`
- `perf/decode_trace_replay_per_device_tt_perf_report.txt`
- `perf/decode_trace_replay_per_device_report.csv`

Operation audit from `perf/perf_provenance.json`:

| Stage | Device ops | Host ops | Device us | Gap us | CCL us | Compute us | Matmul us | Data movement us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill | 24 | 0 | 15799.536 | 14.659 | 8193.324 | 7343.032 | 3911.922 | 114.888 |
| Traced decode replay | 27 | 0 | 383.039 | 434.421 | 157.644 | 235.322 | 157.445 | 19.543 |

The prefill report shows all-gather and reduce-scatter as the dominant cost
after tensor parallelization. Decode has lower summed device-op time but is
sensitive to Ring collective overhead and replay gaps.

## Limitations

- This state targets only the observed T3K `1x8` mesh.
- The residual stream is replicated between layers; this keeps the stacked
  decoder contract simple but adds gather traffic.
- Decode speedup is real in summed device-op and warmed host timing, but the
  inclusive Tracy range is CCL/gap limited.
- No full-model or vLLM work is included.
- No MoE active-expert path is needed for this dense model.
