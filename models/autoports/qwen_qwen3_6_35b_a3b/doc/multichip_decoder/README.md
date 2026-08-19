# Qwen3.6-35B-A3B Multichip Decoder

This stage adds `tt/multichip_decoder.py` as the 2x2 Blackhole mesh decoder
baseline for the repo-local autoport pipeline. It starts from
`tt/optimized_decoder.py::OptimizedDecoder` and preserves the optimized
decoder's public prefill/decode tensor contract.

## Mesh Plan

Target hardware is the four local Blackhole p300c devices as a `2x2` mesh with
`FABRIC_1D_RING` collectives.

| Dimension | Strategy |
| --- | --- |
| Layer boundary activations | Replicated residual stream on every device |
| Tensor parallelism | Mesh columns, `cluster_axis=1`, TP=2 |
| Expert parallelism | Mesh rows, `cluster_axis=0`, EP=2 |
| Collectives | TP all-reduce after attention/linear-attention outputs and shared expert down projection; EP then TP all-reduce after routed MoE |
| Supported mesh configs | Only the local `2x2` target is implemented |

Rejected alternatives:

- KV-head replication was rejected because the model has exactly two KV heads;
  TP=2 gives one KV head per column and halves full-attention KV memory per
  device.
- Sharded hidden states at decoder boundaries were rejected for this stage
  because layernorm, residual, and stacked-layer contracts are currently lower
  risk with replicated hidden states.
- Physical routed-expert weight sharding across EP rows was rejected because
  the current TTNN sparse-matmul path is driven by a global expert axis. EP rows
  execute disjoint expert masks, while routed expert weights remain replicated
  across rows.

## Per-Device Shapes

Logical input/output shapes remain `[1, batch, seq, 2048]` for prefill and
`[1, 1, batch, 2048]` for decode. Non-aligned logical lengths are publically
valid; padding is internal and sliced back before returning.

| Tensor family | Logical shape | Per-device shape |
| --- | --- | --- |
| Full-attention Q heads | 16 x 256 | 8 x 256 |
| Full-attention KV cache | `[blocks, 2, 32, 256]` | `[blocks, 1, 32, 256]` |
| Linear conv state per tap | `[1, 1, batch, 8192]` | `[1, 1, batch, 4096]` |
| Linear recurrent state | `[1, batch * 32, 128, 128]` | `[1, batch * 16, 128, 128]` |
| Shared MoE intermediate | 512 | 256 |
| Routed MoE intermediate | 512 | 256, all 256 experts present per EP row |

For batch 1 at the advertised 262144-token context, full-attention KV cache is
8192 blocks. Keys plus values are 256 MiB per device per full-attention layer
versus 512 MiB on the single-chip cache layout. Across the 10 full-attention
layers that is about 2.5 GiB per device. Linear recurrent state is 0.5 MiB per
device per batch per layer, about 15 MiB per device for the 30 linear-attention
layers at batch 1.

## Correctness

All PCCs are against the single-chip TTNN optimized decoder baseline with a
0.995 acceptance bar.

| Case | Prefill PCC | Decode PCC |
| --- | ---: | ---: |
| Synthetic linear layer 0, seq 5, traced decode | 0.9999484088 | 0.9999441730 |
| Synthetic full layer 3, seq 33, traced decode | 0.9999434563 | 0.9999454953 |
| Synthetic linear layer 0, non-aligned seq 65, traced decode | 0.9999464360 | 0.9999427555 |
| Synthetic full layer 3, non-aligned seq 33, traced decode | 0.9999434563 | 0.9999454953 |
| Synthetic linear layer 0, batch 2, seq 5 | 0.9999495000 | 0.9999499536 |
| Synthetic full layer 3, batch 2, seq 33 | 0.9999451874 | 0.9999451794 |
| Real weights linear layer 0, seq 1, traced decode | 0.9999731323 | 0.9999286757 |
| Real weights full layer 3, seq 1, traced decode | 0.9999452009 | 0.9998278764 |

Paged full-attention cache behavior is validated by comparing each local
per-column KV cache shard against the optimized baseline through the page table.
Linear-attention conv and recurrent state layout is validated per device. Decode
trace capture/replay is covered for both linear and full-attention layer kinds.

Primary logs:

- `logs/watcher_correctness_disable_eth.log`
- `logs/runtime_fallback_audit_exact_nnz.log`

## Performance

The final performance evidence uses exact decode sparse-matmul `nnz` after EP
routing. Wall times are warmed measurements from signposted Tracy runs; device
times and CCL estimates come from `tt-perf-report`.

| Case | Baseline wall ms | Multichip wall ms | Speedup | 4-chip efficiency | Multichip CCL ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear prefill seq 5 | 21.141 | 19.809 | 1.067 | 0.267 | 0.091 |
| Linear traced decode seq 5 | 1.539 | 1.394 | 1.104 | 0.276 | 0.088 |
| Full prefill seq 33 | 8.894 | 6.692 | 1.329 | 0.332 | 0.169 |
| Full traced decode seq 33 | 1.223 | 1.192 | 1.026 | 0.257 | 0.091 |

Summary CSV:

- `tracy/final_exact_nnz/perf_summary.csv`

Human-readable tables and report CSVs:

- `tracy/final_exact_nnz/linear_attention/*_perf_report.txt`
- `tracy/final_exact_nnz/linear_attention/*_perf_report.csv`
- `tracy/final_exact_nnz/full_attention/*_perf_report.txt`
- `tracy/final_exact_nnz/full_attention/*_perf_report.csv`

Raw and normalized Tracy op CSV provenance is stored as gzip-split parts with
`SHA256SUMS` manifests:

- `tracy/final_exact_nnz_raw/reports/2026_08_19_03_59_38/ops_perf_results_2026_08_19_03_59_38.csv.gz.parts`
- `tracy/final_exact_nnz_raw/reports/2026_08_19_03_59_38/ops_perf_results_2026_08_19_03_59_38_blackhole.csv.gz.parts`

## Limitations

- The implementation is intentionally specialized to the observed 2x2 mesh.
- Routed expert weights are TP-sharded by intermediate width but replicated
  across EP rows; EP reduces routed expert execution and output reduction work,
  not routed expert weight DRAM.
- The stage keeps the advertised 262144-token logical context. The mesh path
  did not rerun the full 262144-token probe, but the KV-cache math reduces
  per-device KV memory and no hard physical context limit was observed.
- Active Ethernet watcher mode hit a fabric-router teardown assertion on this
  p300c system after the correctness run body passed. The accepted watcher
  artifact disables active Ethernet watcher while keeping worker/NOC watcher
  coverage enabled.
