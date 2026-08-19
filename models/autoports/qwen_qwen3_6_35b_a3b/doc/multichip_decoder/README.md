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
| MoE execution | Gate-selected active experts; prefill and decode use per-token sparse routing with 4 active experts per EP row |
| Supported mesh configs | Only the local `2x2` target is implemented |

Rejected alternatives and topology audit:

| Alternative | Status | Evidence |
| --- | --- | --- |
| Replicated residual stream with TP/EP output reductions | Selected | Preserves the optimized decoder layer-boundary contract for stacked layers; watcher, fallback, PCC, trace, and perf artifacts cover this path. |
| TP-sharded residual stream at decoder boundaries | Rejected | It would require distributed RMSNorm plus an all-gather before the current column-parallel QKV/linear/MoE projections, whose weights and TTNN matmul/sparse-matmul contracts consume the full hidden dimension. The immediate all-gather removes the stack-level benefit on the current `2x2` path. |
| Delayed sharded-residual families | Rejected | `residual_topology_audit.md` covers reduce-scatter/delayed-gather, fused all-gather-matmul, fused matmul+reduce-scatter, fully sharded residual, and 2D residual variants with byte estimates, next consumers, persistent-buffer requirements, and exact blockers. |
| Broad token-by-expert active sparse prefill | Rejected | `moe_routing_remap` rejects multi-token routing rows with `TT_FATAL` because it expects routing shape `[1, E]`; a no-remap sparse probe with `A=[5,1,128]`, `B=[8,128,32]`, sparsity `[5,8]`, and `nnz=None` hung in `SparseMatmulDeviceOperation`. Triage is in `triage/active_prefill_sparse_probe_*`. |
| Dense/all-expert routed MoE prefill on EP rows | Rejected | It executes non-selected experts through broad EP-row candidate sets instead of the gate-selected top-k experts and therefore does not satisfy the MoE completion contract for this stage. |

- KV-head replication was rejected because the model has exactly two KV heads;
  TP=2 gives one KV head per column and halves full-attention KV memory per
  device.
- Physical routed-expert weight sharding across EP rows was rejected because
  the current TTNN sparse-matmul path is driven by a global expert axis.
  `moe_routing_remap` partitions the selected sparse rows for execution, while
  routed expert weights remain replicated across EP rows.

## Per-Device Shapes

Logical input/output shapes remain `[1, batch, seq, 2048]` for prefill and
`[1, 1, batch, 2048]` for decode. Non-aligned logical lengths are publicly
valid; padding is internal and sliced back before returning.

| Tensor family | Logical shape | Per-device shape |
| --- | --- | --- |
| Full-attention Q heads | 16 x 256 | 8 x 256 |
| Full-attention KV cache | `[blocks, 2, 32, 256]` | `[blocks, 1, 32, 256]` |
| Linear conv state per tap | `[1, 1, batch, 8192]` | `[1, 1, batch, 4096]` |
| Linear recurrent state | `[1, batch * 32, 128, 128]` | `[1, batch * 16, 128, 128]` |
| Shared MoE intermediate | 512 | 256 |
| Routed MoE intermediate | 512 | 256, 4 active experts per token per EP row after `moe_routing_remap`; expert weights remain replicated across EP rows |

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
Routed MoE prefill uses the same active sparse route as decode by slicing the
logical token dimension and applying `moe_routing_remap` per token; no public
sequence-length alignment is exposed by this internal token loop.

Primary logs:

- `logs/watcher_correctness_disable_eth.log`
- `logs/watcher_correctness_active_eth.log`
- `logs/active_eth_isolated/linear_seq5.log`
- `logs/active_eth_isolated/summary.log`
- `logs/tt_smi_post_active_eth_reset.log`
- `logs/post_active_eth_reset_mesh_smoke.log`
- `logs/runtime_fallback_audit_exact_nnz.log`

## Performance

The final performance evidence uses exact sparse-matmul `nnz`. The optimized
single-chip baseline linear seq5 prefill keeps its original all-expert rows
(`active=256/256`). Optimized baseline full seq33 prefill contains both the
32-token all-expert prefill chunk rows (`active=256/256`) and a 1-token
active-routed tail (`active=8/256`). Optimized decode uses active routing
(`active=8/256`). The mesh path uses gate-selected active prefill and decode
after EP routing (`active=4/256` per EP row). Wall times are warmed measurements
from signposted Tracy runs; device times, CCL, data movement, compute, tensor
movement, and modeled DRAM-operation estimates come from `tt-perf-report`.

| Case | Baseline wall ms | Multichip wall ms | Speedup | 4-chip efficiency | Multichip CCL ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear prefill seq 5 | 19.943 | 22.971 | 0.868 | 0.217 | 0.265 |
| Linear traced decode seq 5 | 1.540 | 1.397 | 1.102 | 0.276 | 0.087 |
| Full prefill seq 33 | 8.949 | 39.253 | 0.228 | 0.057 | 1.530 |
| Full traced decode seq 33 | 1.220 | 1.154 | 1.057 | 0.264 | 0.086 |

Prefill is slower after enforcing gate-selected active-expert execution because
the supported sparse routing primitive is single-token. The earlier broad EP
mask path was faster for prefill but executed non-selected experts; the
multi-token active sparse probe hung and is recorded as rejected evidence.

Summary CSV:

- `tracy/final_exact_nnz/perf_summary.csv`

Human-readable tables and report CSVs:

- `tracy/final_exact_nnz/linear_attention/*_perf_report.txt`
- `tracy/final_exact_nnz/linear_attention/*_perf_report.csv`
- `tracy/final_exact_nnz/full_attention/*_perf_report.txt`
- `tracy/final_exact_nnz/full_attention/*_perf_report.csv`
- `logs/residual_topology_audit.log`
- `residual_topology_audit.md`

Raw and normalized Tracy op CSV provenance is stored as gzip-split parts with
`SHA256SUMS` manifests:

- `tracy/final_exact_nnz_raw/reports/2026_08_19_05_09_06/ops_perf_results_2026_08_19_05_09_06.csv.gz.parts`
- `tracy/final_exact_nnz_raw/reports/2026_08_19_05_09_06/ops_perf_results_2026_08_19_05_09_06_blackhole.csv.gz.parts`

## Limitations

- The implementation is intentionally specialized to the observed 2x2 mesh.
- Routed expert weights are TP-sharded by intermediate width but replicated
  across EP rows; EP reduces routed expert execution and output reduction work,
  not routed expert weight DRAM.
- Active MoE prefill is token-serialized until `moe_routing_remap` and
  `SparseMatmulDeviceOperation` support a watcher-clean multi-token active
  sparse geometry.
- The stage keeps the advertised 262144-token logical context. The mesh path
  did not rerun the full 262144-token probe, but the KV-cache math reduces
  per-device KV memory and no hard physical context limit was observed.
- Active Ethernet watcher mode is not the accepted clean watcher artifact on
  this p300c system. A rerun with active Ethernet watcher enabled is captured in
  `logs/watcher_correctness_active_eth.log`: it initialized with `disabled
  features: None`, passed the first selected hardware test, then failed later
  device opens with `Timed out while waiting for active ethernet core 28-25 to
  become active again`. A single-test isolated active-ETH rerun is captured in
  `logs/active_eth_isolated/linear_seq5.log`; the decoder test body passed and
  printed PCC, then `MetalContext` teardown hit the same active-Ethernet watcher
  timeout. The accepted watcher artifact is the ETH-disabled worker/NOC watcher
  run, and post-failure reset/smoke artifacts are recorded in
  `logs/tt_smi_post_active_eth_reset.log` and
  `logs/post_active_eth_reset_mesh_smoke.log`.
