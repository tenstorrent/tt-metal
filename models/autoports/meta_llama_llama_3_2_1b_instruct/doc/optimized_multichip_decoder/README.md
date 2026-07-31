# Optimized Multichip Decoder

Status: complete for the optimized multichip decoder stage.

This directory records the optimization pass for the
`meta-llama/Llama-3.2-1B-Instruct` decoder layer on the repo-local TTNN
autoport pipeline. The measured path is the `MultichipDecoder` layer on a T3K
`1x8` mesh with `FABRIC_1D_RING`; no full-model or vLLM work was started.

## Selected Path

| Item | Final contract |
| --- | --- |
| Mesh | T3K `1x8`, `ttnn.Topology.Ring`, `ttnn.FabricConfig.FABRIC_1D_RING` |
| Decoder boundary | Input and output are replicated full-hidden mesh tensors |
| Inter-layer residual | No gather, reshard, or all-reduce is required between decoder layers |
| WQKV, W1, W3 | Output-dim tensor-parallel over 8 devices |
| WO | Fused `AllGatherMatmulAsyncDeviceOperation` on Ring |
| W2 | Row/input sharded, 16-core decode target, reduce-scatter partials |
| KV cache | BFP8 local KV cache, one local KV head per chip |
| Residual CCL payloads | BFP8 all-gather and BFP8 reduce-scatter by default |
| CCL buffers | Persistent ping-pong output buffers enabled by default |

The full-model bringup contract is to preserve the replicated full-hidden
decoder boundary. A hidden-sharded inter-layer stream was inspected and rejected
for this stage because current `RMSNorm1D` traced decode and QKV/W1/W3 input
contracts still require gathered hidden-width tensors. The remaining collectives
are internal to a layer; none are inserted between decoder layers.

## Before And After Latency

Non-profiler warmed host timings:

| Path | Artifact | Prefill 8192 host ms | Traced decode host ms |
| --- | --- | ---: | ---: |
| Completed multichip decoder before this pass | `../multichip_decoder/perf_trace_contract.json` | 15.922505 | 0.674415 |
| Final optimized multichip decoder | `perf_trace_contract.json` | 13.971094 | 0.648592 |

This is a 12.26% prefill host improvement and a 3.83% traced decode host
improvement versus the completed multichip decoder baseline.

Key optimization trials:

| Trial | Artifact | Prefill host ms | Decode host ms | Decision |
| --- | --- | ---: | ---: | --- |
| BF16 cleanup baseline | `trials/partial_memcfg_default/perf_trace_contract.json` | 15.884992 | 0.665218 | Kept W2 output memcfg fix |
| All-gather BFP8 only | `trials/ag_bfp8/perf_trace_contract.json` | 14.467429 | 0.676863 | Rejected: decode regression |
| Reduce-scatter BFP8 only | `trials/rs_bfp8/perf_trace_contract.json` | 15.041273 | 0.826221 | Rejected: decode regression |
| All-gather plus reduce-scatter BFP8 | `trials/ccl_bfp8_both/perf_trace_contract.json` | 13.916712 | 0.668820 | Selected precision policy |
| W2 16-core plus BFP8 CCL | `trials/w2_16_bfp8_default/perf_trace_contract.json` | 13.866011 | 0.665769 | Selected W2 tiling |
| Persistent CCL buffers | `trials/persistent_ccl_buffers/perf_trace_contract.json` | 13.874695 | 0.645149 | Selected default |

## Performance Accounting

Same-run Tracy/`tt-perf-report` accounting from `perf/perf_provenance.json`:

| Phase | Profile host signpost ms | Device time ms | Reported gap ms | Device window ms |
| --- | ---: | ---: | ---: | ---: |
| Prefill 8192 | 14.499375 | 13.630363 | 0.022878 | 13.653241 |
| Traced decode replay | 0.617592 | 0.352362 | 0.261231 | 0.613593 |

The decode lower-bound DRAM roofline estimate is 0.019118 ms/token for one
decoder layer at position 8192, assuming BFP8 attention weights, BFP4 MLP
weights, BFP8 KV reads, and 8 x 288 GB/s aggregate DRAM bandwidth. The measured
device window is much higher because this isolated layer is dominated by many
small kernels and CCL synchronization, not by a single large DRAM streaming
matmul. In the Tracy run, host signpost time minus device window for traced
decode is only 0.003999 ms, so the replay path is not host dominated.

## Correctness

| Check | Artifact | Prefill PCC | Decode PCC | Repeated trace PCC |
| --- | --- | ---: | ---: | ---: |
| Synthetic 128 | `synthetic_correctness.json` | 0.9999905635 | 0.9999904892 | 1.0 |
| Real weights 128 | `real_weight_correctness.json` | 0.9999908910 | 0.9999914077 | 0.9999999999999881 |
| Real weights 8192 | `real_weight_correctness_prefill_8192.json` | 0.9999913309 | 0.9999914965 | 1.0 |

Runtime fallback audit passed in `runtime_fallback_audit.json`, guarding
`ttnn.from_torch` and `ttnn.to_torch` during measured multichip prefill and
trace capture/replay. Repeated-run stress passed 5 iterations in
`stress_repeated_runs.json`.

Watcher evidence is clean for the final path in
`watcher/watcher_clean_final_persistent_eth_disabled_summary.json`. The final
watcher run used `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1` because a
prior full watcher attempt overflowed `idle_erisc.elf`; the watcher log was
audited for fatal/error/timeout signatures with zero matches.

## TT-Perf-Report Conclusions

Final report artifacts:

- `perf/ops_perf_results_raw.csv`
- `perf/prefill_8192_tt_perf_report.txt`
- `perf/prefill_8192_report.csv`
- `perf/prefill_8192_per_device_tt_perf_report.txt`
- `perf/prefill_8192_per_device_report.csv`
- `perf/decode_trace_replay_tt_perf_report.txt`
- `perf/decode_trace_replay_report.csv`
- `perf/decode_trace_replay_per_device_tt_perf_report.txt`
- `perf/decode_trace_replay_per_device_report.csv`
- `perf/perf_provenance.json`

Prefill conclusions:

- The completed multichip baseline had 8.193 ms of prefill CCL device time.
  The final profile has 5.844 ms of explicit CCL device time.
- Final prefill still has one BF16 internal Attention1D prefill all-gather
  around 2.107 ms. The BFP8 changes apply to the residual all-gather and
  reduce-scatter helpers in this multichip decoder layer.
- Final prefill advice still asks to increase grid size for 8-core matmuls.
  The code is already using the largest 8-wide grid legal for those 1x8
  per-device paths, and the 64-core MLP prefill matmuls have valid
  `in0_block_w` and `1x4` output subblocks.
- Final prefill advice also asks to place MLP input 0 in L1. This was rejected
  for 8192-token prefill because the intended contract keeps large prefill
  activations DRAM interleaved and the report shows those matmuls are not the
  dominant remaining cost after CCL optimization.

Decode conclusions:

- Final decode matmuls are DRAM-sharded where applicable: WQKV, W1, W3, and W2
  report `DRAM Sharded=True`, L1 width-sharded input, and `in0_block_w=2`.
- Final decode uses fused WO all-gather matmul. The report still flags output
  subblock `1x1`; this is constrained by common `Attention1D` because
  `do_per_core_N=1` for hidden 2048, TP 8, tile 32, and an `8x1` fused grid.
- Standalone decode CCLs are BFP8: two all-gathers total 43.316 us and one
  reduce-scatter is 41.744 us. The fused WO all-gather matmul is 33.466 us.
- The report flags a 234 us gap before the first replayed `LayerNorm`. The
  measured decode path already uses `ttnn.execute_trace`, and the same raw
  Tracy signposts show only 0.003999 ms host-minus-device-window for the whole
  replay. This is recorded as a report/window synchronization limitation rather
  than an untraced host fallback.

## Advice And Option Disposition

| Item | Evidence | Decision |
| --- | --- | --- |
| Async CCLs | Final reports show `AllGatherAsyncDeviceOperation` and `ReduceScatterMinimalAsyncDeviceOperation` on Ring | Used |
| BFP8 CCL payloads | `trials/ag_bfp8`, `trials/rs_bfp8`, `trials/ccl_bfp8_both`, final reports | Combined BFP8 selected |
| Persistent/preallocated CCL buffers | `trials/persistent_ccl_buffers`, final watcher summary | Selected |
| Semaphore reuse | Helpers use `TT_CCL` cyclic semaphores; persistent path skips per-op barrier semaphore | Selected |
| Fused matmul-CCL | Decode report has `AllGatherMatmulAsyncDeviceOperation`; code populates `allowed_worker_cores` | Used for WO |
| Additional fused QK decode | `trials/qk_fused/rejection.log` | Rejected: rotary cos/sin batch contract mismatch |
| W2 `in0_block_w` advice | `trials/w2_16_bfp8_default`, final decode report | Selected W2 16-core target; final `in0_block_w=2` |
| Fused WO subblock advice | `attention_1d.py` config evidence and final report | Rejected: `per_core_N=1` makes `1x1` the only legal common-module subblock |
| Activation sharding | `mesh_strategy.json`, final decode report matmul inputs | Width-sharded L1 decode activations kept inside layer |
| Residual layout | `mesh_strategy.json`, final correctness topology | Replicated inter-layer boundary preserved; no inter-layer collective |
| DRAM-sharded decode matmuls | Final decode CSV | Used for WQKV, W1, W3, W2 |
| HiFi4 advice | Final reports, PCC artifacts | Rejected: increases fidelity/cost; PCC already > 0.99999 |
| Hidden-sharded inter-layer residual | Mesh plan and code inspection | Rejected for this stage: norm and gathered-input contracts block it |
| MoE routed experts | Model architecture | Not applicable; Llama-3.2-1B-Instruct decoder is dense |
| LM head and sampling | Goal scope | Not applicable; decoder-layer optimization only |

## Optimize Checklist

All applicable prompt and `$optimize` checklist items were tried; no applicable
optimization is deferred.

- Functional checks: passed.
- Prefill/decode PCC: passed for synthetic 128, real 128, and real 8192.
- Paged KV-cache and warmed trace replay: passed.
- Runtime fallback audit: passed for measured multichip prefill and traced decode.
- Stress coverage: passed 5 repeated runs.
- Warmed before/after latency: reported above.
- `tt-perf-report` human tables, CSVs, and provenance: present in `perf/`.
- Watcher clean: passed with scoped ETH watcher disabled after documented
  `idle_erisc.elf` watcher overflow.
- Decoder path traced with no host fallbacks: passed; decode measurement is a
  warmed `ttnn.execute_trace` replay.
- Decode activation sharding: width-sharded L1 within the layer where common
  modules support it.
- Prefill activations and matmul configs: DRAM interleaved prefill activations
  and explicit 2D prefill configs are used.
- Optimized composites: inherited paged SDPA/FlashDecode-style common
  Attention1D paths.
- Explicit memory, program, and compute configs: configured in the optimized
  decoder, multichip MLP, fused WO, and CCL helpers.
- Shard specs and grids: selected grids divide tile dimensions; W2 decode uses
  16 cores to avoid the earlier `in0_block_w=1` advice.
- DRAM-sharded decode matmuls: used for attention and MLP matmuls where
  applicable.
- Fused matmul-CCL: used for WO; other fused opportunities are recorded above.
- Reduced precision/fidelity: BFP8 residual CCL payloads selected; HiFi4 and
  broader datatype frontier left out because they are not speed reductions for
  this decoder stage.
- Performance accounting: roofline, device-time decode, same-run Tracy host
  window, and non-profiler warmed host latency are recorded.

## Artifacts

- Mesh and layout: `mesh_strategy.json`
- Latency: `perf_trace_contract.json`
- Correctness: `synthetic_correctness.json`, `real_weight_correctness.json`,
  `real_weight_correctness_prefill_8192.json`
- Fallback and stress: `runtime_fallback_audit.json`,
  `stress_repeated_runs.json`
- Perf reports and provenance: `perf/`
- Optimization trials and rejected options: `trials/`
- Watcher logs: `watcher/`
- Detailed command log: `work_log.md`
