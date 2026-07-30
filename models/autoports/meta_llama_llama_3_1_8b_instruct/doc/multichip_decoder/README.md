# Llama 3.1 8B Instruct Multichip Decoder

Model: `meta-llama/Llama-3.1-8B-Instruct`

Autoport path: `models/autoports/meta_llama_llama_3_1_8b_instruct`

Multichip implementation: `tt/multichip_decoder.py`

Single-chip baseline: `tt/optimized_decoder.py`, policy
`llama31_8b_single_chip_bfp8_attn_bfp4_mlp_decode_v1`. The multichip class
records this as `MultiChipDecoder.single_chip_baseline_cls = OptimizedDecoder`.

## Target Mesh Plan

Hardware detected on 2026-06-15 with `tt-smi -s`: eight Wormhole B0 ASICs
in a T3K-class system, exposed as four N300 boards. The target path is fixed
to a `1x8` `ttnn.MeshDevice` with `ttnn.FabricConfig.FABRIC_1D_RING` and
`ttnn.Topology.Ring`. The decoder intentionally does not support smaller mesh
configurations.

The selected strategy is 1D tensor parallelism with `TP=8`. The decoder
boundary is a replicated full-hidden residual stream. Attention and MLP
internals are tensor-parallel; each sublayer gathers its hidden shard back to
the replicated boundary before the residual add and next stacked layer.

| Tensor or op | Global shape | Per-device shape | Placement | Padding |
| --- | ---: | ---: | --- | --- |
| Residual boundary | `[1, 1, S, 4096]` | replicated full hidden | replicated mesh, width-sharded L1 for decode | none |
| RMSNorm weights | `[4096]` | replicated full hidden | replicated mesh | none |
| Q projection | `[4096, 4096]` | `[4096, 512]` | column parallel | none |
| K projection | `[4096, 1024]` | `[4096, 128]` | column parallel | none |
| V projection | `[4096, 1024]` | `[4096, 128]` | column parallel | none |
| Grouped QKV | `[4096, 6144]` | `[4096, 768]` | grouped Q/K/V local heads | none |
| Q heads | 32 heads | 4 heads | local attention | none |
| KV heads | 8 heads | 1 head | local paged cache | none |
| Paged K cache | `[blocks, 8, 64, 128]` logical | `[blocks, 1, 64, 128]` | local KV-head cache | none |
| Paged V cache | `[blocks, 8, 64, 128]` logical | `[blocks, 1, 64, 128]` | local KV-head cache | none |
| WO | `[4096, 4096]` | `[4096, 512]` local input shard | fused AG+matmul, output-column parallel | none |
| MLP gate | `[4096, 14336]` | `[4096, 1792]` | column parallel | none |
| MLP up | `[4096, 14336]` | `[4096, 1792]` | column parallel | none |
| MLP down | `[14336, 4096]` | `[1792, 4096]` | row parallel, then reduce-scatter | none |

Collectives:

- Attention decode: local QKV, local RoPE, local paged KV update, local SDPA,
  fused all-gather plus WO matmul, then hidden all-gather to restore the
  replicated decoder boundary.
- Attention prefill: local heads, paged cache fill, SDPA, fused all-gather
  before WO, then hidden all-gather to restore the replicated boundary.
- MLP decode and prefill: gate/up are output-sharded, down produces partials,
  reduce-scatter creates a hidden shard, then hidden all-gather restores the
  replicated boundary.
- RMSNorm: local full-hidden RMSNorm. This is correct because the decoder
  boundary is replicated full hidden on every chip.
- MoE/expert strategy: not applicable; Llama 3.1 8B is dense.

Rejected alternatives:

- Hidden-sharded decoder boundary: it keeps less activation data resident but
  still requires two full-hidden all-gathers before the next layer's column
  parallel QKV and MLP gate/up matmuls. It also adds distributed RMSNorm stats
  collectives. For this dense decoder, the replicated boundary avoids those
  norm collectives with the same major hidden all-gather count.
- 2D mesh strategy: not applicable on this T3K-class `1x8` target.
- Padding MLP intermediate from 14336 to 16384: this would add 14.3% extra MLP
  weight traffic and compute. The native `14336 / 8 = 1792` shard is tile
  aligned.

## Validation

Primary command:

```bash
MULTICHIP_DECODER_DECODE_REPLAYS=4 python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py::test_multichip_decoder_synthetic_paged_prefill_decode_trace_against_optimized \
  -vv -s
```

Final non-profiler result from the full test-file run, 2026-06-15:

| Metric | Value |
| --- | ---: |
| Prefill PCC vs optimized decoder | 0.9999957546448677 |
| Decode trace PCC vs optimized decoder trace | 0.9999870318423426 |
| Decode trace determinism PCC | 1.0 |
| Decode eager vs trace PCC | 1.0 |
| Single-chip warmed decode min | 0.790356193 ms |
| Multichip warmed decode min | 0.441294163 ms |
| Decode speedup | 1.790996252x |
| TP efficiency | 0.223874531 |
| Single-chip warmed prefill | 3.402695060 ms |
| Multichip warmed prefill | 2.707116306 ms |

The synthetic case validates a non-identity page table, current positions,
local KV-cache layout, replicated multichip input/output layout, runtime
fallback audit, warmed trace capture, and repeated trace replay. The local
KV cache shape is `[max_num_blocks, 1, 64, 128]` per chip for the 1x8 target.

Additional contract coverage:

- `test_multichip_decoder_contract_and_policy`: verifies the public decoder
  contract, fixed mesh constants, and dtype policy inheritance from the
  optimized decoder.
- `test_multichip_decoder_full_context_cache_contract`: verifies full-context
  cache allocation for `128k` tokens, `2048` pages, and one local KV head per
  device.

Final full test-file command:

```bash
python_env/bin/python -m py_compile \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/multichip_decoder.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py
MULTICHIP_DECODER_DECODE_REPLAYS=4 python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -vv -s
```

Result: `3 passed in 18.63s`.

## Profiling

Profiler command:

```bash
MULTICHIP_DECODER_DECODE_REPLAYS=4 python_env/bin/python -m tracy -r -p -v \
  -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/multichip_decoder/tracy/synthetic/.logs \
  -m pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k synthetic_paged_prefill_decode_trace -q -s
```

Profiler result, 2026-06-15:

| Metric | Value |
| --- | ---: |
| Prefill PCC vs optimized decoder | 0.9999957546448677 |
| Decode trace PCC vs optimized decoder trace | 0.9999870318423426 |
| Decode trace determinism PCC | 1.0 |
| Decode eager vs trace PCC | 1.0 |
| Single-chip profiled decode min | 0.812513288 ms |
| Multichip profiled decode min | 0.499652233 ms |
| Decode speedup under profiler | 1.626157624x |
| TP efficiency under profiler | 0.203269703 |

`tt-perf-report` summary from
`tracy/synthetic/.logs/reports/2026_06_15_14_33_14/ops_perf_results_2026_06_15_14_33_14.csv`:

| Slice | Rows | Merged device time | Host rows |
| --- | ---: | ---: | ---: |
| Baseline prefill | 23 | 2451.073 us | 0 |
| Baseline decode | 19 | 749.453 us | 0 |
| Multichip prefill | 27 | 1184.215 us | 0 |
| Multichip decode | 26 | 418.317 us | 0 |
| Multichip decode per-device | 208 | 3261.715 us summed across 8 devices | 0 |

Multichip decode merged hot spots:

| Op category | Count | Device time |
| --- | ---: | ---: |
| `AllGatherAsyncDeviceOperation` | 2 | 77.510 us |
| `ReduceScatterMinimalAsyncDeviceOperation` | 1 | 69.983 us |
| `MatmulDeviceOperation 32 x 4096 x 1792` | 2 | 58.606 us |
| `AllGatherMatmulAsyncDeviceOperation 32 x 512 x 512` | 1 | 44.108 us |
| `MatmulDeviceOperation 32 x 4096 x 768` | 1 | 36.599 us |
| `MatmulDeviceOperation 32 x 1792 x 4096` | 1 | 31.233 us |
| `BinaryNgDeviceOperation` | 3 | 28.153 us |
| `LayerNormDeviceOperation` | 2 | 23.278 us |

Per-device multichip decode balance:

| Op | Per-device range |
| --- | ---: |
| Fused all-gather matmul | 40.208-46.687 us |
| Hidden all-gathers | 36.926-40.641 us |
| Reduce-scatter | 62.437-75.386 us |
| Gate/up matmuls | 27.037-29.484 us |
| QKV matmul | 36.338-36.599 us |
| Down matmul | 22.451-31.233 us |

The final profiler log has no `allowed_worker_cores not populated` warnings
from the multichip path. `tt-perf-report` emits category warnings for CCL ops
that are not yet in its classification table. The merged multichip decode text
report also shows a large op-to-op gap before a layernorm; this is a
cross-device/signpost merge artifact. The host replay latency and merged device
time are the accepted latency evidence.

Stable artifact names:

- `tracy/synthetic/tracy_run.log`
- `tracy/synthetic/multichip_ops_perf_results.csv`
- `tracy/synthetic/multichip_profile_log_device.csv`
- `tracy/synthetic/tracy_ops_data.csv`
- `tracy/synthetic/tracy_ops_times.csv`
- `tracy/synthetic/baseline_prefill_perf_report.{txt,csv,console.log}`
- `tracy/synthetic/baseline_decode_perf_report.{txt,csv,console.log}`
- `tracy/synthetic/multichip_prefill_perf_report.{txt,csv,console.log}`
- `tracy/synthetic/multichip_decode_perf_report.{txt,csv,console.log}`
- `tracy/synthetic/multichip_decode_perf_report_per_device.{txt,csv,console.log}`

## Watcher

Final watcher command:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/multichip_decoder/watcher/synthetic_ring_final \
MULTICHIP_DECODER_DECODE_REPLAYS=1 python_env/bin/pytest --timeout=1200 \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k synthetic_paged_prefill_decode_trace -vv -s
```

Result: `1 passed, 2 deselected in 16.00s`. Watcher attached to all eight
devices with disabled features `None`. PCCs matched the non-watcher run:
prefill `0.9999957546448677`, decode `0.9999870318423426`, trace determinism
`1.0`, and eager-vs-trace `1.0`. The runtime fallback audit reported
`multichip_prefill_decode_clean`.

The watcher/inspector scan found no fatal/assert/exception/hang/fault/illegal/
heartbeat/overflow/sanitize/out-of-bounds signatures in:

- `watcher/synthetic_ring_final/generated/watcher/watcher.log`
- `watcher/synthetic_ring_final/generated/inspector/startup.yaml`
- `watcher/synthetic_ring_final/generated/inspector/mesh_devices_log.yaml`
- `watcher/synthetic_ring_final/generated/inspector/mesh_workloads_log.yaml`
- `watcher/synthetic_ring_final/generated/inspector/programs_log.yaml`

## Limitations

- The implementation is specialized to the local `1x8` T3K mesh.
- The layer boundary is replicated full hidden. This is correct and faster for
  the measured single-layer decode case than adding distributed RMSNorm stats,
  but stacked full-model throughput may still be limited by two hidden
  all-gathers and one reduce-scatter per layer.
- Only the decoder-layer baseline is implemented here. Full-model assembly and
  vLLM integration are intentionally out of scope for this multichip-decoder
  stage.
