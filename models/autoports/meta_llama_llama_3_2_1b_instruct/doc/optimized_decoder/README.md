# Optimized Decoder

Model: `meta-llama/Llama-3.2-1B-Instruct`

Scope: one repo-local TTNN decoder layer at
`models/autoports/meta_llama_llama_3_2_1b_instruct/tt/optimized_decoder.py`.
No multichip decoder, full-model, or vLLM work was started.

## Result

The optimized decoder keeps the functional decoder's paged prefill/decode
contract while replacing the functional MLP path with `_OptimizedLlamaMLP` and
using optimized Attention1D/RMSNorm1D configs.

| Measurement | Functional before | Optimized after |
| --- | ---: | ---: |
| 8192 prefill PCC | 0.9999664355 | 0.9995248960 |
| traced decode replay PCC at pos 8192 | 0.9999890750 | 0.9995249460 |
| repeated traced replay PCC | 1.0 | 1.0 |
| warmed prefill device time | 36560 us | 28849.668 us |
| warmed traced decode replay device time | 864 us | 519.718 us |
| decode replay host wall, same optimized run | n/a | 0.843562 ms |

The PCC delta is expected from promoting the dense MLP weights to BFP4. The
optimized path stays above the 0.995 functional acceptance bar, and the precision
sweep is recorded in `precision_experiments.json`.

## Chosen Runtime Policy

| Area | Final setting |
| --- | --- |
| attention weights | BFP8 |
| MLP gate/up/down weights | BFP4 |
| paged KV cache | BFP8 |
| activations, residuals, norms, MLP mul output | BF16 |
| decode attention matmuls | DRAM-sharded `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` |
| decode MLP matmuls | DRAM-sharded, width-sharded L1 activations |
| prefill attention matmuls | 2D multicast, `in0_block_w=8`, output subblock `1x4` |
| prefill MLP matmuls | 2D multicast, `in0_block_w=8` where applicable |
| SDPA | Attention1D paged prefill SDPA and decode SDPA |
| MoE | not applicable; Llama-3.2-1B is dense |

`optimized_config_summary.json` records the exact generated configs.

## Correctness Coverage

| Artifact | Coverage |
| --- | --- |
| `synthetic_correctness.json` | synthetic paged prefill, eager decode, traced decode replay, repeated replay determinism |
| `real_weight_correctness.json` | real-weight 128-token paged prefill/decode smoke |
| `real_weight_correctness_prefill_8192.json` | real-weight representative 8192-token prefill and traced decode |
| `stress_repeated_runs.json` | 5 repeated optimized prefill runs with identical PCC |
| `runtime_fallback_audit.json` | guarded optimized prefill plus trace capture/replay with `ttnn.from_torch` and `ttnn.to_torch` patched to fail inside measured passes |
| `watcher/watcher_summary.json` | watcher-clean optimized correctness run |

## Perf Report Summary

`tt-perf-report` was run on the final signposted Tracy CSV for both windows.

| Window | Device ops | Host ops | Device time | Notes |
| --- | ---: | ---: | ---: | --- |
| `PERF_PREFILL` to `PERF_PREFILL_END` | 20 | 0 | 28849.668 us | no measured Torch/from_torch/to_torch/tilize/untilize rows |
| `PERF_DECODE` to `PERF_DECODE_END` | 19 | 0 | 519.718 us | no measured Torch/from_torch/to_torch/tilize/untilize rows |

Decode has two tiny TTNN layout transitions from the current Attention1D
composite path: `ShardedToInterleavedDeviceOperation` and
`InterleavedToShardedDeviceOperation`, totaling about 2.6 us. They are required
by the current QKV-head/concat/SDPA interfaces and are not host fallback.

Primary artifacts:

- `perf/ops_perf_results_raw.csv`
- `perf/prefill_8192_tt_perf_report.txt`
- `perf/prefill_8192_report.csv`
- `perf/prefill_8192_human_summary.csv.csv`
- `perf/decode_trace_replay_tt_perf_report.txt`
- `perf/decode_trace_replay_report.csv`
- `perf/decode_trace_replay_human_summary.csv.csv`
- `perf/perf_provenance.json`
- `perf/decode_performance_accounting.json`

## Advice Handling

Kept:

| Advice | Evidence | Outcome |
| --- | --- | --- |
| Increase small attention prefill `in0_block_w` and output subblock | `perf/advice_trials/attention_prefill_default_in0_block1` vs final `perf/perf_provenance.json` | prefill improved from 32202.135 us to 28849.668 us; QKV prefill dropped from 3873 us to 1524 us and WO from 1648 us to 1072 us |

Rejected or not applicable:

| Advice or option | Reason |
| --- | --- |
| Put prefill matmul input0 in L1 | The 8192-token decoder prefill boundary is full-sequence DRAM-interleaved. Moving the full hidden stream, and especially the MLP fused intermediate, into L1 would require extra measured data-movement/reshard ops or a different chunked decoder API. Current Attention1D/RMSNorm1D/MLP1D prefill defaults also use DRAM for this path. |
| Use HiFi4 for non-FLOP-bound matmuls | The perf-report advice is an accuracy suggestion. HiFi2 is the canonical Attention1D/MLP1D matmul policy here, and the BFP4/BFP8 policy remains above the acceptance bar while improving latency. |
| QK fused decode rotary | `precision_experiments.json` records a TT_FATAL API mismatch: fused rotary expects cos/sin batch coverage for Q and K, while the current decode rotary helper supplies the Q batch only. |
| MoE active-expert execution | Not model-applicable; this Llama decoder has a dense MLP and no experts. |

## Limits

This is an optimized single-decoder-layer state for the autoport pipeline. It
does not include multichip, full-model generator, or vLLM serving integration.
Those stages should start from this optimized decoder state.
