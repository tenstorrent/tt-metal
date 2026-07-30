# Multichip Decoder Work Log

Date: 2026-06-15

Scope: create the repo-local multichip decoder state for
`meta-llama/Llama-3.2-1B-Instruct` only. Full-model and vLLM work were not
started.

## Implementation Summary

- Added `../../tt/multichip_decoder.py`.
- Used `OptimizedDecoder.from_state_dict(..., materialize=False)` as the
  single-chip baseline for attention configuration, paged KV behavior, and
  precision choices.
- Selected the available T3K `1x8` mesh with `FABRIC_1D_RING` and Ring
  topology before finalizing the code path.
- Implemented 8-way tensor parallel WQKV, local Q/K/V heads, local paged KV,
  fused all-gather matmul for WO, and MLP W1/W3 column parallel plus W2
  row-parallel reduce-scatter/all-gather.
- Reordered global Q/K/V weight chunks at load time so each device receives
  4 Q heads and 1 KV head.
- Bound paged attention overrides so fill, update, and SDPA decode use the
  explicit page block size and local KV-head count required by the mesh path.
- Kept decoder input/output replicated so stacked decoders can use the same
  layer contract.
- Added multichip tests in `../../tests/test_multichip_decoder.py`.
- Updated `../../tests/conftest.py` so mesh tests can enable and reset fabric.

## Mesh, Tensor, KV, Collective, And Expert Plan

Target plan:

- Hardware: 8 Wormhole B0 devices, `ClusterType.T3K`.
- Mesh shape: `1x8`.
- Tensor parallel degree: 8.
- Fabric: `ttnn.FabricConfig.FABRIC_1D_RING`.
- Topology: `ttnn.Topology.Ring`.
- Hidden size: 2048.
- Intermediate size: 8192.
- Attention heads: 32 total, 4 per device.
- KV heads: 8 total, 1 per device.
- Head dim: 64.
- Page block size: 64.

Per-device tensor shapes:

| Tensor | Global shape | Per-device shape | Strategy | Padding |
| --- | ---: | ---: | --- | --- |
| WQKV | `[2048, 3072]` | `[2048, 384]` | Column/output sharded with Q/K/V reorder | none |
| Q heads | `[32, 64]` | `[4, 64]` | Head sharded | none |
| K/V heads | `[8, 64]` | `[1, 64]` | KV head sharded | none |
| WO | `[2048, 2048]` | `[2048, 256]` | Column/output sharded after fused all-gather matmul | none |
| W1/W3 | `[2048, 8192]` | `[2048, 1024]` | Column/output sharded | none |
| W2 | `[8192, 2048]` | `[1024, 2048]` | Row/input sharded | none |
| RMSNorm | `[2048]` | `[2048]` | Replicated | none |

Shard specs and placement records:

- WQKV is reordered at load time into 8 global Q/K/V chunks and mapped with
  `ShardTensorToMesh(..., dim=-1)`; local shape is `[2048, 384]`.
- W1 and W3 are mapped with `ShardTensorToMesh(..., dim=-1)`; local shape is
  `[2048, 1024]`.
- W2 is mapped with `ShardTensorToMesh(..., dim=-2)`; local shape is
  `[1024, 2048]`.
- RMSNorm weights are replicated through the `RMSNorm1D` path.
- Page table and current position tensors use `ReplicateTensorToMesh`.
- Decode full-hidden activations use the optimized decoder residual memory
  config: width-sharded L1, grid `x=0..7, y=0..3`, shard shape `[32, 64]`.
- Decode W2 partial/reduce-scatter outputs use width-sharded L1, grid
  `x=0..7, y=0..0`, shard shape `[32, 32]`.
- Prefill matmul and CCL intermediates use DRAM interleaved outputs before the
  Ring collectives.
- No weight padding is needed because `2048/8=256`, `8192/8=1024`,
  `32/8=4`, and `8/8=1`. Paged-KV sequence capacity is page-rounded to
  64-token blocks.

Activation and KV contracts:

- Decoder input: replicated full hidden stream.
- Decoder output: replicated full hidden stream.
- Attention output: hidden-sharded internally, all-gathered before residual add.
- MLP output: reduce-scattered hidden shard, all-gathered before residual add.
- RMSNorm: local on replicated hidden stream.
- Page table: replicated `int32`.
- Current positions: replicated `int32`.
- KV cache dtype: `bfloat8_b`.
- Real 8192-token test cache shape per device: `[129, 1, 64, 64]`.
- Prefill and decode output topology: replicated across all 8 devices.

Collectives:

- Attention WO path uses fused all-gather matmul on the Ring.
- Attention output uses all-gather on hidden dim.
- MLP W2 uses `reduce_scatter_minimal_async`.
- MLP output uses all-gather on hidden dim.

MoE/expert strategy:

- Not applicable. Llama 3.2 1B Instruct is a dense decoder and has no experts.

Rejected alternatives:

- `1x1`: baseline only; no multichip utilization.
- `1x4`: valid, but leaves half the available T3K mesh unused.
- 2D/Galaxy plan: hardware mismatch for this machine.
- Hidden-sharded residual stream: rejected because this state needs a reliable
  traced stacked-decoder contract and the current path still gathers for QKV,
  W1/W3, and RMSNorm.

Serialized artifact: `mesh_strategy.json`.

## Commands And Results

Compile check:

```bash
python -m py_compile models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py models/autoports/meta_llama_llama_3_2_1b_instruct/tt/multichip_decoder.py models/autoports/meta_llama_llama_3_2_1b_instruct/tests/conftest.py
```

Result: passed.

Synthetic correctness, trace replay, and determinism:

```bash
pytest --timeout=600 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_synthetic_multichip_paged_prefill_decode_trace_and_determinism
```

Result: passed. Artifact: `synthetic_correctness.json`.

Real weights, short/default prefill:

```bash
pytest --timeout=600 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_real_weights_multichip_paged_prefill_and_decode_trace
```

Result: passed. Artifact: `real_weight_correctness.json`.

Real weights, 4096-token prefill:

```bash
MD_PREFILL_SEQ_LEN=4096 pytest --timeout=600 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_real_weights_multichip_paged_prefill_and_decode_trace
```

Result: passed. Artifact: `real_weight_correctness_prefill_4096.json`.

Real weights, 8192-token prefill:

```bash
MD_PREFILL_SEQ_LEN=8192 pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_real_weights_multichip_paged_prefill_and_decode_trace
```

Result: passed. Artifact: `real_weight_correctness_prefill_8192.json`.

Static plan, fallback audit, and repeated-run stress:

```bash
pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_multichip_decoder_contract_and_runtime_fallback_audit models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_multichip_static_mesh_plan_uses_optimized_baseline models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_runtime_fallback_audit_measured_multichip_prefill_and_traced_decode models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_multichip_repeated_run_stress
```

Result: 4 passed. Artifacts: `mesh_strategy.json`,
`runtime_fallback_audit.json`, and `stress_repeated_runs.json`.

Watcher run:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_APPEND=1 pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_runtime_fallback_audit_measured_multichip_prefill_and_traced_decode
```

Result: passed. Watcher audit:

```bash
rg -n "ERROR|FATAL|ASSERT|TIMEOUT|Watchdog|hang|unhealthy" generated/watcher/watcher.log
```

Result: no matches. Artifacts copied under `watcher/`.

Tracy perf run:

```bash
MD_PERF_PREFILL_SEQ_LEN=8192 python -m tracy -r -p -v -o models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/tracy/run_prefill_decode -m pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_perf_artifact_signposted_multichip_prefill_and_decode
```

Result: passed. Raw Tracy CSV was copied to
`perf/ops_perf_results_raw.csv`; device profile log was copied to
`perf/profile_log_device_raw.csv`.

Perf contract refresh:

```bash
MD_PERF_PREFILL_SEQ_LEN=8192 pytest --timeout=600 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_multichip_decoder.py::test_perf_artifact_signposted_multichip_prefill_and_decode
```

Result: passed. Artifact: `perf_trace_contract.json`.

Report generation commands:

```bash
tt-perf-report --no-color --no-host-ops --start-signpost PERF_MULTICHIP_PREFILL --end-signpost PERF_MULTICHIP_PREFILL_END --csv models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/prefill_8192_report.csv --summary-file models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/prefill_8192_summary.csv models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/ops_perf_results_raw.csv > models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/prefill_8192_tt_perf_report.txt

tt-perf-report --no-color --no-host-ops --no-merge-devices --start-signpost PERF_MULTICHIP_PREFILL --end-signpost PERF_MULTICHIP_PREFILL_END --csv models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/prefill_8192_per_device_report.csv --summary-file models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/prefill_8192_per_device_summary.csv models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/ops_perf_results_raw.csv > models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/prefill_8192_per_device_tt_perf_report.txt

tt-perf-report --no-color --no-host-ops --start-signpost PERF_MULTICHIP_DECODE --end-signpost PERF_MULTICHIP_DECODE_END --csv models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/decode_trace_replay_report.csv --summary-file models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/decode_trace_replay_summary.csv models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/ops_perf_results_raw.csv > models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/decode_trace_replay_tt_perf_report.txt

tt-perf-report --no-color --no-host-ops --no-merge-devices --start-signpost PERF_MULTICHIP_DECODE --end-signpost PERF_MULTICHIP_DECODE_END --csv models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/decode_trace_replay_per_device_report.csv --summary-file models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/decode_trace_replay_per_device_summary.csv models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/ops_perf_results_raw.csv > models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder/perf/decode_trace_replay_per_device_tt_perf_report.txt
```

## Correctness Evidence

Synthetic 128-token run:

- Prefill PCC: `0.9999958887325306`.
- Eager decode PCC: `0.9999932200355885`.
- Traced decode replay PCC: `0.9999932200355885`.
- Repeated trace replay PCC: `1.0`.
- Baseline repeated trace replay PCC: `0.9999999999999931`.

Real-weight 8192-token run:

- Prefill PCC: `0.9999963551395705`.
- Traced decode replay PCC: `0.9999946874742001`.
- Repeated trace replay PCC: `0.9999999999999873`.
- Local key/value cache shape on each device: `[129, 1, 64, 64]`.
- Page table and current position tensors were replicated.
- Prefill and decode outputs were replicated across the mesh.

Runtime fallback and stress:

- `runtime_fallback_audit.json`: measured prefill and traced decode passed with
  guarded Python bridge calls.
- `stress_repeated_runs.json`: 3 repeated runs, each PCC
  `0.9999964050234625`.
- `watcher/watcher_summary.json`: watcher-enabled measured pass status
  `passed`; watcher log had no error matches.

## Performance Evidence

Baseline source:

- `../optimized_decoder/perf/perf_provenance.json`

Multichip source:

- `perf/perf_provenance.json`

Measured performance:

| Stage | Single-chip device us | Multichip device us | Speedup | Efficiency vs 8 | Single-chip host ms | Multichip host ms | Host speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill 8192 | 28849.668 | 15799.536 | 1.8260x | 0.2282 | 36.0752753913 | 15.9225054085 | 2.2657x |
| Traced decode replay | 519.718 | 383.039 | 1.3568x | 0.1696 | 0.8435621858 | 0.6744153798 | 1.2508x |

Perf operation audit:

- Prefill: 24 device ops, 0 host ops, 15799.536 us device time, 14.659 us
  gap time, 8193.324 us CCL time, 7343.032 us compute time, 114.888 us data
  movement time.
- Decode: 27 device ops, 0 host ops, 383.039 us device time, 434.421 us gap
  time, 157.644 us CCL time, 235.322 us compute time, 19.543 us data
  movement time.
- Prefill top costs are Ring all-gathers and reduce-scatter near 2 ms each,
  followed by SDPA and local matmuls.
- Decode top costs are MLP reduce-scatter, SDPA decode, fused all-gather matmul,
  and hidden all-gathers.

Limitation:

- Decode inclusive device-plus-gap speedup is `0.6599x`, so the Tracy inclusive
  range is not faster. Summed device-op time and warmed host replay are faster.

## Notes

The long-context real-weight test initially exposed a test harness page-table
construction bug. The fix was to use the shared functional page-table helper so
the single-chip optimized baseline and multichip path receive identical host and
TT page tables. After that change, 4096-token and 8192-token real-weight PCC
passed.
