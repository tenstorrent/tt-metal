# Phi-3.5 Mini Multichip Decoder Work Log

## 2026-06-15 Plan

Read:

- `.agents/skills/multichip/SKILL.md`
- `tech_reports/LLMs/llms.md` section 3.3 Multi-Device
- `tt/optimized_decoder.py`
- `tests/test_optimized_decoder.py`
- `doc/optimized_decoder/README.md`
- `models/common/modules/attention/attention_1d.py`
- `models/common/modules/mlp/mlp_1d.py`
- `models/common/modules/rmsnorm/rmsnorm_1d.py`
- `models/common/modules/tt_ccl.py`

Hardware commands:

```bash
python -c "import ttnn; print('arch', ttnn.get_arch_name()); print('cluster', ttnn.cluster.get_cluster_type())"
tt-smi -ls
```

Observed hardware:

- `arch wormhole_b0`
- `cluster ClusterType.T3K`
- 8 Wormhole devices exposed as four N300 boards

Chosen target:

- `ttnn.MeshShape(1, 8)`
- `ttnn.FabricConfig.FABRIC_1D_RING`
- `ttnn.Topology.Ring`
- TP axis: mesh axis 1
- TP factor: 8

Chosen runtime contract:

- Input/output hidden states replicated across the mesh.
- Norm weights, RoPE tables, page tables, `current_pos`, and `position_ids` replicated.
- QKV and gate/up weights reordered at load time so each contiguous mesh shard has the chip-local semantic block.
- QKV and gate/up sharded on output dim.
- O and down sharded on input dim.
- Local KV cache shape: `[num_blocks, 4, block_size, 96]` per chip.
- Decode trace target remains a warmed `ttnn.begin_trace_capture` / `execute_trace` path.

Rejected alternatives are recorded in `README.md`.

## Implementation

Files added:

- `tt/multichip_decoder.py`
- `tests/test_multichip_decoder.py`

The implementation imports `OptimizedDecoder` and exposes it as the single-chip baseline class. The multichip path keeps the same stacked decoder input/output contract as the optimized decoder, but requires a `1x8` mesh and ring fabric. The hot prefill/decode paths use TTNN operations only; load-time PyTorch work is limited to weight validation and reordering before TTNN tensor creation.

The real multichip path uses:

- replicated residual stream, norm weights, RoPE tables, page table, `current_pos`, and `position_ids`;
- QKV and gate/up load-time reordering before output-dim sharding;
- local-head Q/K/V ownership with 4 Q heads and 4 KV heads per chip;
- local paged KV cache shape `[num_blocks, 4, 32, 96]` per chip;
- row-sharded attention O and MLP down weights;
- `ttnn.all_reduce(..., cluster_axis=1, topology=Ring)` after attention O and MLP down partials.

MoE strategy: not applicable. Phi-3.5-mini is dense and has no expert layers.

## Validation Commands And Results

Compile:

```bash
python -m py_compile models/autoports/microsoft_phi_3_5_mini_instruct/tt/multichip_decoder.py
python -m py_compile models/autoports/microsoft_phi_3_5_mini_instruct/tt/multichip_decoder.py models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py
```

Result: passed.

Static mesh plan and fallback audit:

```bash
pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_mesh_plan_static \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_runtime_forward_fallback_audit_static -s
```

Result: `2 passed in 3.86s`. The static audit found no `torch.`, `ttnn.from_torch`, `ttnn.to_torch`, `from_device`, or `.cpu(` usage in the runtime multichip hot path.

Synthetic layer PCC against the optimized TTNN baseline:

```bash
pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_vs_single_chip_synthetic_prefill_decode_pcc_1x8_ring -s
```

Result: `1 passed in 53.17s`.

- Prefill PCC: `0.9999945714628357`
- Decode PCC: `0.9999955128943894`
- Warmed decode trace capture/replay executed.

Real layer-0 PCC against the optimized TTNN baseline:

```bash
pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_dense_layer_real_weights_prefill_decode_pcc_1x8_ring -s
```

Result: `1 passed in 14.09s`.

- Prefill PCC: `0.999991788840335`
- Decode PCC: `0.9999935080298819`
- Warmed decode trace capture/replay executed.

Determinism coverage:

```bash
pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_repeated_input_determinism_1x8_ring -s
```

Result: `1 passed in 23.24s`.

- Baseline comparison PCCs were stable across repeated runs: prefill `0.9999947598886589`, decode `0.9999956354943992`.
- Repeated multichip outputs were checked at PCC `>= 0.9999`.

Full-context decode stress:

```bash
PHI35_RUN_LONG_CONTEXT=1 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_full_context_decode_current_position_and_page_table_1x8_ring -s
```

Result: `1 passed in 13.55s`.

- Exercises `current_pos=131071`, `max_seq_len=131072`, full page table, and local cache layout `[4096, 4, 32, 96]` per chip.

Long prefill page-table stress:

```bash
PHI35_RUN_LONG_PREFILL=1 PHI35_LONG_PREFILL_LEN=32768 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_long_prefill_page_table_1x8_ring -s
```

Result: `1 passed in 33.33s`.

Default multichip test file after adding the gated perf-only test:

```bash
pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py -s \
  2>&1 | tee models/autoports/microsoft_phi_3_5_mini_instruct/doc/multichip_decoder/perf/default_test_file.log
```

Result: `5 passed, 3 skipped in 41.52s`.

Skipped tests are the explicit gated perf, long-context, and long-prefill tests.

## Latency And Perf

Host-timed warmed multichip trace replay:

```bash
PHI35_HOST_TIMING_ITERS=100 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_dense_layer_real_weights_prefill_decode_pcc_1x8_ring -s \
  2>&1 | tee models/autoports/microsoft_phi_3_5_mini_instruct/doc/multichip_decoder/perf/host_timing_real_layer0_after_reset.log
```

Result: `1 passed in 14.12s`.

- Multichip warmed trace replay: `580.549 us/token`.
- PCCs in this timing run matched the real layer-0 evidence: prefill `0.999991788840335`, decode `0.9999935080298819`.

Single-chip optimized baseline reference from `doc/optimized_decoder/README.md`:

- Warmed traced decode device + gap: `923.470 us`
- Warmed host trace replay: `0.983737 ms/token`
- Warmed prefill device + gap: `2612.095 us`

Speedup:

| Measurement | Single-chip | Multichip | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: |
| Warmed decode host trace replay | `983.737 us/token` | `580.549 us/token` | `1.694x` | `21.2%` |
| Decode profiled device + gap | `923.470 us` | `1205.575 us` | `0.766x` | `9.6%` |
| Prefill profiled device + gap, `T=32` | `2612.095 us` | `4585.328 us` | `0.570x` | `7.1%` |

Profiler collection:

```bash
PHI35_RUN_PERF=1 python -m tracy -r -p -v \
  -o models/autoports/microsoft_phi_3_5_mini_instruct/doc/multichip_decoder/tracy/host_only \
  -m pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_dense_layer_real_weights_perf_profile_1x8_ring -s \
  2>&1 | tee models/autoports/microsoft_phi_3_5_mini_instruct/doc/multichip_decoder/perf/tracy_host_only_perf.log
```

Result: `1 passed in 87.86s`; merged ops CSV generated:

`tracy/host_only/reports/2026_06_15_14_13_08/ops_perf_results_2026_06_15_14_13_08.csv`

`tt-perf-report` commands:

```bash
CSV=models/autoports/microsoft_phi_3_5_mini_instruct/doc/multichip_decoder/tracy/host_only/reports/2026_06_15_14_13_08/ops_perf_results_2026_06_15_14_13_08.csv
OUT=models/autoports/microsoft_phi_3_5_mini_instruct/doc/multichip_decoder/perf

tt-perf-report "$CSV" --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
  --no-color --csv "$OUT/prefill_perf_report.csv" --summary-file "$OUT/prefill_perf_summary" \
  2>&1 | tee "$OUT/prefill_perf_report.txt"

tt-perf-report "$CSV" --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --tracing-mode --no-color --csv "$OUT/decode_perf_report.csv" --summary-file "$OUT/decode_perf_summary" \
  2>&1 | tee "$OUT/decode_perf_report.txt"

tt-perf-report "$CSV" --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
  --no-color --no-summary 2>&1 | tee "$OUT/prefill_perf_human.txt"

tt-perf-report "$CSV" --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --tracing-mode --no-color --no-summary 2>&1 | tee "$OUT/decode_perf_human.txt"
```

Perf checks:

- Prefill report: 50 device ops, `797.647 us` device time, `3787.681 us` op-to-op gap, `4585.328 us` total. CCL summary includes `137.89 us` ReduceScatter and `78.06 us` AllGather. Matmuls are only `165.15 us` total with low FLOPs utilization; data movement and host gaps dominate.
- Decode report: 63 device ops, `571.840 us` device time, `633.735 us` op-to-op gap, `1205.575 us` total under profiling. CCL summary includes `135.28 us` ReduceScatter and `71.66 us` AllGather. Matmuls are `125.00 us` total; the largest reported gap is `546.701 us` before an `InterleavedToShardedDeviceOperation`.
- Paged cache ops are present in the reports: `PagedFillCacheDeviceOperation` during prefill and `PagedUpdateCacheDeviceOperation` during decode.
- DRAM and compute checks show the path is not compute-bound. Example decode matmuls report 16.5%-35.7% DRAM utilization and 19.4%-39.1% FLOPs utilization, while CCL and layout movement are the main optimization targets.

Profiler caveat:

- A first Tracy run with `PHI35_READ_DEVICE_PROFILER=1 PHI35_HOST_TIMING_ITERS=100` overflowed device profiler buffers and failed report generation.
- A second run with `PHI35_READ_DEVICE_PROFILER=1` on the full baseline-comparison test crashed while closing the single-chip baseline mesh under profiler.
- A third run with `PHI35_READ_DEVICE_PROFILER=1` on the perf-only multichip test stalled in profiler reads.
- Accepted perf artifacts come from `python -m tracy -p` without explicit `ReadDeviceProfiler`; this still produced a v2.1 ops CSV with device arch, device IDs, device times, op-to-op gaps, DRAM, FLOPs, CCL, and data movement fields.

## Watcher

Command:

```bash
RUN=2026_06_15_1x8_ring_real
mkdir -p models/autoports/microsoft_phi_3_5_mini_instruct/doc/multichip_decoder/watcher/$RUN
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_APPEND=1 \
TT_METAL_LOGS_PATH=models/autoports/microsoft_phi_3_5_mini_instruct/doc/multichip_decoder/watcher/$RUN \
pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q \
  models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_multichip_decoder.py::test_multichip_dense_layer_real_weights_prefill_decode_pcc_1x8_ring -s \
  2>&1 | tee models/autoports/microsoft_phi_3_5_mini_instruct/doc/multichip_decoder/watcher/$RUN/pytest.log
```

Result: `1 passed in 286.35s`.

Runtime scan:

```bash
RUN=2026_06_15_1x8_ring_real
rg -n -i "TT_FATAL|TT_THROW|watcher[^\\n]*error|exception|out[ -]?of[ -]?bounds|stack overflow|l1[^\\n]*overflow|noc[^\\n]*(bad|error)|bad noc|retraining events: [1-9]" \
  models/autoports/microsoft_phi_3_5_mini_instruct/doc/multichip_decoder/watcher/$RUN/generated/watcher/watcher.log
```

Result: no matches.

Watcher tail evidence:

- All eight devices detached cleanly.
- Ethernet retraining events were `0` on all reported cores.
- Minimum reported free stack was `456 bytes` on TRISC0 in `sdpa_flash_decode.cpp`; no stack overflow was reported.

Artifacts:

- `watcher/2026_06_15_1x8_ring_real/pytest.log`
- `watcher/2026_06_15_1x8_ring_real/generated/watcher/watcher.log`

## Requirement Status

| Requirement | Status | Evidence |
| --- | --- | --- |
| `tt/multichip_decoder.py` exists and uses optimized baseline | Done | `MultichipDecoder.single_chip_baseline_cls = OptimizedDecoder` |
| Mesh plan and calculated per-device shapes recorded | Done | `README.md` |
| Prefill PCC vs optimized baseline | Done | Synthetic `0.9999945714628357`; real layer-0 `0.999991788840335` |
| Decode PCC vs optimized baseline | Done | Synthetic `0.9999955128943894`; real layer-0 `0.9999935080298819` |
| Paged KV/page-table/current-pos/local-head layout validated | Done | Default correctness tests, full-context decode stress, long prefill stress |
| Stacked input/output layout contract validated | Done | Replicated mesh output checks across devices and shape checks in tests |
| Warmed decode trace replay works | Done | Synthetic, real, host-timed, perf-only, and watcher runs |
| Single-chip and multichip latency, speedup, efficiency | Done | Latency table above |
| `tt-perf-report` human-readable and CSV/provenance artifacts | Done | `perf/*perf_human.txt`, `perf/*perf_report.csv`, `perf/*perf_summary.*`, Tracy CSV |
| Runtime fallback audit clean | Done | Static audit passed |
| Determinism or stress coverage | Done | Repeated-input determinism, full-context decode, long prefill |
| Watcher-clean evidence | Done | Watcher real-weight run passed; runtime watcher log scan clean |
