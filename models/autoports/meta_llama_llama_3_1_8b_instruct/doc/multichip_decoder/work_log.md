# Multichip Decoder Work Log

## 2026-06-15

### Setup And Strategy

- Used `$multichip`.
- Read `tech_reports/LLMs/llms.md` section 3.3 Multi-Device.
- Read the optimized decoder baseline in
  `models/autoports/meta_llama_llama_3_1_8b_instruct/tt/optimized_decoder.py`.
- Read common TTTv2 references:
  - `models/common/modules/attention/attention_1d.py`
  - `models/common/modules/rmsnorm/rmsnorm_1d.py`
  - `models/common/modules/mlp/mlp_1d.py`
  - `models/common/modules/tt_ccl.py`
  - `models/demos/gpt_oss/tests/test_factory.py`
- Hardware probe:
  - command: `tt-smi -s`
  - result: eight Wormhole B0 ASICs, T3K cluster type, four N300 boards.
  - TTNN probe reported `ttnn.cluster.get_cluster_type() == ClusterType.T3K`
    and `ttnn.FabricConfig.FABRIC_1D_RING` is available.
- Chosen final strategy before coding: fixed `1x8` Ring TP, optimized
  single-chip policy, local full-hidden RMSNorm, column-parallel QKV/gate/up,
  fused attention WO path, row-parallel MLP down, local KV-head paged cache,
  replicated full-hidden decoder input/output boundary.
- Rejected hidden-sharded decoder boundary because it still needs two major
  hidden all-gathers and adds distributed RMSNorm stats collectives.
- Rejected MLP padding to 16384 intermediate because native `14336 / 8 = 1792`
  is tile aligned and padding would add 14.3% extra MLP work.
- MoE/expert strategy is not applicable; Llama 3.1 8B is dense.

### Implementation

- Added `models/autoports/meta_llama_llama_3_1_8b_instruct/tt/multichip_decoder.py`.
- The decoder uses `OptimizedDecoder` as the recorded single-chip baseline.
- The dtype/fidelity policy inherits the optimized decoder defaults:
  - BF16 activations.
  - BFP8 attention weights.
  - BFP4 MLP gate/up/down weights.
  - BFP8 paged KV cache.
  - LoFi MLP math fidelity.
- Added `_TensorParallelMLP` with column-parallel gate/up, row-parallel down,
  `reduce_scatter_minimal_async`, and hidden all-gather at the decoder boundary.
- Built attention through `models.common.modules.attention.attention_1d.Attention1D`
  with grouped local Q/K/V chunks, local one-KV-head paged cache, Ring topology,
  and fused all-gather matmul.
- Added explicit `allowed_worker_cores` normalization for the local multichip
  1D/2D matmul configs to avoid the TTNN future-hard-error warning.
- Added `models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py`.
  Coverage includes policy/contract, full-context cache shape, paged prefill,
  paged decode, current positions, non-identity page table, local cache layout,
  replicated output layout, no-host-fallback audit, warmed trace capture,
  replay determinism, and optimized-baseline PCC.

### Correctness And Latency

Compile command:

```bash
python_env/bin/python -m py_compile \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/multichip_decoder.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py
```

Result: passed.

Focused validation command:

```bash
MULTICHIP_DECODER_DECODE_REPLAYS=4 python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py::test_multichip_decoder_synthetic_paged_prefill_decode_trace_against_optimized \
  -vv -s
```

Focused result: `1 passed in 15.60s`.

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

Final non-profiler metrics:

- `prefill_pcc_vs_optimized`: `0.9999957546448677`
- `decode_pcc_vs_optimized`: `0.9999870318423426`
- `determinism_pcc`: `1.0`
- `eager_trace_pcc`: `1.0`
- `runtime_fallback_audit`: `multichip_prefill_decode_clean`
- single-chip warmed prefill: `3.402695060 ms`
- multichip warmed prefill: `2.707116306 ms`
- single-chip warmed decode min/avg: `0.790356193 ms / 0.793234794 ms`
- multichip warmed decode min/avg: `0.441294163 ms / 0.444971491 ms`
- decode speedup: `1.790996252x`
- TP efficiency: `0.223874531`

The synthetic validation used `seq_len=128`, `max_seq_len=256`,
`page_block_size=64`, and `max_num_blocks=4`. The full-cache contract test
uses `FULL_CACHE_SEQ_LEN=128 * 1024`, `2048` pages, and one local KV head per
chip.

### Tracy And tt-perf-report

Profiler command:

```bash
mkdir -p models/autoports/meta_llama_llama_3_1_8b_instruct/doc/multichip_decoder/tracy/synthetic/.logs
MULTICHIP_DECODER_DECODE_REPLAYS=4 python_env/bin/python -m tracy -r -p -v \
  -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/multichip_decoder/tracy/synthetic/.logs \
  -m pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k synthetic_paged_prefill_decode_trace -q -s \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/multichip_decoder/tracy/synthetic/tracy_run.log
```

Result: `1 passed, 2 deselected in 20.82s`. Raw report:

- `tracy/synthetic/.logs/reports/2026_06_15_14_33_14/ops_perf_results_2026_06_15_14_33_14.csv`
- `tracy/synthetic/.logs/reports/2026_06_15_14_33_14/profile_log_device.csv`

Profiler metrics:

- `prefill_pcc_vs_optimized`: `0.9999957546448677`
- `decode_pcc_vs_optimized`: `0.9999870318423426`
- `determinism_pcc`: `1.0`
- `eager_trace_pcc`: `1.0`
- single-chip profiled decode min/avg: `0.812513288 ms / 0.817927881 ms`
- multichip profiled decode min/avg: `0.499652233 ms / 0.503186835 ms`
- profiler decode speedup: `1.626157624x`
- profiler TP efficiency: `0.203269703`

Report generation commands:

```bash
PROFILE_DIR=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/multichip_decoder/tracy/synthetic
RAW=$PROFILE_DIR/.logs/reports/2026_06_15_14_33_14/ops_perf_results_2026_06_15_14_33_14.csv
DEV=$PROFILE_DIR/.logs/reports/2026_06_15_14_33_14/profile_log_device.csv
cp "$RAW" "$PROFILE_DIR/multichip_ops_perf_results.csv"
cp "$DEV" "$PROFILE_DIR/multichip_profile_log_device.csv"
cp "$PROFILE_DIR/.logs/.logs/tracy_ops_data.csv" "$PROFILE_DIR/tracy_ops_data.csv"
cp "$PROFILE_DIR/.logs/.logs/tracy_ops_times.csv" "$PROFILE_DIR/tracy_ops_times.csv"
tt-perf-report "$RAW" --start-signpost PERF_BASELINE_PREFILL --end-signpost PERF_BASELINE_PREFILL_END --csv "$PROFILE_DIR/baseline_prefill_perf_report.csv" > "$PROFILE_DIR/baseline_prefill_perf_report.console.log"
tt-perf-report "$RAW" --start-signpost PERF_BASELINE_PREFILL --end-signpost PERF_BASELINE_PREFILL_END --no-summary > "$PROFILE_DIR/baseline_prefill_perf_report.txt"
tt-perf-report "$RAW" --start-signpost PERF_BASELINE_DECODE --end-signpost PERF_BASELINE_DECODE_END --csv "$PROFILE_DIR/baseline_decode_perf_report.csv" > "$PROFILE_DIR/baseline_decode_perf_report.console.log"
tt-perf-report "$RAW" --start-signpost PERF_BASELINE_DECODE --end-signpost PERF_BASELINE_DECODE_END --no-summary > "$PROFILE_DIR/baseline_decode_perf_report.txt"
tt-perf-report "$RAW" --start-signpost PERF_MULTICHIP_PREFILL --end-signpost PERF_MULTICHIP_PREFILL_END --csv "$PROFILE_DIR/multichip_prefill_perf_report.csv" > "$PROFILE_DIR/multichip_prefill_perf_report.console.log"
tt-perf-report "$RAW" --start-signpost PERF_MULTICHIP_PREFILL --end-signpost PERF_MULTICHIP_PREFILL_END --no-summary > "$PROFILE_DIR/multichip_prefill_perf_report.txt"
tt-perf-report "$RAW" --start-signpost PERF_MULTICHIP_DECODE --end-signpost PERF_MULTICHIP_DECODE_END --csv "$PROFILE_DIR/multichip_decode_perf_report.csv" > "$PROFILE_DIR/multichip_decode_perf_report.console.log"
tt-perf-report "$RAW" --start-signpost PERF_MULTICHIP_DECODE --end-signpost PERF_MULTICHIP_DECODE_END --no-summary > "$PROFILE_DIR/multichip_decode_perf_report.txt"
tt-perf-report "$RAW" --start-signpost PERF_MULTICHIP_DECODE --end-signpost PERF_MULTICHIP_DECODE_END --no-merge-devices --csv "$PROFILE_DIR/multichip_decode_perf_report_per_device.csv" > "$PROFILE_DIR/multichip_decode_perf_report_per_device.console.log"
tt-perf-report "$RAW" --start-signpost PERF_MULTICHIP_DECODE --end-signpost PERF_MULTICHIP_DECODE_END --no-merge-devices --no-summary > "$PROFILE_DIR/multichip_decode_perf_report_per_device.txt"
```

`tt-perf-report` summary:

- baseline prefill: `23` rows, `2451.073 us` merged device time, `0` host rows.
- baseline decode: `19` rows, `749.453 us` merged device time, `0` host rows.
- multichip prefill: `27` rows, `1184.215 us` merged device time, `0` host rows.
- multichip decode: `26` rows, `418.317 us` merged device time, `0` host rows.
- multichip decode per-device: `208` rows, `3261.715 us` summed across eight
  devices, `0` host rows.

Multichip decode hot spots:

- all-gathers: `77.510 us`
- reduce-scatter: `69.983 us`
- gate/up matmuls: `58.606 us`
- fused all-gather matmul: `44.108 us`
- QKV matmul: `36.599 us`
- down matmul: `31.233 us`
- binary adds/mul: `28.153 us`
- RMSNorms: `23.278 us`

Per-device ranges:

- fused all-gather matmul: `40.208-46.687 us`
- all-gathers: `36.926-40.641 us`
- reduce-scatter: `62.437-75.386 us`
- gate/up matmuls: `27.037-29.484 us`
- QKV matmul: `36.338-36.599 us`
- down matmul: `22.451-31.233 us`

The final profiler log has no `allowed_worker_cores not populated` hits.
`tt-perf-report` category warnings for CCL ops are classification gaps in the
report tool. The large merged op-to-op gap before one layernorm is a report
merge artifact across devices/signposts, not accepted as device compute time.

### Watcher

Final watcher command:

```bash
mkdir -p models/autoports/meta_llama_llama_3_1_8b_instruct/doc/multichip_decoder/watcher/synthetic_ring_final
TT_METAL_WATCHER=10 \
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/multichip_decoder/watcher/synthetic_ring_final \
MULTICHIP_DECODER_DECODE_REPLAYS=1 \
python_env/bin/pytest --timeout=1200 \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k synthetic_paged_prefill_decode_trace -vv -s \
  2>&1 | tee models/autoports/meta_llama_llama_3_1_8b_instruct/doc/multichip_decoder/watcher/synthetic_ring_final/watcher_run.log
```

Result: `1 passed, 2 deselected in 16.00s`. Watcher attached to all eight
devices with disabled features `None`.

Watcher metrics:

- `prefill_pcc_vs_optimized`: `0.9999957546448677`
- `decode_pcc_vs_optimized`: `0.9999870318423426`
- `determinism_pcc`: `1.0`
- `eager_trace_pcc`: `1.0`
- `runtime_fallback_audit`: `multichip_prefill_decode_clean`

Watcher artifact files:

- `watcher/synthetic_ring_final/watcher_run.log`
- `watcher/synthetic_ring_final/generated/watcher/watcher.log`
- `watcher/synthetic_ring_final/generated/watcher/kernel_names.txt`
- `watcher/synthetic_ring_final/generated/watcher/kernel_elf_paths.txt`
- `watcher/synthetic_ring_final/generated/inspector/kernels.yaml`
- `watcher/synthetic_ring_final/generated/inspector/mesh_devices_log.yaml`
- `watcher/synthetic_ring_final/generated/inspector/mesh_workloads_log.yaml`
- `watcher/synthetic_ring_final/generated/inspector/programs_log.yaml`
- `watcher/synthetic_ring_final/generated/inspector/startup.yaml`

Watcher/inspector scans:

```bash
rg -n -i '\b(fatal|assert|exception|timeout|hang|failed|fault|illegal|heartbeat|overflow|sanitize)\b|out[ ._-]?of[ ._-]?bounds|watcher.*\b(warn|fail)\b|allowed_worker_cores not populated' \
  watcher/synthetic_ring_final/generated/watcher/watcher.log \
  watcher/synthetic_ring_final/watcher_run.log || true
```

Only pytest header lines containing the configured timeout were returned.

```bash
rg -n -i '\b(fatal|assert|exception|timeout|hang|failed|fault|illegal|heartbeat|overflow|sanitize)\b|out[ ._-]?of[ ._-]?bounds' \
  watcher/synthetic_ring_final/generated/inspector/startup.yaml \
  watcher/synthetic_ring_final/generated/inspector/mesh_devices_log.yaml \
  watcher/synthetic_ring_final/generated/inspector/mesh_workloads_log.yaml \
  watcher/synthetic_ring_final/generated/inspector/programs_log.yaml || true
```

No output.

### Current Limitations

- Fixed target only: `1x8` T3K Ring.
- Decoder boundary is replicated full hidden. This is the chosen layer-stack
  contract for this stage; full-model integration can reuse it directly, but
  decode efficiency is limited by two hidden all-gathers and one reduce-scatter
  per layer.
- No full-model or vLLM work was started in this goal.
