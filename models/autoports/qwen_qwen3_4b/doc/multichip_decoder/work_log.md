# Multichip Decoder Work Log

## Implementation

- Added `models/autoports/qwen_qwen3_4b/tt/multichip_decoder.py`.
- Exported `MultichipDecoder` from `models/autoports/qwen_qwen3_4b/tt/__init__.py`.
- Added `models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py`.
- Updated `models/autoports/qwen_qwen3_4b/doc/context_contract.json` for the
  multichip decoder stage and full HF context KV-cache capacity.

The implementation uses `OptimizedDecoder` as `baseline_cls`. It targets only a
1x4 local Blackhole mesh and intentionally does not support smaller meshes.

## Commands

Hardware discovery and smoke:

```bash
tt-smi -ls --local
tt-smi -r
python - <<'PY'
import ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
ttnn.close_mesh_device(mesh)
print("mesh smoke ok")
PY
```

Correctness:

```bash
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py --tb=short
```

Result: `8 passed, 3 skipped in 74.56s`.

Perf:

```bash
QWEN3_4B_MULTICHIP_RUN_PERF=1 \
QWEN3_4B_MULTICHIP_PERF_OUT=models/autoports/qwen_qwen3_4b/doc/multichip_decoder/perf_host_timings.csv \
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_perf_signposts --tb=short
```

Tracy:

```bash
QWEN3_4B_MULTICHIP_RUN_PERF=1 \
QWEN3_4B_MULTICHIP_PERF_OUT=models/autoports/qwen_qwen3_4b/doc/multichip_decoder/perf_host_timings_tracy.csv \
python -m tracy -r -p -v --sync-host-device --dump-device-data-mid-run --check-exit-code \
  -o models/autoports/qwen_qwen3_4b/doc/multichip_decoder/tracy/perf_capture_tp4 \
  -n qwen3_4b_multichip_decoder_tp4 \
  -m pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_perf_signposts --tb=short
```

`tt-perf-report`:

```bash
~/.local/bin/tt-perf-report models/autoports/qwen_qwen3_4b/doc/multichip_decoder/tracy/multichip_ops_final.csv \
  --start-signpost PERF_MULTICHIP_PREFILL_WARMED \
  --end-signpost PERF_MULTICHIP_PREFILL_WARMED_END \
  --csv models/autoports/qwen_qwen3_4b/doc/multichip_decoder/tt_perf_report_prefill.csv \
  > models/autoports/qwen_qwen3_4b/doc/multichip_decoder/tt_perf_report_prefill.txt

~/.local/bin/tt-perf-report models/autoports/qwen_qwen3_4b/doc/multichip_decoder/tracy/multichip_ops_final.csv \
  --start-signpost PERF_MULTICHIP_TRACE_DECODE \
  --end-signpost PERF_MULTICHIP_TRACE_DECODE_END \
  --csv models/autoports/qwen_qwen3_4b/doc/multichip_decoder/tt_perf_report_traced_decode.csv \
  > models/autoports/qwen_qwen3_4b/doc/multichip_decoder/tt_perf_report_traced_decode.txt
```

The raw Tracy ops CSV was an intermediate profiler export used to generate the
committed `tt_perf_report_*.csv` tables. It was excluded from the final commit
because it exceeds the repository hook's 500 KB artifact limit.

Watcher stress:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
QWEN3_4B_MULTICHIP_RUN_WATCHER_STRESS=1 \
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_watcher_single_mesh_stress --tb=short
```

Result: `1 passed in 73.51s`; process exited cleanly.

Reduce-scatter residual probe:

```bash
QWEN3_4B_MULTICHIP_RUN_RS_PROBE=1 \
pytest -q -s models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py::test_multichip_reduce_scatter_residual_contract_probe --tb=short
```

Result: `1 passed in 10.03s`. Log:
`models/autoports/qwen_qwen3_4b/doc/multichip_decoder/reduce_scatter_residual_probe.log`.

Context contract:

```bash
.agents/scripts/check_context_contract.py --model-dir models/autoports/qwen_qwen3_4b
```

## Artifacts

- `perf_host_timings.csv`
- `perf_host_timings_tracy.csv`
- `tt_perf_report_prefill.*`
- `tt_perf_report_traced_decode.*`
- `watcher/watcher_failed_active_eth_program_size.log`
- `watcher/watcher_failed_repeated_fabric_reinit.log`
- `watcher/watcher_failed_eth_noc_sanitize_after_stress.log`
- `watcher/watcher_failed_noc_sanitize_disabled_eth_teardown.log`
- `watcher/watcher_pass_eth_disabled_single_mesh_stress.log`
- `reduce_scatter_residual_probe.log`

## Results

PCC:

| Case | PCC |
| --- | ---: |
| Prefill seq 16 | `0.9997333805764328` |
| Prefill seq 17 | `0.9997359676733848` |
| Prefill seq 64 | `0.999684148069019` |
| Paged decode prefix 16 | `0.9980224018654454` |
| Paged decode prefix 17 | `0.9978650895682959` |
| Trace replay | `1.0` |

Latency:

| Mode | Single-chip ms | Multichip ms | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: |
| Prefill seq 16, uninstrumented | `1.743231` | `2.514297` | `0.693327` | `0.173332` |
| Decode traced, uninstrumented | `0.504047` | `0.440318` | `1.144734` | `0.286184` |
| Prefill seq 16, Tracy | `1.818641` | `2.913875` | `0.624131` | `0.156033` |
| Decode traced, Tracy | `0.537467` | `0.477777` | `1.124933` | `0.281233` |

`tt-perf-report`:

| Mode | DRAM roofline | Matmul | CCL |
| --- | ---: | ---: | ---: |
| Prefill | `6.7%`, `34 GB/s` | `213.678 us` | `77.639 us` |
| Traced decode | `6.8%`, `35 GB/s` | `211.261 us` | `79.161 us` |

## Stage Review Follow-up

First stage review returned `more-work-needed` for two documentation/evidence
gaps:

- The context contract needed per-device decoder weight, activation, and trace
  estimates in addition to KV-cache capacity. Added those fields to
  `doc/context_contract.json`.
- The replicated residual topology needed stronger evidence against
  reduce-scatter-only alternatives. Added the topology/adapted-evidence table to
  `README.md`, including measured CCL time.
- Second review asked for a shape-faithful adapted probe. Added
  `test_multichip_reduce_scatter_residual_contract_probe` and recorded its log.
  The probe shows reduce-scatter produces local width `640`; full residual add
  fails, full-width RMSNorm gamma fails, sharded RMSNorm can produce local width
  `640`, and the current gate/up matmul then fails because its K dimension is
  `2560`. That is the exact op-contract blocker for a local decoder-layer change.

The selected path remains replicated at layer input/output. The profiler already
shows the all-reduce implementation cost through its ReduceScatter and AllGather
components, so full-model work can decide whether a broader distributed-RMSNorm
and sharded-stack redesign is worth the extra integration risk.

## Limitations

- The target is fixed to a local 1x4 Blackhole ring. Smaller meshes are not
  implemented for this stage.
- Layer input/output remain replicated. A future sharded residual contract would
  need distributed RMSNorm and a stacked-layer contract change.
- Full active-Ethernet watcher coverage is blocked by fabric-router watcher
  issues outside this decoder. See `AUTOFIX.md`.
