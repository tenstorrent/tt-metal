# Optimized Decoder Work Log

## Commands

Syntax and collection:

```bash
python -m py_compile models/autoports/meta_llama_llama_3_2_1b_instruct/tt/optimized_decoder.py models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_decoder.py
pytest --collect-only models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_decoder.py
```

Config and static path checks:

```bash
pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_decoder.py::test_optimized_static_config_uses_optimized_path
```

Correctness:

```bash
pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_decoder.py::test_synthetic_optimized_paged_prefill_decode_trace_and_determinism
pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_decoder.py::test_real_weights_optimized_paged_prefill_and_decode_trace
OD_PREFILL_SEQ_LEN=8192 pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_decoder.py::test_real_weights_optimized_paged_prefill_and_decode_trace
OD_STRESS_ITERS=5 pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_decoder.py::test_synthetic_optimized_paged_prefill_decode_trace_and_determinism models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_decoder.py::test_optimized_repeated_run_stress
```

Runtime fallback and watcher:

```bash
pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_decoder.py::test_runtime_fallback_audit_measured_optimized_prefill_and_traced_decode
TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_APPEND=1 pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_decoder.py::test_runtime_fallback_audit_measured_optimized_prefill_and_traced_decode
```

Final Tracy profile and reports:

```bash
OD_PERF_PREFILL_SEQ_LEN=8192 python -m tracy -r -p -v -o models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_decoder/tracy/run_prefill_decode -m pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_decoder.py::test_perf_artifact_signposted_optimized_prefill_and_decode
tt-perf-report --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --no-color --csv models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_decoder/perf/prefill_8192_report.csv --summary-file models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_decoder/perf/prefill_8192_summary.csv models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_decoder/perf/ops_perf_results_raw.csv
tt-perf-report --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --no-color --summary-file models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_decoder/perf/prefill_8192_human_summary.csv models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_decoder/perf/ops_perf_results_raw.csv
tt-perf-report --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-color --csv models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_decoder/perf/decode_trace_replay_report.csv --summary-file models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_decoder/perf/decode_trace_replay_summary.csv models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_decoder/perf/ops_perf_results_raw.csv
tt-perf-report --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-color --summary-file models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_decoder/perf/decode_trace_replay_human_summary.csv models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_decoder/perf/ops_perf_results_raw.csv
```

## Before And After

| Metric | Before | After |
| --- | ---: | ---: |
| prefill PCC, real 8192 | 0.9999664355 | 0.9995248960 |
| traced decode PCC, real 8192 | 0.9999890750 | 0.9995249460 |
| repeated traced decode PCC | 1.0 | 1.0 |
| warmed prefill device time | 36560 us | 28849.668 us |
| warmed traced decode replay device time | 864 us | 519.718 us |

Final optimized prefill speedup is 1.2673x by device time. Final optimized
traced decode replay speedup is 1.6624x by device time.

## Optimization Checklist

| Requirement | Evidence |
| --- | --- |
| optimized decoder exists and is not a functional fallback | `tt/optimized_decoder.py`, `test_optimized_decoder_contract_and_runtime_fallback_audit` |
| prefill/decode semantics and paged KV behavior preserved | synthetic and real correctness artifacts, non-monotonic synthetic page table |
| determinism | repeated trace replay PCC 1.0 and 5 repeated stress PCCs |
| representative layer-kind coverage | synthetic layer, real-weight layer at 128 and 8192 tokens |
| warmed prefill and traced warmed decode latency | `perf/perf_provenance.json` |
| tt-perf-report text and CSV with advice | `perf/*tt_perf_report.txt`, `perf/*report.csv`, `perf/*summary*.csv` |
| advice tried | attention prefill `in0_block_w`/subblock trial kept |
| canonical precision/fidelity policy | BFP8 attention/KV, BFP4 MLP, BF16 activations, HiFi2 matmuls, HiFi4 prefill SDPA |
| sharded layouts and DRAM-sharded decode matmuls | `optimized_config_summary.json`, final decode report shows five DRAM-sharded matmuls |
| large prefill configs | 8192-token prefill uses chunked 2D prefill matmul configs with `in0_block_w=8` |
| SDPA/composite ops | Attention1D paged SDPA prefill and decode in final reports |
| memory/program/compute configs | `optimized_config_summary.json` and `optimized_decoder.py` |
| runtime data movement audited | `perf/perf_provenance.json` records no host ops and no measured tilize/untilize/Torch bridge ops |
| MoE active experts | not applicable; dense Llama MLP |
| stress coverage | `stress_repeated_runs.json` |
| watcher-clean run | `watcher/watcher_summary.json` |

## Artifacts

- `optimized_config_summary.json`
- `synthetic_correctness.json`
- `real_weight_correctness.json`
- `real_weight_correctness_prefill_8192.json`
- `runtime_fallback_audit.json`
- `stress_repeated_runs.json`
- `perf_trace_contract.json`
- `precision_experiments.json`
- `perf/perf_provenance.json`
- `perf/decode_performance_accounting.json`
- `perf/advice_trials/attention_prefill_default_in0_block1/`
- `watcher/watcher_summary.json`
