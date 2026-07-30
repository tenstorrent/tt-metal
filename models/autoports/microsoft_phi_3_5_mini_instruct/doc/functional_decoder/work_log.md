# Functional Decoder Work Log

Model: `microsoft/Phi-3.5-mini-instruct`

Autoport directory: `models/autoports/microsoft_phi_3_5_mini_instruct`

Date: 2026-06-15 UTC

## Implementation

Created:

- `tt/functional_decoder.py`
- `tests/test_functional_decoder.py`
- `doc/functional_decoder/weight_stats_layer0.json`
- `doc/functional_decoder/tracy/`
- `doc/functional_decoder/watcher/`

The implementation is a single dense Phi-3.5-mini decoder layer. It supports paged prefill and paged decode, LongRoPE short/long tables, real HF weight loading, trace-safe decode with tensor `current_pos`, and a batch-size-1 decode contract.

## Weight Evidence

Real weights were loaded from:

`/home/moconnor/.cache/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/snapshots/2fe192450127e6a83f7441aef6e3ca586c338b77`

Stats artifact:

`models/autoports/microsoft_phi_3_5_mini_instruct/doc/functional_decoder/weight_stats_layer0.json`

The stats file records all six layer-0 tensors used by the TTNN decoder: tensor name, checkpoint key, shape, dtype, mean, and std. Synthetic tests generate deterministic real-shape tensors from these stats.

## Final Commands And Results

Syntax and default suite:

```bash
python -m py_compile models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py models/autoports/microsoft_phi_3_5_mini_instruct/tt/functional_decoder.py
pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py -s
```

Result: `4 passed, 2 skipped in 12.99s`.

Printed PCCs:

- Synthetic stats-derived weights: prefill PCC `0.9999970054274875`, decode PCC `0.9999976050540407`.
- Real layer-0 weights: prefill PCC `0.9999957910376245`, decode PCC `0.9999965913259444`.
- Determinism test HF comparisons in each repeated run: prefill PCC `0.9999970458001098`, decode PCC `0.9999975174967267`; repeated identical TTNN outputs asserted PCC >= 0.9999 and passed.

Full context decode:

```bash
PHI35_RUN_LONG_CONTEXT=1 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py::test_full_context_decode_current_position_and_page_table -s
```

Result: `1 passed in 8.40s`.

Long prefill:

```bash
PHI35_RUN_LONG_PREFILL=1 PHI35_LONG_PREFILL_LEN=32768 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py::test_long_prefill_page_table -s
```

Result: `1 passed in 8.84s`.

The default suite was also run before final fallback-audit tightening and passed with equivalent PCCs. Long prefill was previously exercised at 4096, 8192, and 32768; the final recorded long prefill is 32768.

## Trace Evidence

`test_dense_layer_synthetic_prefill_decode_pcc_and_traced_decode` and `test_dense_layer_real_weights_prefill_decode_pcc` both capture and execute TTNN traces for decode. The decode PCC is measured from the trace replay output.

The harness performs:

1. Eager decode compile/warmup.
2. Trace capture of `decode_forward`.
3. One trace replay warmup.
4. Signposted measured trace replay.
5. PCC comparison using the replay output tensor.

## Performance Collection

Profiler command:

```bash
PHI35_READ_DEVICE_PROFILER=1 PHI35_SKIP_MESH_CLOSE=1 python -m tracy -r -p -v --dump-device-data-mid-run -o models/autoports/microsoft_phi_3_5_mini_instruct/doc/functional_decoder/tracy/raw_real -m pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py::test_dense_layer_real_weights_prefill_decode_pcc -s
```

Result: `1 passed in 6.49s`; real-weight PCCs were prefill `0.9999957910376245`, decode `0.9999965913259444`.

Source profiler CSV:

`tracy/raw_real/reports/2026_06_15_12_45_42/ops_perf_results_2026_06_15_12_45_42.csv`

Report commands:

```bash
tt-perf-report models/autoports/microsoft_phi_3_5_mini_instruct/doc/functional_decoder/tracy/dense/prefill_ops.csv --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --csv models/autoports/microsoft_phi_3_5_mini_instruct/doc/functional_decoder/tracy/dense/prefill_perf_report.csv --no-advice
tt-perf-report models/autoports/microsoft_phi_3_5_mini_instruct/doc/functional_decoder/tracy/dense/prefill_ops.csv --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --no-summary --no-advice
tt-perf-report models/autoports/microsoft_phi_3_5_mini_instruct/doc/functional_decoder/tracy/dense/decode_ops.csv --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --csv models/autoports/microsoft_phi_3_5_mini_instruct/doc/functional_decoder/tracy/dense/decode_perf_report.csv --no-advice --tracing-mode
tt-perf-report models/autoports/microsoft_phi_3_5_mini_instruct/doc/functional_decoder/tracy/dense/decode_ops.csv --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-summary --no-advice --tracing-mode
```

Summary from `tracy/dense/perf_summary.json`:

| Window | Device time | Op-to-op gap | Device + gap | Rows | Host ops |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill | 1807.085 us | 944.933 us | 2752.018 us | 42 | 0 |
| decode | 1752.534 us | 73.842 us | 1826.376 us | 54 | 0 |

The first profiler attempt with `python -m tracy -r -p -v -m pytest ...` passed the test but segfaulted during mesh close and did not produce device data. The final profiler run used `PHI35_READ_DEVICE_PROFILER=1` to flush device profiler data before teardown and `PHI35_SKIP_MESH_CLOSE=1` for that profiler-only path. The process still closed device drivers at interpreter shutdown and produced the required ops CSV and reports.

## Watcher Collection

First attempt:

```bash
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=models/autoports/microsoft_phi_3_5_mini_instruct/doc/functional_decoder/watcher/2026_06_15_1246 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py::test_dense_layer_real_weights_prefill_decode_pcc -s
```

This attempt hit:

`TT_THROW: ... idle_erisc.elf: segment[0] [0x3f10,+0x5a88) overflows region:0 limit of 0x54c0 bytes`

The process stopped making progress and was terminated. A second watcher attempt before reset failed during device open because remote ETH dispatch state was stale. I reset the devices with:

```bash
tt-smi -r all --eth_train_skip
```

Clean watcher run:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_APPEND=1 TT_METAL_LOGS_PATH=models/autoports/microsoft_phi_3_5_mini_instruct/doc/functional_decoder/watcher/2026_06_15_1253 pytest --confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct -q models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_functional_decoder.py::test_dense_layer_real_weights_prefill_decode_pcc -s
```

Result: `1 passed in 31.96s`; real-weight PCCs remained prefill `0.9999957910376245`, decode `0.9999965913259444`.

Clean audit:

```bash
rg -n -i "TT_FATAL|TT_THROW|exception|assert|out.of.bounds|overflow|sanit|stack overflow|noc .*bad|bad noc|l1 .*overflow|watcher.*error" models/autoports/microsoft_phi_3_5_mini_instruct/doc/functional_decoder/watcher/2026_06_15_1253/generated/watcher/watcher.log
```

Result: no matches. The watcher log records normal stack-usage summaries, zero Ethernet retraining events, and detach lines for all eight devices.

## Limitations

- Decode is limited to batch size 1 in this functional decoder.
- Prefill requires sequence length to be a multiple of the paged-cache block size.
- Full 131072-token prefill is not feasible in this unoptimized nonchunked path: the attention score tensor `[1, 32, 131072, 131072]` at BF16 is about 1.0 TiB before other live tensors. The largest final tested prefill length is 32768.
- Tests use `--confcutdir=models/autoports/microsoft_phi_3_5_mini_instruct` because this local checkout's root `conftest.py` imports a missing `models.tt_transformers.demo.trace_region_config`.
