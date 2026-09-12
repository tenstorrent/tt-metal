# Functional decoder evidence

Hardware and numerical validation passed; independent stage review returned clean-pass.
Implementation/API, acceptance status, and source
artifacts are documented in [functional_decoder/README.md](functional_decoder/README.md).

## Warmed performance

Single-device Blackhole mesh, 110 worker cores, real checkpoint weights, batch
1, 128-token prefill and traced one-token decode at context 129. One measured
warmed execution per mode. These are device timings, not end-to-end host
latency or full-model performance.

| Layer kind | Prefill kernel sum | Prefill op gaps | Decode kernel sum | Decode op gaps |
|---|---:|---:|---:|---:|
| Linear attention | 4.623045 ms | 0.350179 ms | 3.009414 ms | 0.363198 ms |
| Full attention | 3.407822 ms | 0.088856 ms | 2.378159 ms | 0.060647 ms |

The source is tt-perf-report 1.3.0's filtered CSV **Device Time** column in
**microseconds**, divided by 1000 above. `Op-to-Op Gap` is also microseconds
and reported separately. Details and paths are in
[performance.json](functional_decoder/performance.json).

Human-readable tables:

- [Linear prefill](functional_decoder/tracy/linear_attention/prefill_perf_report.txt)
- [Linear traced decode](functional_decoder/tracy/linear_attention/decode_perf_report.txt)
- [Full prefill](functional_decoder/tracy/full_attention/prefill_perf_report.txt)
- [Full traced decode](functional_decoder/tracy/full_attention/decode_perf_report.txt)

Each table has adjacent filtered `*_perf_report.csv`, console log, and original
`*_ops.csv`. Original Tracy collection remains under each kind's `reports/`
and `.logs/` directories. Collection was separate from watcher. Both measured
forwards pass the Torch/host-conversion guard and HF PCC; `linear_profile.json`
and `full_profile.json` record the associated correctness.

Collection command, substituting layer 0/3, kind linear_attention/full_attention,
and matching result/log names:

```bash
PYTHONPATH=. timeout -k 10 900 python_env/bin/python -m tracy -r -p -v \
  -o models/autoports/qwen_qwen3_8_27b/doc/functional_decoder/tracy/linear_attention \
  -m models.autoports.qwen_qwen3_8_27b.tests.run_decoder \
  --snapshot /home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --layer 0 --length 128 --profile \
  --output models/autoports/qwen_qwen3_8_27b/doc/functional_decoder/linear_profile.json
```

The measured prefill window is `PERF_PREFILL` to `PERF_PREFILL_END`; decode is
`PERF_DECODE` to `PERF_DECODE_END`. Prefill warmup/capture state is restored
outside the measured window. Decode is warmed, captured, replayed for PCC and
determinism, then restored and replayed nonblocking with one synchronization
inside the timing window.

Report commands (use the corresponding directory and PREFILL/DECODE):

```bash
python_env/bin/tt-perf-report "$KIND_DIR/decode_ops.csv" \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --csv "$KIND_DIR/decode_perf_report.csv" --no-advice \
  > "$KIND_DIR/decode_perf_report.console.log" 2>&1
python_env/bin/tt-perf-report "$KIND_DIR/decode_ops.csv" \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --no-summary --no-advice --no-color \
  > "$KIND_DIR/decode_perf_report.txt" 2>&1
```
