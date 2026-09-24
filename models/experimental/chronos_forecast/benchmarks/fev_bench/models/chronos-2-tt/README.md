# Chronos-2 TTNN fev adapter

Run from the `tt-metal` repository root with the repository Python environment:

```bash
source python_env/bin/activate
TT_VISIBLE_DEVICES=0 \
HF_DATASETS_CACHE=$PWD/generated/fev_hf_cache \
PYTHONPATH=models/experimental/chronos_forecast/third_party/fev/src:\
models/experimental/chronos_forecast/third_party/chronos-forecasting/src:. \
python models/experimental/chronos_forecast/benchmarks/fev_bench/models/evaluate.py \
  -m chronos-2-tt \
  -b models/experimental/chronos_forecast/benchmarks/fev_bench/tasks_smoke.yaml \
  -n chronos-2-tt-smoke \
  -k '{"batch_size":16,"model_path":"models/experimental/chronos_forecast/weights/chronos-2"}'
```

The adapter uses one Blackhole chip and the device-resident TTNN forward. It
supports Chronos-2 trained quantile levels and all-valid contexts. The current
v1 model does not support masked/padded context patches, so unsupported fev
tasks fail instead of silently changing the benchmark.

The first run includes kernel compilation. `inference_time_s` from a subsequent
run with a warm kernel cache is the steady-state number.
