# Chronos-2 Forecast

[Chronos](https://github.com/amazon-science/chronos-forecasting) pretrained time-series models. This tree vendors **Chronos-2** inference sources as a golden PyTorch reference for a later TTNN port.

## Setup

```bash
git submodule update --init models/experimental/chronos_forecast/third_party/chronos-forecasting
pip install -r models/experimental/chronos_forecast/requirements.txt
```

Pinned Chronos-2 copy: commit `10afa9ebe016e514f9d7dc1aa873f66af57e116b`. See [reference/PROVENANCE.md](reference/PROVENANCE.md).



## Weights

Real Chronos-2 weights are **not** committed, only testing for model shape right now will change when real infra

## Demo

Single `Chronos2Model.forward` (CPU):

```bash
PYTHONPATH=. python models/experimental/chronos_forecast/demo/demo.py
```

Chronos-1 tokenizer only:

```bash
PYTHONPATH=. python models/experimental/chronos_forecast/demo/demo.py --tokenizer-only --context-length 16
```

## Single-chip TTNN trace

The fixed paper benchmark (`batch=1024`, context `2048`, forecast `64`) has a
device-resident path and an address-stable TTNN trace runner. It keeps the
embeddings, 12-layer encoder, and output head on device, specializes group
attention when every series has a unique group ID, and refreshes fixed input
slots between replays.

Accuracy and lifecycle:

```bash
source python_env/bin/activate
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/pcc/test_modules.py \
  models/experimental/chronos_forecast/tests/pcc/test_trace.py
```

Paper-shape performance (20 replay and 20 end-to-end iterations):

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/perf/test_paper_forward_trace.py \
  -s
```


## tests/meanings


From `/home/andy/tt-metal`:

```bash
source python_env/bin/activate
```

Core Python/unit tests:

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/test_*.py \
  -s
```

TTNN module PCC tests:

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/pcc/test_modules.py \
  -s
```

TTNN trace PCC tests:

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/pcc/test_trace.py \
  -s
```

Eager paper-shape performance test:

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/perf/test_paper_forward.py \
  -s \
  --timeout=3600
```

Trace-replay performance test—the approximately 2.7 s benchmark(0.26 now):

```bash
PYTHONPATH=. pytest \
  models/experimental/chronos_forecast/tests/perf/test_paper_forward_trace.py \
  -s \
  --timeout=3600
```

```
pytest -s "models/experimental/chronos_forecast/tests/perf/test_paper_forward_trace.py" -k "performance_l1 and not groups"
```

The FEV benchmark is separate and is not part of these pytest commands:

```bash
PYTHONPATH=. python \
  models/experimental/chronos_forecast/benchmarks/fev_bench/models/evaluate.py \
  -m chronos-2
```

`TtChronosTraceRunner` is TT Metal trace capture/replay, not a resident
persistent compute kernel. A true persistent kernel would require porting the
full transformer into unified device kernels under slow dispatch.

## References
- Paper: [Chronos](https://arxiv.org/abs/2403.07815)
- Chronos-2: [arXiv:2510.15821](https://arxiv.org/abs/2510.15821)
- Upstream: [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
