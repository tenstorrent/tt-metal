# Informer Time-Series Forecasting (TTNN)

## Platforms
- Wormhole (`n150`, `n300`)
- Blackhole

## Overview
This demo provides a TTNN implementation of Informer for long-sequence time-series forecasting, with:
- ProbSparse attention
- Distilling encoder blocks
- Generative decoder inference
- HF checkpoint compatibility (`InformerForPrediction`)
- Streaming inference API for long contexts

## Directory Layout
```text
informer/
├── README.md
├── PERF.md
├── conftest.py
├── requirements.txt
├── scripts/
│   └── prepare_assets.py
├── demo/
│   └── demo.py
├── reference/
│   ├── torch_reference.py
│   └── eval_utils.py
├── tests/
│   ├── pcc/
│   │   ├── test_embeddings.py
│   │   ├── test_attention.py
│   │   ├── test_layers.py
│   │   └── test_e2e_model.py
│   └── perf/
│       ├── perf_common.py
│       ├── test_perf.py
│       └── test_accuracy_hf.py
└── tt/
    ├── config.py
    ├── ops.py
    ├── embeddings.py
    ├── attention.py
    ├── transformer.py
    ├── model.py
    ├── hf_runtime.py
    ├── hf_common.py
    └── state_io.py
```

## Setup
```bash
./create_venv.sh
source python_env/bin/activate
./build_metal.sh
pip install -r models/demos/informer/requirements.txt
```

## Assets
Prepare local ETTh1 assets and checkpoint:
```bash
python models/demos/informer/scripts/prepare_assets.py
```
Generated files:
- `models/demos/informer/.assets/ETTh1.csv`
- `models/demos/informer/.assets/etth1_ttnn.pt`

Optional explicit HF cache pre-download:
```bash
huggingface-cli download huggingface/informer-tourism-monthly --repo-type model
huggingface-cli download monash_tsf --repo-type dataset
```

## Run Demo
HF dataset path:
```bash
python models/demos/informer/demo/demo.py \
  --hf-model-id huggingface/informer-tourism-monthly \
  --hf-dataset monash_tsf --hf-dataset-config tourism_monthly \
  --hf-split test --hf-freq M --hf-max-series 128
```

CSV path (ETTh1-style):
```bash
python models/demos/informer/demo/demo.py \
  --hf-model-id huggingface/informer-tourism-monthly \
  --csv /absolute/path/ETTh1.csv --csv-freq H
```

Pytest demo entrypoint:
```bash
pytest models/demos/informer/demo/demo.py::test_demo -v -s
```

## Test Commands
Run full Informer suite:
```bash
pytest models/demos/informer/tests/pcc models/demos/informer/tests/perf -v -s
```

Run by objective:
```bash
# Core throughput/latency and summary metrics
pytest models/demos/informer/tests/perf/test_perf.py::TestInformerPerformance -v -s

# Advanced capability checks (long sequence, streaming, high-dimensional, cache path)
pytest models/demos/informer/tests/perf/test_perf.py::TestInformerAdvancedCapabilities -v -s

# Real-data ETTh1 and HF checkpoint accuracy
pytest models/demos/informer/tests/perf/test_accuracy_hf.py -v -s
```

## Measured Performance (Wormhole n300, 2026-02-26)

| Metric | Value | Source |
|---|---:|---|
| Throughput (`batch=8`) | `1584.9 seq/s` | `test_perf.py::TestInformerPerformance::test_benchmark_summary` |
| Latency (`batch=1`) | `4.01 ms` | `test_perf.py::TestInformerPerformance::test_benchmark_summary` |
| Correlation (TTNN vs torch ref) | `0.9964` | `test_perf.py::TestInformerPerformance::test_benchmark_summary` |
| MSE (TTNN vs torch ref) | `0.000106` | `test_perf.py::TestInformerPerformance::test_benchmark_summary` |
| MAE (TTNN vs torch ref) | `0.008176` | `test_perf.py::TestInformerPerformance::test_benchmark_summary` |

Additional details and benchmark context are in [PERF.md](PERF.md).

## Streaming Inference Example
```python
from models.demos.informer.tt.config import InformerConfig
from models.demos.informer.tt.model import create_informer

cfg = InformerConfig(
    enc_in=7,
    dec_in=7,
    c_out=7,
    seq_len=336,
    label_len=168,
    pred_len=96,
    d_model=64,
    n_heads=2,
    d_ff=256,
    e_layers=2,
    d_layers=1,
    time_feature_dim=4,
    dtype="bfloat16",
)

model = create_informer(cfg, device=device)
out = model.stream_forecast(past_values, past_time, future_time, chunk_size=128)
model.release_trace()
```

## Notes
- `d_model` should be a multiple of `32` for best TTNN tiling behavior.
- ProbSparse currently uses selected-query sparse routing over full keys; there is no dedicated sparse-only kernel path yet.
