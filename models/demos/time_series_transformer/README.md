# Time Series Transformer (TTNN)

## Platforms
- Wormhole (`n150`, `n300`)

## Overview
This directory contains the `ttnn` implementation and validation suite for the
`TimeSeriesTransformerForPrediction` model used in monthly tourism forecasting.

- **Architecture**: Vanilla Encoder-Decoder Transformer with Student-T probabilistic distribution head
- **HF checkpoint**: `huggingface/time-series-transformer-tourism-monthly`
- **Dataset**: Tourism Monthly (`hf-internal-testing/tourism-monthly-batch`)
- **Validation**: Per-layer PCC + end-to-end NLL/CRPS within 5% of HF reference

## Directory Layout

```text
time_series_transformer/
├── README.md
├── requirements.txt
├── scripts/
│   └── save_reference_tensors.py   # Generates PCC/e2e validation artifacts
├── reference/
│   ├── config.json                 # Static model provenance (committed)
│   └── config_runtime.json         # Dynamic environment versions
├── tests/
│   ├── test_tst_pcc.py             # Per-layer PCC validation (encoder + decoder)
│   ├── test_tst_e2e.py             # End-to-end NLL/CRPS vs HF reference
│   └── test_tst_perf.py            # Latency and throughput benchmarks
└── tt/
    ├── tst_model.py                # Weight loading, run_encoder, run_decoder_step, generate()
    ├── tst_embedding.py            # Value projection, lag features, mean scaler
    ├── tst_encoder_layer.py        # Encoder block (self-attention + FFN)
    ├── tst_decoder_layer.py        # Decoder block (masked self-attn + cross-attn + FFN)
    ├── tst_attention.py            # Attention mechanism
    ├── tst_distribution.py         # Student-T parameter projection (squareplus activation)
    └── ttnn_utils.py               # TTNN helper functions (layer norm, etc.)
```

## Setup

```bash
source .tenstorrent-venv/bin/activate
pip install -r models/demos/time_series_transformer/requirements.txt
```

## Validation & Usage

### 1. Generate Reference Tensors

Downloads the pinned HF model and tourism batch, runs a forward pass, and saves
reference tensors to `reference/` (gitignored except `config.json`).

```bash
python models/demos/time_series_transformer/scripts/save_reference_tensors.py
```

### 2. Run All Tests

```bash
cd models/demos/time_series_transformer && \
PYTHONPATH=/path/to/tt-metal/ttnn:/path/to/tt-metal/tools:/path/to/tt-metal/build_Release/lib:. \
ARCH_NAME=wormhole_b0 pytest tests/ -v -s --noconftest
```

Expected results on Wormhole n300:

| Test | Result |
|------|--------|
| `test_encoder_pcc` | PCC 0.9999968 ✓ |
| `test_decoder_pcc` | PCC 0.9999934 ✓ |
| `test_e2e_generate` | CRPS diff 2.7%, NLL diff 0.6% ✓ |
| `test_single_sequence_latency` | ~1553ms ✓ |
| `test_batch_throughput` | ~5 seq/s ✓ |

## Provenance

- **Model**: pinned to revision `2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35`
- **Dataset**: pinned to revision `81c7ee3cf3317e51beb97327df55926cd5bbfadb`
- See `reference/config.json` for full architectural parameters
- Runtime environment logged in `reference/config_runtime.json`
