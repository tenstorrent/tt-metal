# Time Series Transformer (TTNN)

## Platforms
- Wormhole (`n150`, `n300`)

## Overview
This directory contains the `ttnn` implementation and validation suite for the `TimeSeriesTransformerForPrediction` model used in monthly tourism forecasting.

- **Architecture**: Vanilla Encoder-Decoder Transformer (Student-T probabilistic distribution head).
- **HF checkpoint**: `huggingface/time-series-transformer-tourism-monthly`
- **Dataset**: Tourism Monthly (`hf-internal-testing/tourism-monthly-batch`)
- **Validation**: Per-layer PCC via forward hooks + 5% tolerance end-to-end metrics (NLL/CRPS/MAE).

## Directory Layout

```text
time_series_transformer/
├── README.md
├── requirements.txt
├── scripts/
│   └── save_reference_tensors.py # Generates PCC validation artifacts
├── reference/
│   ├── config.json               # Static model provenance
│   └── config_runtime.json       # Dynamic environment versions
├── tests/
│   └── test_tst_pcc.py           # TTNN implementation PCC validation
└── tt/
    ├── tst_model.py              # Main model entry point
    ├── tst_encoder_layer.py      # Encoder block implementation
    ├── tst_decoder_layer.py      # Decoder block implementation
    ├── tst_attention.py          # Attention mechanisms
    └── ttnn_utils.py             # TTNN-specific helper functions

```

## Setup

Run commands from the repository root:

```bash
source .tenstorrent-venv/bin/activate
pip install -r models/demos/time_series_transformer/requirements.txt

```

## Validation & Usage

### 1. Generate Reference Tensors

Generates frozen HuggingFace tensors for PCC validation. Files are saved to `reference/` (ignored by git).

```bash
python models/demos/time_series_transformer/scripts/save_reference_tensors.py

```

### 2. Run PCC Validation

Execute the functional parity tests on Wormhole hardware:

```bash
ARCH_NAME=wormhole_b0 pytest models/demos/time_series_transformer/tests/test_tst_pcc.py -v -s

```

## Provenance

* **Model**: Pinned to revision `2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35`.
* **Dataset**: Pinned to revision `81c7ee3cf3317e51beb97327df55926cd5bbfadb`.
* See `reference/config.json` for full architectural parameters.
* Runtime environments are logged in `reference/config_runtime.json`.

```

```
