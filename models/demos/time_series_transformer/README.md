# Time Series Transformer (TTNN)

## Platforms
- Wormhole (`n150`, `n300`)

## Overview
This directory contains reference tensor generation and scaffolding for the
upcoming TTNN port of `TimeSeriesTransformerForPrediction` for monthly tourism
forecasting.

- HF checkpoint: `huggingface/time-series-transformer-tourism-monthly`
- Dataset: Tourism Monthly (`hf-internal-testing/tourism-monthly-batch`)
- Distribution output: Student-T
- Validation plan: PCC per-layer + 5% tolerance end-to-end (NLL/CRPS/MAE)

## Directory Layout
\`\`\`text
time_series_transformer/
├── README.md
├── requirements.txt
├── scripts/
│   └── save_reference_tensors.py
└── reference/
    └── config.json
\`\`\`

## Setup
\`\`\`bash
source python_env/bin/activate
pip install -r models/demos/time_series_transformer/requirements.txt
\`\`\`

## Generate Reference Tensors
Reference \`.pt\` tensors are generated locally and not committed to the repo.
Run from the repo root:
\`\`\`bash
python models/demos/time_series_transformer/scripts/save_reference_tensors.py
\`\`\`

Generated files appear in \`reference/\` and include all encoder/decoder
intermediate outputs for PCC validation against the TTNN port.

Model pinned to revision \`2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35\`.
Dataset pinned to revision \`81c7ee3cf3317e51beb97327df55926cd5bbfadb\`.
See \`reference/run_metadata.json\` for exact runtime package versions used.