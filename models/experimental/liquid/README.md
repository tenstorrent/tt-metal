# LFM2.5-VL-1.6B - LiquidAI Vision-Language Model

LFM2.5-VL-1.6B is a general-purpose vision-language model by Liquid AI for OCR, document comprehension, and visual question answering.

## Architecture

- **LM Backbone**: LFM2.5-1.2B-Base
- **Vision Encoder**: SigLIP2 NaFlex 400M
- **Projector**: Linear projection
- **Context Length**: 32,768 tokens
- **Image Resolution**: 512x512, with tiling for larger images

## Supported Hardware

| Device | Configuration |
|--------|---------------|
| N150/P150 | 1 device |
| N300 | 2 devices (1x2 mesh) |

## Usage

### Prerequisites

```bash
pip install transformers pillow requests
```

### Download Weights

```bash
bash models/experimental/liquid/references/setup_weights.sh
```

### Run Demo

```bash
LIQUID_WEIGHTS=~/liquid_weights/ python models/experimental/liquid/demo/demo.py
```

### Run Tests

```bash
LIQUID_WEIGHTS=~/liquid_weights/ pytest models/experimental/liquid/tests/ -v
```
