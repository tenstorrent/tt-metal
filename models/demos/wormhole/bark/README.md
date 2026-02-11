# Bark Small - TTNN Model Bring-up

## Overview

[Bark](https://github.com/suno-ai/bark) is a transformer-based text-to-audio model created by Suno.
This implementation brings up the **Bark Small** (80M params per stage, 240M total) on Tenstorrent
Wormhole hardware using TTNN APIs.

Bark uses a three-stage architecture:
1. **Text-to-Semantic**: BERT tokenizer → Causal Transformer → 10k semantic vocab
2. **Semantic-to-Coarse**: Semantic tokens → Causal Transformer → 2×1024 EnCodec codebooks
3. **Coarse-to-Fine**: Coarse tokens → Non-causal Transformer → 6×1024 EnCodec codebooks

The final tokens are decoded to a 24 kHz mono waveform via Facebook's EnCodec decoder.

## How to Run

### Prerequisites
- Tenstorrent Wormhole N150 or N300 hardware
- Python 3.10+
- Install dependencies: `pip install transformers scipy encodec`

### Demo
```bash
pytest models/demos/wormhole/bark/tests/test_bark_demo.py -svv
```

### Unit Tests
```bash
# Test individual stages
pytest models/demos/wormhole/bark/tests/test_bark_semantic.py -svv
pytest models/demos/wormhole/bark/tests/test_bark_coarse.py -svv
pytest models/demos/wormhole/bark/tests/test_bark_fine.py -svv

# Test full pipeline
pytest models/demos/wormhole/bark/tests/test_bark_model.py -svv
```

## Architecture

Each of the three stages uses a GPT-style transformer with:
- 4 attention heads
- 512 embedding dimension
- 8 transformer layers
- GELU activation
- Causal attention (stages 1-2), Non-causal (stage 3)

## Bounty

Fixes #32069

## References
- [Bark Small on HuggingFace](https://huggingface.co/suno/bark-small)
- [Bark Official Repository](https://github.com/suno-ai/bark)
- [EnCodec Repository](https://github.com/facebookresearch/encodec)
- [TTNN Model Bring-up Tech Report](https://docs.tenstorrent.com)
