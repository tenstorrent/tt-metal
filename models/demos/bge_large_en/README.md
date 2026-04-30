# BGE-Large-EN-v1.5

## Model Information
This directory contains the core implementation for BGE-large-en-v1.5 (BAAI General Embedding).

## Structure
- `common.py` - Model loading utilities for BGE
- `reference/` - PyTorch reference implementation (reuses sentence_bert)
- `ttnn/` - TTNN model implementation (reuses sentence_bert)
- `runner/` - Performance runner infrastructure (reuses sentence_bert)

## Device-Specific Implementations
- Wormhole: See `models/demos/wormhole/bge_large_en/`

## Model Specifications
- Model: BAAI/bge-large-en-v1.5
- Hidden Size: 1024
- Layers: 24
- Attention Heads: 16
- Intermediate Size: 4096
- Max Sequence Length: 512
