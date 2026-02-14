# Bark Small Bring-up for Tenstorrent Wormhole

This directory contains the implementation of the **Bark Small** text-to-audio model optimized for Tenstorrent hardware using the TTNN API.

## Bark Model Overview
Bark is a transformer-based text-to-audio model by Suno AI. It consists of three stages:
1.  **Stage 1 (Semantic)**: Text → Semantic tokens.
2.  **Stage 2 (Coarse)**: Semantic tokens → Coarse acoustic codebooks (EnCodec).
3.  **Stage 3 (Fine)**: Coarse codebooks → Fine acoustic codebooks (EnCodec).
4.  **EnCodec Decoder**: Fine codebooks → Audio waveform.

## Requirements
-   Tenstorrent Wormhole card.
-   HuggingFace Transformers library.
-   PyTorch.
-   Numpy, Scipy.

## Installation
Ensure you have the Tenstorrent SDK and environment set up.
```bash
python -m pip install transformers torch numpy scipy loguru
```

## Running the Demo
A demo script is provided to generate audio from sample text.
```bash
# Run the demo script
python models/demos/wormhole/bark/demo/demo.py
```

## Running Tests
Comprehensive unit tests and accuracy validations are provided in a single test suite.
```bash
# Run all Bark unit tests
pytest models/demos/wormhole/bark/tests/test_bark_model.py -svv
```

The tests cover:
- Individual stage forward passes.
- End-to-end audio generation.
- Multilingual support.
- Emotion annotations ([laughs], [sighs]).
- PCC validation against PyTorch reference.
- Performance benchmarks.

## Implementation Details
-   **Hardware Acceleration**: MLP and LayerNorm layers are executed on the TT device.
-   **Hybrid Attention**: Attention computation uses PyTorch SDPA for correctness during bring-up, with TT device handling projections.
-   **Weight Loading**: Uses `from_pretrained` to load weights directly from HuggingFace `suno/bark-small`.
-   **Configuration**: Automatically detects model dimensions from state dict keys to ensure exact matching.

## Bounty Info
Fixes #32069
Bounty: Bark Small Bring-up using TTNN APIs.
Target: Stage 1 (Bring-up) implementation and validation.
Throughput Target: ≥ 20 tokens/sec for Stage 1.
RTF Target: < 0.8.
