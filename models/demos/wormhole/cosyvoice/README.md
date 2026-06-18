# CosyVoice - Multi-lingual Voice Generation Model

[CosyVoice](https://github.com/FunAudioLLM/CosyVoice) is a multi-lingual large voice generation model from Alibaba's FunAudioLLM that provides full-stack TTS capabilities. This directory contains the TTNN-based bring-up of CosyVoice-300M for Tenstorrent hardware.

## Architecture

CosyVoice-300M consists of three main components:

1. **LLM Backbone** - Transformer-based model for semantic token generation from text
2. **Flow-based Decoder** - Conditional flow matching (CFM) with Diffusion Transformer (DiT) for mel-spectrogram generation
3. **HiFi-GAN Vocoder** - High-fidelity waveform generation from mel-spectrograms

### Supported Inference Modes

- **SFT mode**: Generate speech with predefined speakers
- **Zero-shot mode**: Generate speech with reference audio (voice cloning)
- **Cross-lingual mode**: Generate speech in a different language from the reference
- **Instruct mode**: Generate expressive speech with instructions

### Supported Languages

- Chinese, English, Japanese, Cantonese, Korean

## File Structure

```
cosyvoice/
├── README.md                           # This file
├── demo/
│   └── demo.py                         # Demo entry point
├── tt/
│   ├── __init__.py
│   ├── model_config.py                 # TTNN memory/config management
│   ├── pipeline.py                     # End-to-end inference pipeline
│   ├── llm/
│   │   ├── __init__.py
│   │   └── cosyvoice_llm.py            # LLM backbone (transformer)
│   ├── transformer/
│   │   ├── __init__.py
│   │   └── transformer.py              # Core transformer blocks
│   ├── flow/
│   │   ├── __init__.py
│   │   └── flow_matching.py            # CFM + DiT decoder
│   └── vocoder/
│       ├── __init__.py
│       └── hifigan.py                  # HiFi-GAN vocoder
├── tests/
│   └── test_demo.py                    # CI test harness
├── scripts/
│   ├── download_weights.py             # Weight downloader
│   └── infer.py                        # Standalone inference script
```

## Setup

### Prerequisites

- Tenstorrent hardware (N150 or N300)
- tt-metal environment configured
- Python 3.10+

### Download Weights

```bash
# Download CosyVoice-300M pretrained weights
python models/demos/wormhole/cosyvoice/scripts/download_weights.py
```

### Run Demo

```bash
# Single utterance TTS
python models/demos/wormhole/cosyvoice/demo/demo.py \
  --text "Hello, welcome to CosyVoice on Tenstorrent." \
  --mode sft

# Zero-shot voice cloning
python models/demos/wormhole/cosyvoice/demo/demo.py \
  --text "Hello world." \
  --mode zero_shot \
  --ref_audio /path/to/reference.wav

# Cross-lingual generation
python models/demos/wormhole/cosyvoice/demo/demo.py \
  --text "你好，欢迎使用CosyVoice。" \
  --mode cross_lingual \
  --ref_audio /path/to/reference.wav

# Instruct-based expressive speech
python models/demos/wormhole/cosyvoice/demo/demo.py \
  --text "Hello, this is an expressive voice." \
  --mode instruct \
  --instruct "Speak excitedly"
```

## Stage 1 Deliverables (This PR)

- [x] Full model architecture implemented using TTNN APIs
- [x] LLM backbone for semantic token generation
- [x] Flow-based decoder for acoustic modeling
- [x] HiFi-GAN vocoder for waveform generation
- [x] Inference pipeline with all 4 modes
- [x] Weight download and conversion scripts
- [x] Demo and test harness

### Notes

- This is a **Draft PR** for Stage 1 (Bring-Up).
- Hardware access is required for end-to-end validation. Testing performed locally without TT hardware shows no import/runtime errors in the Python graph.
- Stage 2 and 3 optimizations (sharding, flash attention, KV-cache optimization) are planned for follow-up PRs.
- Reference PyTorch implementation: https://github.com/FunAudioLLM/CosyVoice
