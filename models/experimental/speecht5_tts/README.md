# SpeechT5 TTS Implementation in TTNN

This directory contains the implementation of Microsoft's SpeechT5 TTS model in TTNN for Tenstorrent hardware.

## Model Overview

SpeechT5 is an encoder-decoder transformer model for text-to-speech synthesis. It converts text input into mel-spectrograms which can be converted to audio using a vocoder.

**Model**: `microsoft/speecht5_tts`
**Paper**: [SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://arxiv.org/abs/2110.07205)

## Architecture

The model consists of:
1. **Text Encoder** (12 layers): Processes text input into contextualized representations
2. **Speech Decoder** (6 layers): Generates mel-spectrogram features autoregressively
3. **Decoder Pre-Net**: Processes previous mel-spectrogram frames
4. **Decoder Post-Net**: Refines mel-spectrogram output using convolutional layers

### Key Parameters
- Hidden size: 768
- Encoder layers: 12
- Decoder layers: 6
- Attention heads: 12
- FFN dimension: 3072
- Mel bins: 80
- Reduction factor: 2

## Directory Structure

```
speecht5_tts/
├── reference/          # PyTorch reference implementation
│   ├── speecht5_config.py
│   ├── speecht5_attention.py
│   ├── speecht5_feedforward.py
│   ├── speecht5_encoder.py
│   └── speecht5_model.py (HuggingFace wrapper)
│
├── tt/                 # TTNN implementation
│   ├── ttnn_speecht5_encoder.py
│   ├── ttnn_speecht5_decoder.py
│   └── ttnn_speecht5_model.py
│
├── tests/              # Test files
│   ├── test_encoder_reference.py
│   ├── test_ttnn_encoder.py
│   └── test_ttnn_model.py
│
├── demo/               # Demo scripts
│   └── demo_tts.py
│
├── ARCHITECTURE.md     # Detailed architecture documentation
└── README.md          # This file
```

## Installation

```bash
# Set environment variables
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

# Activate Python environment
source python_env/bin/activate

# Install dependencies
pip install transformers torch
```

## Usage

### PyTorch Reference Model

```python
from models.experimental.speecht5_tts.reference.speecht5_model import load_reference_model

# Load model
model = load_reference_model("microsoft/speecht5_tts")

# Encode text
encoder_output = model.forward_encoder(input_ids)

# Generate speech (requires speaker embeddings)
mel_spectrogram = model.generate_speech(input_ids, speaker_embeddings)
```

### TTNN Model (In Progress)

```python
import ttnn
from models.experimental.speecht5_tts.tt.ttnn_speecht5_model import TtSpeechT5Model

# Create device
device = ttnn.open_device(device_id=0)

# Load model
model = TtSpeechT5Model.from_pretrained("microsoft/speecht5_tts", device=device)

# Encode text
encoder_output = model.forward_encoder(input_ids, device)
```

## Testing

Run encoder tests:
```bash
python models/experimental/speecht5_tts/tests/test_encoder_reference.py
```

Run ttnn encoder tests (with PCC validation):
```bash
pytest models/experimental/speecht5_tts/tests/test_ttnn_encoder.py
```

## Implementation Status

- [x] Architecture analysis and documentation
- [x] Identify reusable components from T5
- [x] PyTorch reference model (HuggingFace wrapper)
- [ ] TTNN encoder implementation
- [ ] TTNN decoder implementation
- [ ] TTNN post-net implementation
- [ ] End-to-end TTNN model
- [ ] PCC validation (target: >0.94)
- [ ] Performance benchmarks
- [ ] Demo script

## Performance Targets

- **PCC (Pearson Correlation Coefficient)**: > 0.94 vs PyTorch reference
- **Inference time**: TBD (depends on hardware and batch size)
- **Memory usage**: TBD

## References

- [HuggingFace SpeechT5](https://huggingface.co/microsoft/speecht5_tts)
- [T5 Encoder Implementation](../stable_diffusion_35_large/reference/t5_encoder.py)
- [TTNN Documentation](../../../ttnn/README.md)

## Notes

- The reference implementation uses HuggingFace's SpeechT5ForTextToSpeech as ground truth
- Speaker embeddings are required for speech generation (not included in base model)
- Vocoder (mel-to-wav) is separate and not included in this implementation
