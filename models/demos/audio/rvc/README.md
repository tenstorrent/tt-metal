# RVC (Retrieval-based Voice Conversion)

## Platforms:
    Wormhole (n150, n300, T3K, Galaxy), Blackhole (p100, p150)

## Introduction

RVC (Retrieval-based Voice Conversion) is an easy-to-use voice conversion framework based on VITS architecture. It enables high-quality voice conversion by combining VITS architecture with retrieval-based feature matching.

This implementation brings RVC to Tenstorrent hardware using TTNN APIs, enabling high-throughput, low-latency voice conversion.

### Key Features
- **VITS-based posterior encoder** for extracting speaker features
- **RMVPE pitch extraction** for accurate F0 estimation
- **Index-based feature retrieval** for accent/speaker style transfer
- **Flow-based decoder** using normalizing flows
- **HiFi-GAN vocoder** for high-quality waveform generation

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Tenstorrent hardware (Wormhole N150/N300 or Blackhole P100/P150)

## How to Run

### Full Demo

```sh
pytest --disable-warnings models/demos/audio/rvc/demo/demo.py::test_rvc_demo
```

### Component Validation

```sh
pytest --disable-warnings models/demos/audio/rvc/demo/demo.py::test_rvc_component_validation
```

### Pitch Transposition Test

```sh
pytest --disable-warnings models/demos/audio/rvc/demo/demo.py::test_rvc_pitch_transposition
```

### Index Rate Sweep Test

```sh
pytest --disable-warnings models/demos/audio/rvc/demo/demo.py::test_rvc_index_rate_sweep
```

### Unit Tests

```sh
pytest --disable-warnings models/demos/audio/rvc/tests/test_rvc_modules.py
```

## Architecture

```
Source Audio
     │
     ▼
┌─────────────────────┐
│   Mel Spectrogram    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Posterior Encoder   │  ← VITS encoder (conv + transformer layers)
│  (z, mean, logs)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Pitch Extraction    │  ← RMVPE / CREPE
│  (F0 estimation)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Feature Retrieval    │  ← Index-based cosine similarity search
│ (accent blending)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Flow-based Decoder  │  ← Affine coupling normalizing flows
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  HiFi-GAN Vocoder    │  ← Transposed conv + ResBlock MRF
│  (waveform output)   │
└─────────┬───────────┘
          │
          ▼
   Converted Audio
```

## Model Components

### Posterior Encoder
- Conv1d pre-processing
- 6-layer transformer encoder (2-head attention, 768 filter channels)
- Projects to latent mean and log-variance
- Reparameterization trick for sampling

### Pitch Extraction (RMVPE)
- 3-layer CNN backbone (128→256 channels)
- Sigmoid-activated F0 prediction head
- Supports transposition (-12 to +12 semitones)

### Feature Retrieval
- Cosine similarity-based nearest neighbor search
- Adjustable index rate (0.0 to 1.0) for accent strength control
- Blends source and retrieved features

### Flow Decoder
- 4 affine coupling layers
- Invertible normalizing flows
- Forward/reverse pass support

### HiFi-GAN Vocoder
- 5 transposed convolution upsampling layers (10×, 6×, 2×, 2×, 2× = 480×)
- Multi-receptive field fusion (MRF) residual blocks
- Kernel sizes: [3, 7, 11], Dilation sizes: [[1,3,5], [1,3,5], [1,3,5]]

## Performance Targets

| Metric | Target | Stage 3 Target |
|--------|--------|----------------|
| Flow generation speed | >30 tok/s | >60 tok/s |
| Real-time factor (RTF) | <0.5 | <0.2 |
| Speaker similarity | >75% | >85% |
| Content preservation (WER) | <2.5 | <1.5 |
| Token accuracy vs PyTorch | >95% | >99% |

## File Structure

```
models/demos/audio/rvc/
├── tt/
│   ├── ttnn_rvc.py                  # TTNN implementation of RVC components
│   ├── reference_rvc.py             # PyTorch reference model
│   └── rvc_parameter_preprocessing.py  # Parameter conversion utilities
├── demo/
│   └── demo.py                      # Demo and validation tests
├── tests/
│   └── test_rvc_modules.py          # Unit tests
└── README.md                        # This file
```

## References
- [RVC Official Repository](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion)
- [VITS Paper](https://arxiv.org/abs/2106.06103)
- [HiFi-GAN Paper](https://arxiv.org/abs/2010.05646)
- [RMVPE Pitch Extraction](https://github.com/Dream-High/RMVPE)
- [TTNN Model Bring-up Guide](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md)
