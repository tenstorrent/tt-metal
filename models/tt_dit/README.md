# TT-DiT: Tenstorrent Diffusion Transformers

This directory contains the implementation of the Tenstorrent Diffusion Transformer (TT-DiT) architecture, designed for optimized parallelism of DiTs for image and video generation.

## Supported Models

For detailed information about each model including performance metrics, usage instructions, and specific requirements, see:

- **[Stable Diffusion 3.5 Large](models/StableDiffusion35.md)** - Text-to-image generation
- **[Flux 1](models/Flux1.md)** - Text-to-image generation (schnell & dev variants)
- **[Motif](models/Motif.md)** - Text-to-image generation model
- **[Qwen-Image](models/QwenImage.md)** - Text-to-image generation model
- **[Mochi-1](models/Mochi_1.md)** - Video generation model
- **[Wan2.2-T2V-A14B](models/Wan2_2.md)** - Text-to-video generation model

## Directory Structure

```
tt_dit/
├── layers/              # Core neural network layers
├── models/              # Model architectures and documentation
│   ├── transformers/    # Transformer implementations (SD35, Mochi, Wan, Flux1, Motif, QwenImage)
│   ├── vae/            # VAE/Autoencoder implementations
│   ├── StableDiffusion35.md  # SD3.5 model documentation
│   ├── Flux1.md         # Flux 1 model documentation
│   ├── Motif.md         # Motif model documentation
│   ├── QwenImage.md     # Qwen-Image model documentation
│   ├── Mochi_1.md       # Mochi-1 model documentation
│   └── Wan2_2.md        # Wan2.2 model documentation
├── encoders/            # Text encoder implementations
│   ├── clip/           # CLIP encoder
│   └── t5/             # T5 encoder
├── parallel/            # Parallelization utilities
│   ├── config.py        # Parallel configuration
│   └── manager.py       # Parallel execution management
├── pipelines/           # End-to-end model pipelines
│   ├── stable_diffusion_35_large/
│   ├── mochi/
│   ├── wan/
│   ├── flux1/
│   ├── motif/
│   └── qwenimage/
├── tests/              # Test suite
│   ├── models/         # Model-level tests (sd35, mochi, wan2_2, flux1, motif, qwenimage)
│   ├── encoders/       # Encoder tests
│   ├── blocks/         # Block-level tests
│   └── unit/          # Unit tests for layers
└── utils/             # Utility functions
    ├── check.py       # Validation utilities
    ├── padding.py     # Padding operations
    ├── substate.py    # State management
    └── tensor.py      # Tensor operations
```

## Core Components

### Layers
- **Embeddings**: Position and token embedding implementations
- **Feedforward**: MLP and feed-forward network components
- **Linear**: Optimized linear transformation layers
- **Normalization**: Layer normalization implementations

### Models
- **Transformers**: DiT transformer implementations for various generative models
  - Support for multiple architectures including SD3.5, Mochi, Wan2.2, Flux1, Motif, and Qwen-Image
  - Model-specific attention mechanisms and transformer architectures
- **Autoencoders**: VAE implementations for different models

### Parallel Processing
- **Config**: Parallel configuration management with `DiTParallelConfig`
- **Manager**: Parallel execution and device management

### Pipelines
End-to-end pipeline implementations for multiple generative models:
- **Stable Diffusion 3.5 Large**: Text-to-image generation (1024x1024px)
- **Flux 1**: Text-to-image generation (schnell & dev variants, 1024x1024px)
- **Motif**: Text-to-image generation (6B model, 1024x1024px)
- **Qwen-Image**: Text-to-image generation (1024x1024px)
- **Mochi-1**: Video generation model (824x480px, 168 frames)
- **Wan2.2-T2V-A14B**: Text-to-video generation

Each pipeline includes:
- Automatic parallel configuration for different device meshes
- Optimized execution on Wormhole systems
- Comprehensive timing collection and profiling

## Testing

The test suite is organized into three main categories:

1. **Unit Tests** (`tests/unit/`):
   - Layer-level testing
   - Individual component validation
   - Coverage for embeddings, feedforward, linear, and normalization layers

2. **Model Tests** (`tests/models/`):
   - Smoke tests for model components
   - Performance testing
   - Pipeline validation

Running tests:
```bash
# Run unit tests
python -m pytest tests/unit/

# Run all model tests
python -m pytest tests/models/

# Run specific model pipeline tests
python -m pytest tests/models/sd35/test_pipeline_sd35.py -v
python -m pytest tests/models/flux1/test_pipeline_flux1.py -v
python -m pytest tests/models/motif/test_pipeline_motif.py -v
python -m pytest tests/models/qwenimage/test_pipeline_qwenimage.py -v
python -m pytest tests/models/mochi/test_pipeline_mochi.py -v
python -m pytest tests/models/wan2_2/test_pipeline_wan.py -v
```

## Key Features

- **Modular Architecture**: Clean separation between layers, models, and parallel processing
- **Optimized Parallelism**:
  - Automatic configuration of tensor and sequence parallelism
  - Efficient device mesh management
  - Dedicated parallel managers for different model components
- **Performance Monitoring**:
  - Built-in timing collection
  - Performance profiling support
  - Tracing capabilities for optimization

## Contributing

When adding new components:
1. Place implementation in appropriate subdirectory
2. Add corresponding unit tests
3. Update model tests if necessary
4. Ensure parallel processing compatibility
