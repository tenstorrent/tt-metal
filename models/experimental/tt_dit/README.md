# TT-DiT: Tenstorrent Diffusion Transformers

This directory contains the implementation of the Tenstorrent Diffusion Transformer (TT-DiT) architecture, designed for optimized parallelism of DiTs for image and video generation.

## Directory Structure

```
tt_dit/
├── layers/              # Core neural network layers
├── models/              # Model architectures
│   ├── autoencoders/    # Autoencoder implementations
│   └── transformers/    # Transformer model implementations
├── parallel/            # Parallelization utilities
│   ├── config.py        # Parallel configuration
│   └── manager.py       # Parallel execution management
├── pipelines/           # Model pipelines
│   └── stable_diffusion_35_large/
├── tests/              # Test suite
│   ├── models/         # Model-level tests
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
- **Transformers**: SD3.5-specific transformer implementations
  - `attention_sd35.py`: Attention mechanisms
  - `transformer_sd35.py`: Main transformer architecture

### Parallel Processing
- **Config**: Parallel configuration management with `DiTParallelConfig`
- **Manager**: Parallel execution and device management

### Pipelines
- **Stable Diffusion 3.5**: Large model pipeline implementation
  - Automatic parallel configuration
  - Optimized for different device configurations
  - Comprehensive timing collection

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

# Run model tests
python -m pytest tests/models/

# Run specific test file
python -m pytest tests/models/test_pipeline_sd35.py -v
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
