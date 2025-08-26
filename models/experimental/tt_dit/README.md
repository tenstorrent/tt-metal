# TT-DiT: Tenstorrent Diffusion Transformers

This directory contains the implementation of the Tenstorrent Diffusion Transformer (TT-DiT) architecture, designed for optimized parallelism of DiTs for image and video generation.


## Models

| Model                   | Hardware          | taft (ms) | t/s/u | target t/s/u | t/s | TT-metalium release |
|--------------------------|------------------|-----------|-------|---------------|-----|---------------------|
| [Stable Diffusion 3.5 Large](#stable-diffusion-35-large)   | T3K (Wormhole)           |       |               |     |                     |
| [Stable Diffusion 3.5 Large](#stable-diffusion-35-large) |  Galaxy (Wormhole)          |       |               |     |                     |

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

<br>

# Models

### Stable Diffusion 3.5 Large

#### Platforms:
    LoudBox, QuietBox (WH), Galaxy (Wormhole)

## Introduction
[Stable Diffusion 3.5](https://stability.ai/news/introducing-stable-diffusion-3-5) is a generative model for image synthesis guided by text prompts.

## Details

The architecture is described in the paper
[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).

The model consists of two different text encoders together with their tokenizers, a scheduler, a trasformer and a VAE. The core component is the transformer, called MMDiT (Multimodal Diffusion Transformer). The transformer is made up of spatial, prompt and time embeddings, and a series of transformer blocks. Transformer blocks mainly contain attention layers, that operate either on the spatial embedding only, or on the spatial and prompt embeddings together.

## Scalability

SD3.5-Large has been implemented to support execution on 8-chip (LoudBox and QuietBox) as well as 32-chip (Galaxy) systems.
The model has only been tested on Wormhole. Blackhole support is coming soon.

The DiT model can be parallelized on 3 axes:
1. `cfg` (classifier-free guidance) - execute conditional and unconditional steps in parallel
2. `sp` (sequence parallel) - the input sequence is fractured across a mesh axis. FeedForward layers execute in parallel across different chunks of the sequence. Attention is implemented with ring attention, overlapping KV all-gather with computation. See the [reference implementation](https://github.com/feifeibear/long-context-attention) of Unified Sequence Parallel for more information.
3. `tp` (tensor parallel) - weights are fractured across a mesh axis. CCLs such as AllGather and ReduceScatter are used to gather and scatter activations.

There are two additional axes of parallelism: `rp` (ring parallel) is tied to `sp`, and `up` (ulysses parallel) is tied to `tp`. These are the equivalents of `sp` and `tp` for the attention module.

A parallel config is defined by a tuple `((cfg_factor, cfg_axis), (sp_factor, sp_axis), (tp_factor, tp_axis))`.

An example parallel config on a 2x4 mesh is `((2, 1), (2, 0), (2, 1))`. This gives us `cfg` parallelism with factor 2 on axis 1, yielding 2 2x2 submeshes. `sp` is factor 2 on axis 0, meaning that activations are sequence-fractured on the `2x2` submesh on axis 0. `tp` is factor 2 on axis 1, meaning weights are tensor-fractured on the `2x2` submesh on axis 1.

Another example parallel config on a 4x8 mesh is `((2, 1), (4, 0), (4, 1))`. `cfg` factor 2 on axis 1 yields 2 4x4 submeshes. `sp` is on axis 0 and `tp` is on axis 1, giving us `sp` factor 4 and `tp` factor 4.

The text embedding models and the VAE decoder are parallelized with tensor parallelism on one or both of the cfg submeshes.



## Performance

Current performance and target performance for two systems are detailed below. Performance is measured in seconds per image, where the image size is 1024x1024px.

| System    | CFG | SP | TP | Current Performance | Target Performance |
|-----------|-----|----|----|---------------------|--------------------|
| QuietBox  | 2   | 2  | 2  | 12.2s               | 14.4s              |
| Galaxy    | 2   | 4  | 4  | 5.9s                | 3.6s               |

Reproduce these performance numbers with our performance tests.
```
# On QuietBox
pytest models/experimental/stable_diffusion_35_large/tests/test_performance.py -k "t3k_cfg2_sp2_tp2"

# On Galaxy
TT_MM_THROTTLE_PERF=5 pytest models/experimental/stable_diffusion_35_large/tests/test_performance.py -k "tg_cfg2_sp4_tp4"
```

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)


## How to Run

## Running the Demo

1. Visit [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) to grant access to the model weights
2. Login with the HuggingFace token: `huggingface-cli login`

Finally, run the demo.
```
# On QuietBox
pytest models/experimental/stable_diffusion_35_large/fun_demo.py -k "t3k_cfg2_sp2_tp2 and yes_trace"

# On Galaxy
TT_MM_THROTTLE_PERF=5 pytest models/experimental/stable_diffusion_35_large/fun_demo.py -k "tg_cfg2_sp4_tp4 and yes_trace"
```

## Serving the model

Coming soon!
Serve the model with our inference server and test it with a simple GUI.

## Disclaimers

- Output correctness validation is underway.
- On Galaxy, avoid hangs by setting `TT_MM_THROTTLE_PERF=5`. This has a slight impact on performance but stabilizes the demo. We are working on a fix.
