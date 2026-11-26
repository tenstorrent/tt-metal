# TT-DiT: Tenstorrent Diffusion Transformers

This directory contains the implementation of the Tenstorrent Diffusion Transformer (TT-DiT) architecture, designed for optimized parallelism of DiTs for image and video generation.

## Supported Models

For detailed information about each model including performance metrics, usage instructions, and specific requirements, see:

- **[Stable Diffusion 3.5 Large](models/StableDiffusion35.md)** - Text-to-image generation
- **[Flux 1](models/Flux1.md)** - Text-to-image generation (schnell & dev variants)
- **[Motif](models/Motif.md)** - Text-to-image generation model
- **[Qwen-Image](models/QwenImage.md)** - Text-to-image generation model
- **[Ideogram 4.0](models/Ideogram4.md)** - Text-to-image generation model (best-in-class typography; Blackhole)
- **[FIBO](models/Fibo.md)** - Text-to-image generation model (structured JSON prompts, expanded from text by a VLM)
- **[Mochi-1](models/Mochi_1.md)** - Video generation model
- **[Wan2.2-T2V-A14B](models/Wan2_2.md)** - Text-to-video generation model
- See **[experimental/](experimental/)** for in-progress model variants (AniSora, Wan2.2-Distill, LoRA adapters)
- **[LTX-2.3](models/LTX2.md)** - Joint audio-video generation model (Pro and Fast variants)

## Directory Structure

```
tt_dit/
├── layers/              # Core neural network layers
├── models/              # Model architectures and documentation
│   ├── transformers/    # Transformer implementations (SD35, Mochi, Wan, LTX, Flux1, Motif, QwenImage)
│   ├── vae/            # VAE/Autoencoder implementations
│   ├── StableDiffusion35.md  # SD3.5 model documentation
│   ├── Flux1.md         # Flux 1 model documentation
│   ├── Motif.md         # Motif model documentation
│   ├── QwenImage.md     # Qwen-Image model documentation
│   ├── Mochi_1.md       # Mochi-1 model documentation
│   ├── Wan2_2.md        # Wan2.2 model documentation
│   └── LTX2.md          # LTX-2.3 model documentation
├── encoders/            # Text encoder implementations
│   ├── clip/           # CLIP encoder
│   ├── gemma/          # Gemma encoder
│   └── t5/             # T5 encoder
├── parallel/            # Parallelization utilities
│   ├── config.py        # Parallel configuration
│   └── manager.py       # Parallel execution management
├── pipelines/           # End-to-end model pipelines
│   ├── stable_diffusion_35_large/
│   ├── mochi/
│   ├── wan/
│   ├── ltx/
│   ├── flux1/
│   ├── motif/
│   └── qwenimage/
├── tests/              # Test suite
│   ├── models/         # Model-level tests (sd35, mochi, wan2_2, ltx, flux1, motif, qwenimage)
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
  - Support for multiple architectures including SD3.5, Mochi, Wan2.2, LTX-2, Flux1, Motif, and Qwen-Image
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
- **LTX-2.3**: Joint audio-video generation (Pro one-stage and Fast distilled two-stage)

Each pipeline includes:
- Automatic parallel configuration for different device meshes
- Optimized execution on Wormhole systems
- Comprehensive timing collection and profiling

### Device residency and traces
`pipelines/residency.py` owns which modules are on device. A pipeline registers each module once in
`__init__` with the keywords `cache.load_model` takes (`spec`), the modules that must be gone before it is
resident (`evicts`), how it is evicted when another module lists it (`mode`: `Offload()` deallocates,
`Shard(axis, ccl_manager)` shards and later gathers) and what runs after it is evicted (`on_evict`):

```
self.residency.register(a, spec, evicts=(...), mode=Offload(), on_evict=(self.release_traces,))

self.residency.load(a):
    for m in evicts[a] that are loaded:
        m.mode.evict(m); run m.on_evict in order
    if a is transiently sharded:  a.gather(...)
    elif a is not loaded:         cache.load_model(a, **spec)
```

There is no `unload`: a module is evicted only because another module's `load` lists it. Residency releases
traces only through `on_evict`, never on a load.

- **Trace interface.** `PipelineAPIMixin.release_traces` releases every live `Tracer` (`Tracer.release_all`),
  however it was built. Override it only to also release traces that are not `Tracer`s; releasing less is
  unsound, since a load after any capture can land in any live trace's scratch.
- **Permanent modules.** A module that no registration lists is never evicted, and loading it releases
  nothing. A module first loaded after a capture has its buffers live during every later replay; if that can
  land in a trace's scratch, release traces at that load site.
- **`on_evict`.** The designer's declaration of what the eviction releases; it is trusted, not checked. The
  default `(self.release_traces,)` is for a new module. `()` keeps a pipeline whose eviction released nothing;
  `(tracer.release_trace,)` one that released exactly that tracer; `(self.release_traces, extra, ...)` composes
  per-module cleanup after the release. A declaration narrower than "release all" keeps an existing pipeline's
  behavior; where that behavior is unsound (a trace replayed after a reload moved a buffer it reads), say so in
  a comment at the registration.

Rules:
1. Register each module once in the pipeline's `__init__`, after construction, before any load that is not
   construction-time.
2. Residency changes only through `self.residency.load(m)`. No module loads, reloads, pages, shards or evicts
   itself.
3. `on_evict=()` only for an untraced module whose eviction frees everything it allocated.
4. Trace strategy lives in `release_traces`; `on_evict` composes it with per-module cleanup.
5. Persistent device memory that is not a module's weights is a `StateTensor` allocated before the first
   capture, or goes with the module's eviction (`deallocate_weights` override or `register_eviction_hook`).

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
# Run tests from tt-metal/

# Run unit tests
python -m pytest  models/tt_dit/tests/unit/

# Run all model tests
python -m pytest  models/tt_dit/tests/models/

# Run specific model pipeline tests
python -m pytest models/tt_dit/tests/models/sd35/test_pipeline_sd35.py -v
python -m pytest models/tt_dit/tests/models/flux1/test_pipeline_flux1.py -v
python -m pytest models/tt_dit/tests/models/motif/test_pipeline_motif.py -v
python -m pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py -v
python -m pytest models/tt_dit/tests/models/mochi/test_pipeline_mochi.py -v
python -m pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -v
python -m pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py -v
python -m pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_two_stages.py -v
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
