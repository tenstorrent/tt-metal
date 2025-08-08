# SD3.5 Pipeline Demos

This directory contains demonstration scripts for the new Stable Diffusion 3.5 pipeline implementation.

## Files

- `sd35_demo.py`: Standalone demo script for the new SD3.5 pipeline

## Usage

### Basic Usage

Run the demo with default settings (2x4 mesh, 1024x1024 image):

```bash
python models/experimental/tt_dit/demos/sd35_demo.py
```

### Advanced Usage

```bash
# T3K configuration (2x4 mesh)
python models/experimental/tt_dit/demos/sd35_demo.py --mesh 2x4 --size 1024x1024 --steps 28

# TG configuration (4x8 mesh)
python models/experimental/tt_dit/demos/sd35_demo.py --mesh 4x8 --size 1024x1024 --steps 28

# Enable tracing for performance
python models/experimental/tt_dit/demos/sd35_demo.py --traced

# Interactive mode with custom prompts
python models/experimental/tt_dit/demos/sd35_demo.py --interactive

# Custom guidance scale and steps
python models/experimental/tt_dit/demos/sd35_demo.py --guidance 4.0 --steps 40
```

### Command Line Arguments

- `--mesh`: Mesh device shape (2x4 or 4x8)
- `--size`: Image size in WxH format (e.g., 1024x1024, 512x512)
- `--guidance`: Classifier-free guidance scale (default: 3.5)
- `--steps`: Number of inference steps (default: 28)
- `--traced`: Enable tracing for performance optimization
- `--interactive`: Enable interactive prompt mode

## Features

- **Parallel Configuration**: Automatically configures CFG, sequence, and tensor parallelism based on mesh shape
- **Timing Collection**: Detailed timing information for each pipeline stage
- **T5 Text Encoder**: Automatically enables/disables based on device configuration
- **Tracing Support**: Optional tracing for improved performance
- **Interactive Mode**: Generate multiple images with different prompts

## Test Integration

The demo functionality is also available as a pytest test:

```bash
# Run the pipeline test
cd models/experimental/tt_dit/tests/models
python -m pytest test_pipeline_sd35.py -v

# Run with specific configuration
python -m pytest test_pipeline_sd35.py::test_sd35_pipeline -v
```

## Architecture

The new pipeline implementation features:

- **Modular Parallel Config**: Uses `DiTParallelConfig` with separate CFG, tensor, and sequence parallelism
- **Submesh Management**: Automatic submesh creation and device reshaping
- **CCL Managers**: Improved collective communication handling
- **Enhanced Timing**: Comprehensive timing collection and reporting
- **Trace Support**: Optional tracing for production deployments

## Comparison with Old Pipeline

| Feature | Old Pipeline (`fun_pipeline.py`) | New Pipeline (`pipeline_stable_diffusion_35_large.py`) |
|---------|----------------------------------|--------------------------------------------------------|
| Parallel Config | `StableDiffusionParallelManager` | `DiTParallelConfig` with `ParallelFactor` |
| Device Management | Manual submesh setup | Automatic submesh creation |
| CCL Handling | Integrated parallel manager | Separate `CCLManager` instances |
| Timing | Basic timing | Detailed `TimingCollector` |
| Transformer | Old `sd_transformer` function | New `SD35Transformer2DModel` class |

The new implementation provides better modularity, improved performance monitoring, and cleaner separation of concerns.
