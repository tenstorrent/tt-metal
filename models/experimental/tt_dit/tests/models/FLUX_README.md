# FLUX Image Generation with TT-Metal

This guide will help you set up and run the FLUX image generation pipeline using TT-Metal on Tenstorrent hardware.

## Prerequisites

Before starting, ensure you have:
- A Tenstorrent device (Wormhole or Blackhole) properly set up
- Ubuntu 22.04 or compatible Linux distribution
- Python 3.10 or higher

## Installation Steps

### 1. Install TT-Metal

### 2. Set Up Virtual Environment and Activate It

### 3. Set Environment Variables

### 4. Run the FLUX Pipeline Test

Execute the FLUX image generation test using pytest:

```bash
pytest models/experimental/tt_dit/tests/models/test_pipeline_flux1.py::test_flux1_pipeline_fixed_params
```

#### Test Parameters

The test supports various configurations:
- **Model variants**: `schnell` (4 steps) or `dev` (28 steps)
- **Image dimensions**: 1024x1024 (default)
- **Device configurations**: (2x4)
- **Tracing**: Enabled or disabled for performance optimization

#### Running with Specific Parameters

You can run specific test configurations:

```bash
# Run with specific mesh configuration
pytest models/experimental/tt_dit/tests/models/test_pipeline_flux1.py -k "2x4sp0tp1"

# Run without prompts (uses predefined prompts)
NO_PROMPT=1 pytest models/experimental/tt_dit/tests/models/test_pipeline_flux1.py
```

### 5. Enter Prompts to Generate Images

When you run the test without the `NO_PROMPT=1` environment variable, the system will prompt you to enter text descriptions for image generation:

```
Enter the input prompt, or q to exit: A luxury sports car in a futuristic city
```

You can:
- Enter any text description you want to generate as an image
- Press Enter to use the previous prompt again
- Type 'q' and press Enter to quit the generation loop

### 6. Generated Images Location

The generated images are automatically saved in the **root directory** of the tt-metal repository with descriptive filenames that include:

- Model variant (schnell/dev)
- Image dimensions (1024x1024)
- Device configuration (e.g., 2x4sp0tp1)
- Sequential number

Example filename: `flux_schnell_1024_1024_2x4sp0tp1_0.png`


### Performance Monitoring

The test provides detailed timing information:
- CLIP encoding time
- T5 encoding time
- VAE decoding time
- Total pipeline time
- Average denoising step time

## Model Information

- **FLUX.1-schnell**: Fast variant with 4 inference steps
- **FLUX.1-dev**: Development variant with 28 inference steps (higher quality)
- **Text Encoders**: Supports both T5 and CLIP text encoders
- **Parallel Processing**: Configurable tensor and sequence parallelism for multi-device setups
