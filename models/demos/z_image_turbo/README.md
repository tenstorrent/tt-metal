# Z-Image-Turbo

Text-to-image pipeline ([Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)) on Tenstorrent hardware.

## Hardware

- **Board:** QB2 (2x p300 cards)
- **Mesh:** (1, 4) — four chips over 1D fabric

## Parallelism Strategy

| Model | Parallelism |
|---|---|
| Text Encoder (Qwen3) | TP=4 |
| Transformer (DIT) | TP=4 |
| VAE Decoder | Replicated |

All three models are Metal-Traced after compilation for fast inference (no host dispatch overhead).

## Quick Start

```bash
# Install dependencies
uv pip install -r models/forge/z_image_turbo/requirements.txt

# Generate an image
python -m models.demos.z_image_turbo.demo.demo "a misty mountain lake at dawn"

# Interactive server (REPL)
python -m models.demos.z_image_turbo.demo.demo_server
```

## Tests

```bash
# Run all tests
pytest models/demos/z_image_turbo/tests/

# Individual tests
pytest models/demos/z_image_turbo/tests/test_vae.py
pytest models/demos/z_image_turbo/tests/test_text_encoder_trace.py
pytest models/demos/z_image_turbo/tests/test_measure_dram.py
```

## Structure

```
z_image_turbo/
├── demo/
│   ├── demo.py            # Single/batch image generation
│   └── demo_server.py     # Interactive REPL server
├── tests/
│   ├── test_vae.py                  # VAE decoder correctness
│   ├── test_text_encoder_trace.py   # Text encoder trace correctness + perf
│   └── test_measure_dram.py         # DRAM footprint measurement
└── tt/
    ├── z_image_turbo.py   # Full pipeline (TE + DIT + VAE with Metal Trace)
    ├── vae/               # VAE decoder (TTNN + PyTorch reference)
    ├── dit/               # Diffusion Transformer (TTNN + PyTorch reference)
    └── text_encoder/      # Qwen3 text encoder (TTNN)
```
