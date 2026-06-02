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

## Dependencies

Install model-specific dependencies:

```bash
pip install -r models/demos/z_image_turbo/requirements.txt
```

## Quick Start

```bash
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
pytest models/demos/z_image_turbo/tests/test_text_encoder.py
pytest models/demos/z_image_turbo/tests/test_dit.py
pytest models/demos/z_image_turbo/tests/test_vae.py
```
