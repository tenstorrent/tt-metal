#!/usr/bin/env python3
"""
Generate synthetic preprocessed tensor fixtures for end-to-end testing.

Creates N fake .pt files with correct tensor shapes so the full training
pipeline can be exercised without needing real audio data or the VAE/encoder.

Tensor shapes (matching the real preprocessing pipeline):
    target_latents       : [T, 64]    (audio_acoustic_hidden_dim)
    attention_mask       : [T]
    encoder_hidden_states: [L, 2048]  (hidden_size)
    encoder_attention_mask: [L]
    context_latents      : [T, 128]   (64 src_latents + 64 chunk_masks)

Usage:
    python -m acestep.training_v2.make_test_fixtures \\
        --output-dir ./test_fixtures --num-samples 4

Or from another script:
    from acestep.training_v2.make_test_fixtures import generate_fixtures
    generate_fixtures(Path("./test_fixtures"), num_samples=4)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Default dimensions matching the turbo model config
_TARGET_DIM = 64  # audio_acoustic_hidden_dim
_CONTEXT_DIM = 128  # 64 (src_latents) + 64 (chunk_masks)
_ENCODER_DIM = 2048  # hidden_size
_LATENT_LENGTH = 128  # Short sequence for fast testing
_ENCODER_LENGTH = 64  # Encoder sequence length


def generate_single_sample(
    sample_id: str,
    latent_length: int = _LATENT_LENGTH,
    encoder_length: int = _ENCODER_LENGTH,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Generate a single synthetic preprocessed sample.

    All tensors are random (no semantic meaning), but shapes are correct
    for the DiT decoder.
    """
    return {
        "target_latents": torch.randn(latent_length, _TARGET_DIM, dtype=dtype),
        "attention_mask": torch.ones(latent_length, dtype=dtype),
        "encoder_hidden_states": torch.randn(encoder_length, _ENCODER_DIM, dtype=dtype),
        "encoder_attention_mask": torch.ones(encoder_length, dtype=dtype),
        "context_latents": torch.randn(latent_length, _CONTEXT_DIM, dtype=dtype),
        "metadata": {
            "sample_id": sample_id,
            "caption": f"Test sample {sample_id}",
            "lyrics": "[Instrumental]",
            "duration": latent_length * 0.01,
            "is_synthetic": True,
        },
    }


def generate_fixtures(
    output_dir: Path,
    num_samples: int = 4,
    latent_length: int = _LATENT_LENGTH,
    encoder_length: int = _ENCODER_LENGTH,
) -> Path:
    """Generate *num_samples* synthetic .pt files + manifest.json.

    Args:
        output_dir: Directory to write fixtures to.
        num_samples: Number of samples to generate.
        latent_length: Latent sequence length (T dimension).
        encoder_length: Encoder sequence length (L dimension).

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = []
    for i in range(num_samples):
        sample_id = f"test_{i:04d}"
        data = generate_single_sample(
            sample_id=sample_id,
            latent_length=latent_length,
            encoder_length=encoder_length,
        )
        pt_path = output_dir / f"{sample_id}.pt"
        torch.save(data, str(pt_path))
        sample_paths.append(str(pt_path))

    # Write manifest
    manifest = {
        "metadata": {
            "type": "synthetic_test_fixtures",
            "num_samples": num_samples,
            "latent_length": latent_length,
            "encoder_length": encoder_length,
        },
        "samples": sample_paths,
        "num_samples": num_samples,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[OK] Generated {num_samples} synthetic fixtures in {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic preprocessed tensor fixtures for testing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_fixtures",
        help="Output directory (default: ./test_fixtures)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to generate (default: 4)",
    )
    parser.add_argument(
        "--latent-length",
        type=int,
        default=_LATENT_LENGTH,
        help=f"Latent sequence length (default: {_LATENT_LENGTH})",
    )
    parser.add_argument(
        "--encoder-length",
        type=int,
        default=_ENCODER_LENGTH,
        help=f"Encoder sequence length (default: {_ENCODER_LENGTH})",
    )

    args = parser.parse_args()
    generate_fixtures(
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        latent_length=args.latent_length,
        encoder_length=args.encoder_length,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
