# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end parity for the composed LTX-2.5 DiffVAE decoder, on shipped weights.

The halves are covered elsewhere — ``test_diffvae_det`` for the deterministic stages against
captured activations, ``test_diffvae_stage5`` for the diffusion stage against upstream's
modules. What neither covers is the seam: the trailing-ghost pad and crop that bracket the
deterministic stages, the context handoff into stage 5, and whether the two halves agree on
layout and channel order. That is what fails when parts verified separately are joined.

Ground truth is ``capture_stages.py``'s dump, which also supplies the stage-5 noise: it is an
input to a single-step x0 prediction, not an implementation detail, so matching pixels
requires using the reference's own noise rather than reseeding.

  PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src:. python capture_stages.py \
      latents/latent_0_1x128x4x34x60.pt --crop 10 --out stages/crop10.safetensors
"""

import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import ttnn
from models.tt_dit.models.vae.diffvae_ltx import DiffVAEDecoder, decoder_config
from models.tt_dit.utils.check import assert_quality

CAPTURE = Path(os.environ.get("DIFFVAE_CAPTURE", "/home/noblewoodall/ltx25_diffvae/stages/crop10.safetensors"))
CHECKPOINT = Path(
    os.environ.get(
        "DIFFVAE_CHECKPOINT",
        os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors"),
    )
)


def _captured(*names: str) -> tuple[torch.Tensor, ...]:
    """Load tensors from the capture, skipping if it was written without them.

    ``capture_stages.py --pixels-only`` keeps only the endpoints, which is how captures at
    resolutions above the smallest tile are produced — the intermediates would be tens of GB.
    """
    with safe_open(str(CAPTURE), "pt") as handle:
        available = set(handle.keys())
        if missing := [name for name in names if name not in available]:
            pytest.skip(f"{CAPTURE.name} lacks {missing}; regenerate without --pixels-only")
        return tuple(handle.get_tensor(name).float() for name in names)


@pytest.fixture
def decoder(device):
    if not CAPTURE.exists():
        pytest.skip(f"missing {CAPTURE}; run capture_stages.py first")
    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")
    model = DiffVAEDecoder(decoder_config(CHECKPOINT), mesh_device=device)
    model.load_checkpoint(CHECKPOINT)
    return model


def test_context_matches_upstream(*, decoder):
    """All four deterministic stages plus the ghost pad and crop, latent to stage-5 context.

    The frame arithmetic is the point: 4 latent frames become 6 by trailing replication, grow
    to 41 through four upsamples, then crop back to 25. An off-by-one in any of those three
    steps lands here rather than in a stage's own test.
    """
    latent, expected = _captured("input.latent", "stage4.context")

    context, dims = decoder.forward_context(latent)
    assert dims == tuple(expected.shape[1:4]), f"dims {dims} != capture {tuple(expected.shape[1:4])}"
    assert decoder.context_frames(latent.shape[2]) == expected.shape[1]

    actual = ttnn.to_torch(context).reshape(1, *dims, expected.shape[-1])
    assert_quality(expected, actual, pcc=0.99)


def test_decode_matches_upstream(*, decoder):
    """Latent to pixels through the whole decoder, against upstream's own pixels."""
    latent, noise, expected = _captured("input.latent", "stage5.noise", "output.pixels")

    pixels = decoder.decode(latent, noise=noise)

    assert tuple(pixels.shape) == tuple(expected.shape), f"{tuple(pixels.shape)} != {tuple(expected.shape)}"
    if out := os.environ.get("DIFFVAE_DUMP_PIXELS"):
        # A PCC number says the port is right; it does not say the video looks right. Keeping
        # the device pixels lets them be viewed against the reference and the conv decoder.
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(pixels.cpu(), out)
        print(f"\nwrote device pixels {tuple(pixels.shape)} to {out}")
    assert_quality(expected, pixels, pcc=0.99)
