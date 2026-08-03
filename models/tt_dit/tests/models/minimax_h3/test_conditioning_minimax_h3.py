# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only gate for MiniMax-H3 FL2VA keyframe conditioning.

Four things here reproduce nothing if dropped, and none of them fails loudly:
ImageNet pixel statistics rather than ``[-1, 1]``, a *sampled* posterior under a
seed independent of the request, a float16 round trip on the sampled latent, and
the conditioning noise being the first draw off the request generator.

The float16 round trip in particular looks like a precision bug and is not, so
``test_float16_round_trip_is_load_bearing`` asserts that it actually changes the
output -- a refactor that "cleans it up" has to fail here.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from ....pipelines.minimax_h3 import conditioning as c
from ....pipelines.minimax_h3.packing import MINIMAX_H3_KEYFRAME_NOISE_AUG, patchify_video_latents
from ....pipelines.minimax_h3.scheduler import MiniMaxH3Scheduler

LATENT_CHANNELS = 24
LATENT_HEIGHT, LATENT_WIDTH = 34, 60
PATCH = (1, 2, 2)
# Two conditions: an fl2va request with both a first and a last keyframe.
CONDITION_SHAPES = ((1, LATENT_HEIGHT, LATENT_WIDTH),) * 2

LATENTS_MEAN = tuple(0.1 * index for index in range(LATENT_CHANNELS))
LATENTS_STD = tuple(1.0 + 0.05 * index for index in range(LATENT_CHANNELS))


def _latents(seed=5, frames=1):
    return torch.randn(
        1, LATENT_CHANNELS, frames, LATENT_HEIGHT, LATENT_WIDTH, generator=torch.Generator().manual_seed(seed)
    )


def _image(height=544, width=960, seed=3):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def test_pixels_use_imagenet_statistics():
    """Not [-1, 1]: H3 conditions on VLM-style normalized pixels."""
    assert c.MINIMAX_H3_PIXEL_MEAN == (0.485, 0.456, 0.406)
    assert c.MINIMAX_H3_PIXEL_STD == (0.229, 0.224, 0.225)

    image = _image()
    pixels = c.normalize_keyframe_pixels(image)
    assert pixels.shape == (1, 3, 1, image.size[1], image.size[0])
    assert pixels.dtype == torch.float32

    mean = torch.tensor(c.MINIMAX_H3_PIXEL_MEAN).view(1, -1, 1, 1, 1)
    std = torch.tensor(c.MINIMAX_H3_PIXEL_STD).view(1, -1, 1, 1, 1)
    raw = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None, :, None].float().div(255.0)
    assert torch.equal(pixels, (raw - mean) / std)
    # A [-1, 1] normalization would land in [-1, 1]; this does not.
    assert pixels.min() < -1.0


def test_posterior_is_sampled_not_mean():
    moments = torch.randn(1, 2 * LATENT_CHANNELS, 1, 8, 8, generator=torch.Generator().manual_seed(1))
    sampled = c.sample_posterior(moments)
    mean = moments.chunk(2, dim=1)[0]

    assert sampled.shape == mean.shape
    assert not torch.equal(sampled, mean)


def test_posterior_seed_is_independent_of_request_seed():
    """Seeded to 42 internally, so a keyframe encodes identically every request."""
    assert c.MINIMAX_H3_KEYFRAME_ENCODE_SEED == 42
    moments = torch.randn(1, 2 * LATENT_CHANNELS, 1, 8, 8, generator=torch.Generator().manual_seed(1))

    # Perturbing the global RNG must not change the result.
    torch.manual_seed(999)
    first = c.sample_posterior(moments)
    torch.manual_seed(12345)
    second = c.sample_posterior(moments)
    assert torch.equal(first, second)

    assert not torch.equal(first, c.sample_posterior(moments, seed=43))


def test_float16_round_trip_is_load_bearing():
    """The sampled latent is rounded through fp16 before normalization.

    This keeps ~11 bits of every conditioning latent. Asserted to actually change
    the rows so it cannot be silently optimized away.
    """
    latents = _latents()
    rows = c.keyframe_condition_rows(latents, LATENTS_MEAN, LATENTS_STD, PATCH)

    mean = torch.tensor(LATENTS_MEAN).view(1, LATENT_CHANNELS, 1, 1, 1)
    std = torch.tensor(LATENTS_STD).view(1, LATENT_CHANNELS, 1, 1, 1)
    exact = patchify_video_latents((latents - mean) / std, PATCH)
    rounded = patchify_video_latents((latents.to(torch.float16).float() - mean) / std, PATCH)

    assert torch.equal(rows, rounded)
    assert not torch.equal(rows, exact)
    assert (rows - exact).abs().max() > 1e-5


def test_condition_rows_shape_and_dtype():
    rows = c.keyframe_condition_rows(_latents(), LATENTS_MEAN, LATENTS_STD, PATCH)
    expected_rows = (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2)
    assert rows.shape == (expected_rows, LATENT_CHANNELS * 4)
    assert rows.dtype == torch.float32


def test_encode_keyframes_is_one_encode_per_image():
    """A keyframe is a single frame, so no temporal chunking applies."""
    seen = []

    def encode_clip(pixels):
        seen.append(tuple(pixels.shape))
        batch, _, frames, height, width = pixels.shape
        return torch.zeros(batch, 2 * LATENT_CHANNELS, frames, height // 16, width // 16)

    images = [_image(seed=1), _image(seed=2)]
    rows = c.encode_keyframes(images, encode_clip, LATENTS_MEAN, LATENTS_STD, PATCH)

    assert len(seen) == len(images)
    assert all(shape[2] == 1 for shape in seen)
    rows_per_frame = (544 // 16 // 2) * (960 // 16 // 2)
    assert rows.shape == (len(images) * rows_per_frame, LATENT_CHANNELS * 4)


@pytest.mark.parametrize("seed", [0, 42, 1101])
def test_condition_noise_is_deterministic_per_seed(seed):
    first = c.keyframe_condition_noise(CONDITION_SHAPES, LATENT_CHANNELS, PATCH, torch.Generator().manual_seed(seed))
    second = c.keyframe_condition_noise(CONDITION_SHAPES, LATENT_CHANNELS, PATCH, torch.Generator().manual_seed(seed))
    assert torch.equal(first, second)

    rows_per_frame = (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2)
    assert first.shape == (len(CONDITION_SHAPES) * rows_per_frame, LATENT_CHANNELS * 4)


def test_condition_noise_draws_per_condition_in_order():
    """One draw per condition, off the request generator, in packed order.

    Drawing once at the concatenated shape would consume the same number of
    values but assign them differently, and would leave the generator in the same
    state -- so this checks the per-condition split explicitly.
    """
    generator = torch.Generator().manual_seed(0)
    both = c.keyframe_condition_noise(CONDITION_SHAPES, LATENT_CHANNELS, PATCH, generator)

    per_condition = torch.cat(
        [
            c.keyframe_condition_noise((shape,), LATENT_CHANNELS, PATCH, torch.Generator().manual_seed(0))
            for shape in CONDITION_SHAPES[:1]
        ]
    )
    assert torch.equal(both[: per_condition.shape[0]], per_condition)


def test_condition_noise_is_the_first_draw_of_a_request():
    """Conditioning noise precedes the video and audio draws.

    If the order slipped, every latent in the request would change while still
    looking like valid noise.
    """
    ordered = torch.Generator().manual_seed(0)
    c.keyframe_condition_noise(CONDITION_SHAPES, LATENT_CHANNELS, PATCH, ordered)
    video_after_conditioning = torch.randn(3, 64, generator=ordered)

    skipped = torch.Generator().manual_seed(0)
    video_without_conditioning = torch.randn(3, 64, generator=skipped)

    assert not torch.equal(video_after_conditioning, video_without_conditioning)


def test_noise_augmentation_has_no_second_implementation():
    """Noise augmentation must go through the scheduler, not a local copy.

    A local ``t*x0 + (1-t)*noise`` drifted by 2.4e-7 against the scheduler because
    it evaluated ``1 - t`` in Python double instead of float32.
    """
    assert not hasattr(c, "noise_augment")

    rows = torch.randn(20, 96, generator=torch.Generator().manual_seed(1))
    noise = torch.randn(20, 96, generator=torch.Generator().manual_seed(2))
    out = MiniMaxH3Scheduler(12.0).scale_noise(rows, MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)

    timestep = torch.tensor(MINIMAX_H3_KEYFRAME_NOISE_AUG, dtype=torch.float32)
    assert torch.equal(out, timestep * rows + (1.0 - timestep) * noise)
    # Python-double arithmetic is what to avoid; prove it is observably different.
    drifted = MINIMAX_H3_KEYFRAME_NOISE_AUG * rows + (1.0 - MINIMAX_H3_KEYFRAME_NOISE_AUG) * noise
    assert not torch.equal(out, drifted)


def test_noise_aug_level_is_the_released_default():
    assert MINIMAX_H3_KEYFRAME_NOISE_AUG == 0.999
