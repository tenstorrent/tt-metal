# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side validation of the Wan2.2 TI2V-5B image-conditioning math.

No device, no hardware, runs in seconds. Every assertion is exact equality against the
expression in ``diffusers/pipelines/wan/pipeline_wan_i2v.py`` (``WanImageToVideoPipeline``
with ``expand_timesteps=True``), which is the reference implementation for this checkpoint.
The reference expressions are inlined in each test so a diffusers upgrade that changes them
shows up as a diff here rather than as a silent behaviour change on device.
"""

import pytest
import torch

from models.tt_dit.pipelines.wan.ti2v_5b_i2v_math import (
    blend_condition,
    denormalize_latents,
    first_frame_mask,
    normalize_latents,
    pad_timesteps_for_sequence_parallel,
    per_token_timesteps,
)
from models.tt_dit.utils.padding import get_padded_vision_seq_len

# TI2V-5B geometry. vae_scale_factor_spatial=16, temporal=4, transformer patch_size=(1,2,2).
# 1280x704 is the checkpoint's stated 720P size; 704/16=44 is even so patchify is exact.
Z_DIM = 48
SPATIAL = 16
TEMPORAL = 4
PATCH = (1, 2, 2)


def _geometry(height=704, width=1280, num_frames=81):
    num_latent_frames = (num_frames - 1) // TEMPORAL + 1
    return num_latent_frames, height // SPATIAL, width // SPATIAL


class TestFirstFrameMask:
    def test_matches_reference(self):
        """vs diffusers pipeline_wan_i2v.py:461-466."""
        f, h, w = _geometry()

        expected = torch.ones(1, 1, f, h, w, dtype=torch.float32)
        expected[:, :, 0] = 0

        assert torch.equal(first_frame_mask(f, h, w), expected)

    def test_shape_and_contents(self):
        f, h, w = _geometry()
        mask = first_frame_mask(f, h, w)

        assert mask.shape == (1, 1, f, h, w)
        assert (mask[:, :, 0] == 0).all(), "frame 0 must be pinned to the condition"
        assert (mask[:, :, 1:] == 1).all(), "every other frame must be denoised"


class TestPerTokenTimesteps:
    def test_matches_reference(self):
        """vs diffusers pipeline_wan_i2v.py:762."""
        f, h, w = _geometry()
        mask = first_frame_mask(f, h, w)
        t = 731.0

        expected = (mask[0][0][:, ::2, ::2] * t).flatten()

        assert torch.equal(per_token_timesteps(mask, t), expected)

    # (num_frames, latent_F, N) at 1280x704. 81f is the perf-gate clip, 121f the card's 5s@24fps.
    @pytest.mark.parametrize(("num_frames", "latent_f", "expected_n"), [(81, 21, 18480), (121, 31, 27280)])
    def test_length_equals_transformer_token_count(self, num_frames, latent_f, expected_n):
        """The vector must be exactly one entry per transformer token."""
        f, h, w = _geometry(num_frames=num_frames)
        _, ph, pw = PATCH
        assert f == latent_f, "(num_frames - 1) // 4 + 1"

        ts = per_token_timesteps(first_frame_mask(f, h, w), 500.0)

        assert ts.shape == (f * (h // ph) * (w // pw),)
        assert ts.shape == (expected_n,), f"1280x704/{num_frames}f should give {expected_n} tokens"

    def test_frame_zero_tokens_are_zero_rest_are_t(self):
        """Frame-0 tokens are a contiguous prefix, because token order is patch-F major."""
        f, h, w = _geometry()
        _, ph, pw = PATCH
        tokens_per_frame = (h // ph) * (w // pw)
        t = 842.0

        ts = per_token_timesteps(first_frame_mask(f, h, w), t)

        assert (ts[:tokens_per_frame] == 0).all(), "conditioned frame must see timestep 0"
        assert (ts[tokens_per_frame:] == t).all(), "every other token must see t"

    def test_token_order_is_frame_major(self):
        """Guards the ordering contract against preprocess_spatial_input_host.

        That method permutes to (B, patch_F, patch_H, patch_W, ...) before flattening, so
        token index must advance fastest in W, then H, then F. Build a mask with a unique
        value per (f, h, w) cell and check the flattened order matches.
        """
        f, h, w = 3, 4, 6
        _, ph, pw = PATCH
        marker = torch.arange(f * (h // ph) * (w // pw), dtype=torch.float32)
        mask = torch.zeros(1, 1, f, h, w, dtype=torch.float32)
        mask[0, 0, :, ::ph, ::pw] = marker.reshape(f, h // ph, w // pw)

        assert torch.equal(per_token_timesteps(mask, 1.0), marker)


class TestSequenceParallelPadding:
    @pytest.mark.parametrize("sp_factor", [8, 4])
    def test_pads_with_t_not_zero(self, sp_factor):
        """The divergence from diffusers that keeps the all-ones-mask path exact.

        pad_vision_seq_parallel zero-fills the spatial input. Zero is a meaningful timestep
        (fully denoised), so zero-filling the timestep vector would change pad-token AdaLN
        relative to the scalar path. Fill with t instead.
        """
        f, h, w = _geometry()
        t = 617.0
        ts = per_token_timesteps(first_frame_mask(f, h, w), t)

        padded = pad_timesteps_for_sequence_parallel(ts, sp_factor, fill=t)

        n = ts.shape[0]
        padded_n = get_padded_vision_seq_len(n, sp_factor)
        assert padded.shape == (padded_n,)
        assert torch.equal(padded[:n], ts), "existing tokens must be untouched"
        assert (padded[n:] == t).all(), "pad tokens must see t, not 0"

    # These SP-local M values are what the tuned matmul tables must be keyed on.
    @pytest.mark.parametrize(
        ("num_frames", "unpadded", "padded", "per_device"), [(81, 18480, 18688, 2336), (121, 27280, 27392, 3424)]
    )
    def test_padded_length_shards_evenly(self, num_frames, unpadded, padded, per_device):
        f, h, w = _geometry(num_frames=num_frames)
        ts = per_token_timesteps(first_frame_mask(f, h, w), 1.0)
        assert ts.shape[0] == unpadded

        out = pad_timesteps_for_sequence_parallel(ts, 8, fill=1.0)

        assert out.shape[0] % (32 * 8) == 0, "must tile-align on every SP shard"
        assert out.shape[0] == padded, f"1280x704/{num_frames}f pads {unpadded} -> {padded}"
        assert out.shape[0] // 8 == per_device

    def test_no_padding_is_a_passthrough(self):
        ts = torch.full((256,), 5.0)
        assert pad_timesteps_for_sequence_parallel(ts, 8, fill=5.0) is ts

    def test_rejects_non_1d(self, expect_error):
        with expect_error(ValueError, "1-D"):
            pad_timesteps_for_sequence_parallel(torch.zeros(2, 4), 8, fill=0.0)


class TestBlend:
    def test_matches_reference(self):
        """vs diffusers pipeline_wan_i2v.py:758 and :813-814."""
        f, h, w = _geometry(height=64, width=64, num_frames=9)
        torch.manual_seed(0)
        latents = torch.randn(1, Z_DIM, f, h, w)
        condition = torch.randn(1, Z_DIM, 1, h, w)
        mask = first_frame_mask(f, h, w)

        expected = (1 - mask) * condition + mask * latents

        assert torch.equal(blend_condition(condition, latents, mask), expected)

    def test_pins_frame_zero_and_leaves_the_rest(self):
        """The whole point: frame 0 becomes the condition, nothing else changes."""
        f, h, w = _geometry(height=64, width=64, num_frames=9)
        torch.manual_seed(0)
        latents = torch.randn(1, Z_DIM, f, h, w)
        condition = torch.randn(1, Z_DIM, 1, h, w)

        out = blend_condition(condition, latents, first_frame_mask(f, h, w))

        assert torch.equal(out[:, :, 0], condition[:, :, 0]), "frame 0 must be exactly the condition"
        assert torch.equal(out[:, :, 1:], latents[:, :, 1:]), "later frames must be untouched"

    def test_all_ones_mask_is_identity(self):
        """The host-side half of the all-ones-mask equivalence argument.

        With mask=1 everywhere the conditioning drops out entirely and the I2V path must
        reduce to the T2V path. If this is not exact, the on-device equivalence test cannot
        be either.
        """
        f, h, w = _geometry(height=64, width=64, num_frames=9)
        torch.manual_seed(0)
        latents = torch.randn(1, Z_DIM, f, h, w)
        condition = torch.randn(1, Z_DIM, 1, h, w)
        ones = torch.ones(1, 1, f, h, w)

        assert torch.equal(blend_condition(condition, latents, ones), latents)

    def test_all_ones_mask_gives_scalar_timesteps(self):
        f, h, w = _geometry(height=64, width=64, num_frames=9)
        ones = torch.ones(1, 1, f, h, w)
        t = 333.0

        ts = per_token_timesteps(ones, t)

        assert (ts == t).all(), "per-token must collapse to the scalar timestep"


class TestLatentNormalization:
    """diffusers normalizes with a reciprocal std (pipeline_wan_i2v.py:445-460) while the TT
    decode adapter denormalizes with the raw std (vae_wan2_1.py:2262). Prove they are inverses
    so a conditioning latent survives the encode->denoise->decode round trip."""

    @staticmethod
    def _stats():
        torch.manual_seed(0)
        mean = torch.randn(1, Z_DIM, 1, 1, 1)
        std = torch.rand(1, Z_DIM, 1, 1, 1) + 0.5  # keep away from zero
        return mean, std

    def test_matches_reference_formulation(self):
        """Our divide must equal diffusers' multiply-by-reciprocal, bit for bit."""
        mean, std = self._stats()
        x = torch.randn(1, Z_DIM, 3, 4, 4)

        reference = (x - mean) * (1.0 / std)

        assert torch.equal(normalize_latents(x, mean, std), reference)

    def test_denormalize_matches_decode_adapter(self):
        """vs models/tt_dit/models/vae/vae_wan2_1.py:2262."""
        mean, std = self._stats()
        x = torch.randn(1, Z_DIM, 3, 4, 4)

        assert torch.equal(denormalize_latents(x, mean, std), x * std + mean)

    def test_round_trip(self):
        mean, std = self._stats()
        x = torch.randn(1, Z_DIM, 3, 4, 4, dtype=torch.float64)
        m64, s64 = mean.double(), std.double()

        assert torch.allclose(denormalize_latents(normalize_latents(x, m64, s64), m64, s64), x, atol=1e-12)
