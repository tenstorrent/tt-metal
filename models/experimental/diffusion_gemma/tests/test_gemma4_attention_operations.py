# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Guards for the SDPA chunk-size helper the denoise path depends on.

Imported from ``diffusion_gemma.tt.diffusion_attention``, which owns this helper. It used to be
imported from ``models.demos.gemma4.tt.attention.operations``; that module no longer has it, so the
import failed at COLLECTION time and took the whole DiffusionGemma suite down with it -- pytest
aborts the run on a collection error, so one stale line hid 761 passing tests.
"""
from models.experimental.diffusion_gemma.tt.diffusion_attention import _largest_tile_divisor


def test_largest_tile_divisor_never_returns_non_tile_multiple():
    assert _largest_tile_divisor(100, 100) == 32


def test_largest_tile_divisor_prefers_largest_aligned_divisor():
    assert _largest_tile_divisor(384, 256) == 192
    assert _largest_tile_divisor(512, 256) == 256
