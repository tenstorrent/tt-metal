# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Original run 1000 registry metadata; test-only, no generic planner."""

import ttnn


def tag_num_groups(inputs, axes):
    """Per-shape coupled scalar — ``inputs[1]``. Declared first."""
    return inputs[1]


def tag_alignment(inputs, axes):
    """``(N, 1, HW, C)``: last two dims are HW (-2) and C (-1)."""
    shape = inputs[0]
    HW, C = shape[-2], shape[-1]
    if C % 32 != 0:
        return "c_non_aligned"
    if HW % 32 != 0:
        return "hw_non_aligned"
    return "tile_aligned"


INPUT_TAGGERS = {
    "num_groups": tag_num_groups,
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
#
# `num_groups` is intentionally absent: it is an unbounded integer coupled to
# the shape, not a finite axis, and the golden feature_spec's TARGET does not
# enumerate it. Every TARGET axis has an entry.

SUPPORTED = {
    # Refinement 2: bfloat8_b is a block-quantized TILE-only format (bf8b +
    # ROW_MAJOR is INVALID, never reaches here). The activation path needs no
    # kernel change — the CB carries the bfp8 page format and the unpacker /
    # packer convert — while the affine path decodes the tile's shared-exponent
    # row 0 in the reader.
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    # Tracks HW/C tile-alignment only, and is ORTHOGONAL to per-group channel
    # alignment: every SD/SDXL shape tags "tile_aligned" while still having
    # (C/G) % 32 != 0, which the kernel handles natively.
    # Refinement 1: HW % 32 != 0 is handled by a row mask applied to the
    # trailing HW tile in the moment pass; C % 32 != 0 collapses the
    # channel-cluster construction to the degenerate single cluster, where the
    # existing per-(group, channel-tile) interval mask already zeroes the
    # trailing tile's padding lanes.
    "alignment": ["tile_aligned", "hw_non_aligned", "c_non_aligned"],
    "affine": ["gamma_beta", "gamma_only", "no_affine"],
    # "none" = no affine tensor supplied; always legal, never refused.
    "affine_dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, "none"],
    "affine_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []
