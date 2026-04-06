# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Face-view optimization utilities for gated reduce operations.

Face-view packs multiple small tiles into a single hardware face (16x16 = 256 elements),
allowing the reduce kernel to operate on face-sized tiles instead of iterating over
multiple small tiles per K-position.
"""

# Hardware face dimensions
FACE_HEIGHT = 16
FACE_WIDTH = 16
FACE_ELEMENTS = FACE_HEIGHT * FACE_WIDTH  # 256


def can_use_face_view(tile_h, tile_w, tiles_per_k, k_num_tiles):
    """
    Check if face-view optimization can be applied for gated reduce.

    Face-view requires:
    1. Tiles smaller than a face (tile_h * tile_w < 256)
    2. Each collection's k_num_tiles tiles exactly fill one face
    3. tiles_per_k >= 2 (need at least 2 faces to reduce)
    4. tiles_per_k is even (for pairwise addition)
    """
    elements_per_tile = tile_h * tile_w
    if elements_per_tile >= FACE_ELEMENTS:
        return False

    if elements_per_tile * k_num_tiles != FACE_ELEMENTS:
        return False

    if tiles_per_k < 2:
        return False

    if tiles_per_k % 2 != 0:
        return False

    return True
