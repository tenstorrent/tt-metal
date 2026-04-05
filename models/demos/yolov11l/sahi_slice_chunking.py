# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SAHI slice batching helpers (no OpenCV / SAHI imports)."""


def parallel_slice_chunk_bounds(num_slices: int, parallel: int):
    """Yield (start_index, num_valid_in_chunk) for batching SAHI slices across `parallel` devices."""
    i = 0
    while i < num_slices:
        n_valid = min(parallel, num_slices - i)
        yield i, n_valid
        i += parallel
