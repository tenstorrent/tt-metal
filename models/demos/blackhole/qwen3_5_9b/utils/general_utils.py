# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Small host-side helpers shared across the model."""


def get_cache_file_name(tensor_cache_path, name):
    """Build the on-disk path for a cached weight tensor, or None when caching is off.

    The weight loaders pass the returned path to ``ttnn.as_tensor(cache_file_name=...)``;
    a None ``tensor_cache_path`` disables caching (the tensor is reconverted every run),
    which is what the random-weight unit tests want to avoid a stale cross-run cache.
    """
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None
