# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
General utilities for the Gemma4 demo.
"""


def get_cache_file_name(tensor_cache_path, name):
    """Build a tensor-cache path prefix for ``ttnn.as_tensor``.

    Always return the path when a cache root is configured so a writable CI
    pass can populate new names (e.g. DRAM-sharded ``*.ws``); later read-only
    CI then hits those cached tensorbins.
    """
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None
