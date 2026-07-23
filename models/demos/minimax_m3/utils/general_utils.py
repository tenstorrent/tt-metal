# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
General utilities for MiniMax-M3.
"""

from models.common.utility_functions import is_blackhole


def get_cache_file_name(tensor_cache_path, name):
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


def cache_file_exists(cache_file_name):
    """True iff a tilized tensor cache file for `cache_file_name` exists on disk. ttnn appends a
    `_dtype_<DT>_layout_<L>.tensorbin` suffix, so match by prefix. Used to decide whether to load an
    OPTIONAL weight (e.g. the MoE gate's correction bias) from cache when the source state_dict is
    absent (cache-only loading) — its presence can't be known from an empty state_dict."""
    if not cache_file_name:
        return False
    import glob

    return bool(glob.glob(f"{cache_file_name}*.tensorbin"))


def get_default_num_links(mesh_device):
    """Default number of fabric links for CCL ops on the given mesh.

    Blackhole exposes 2 fabric links per device; Wormhole exposes 4. Single-row meshes
    (shape[0] == 1) only need 1 link regardless of arch.
    """
    if mesh_device.shape[0] == 1:
        return 1
    return 2 if is_blackhole() else 4
