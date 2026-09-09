# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
General utilities for Llama-3.1-8B.
"""

from models.common.utility_functions import is_blackhole


def get_cache_file_name(tensor_cache_path, name):
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


def cache_file_exists(cache_file_name):
    """True iff a tilized tensor cache file for `cache_file_name` exists on disk.

    ttnn appends a ``_dtype_<DT>_layout_<L>.tensorbin`` suffix, so match by prefix.

    Deliberately does NOT use glob, and that is load-bearing rather than stylistic: the real cache
    directory name embeds the mesh shape, e.g. ``tensor_cache_bfp8_MeshShape([8, 4])``, and glob
    reads ``[8, 4]`` as a CHARACTER CLASS matching one of {'8', ',', ' ', '4'} — so a glob pattern
    never matches the literal path and reports every cached tensor as absent. listdir + startswith
    has no metacharacter semantics and is correct for any path.
    """
    if not cache_file_name:
        return False
    import os

    directory, prefix = os.path.split(str(cache_file_name))
    if not directory or not prefix:
        return False
    try:
        return any(e.startswith(prefix) and e.endswith(".tensorbin") for e in os.listdir(directory))
    except OSError:
        return False


def get_default_num_links(mesh_device):
    """Default number of fabric links for CCL ops on the given mesh.

    Blackhole exposes 2 fabric links per device; Wormhole exposes 4. Single-row meshes
    (shape[0] == 1) only need 1 link regardless of arch.
    """
    if mesh_device.shape[0] == 1:
        return 1
    return 2 if is_blackhole() else 4
