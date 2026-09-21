# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Small host-side helpers shared by the TT modules."""

from __future__ import annotations

import os
from typing import Optional

from models.common.utility_functions import is_blackhole


def get_cache_file_name(tensor_cache_path: Optional[str], name: str) -> Optional[str]:
    """Compose a tilized-weight cache path, or ``None`` when caching is off."""
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


def cache_file_exists(cache_file_name: Optional[str]) -> bool:
    """Whether a tilized cache entry exists. ``ttnn.as_tensor`` appends a
    ``_dtype_<DT>_layout_<L>.tensorbin`` suffix, so this matches by prefix.

    listdir + startswith rather than glob on purpose: the cache directory name embeds the mesh
    shape (``tensor_cache_bfp8_MeshShape([8, 4])``) and glob reads ``[8, 4]`` as a character class,
    so a glob pattern silently never matches.
    """
    if not cache_file_name:
        return False
    directory, prefix = os.path.split(str(cache_file_name))
    if not directory or not prefix:
        return False
    try:
        return any(e.startswith(prefix) and e.endswith(".tensorbin") for e in os.listdir(directory))
    except OSError:
        return False


def get_default_num_links(mesh_device) -> int:
    """Fabric links per CCL op. Blackhole exposes 2 per device, Wormhole 4; a single-row mesh
    needs only 1 because there is no second axis to ring around."""
    if mesh_device.shape[0] == 1:
        return 1
    return 2 if is_blackhole() else 4
