# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Small shared helpers: the tilized-weight cache filename and the fabric link count."""

from models.common.utility_functions import is_blackhole


def get_cache_file_name(tensor_cache_path, name):
    """``ttnn.as_tensor(cache_file_name=...)`` argument, or None when caching is off."""
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


def get_default_num_links(mesh_device):
    """Fabric links per device for CCL ops. Blackhole exposes 2, Wormhole 4; a single-row mesh
    needs only 1. The target here is an 8x4 Blackhole Galaxy, i.e. 2."""
    if mesh_device.shape[0] == 1:
        return 1
    return 2 if is_blackhole() else 4
