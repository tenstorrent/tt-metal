# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""General utilities for Llama 3.1 8B."""


def get_cache_file_name(tensor_cache_path, name):
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


def get_default_num_links(mesh_device):
    """Default number of fabric links for CCL ops on the given mesh.

    Blackhole Galaxy (8x4): always 1 link.
    """
    return 1
