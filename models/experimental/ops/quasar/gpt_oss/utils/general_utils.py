# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
General utilities for the GPT-OSS demo.
"""

from models.common.utility_functions import is_blackhole


def get_cache_file_name(tensor_cache_path, name):
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


def get_default_num_links(mesh_device):
    """Default number of fabric links for CCL ops on the given mesh.

    Blackhole exposes 2 fabric links per device; Wormhole exposes 4. Single-row meshes
    (shape[0] == 1) only need 1 link regardless of arch.
    """
    if mesh_device.shape[0] == 1:
        return 1
    return 2 if is_blackhole() else 4


def throughput_experts_supported_on_arch():
    """Whether the throughput-optimized experts path (all_to_all dispatch/combine, moe_gpt,
    selective_reduce_combine) is supported on the current arch.

    Currently disabled on Blackhole; only the batch=1 low-latency expert path runs there.
    """
    return not is_blackhole()
