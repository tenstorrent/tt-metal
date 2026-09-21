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
    """Whether the throughput experts path (all_to_all dispatch/combine over a mesh axis)
    is supported on the current arch.

    Supported on both Wormhole and Blackhole. The dense throughput flow only uses generic
    ops (all_to_all_dispatch, matmul, all_to_all_combine, all_reduce), none of which carry
    arch-specific assumptions. The *fused* variant of this path is a separate question --
    see fused_moe_kernels_supported_on_arch().
    """
    return True


def fused_moe_kernels_supported_on_arch():
    """Whether the fused MoE kernels (moe_gpt, all_to_all_dispatch_metadata,
    selective_reduce_combine, topk_router_gpt) are supported on the current arch.

    These kernels shard K across the DRAM-bank-aligned matmul cores returned by
    get_optimal_dram_bank_to_logical_worker_assignment(), and hardcode a 12-bank layout:

      * moe_gpt_program_factory.cpp: `tiles_per_core_table[12] = {8,8,7,7,...}` sums to 90
        (= 2880/32) only at 12 banks; the host-side split in experts_throughput/weights.py
        (_FUSED_FULL_CORES / _FUSED_PAD_CORES) mirrors the same 12-entry layout.
      * moe_gpt combine_dm1.cpp assumes RING_CORES_PER_COMBINE_COL = 12/3 = 4.
      * topk_router_gpt_program_factory.cpp: TT_FATAL(num_cores >= 4*3).

    Wormhole exposes 12 DRAM banks, Blackhole only 8 (soc_descriptors: 12 vs 8 dram_views),
    so on Blackhole these kernels would silently cover 60 of 90 K-tiles, or TT_FATAL.
    Blackhole therefore runs the dense throughput flow instead, which is numerically
    equivalent and uses only generic ops.
    """
    return not is_blackhole()
