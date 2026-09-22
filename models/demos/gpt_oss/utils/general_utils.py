# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
General utilities for the GPT-OSS demo.
"""

import os

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

    Set GPT_OSS_LOW_LATENCY_EXPERTS=1 to force the sparse_matmul (low-latency) expert path
    instead, at any batch. On Blackhole the throughput path runs its *dense* flow, which
    computes every token through every local expert and brackets the CCL ops with
    tilize/untilize; the sparse path does neither. Which one wins at a given batch is a
    measurement, so this knob exists to make the A/B cheap.
    """
    if os.getenv("GPT_OSS_LOW_LATENCY_EXPERTS") == "1":
        return False
    return True


def decode_expert_parallel(mesh_device, users_row_sharded, use_throughput_experts):
    """Expert-parallel degree to use for decode on this mesh.

    Expert parallelism splits each token's active experts across the rows, so it is only
    coherent when every row holds the *same* tokens. The throughput path satisfies that by
    re-gathering tokens across rows with all_to_all_dispatch before the experts run, so it
    can use EP = rows. The sparse (low-latency) path has no such gather: when users are
    row-sharded each row holds different users, so the rows are data-parallel and EP must be
    1 -- otherwise the expert-parallel all_reduce would sum across different users.

    MeshConfig then derives dp = devices / (tp * ep), e.g. tp=8, ep=1 -> dp=4 on a 4x8 mesh,
    and skips the EP-vs-axis check entirely for ep == 1 (see MeshConfig._validate_config).
    """
    if mesh_device.shape[0] == 1:
        return 1
    if users_row_sharded and not use_throughput_experts:
        return 1
    return mesh_device.shape[0]


def fused_moe_kernels_supported_on_arch():
    """Whether the fused MoE kernels (moe_gpt, all_to_all_dispatch_metadata,
    selective_reduce_combine, topk_router_gpt) are supported on the current arch.

    UPDATE: moe_gpt's own 12-bank hardcoding (tiles_per_core_table[12], combine_dm1.cpp's
    RING_CORES_PER_COMBINE_COL) has been generalized -- moe_gpt_ring_common.h now derives its
    per-core tile distribution and combine grid from the live DRAM bank count, reproducing the
    original Wormhole (12-bank) numbers exactly via a compile-time-verified passthrough
    (see MoeGptRingConfig::kLegacyWh12 and the static_asserts in moe_gpt_ring_common.h) and
    using a general Euclidean-rhythm formula for any other ring size. This return value is
    intentionally left `not is_blackhole()` regardless: selecting moe_gpt on Blackhole is a
    separate model-integration decision (wiring mlp.py/experts_throughput/weights.py, which
    still has its own independent 12-bank hardcoding in _FUSED_FULL_CORES/_FUSED_PAD_CORES,
    not yet generalized) and end-to-end device validation on Blackhole is still pending --
    see models/demos/gpt_oss's memory notes for what's been proven (the C++ compiles and its
    Wormhole-equivalence static_asserts pass) versus what hasn't (a real BH numerical run;
    the existing tests/ttnn/nightly/.../test_moe_gpt_e2e.py also hits an unrelated hardcoded
    num_links=4 ethernet-fabric assumption on this topology before moe_gpt is even reached).

    topk_router_gpt_program_factory.cpp's TT_FATAL(num_cores >= 4*3) is UNCHANGED -- it is not
    on moe_gpt's call path and was intentionally left out of this pass (a harvested-Blackhole
    ring of 7 cores doesn't divide evenly into its fixed 4-groups-of-3 topology, which needs a
    real algorithm decision, not just a parametrization).

    Wormhole exposes 12 DRAM banks, Blackhole only 8 (soc_descriptors: 12 vs 8 dram_views).
    Blackhole currently runs the dense throughput flow (or ttnn.experimental.moe_compute,
    opt-in via GPT_OSS_MOE_COMPUTE=1), both numerically equivalent and already validated
    end-to-end on Blackhole this session.
    """
    return not is_blackhole()
