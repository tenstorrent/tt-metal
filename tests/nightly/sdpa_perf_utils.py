# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for SDPA performance testing across architectures.

Used by:
- tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py
"""

import math


# ============================================================================
# Architecture constants
# ============================================================================

ARCH_CONSTANTS = {
    "blackhole": {
        "clock_ghz": 1.35,
        "mm_flops_per_cycle_per_core": 2048,  # HiFi2: 4096 base / 2
    },
    "wormhole_b0": {
        "clock_ghz": 1.0,
        "mm_flops_per_cycle_per_core": 2048,  # HiFi2: 4096 base / 2
    },
}


# ============================================================================
# Tracy ops log processing
# ============================================================================


def post_process_ops_log(
    output_logs_subdir, float_columns=None, columns=None, sum_vals=True, op_name="", has_signposts=False
):
    """Process the ops log CSV and extract performance data."""
    from tracy.process_model_log import get_latest_ops_log_filename
    import pandas as pd

    filename = get_latest_ops_log_filename(output_logs_subdir)
    df = pd.read_csv(filename)

    if has_signposts:
        markers = df[df["OP TYPE"] == "signpost"]["OP CODE"]
        start = markers[markers == "start"].index[0]
        stop = markers[markers == "stop"].index[0]
        df = df.iloc[start + 1 : stop]
    if op_name != "":
        df = df[df["OP CODE"] == op_name]

    results = {}
    if float_columns:
        for col in float_columns:
            df_filtered = df[df[col] != "-"]
            if sum_vals:
                results[col] = df_filtered[col].astype(float).sum()
            else:
                results[col] = df_filtered[col].astype(float).to_numpy()
    if columns:
        for col in columns:
            df_filtered = df[df[col] != "-"]
            results[col] = df_filtered[col]
    else:
        results = df
    return results


# ============================================================================
# Core utilization
# ============================================================================


def compute_cores_used(seqlen, q_chunk_size, compute_cores, num_heads, batch_size=1):
    """
    Compute number of cores actually used for SDPA based on the parallelization scheme.

    Parallelization hierarchy (from sdpa_program_factory.cpp):
    1. batch_parallel_factor = min(B, num_cores)       — always 1 for B=1
    2. nh_parallel_factor = min(num_cores, num_heads)
    3. q_parallel_factor = min(num_cores / nh, q_num_chunks)

    Args:
        seqlen: Query sequence length.
        q_chunk_size: Q chunk size.
        compute_cores: Available compute cores.
        num_heads: Attention heads on this device.
        batch_size: Batch size (default 1).
    """
    q_num_chunks = math.ceil(seqlen / q_chunk_size)

    batch_parallel = min(batch_size, compute_cores)
    nh_parallel = min(compute_cores // batch_parallel, num_heads)
    q_parallel = min(compute_cores // (batch_parallel * nh_parallel), q_num_chunks)

    cores_used = batch_parallel * nh_parallel * q_parallel
    return cores_used


# ============================================================================
# Math utilization
# ============================================================================


def compute_sdpa_flops(sq, sk, d_q, d_v, num_heads, is_causal=False):
    """
    Compute FLOPs for SDPA operation.

    Matches sdpa_perf_model.cpp calculation:
    - Q*K matmul: 2 * d_q * Sq * Sk FLOPs
    - attn*V matmul: 2 * d_v * Sq * Sk FLOPs
    - Causal: divide by 2 (only half the attention matrix computed)

    Args:
        sq: Query sequence length.
        sk: Key/Value sequence length.
        d_q: Query/Key dimension.
        d_v: Value dimension.
        num_heads: Number of attention heads.
        is_causal: Whether causal masking is used (reduces FLOPs by half).
    """
    flops = 2 * sq * sk * (d_q + d_v) * num_heads
    if is_causal:
        flops //= 2
    return flops


def compute_math_utilization(
    local_seqlen,
    total_seqlen,
    d_q,
    d_v,
    num_heads_per_device,
    duration_ns,
    core_count,
    is_causal=False,
    arch="blackhole",
):
    """
    Compute math utilization as a percentage (0-100).

    Args:
        local_seqlen: Query sequence length.
        total_seqlen: Key/Value sequence length.
        d_q: Query/Key dimension.
        d_v: Value dimension.
        num_heads_per_device: Attention heads on this device.
        duration_ns: Measured device kernel duration in nanoseconds.
        core_count: Number of compute cores used.
        is_causal: Whether causal masking is used.
        arch: Architecture name ("blackhole" or "wormhole_b0").
    """
    constants = ARCH_CONSTANTS[arch]
    mm_flops = compute_sdpa_flops(local_seqlen, total_seqlen, d_q, d_v, num_heads_per_device, is_causal)

    cycles = duration_ns * constants["clock_ghz"]
    theoretical_flops = core_count * cycles * constants["mm_flops_per_cycle_per_core"]
    return (mm_flops / theoretical_flops) * 100 if theoretical_flops > 0 else 0
