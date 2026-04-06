# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for SDPA performance testing across architectures.

Used by:
- tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py
- tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py
- tests/nightly/t3000/ccl/test_ring_joint_attention.py
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


def compute_cores_used(seqlen, q_chunk_size, num_cores, num_heads, ring_size=1):
    """
    Compute number of cores actually used based on parallelization scheme.

    Parallelization hierarchy (from sdpa_program_factory.cpp):
    1. batch_parallel_factor = min(B, num_cores)       — always 1 for B=1
    2. nh_parallel_factor = min(num_cores, num_heads)
    3. q_parallel_factor = min(num_cores / nh, q_num_chunks)

    Args:
        seqlen: Total (global) sequence length.
        q_chunk_size: Q chunk size.
        num_cores: Available compute cores.
        num_heads: Attention heads on this device.
        ring_size: Sequence-parallel ring size (1 for single-chip).
    """
    local_seq_len = seqlen // ring_size
    q_num_chunks = math.ceil(local_seq_len / q_chunk_size)

    nh_parallel = min(num_cores, num_heads)
    q_parallel = min(num_cores // nh_parallel, q_num_chunks)

    return nh_parallel * q_parallel


# ============================================================================
# Math utilization
# ============================================================================


def compute_math_utilization(
    local_seqlen, total_seqlen, head_dim, num_heads, duration_ns, core_count, arch="blackhole"
):
    """
    Compute math utilization as a percentage (0-100).

    FLOPs model: 4 * local_seq * total_seq * head_dim * num_heads
      (Q@K^T forward + backward = 2 * local * total * d * nh,
       P@V  forward + backward = 2 * local * total * d * nh)

    For single-chip, local_seqlen == total_seqlen.

    Args:
        local_seqlen: Per-device sequence length (= total_seqlen / ring_size).
        total_seqlen: Global sequence length across all ring devices.
        head_dim: Head dimension (d).
        num_heads: Attention heads on this device.
        duration_ns: Measured device kernel duration in nanoseconds.
        core_count: Number of compute cores used.
        arch: Architecture name ("blackhole" or "wormhole_b0").
    """
    constants = ARCH_CONSTANTS[arch]
    mm_flops = 4 * local_seqlen * total_seqlen * head_dim * num_heads
    cycles = duration_ns * constants["clock_ghz"]
    theoretical_flops = core_count * cycles * constants["mm_flops_per_cycle_per_core"]
    return (mm_flops / theoretical_flops) * 100 if theoretical_flops > 0 else 0
