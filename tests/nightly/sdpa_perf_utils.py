# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for SDPA performance testing across architectures.

Used by:
- tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py
- tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py
- tests/nightly/t3000/ccl/test_ring_joint_attention.py
"""

import math
from dataclasses import dataclass, field
from typing import ClassVar, Tuple, List

import torch


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
# Hardware configuration
# ============================================================================


def _detect_devices_without_opening():
    """
    Detect the number of available TT devices WITHOUT opening them.
    Uses /dev/tenstorrent/* device files to avoid holding device locks.
    This is required for performance tests that use run_device_profiler().
    """
    import glob
    import os

    # Device nodes are named numerically (/dev/tenstorrent/0, .../1, ...). Filter to those
    # so sibling entries like the `by-id/` directory are not miscounted as devices.
    device_files = [p for p in glob.glob("/dev/tenstorrent/*") if os.path.basename(p).isdigit()]
    return len(device_files)


@dataclass
class MeshConfig:
    """Mesh configuration detected at runtime.

    Use MeshConfig.detect() to create an instance with all values resolved.
    """

    # Galaxy hardware constants. A Galaxy is 32 devices on either architecture, so the device count
    # alone does not fix the grid -- Blackhole is 12x10 and Wormhole 8x9. Keyed separately because
    # every core count downstream (`sdpa_cores`, and so the chunk sizes swept against it) is derived
    # from this: 11x10 = 110 SDPA cores on Blackhole against 7x9 = 63 on Wormhole.
    GALAXY_DEVICE_COUNT: ClassVar[int] = 32
    GALAXY_GRID: ClassVar[Tuple[int, int]] = (12, 10)  # (cols, rows), Blackhole
    WH_GALAXY_GRID: ClassVar[Tuple[int, int]] = (8, 9)  # (cols, rows), Wormhole
    GALAXY_TP_SIZE: ClassVar[int] = 4
    GALAXY_SP_SIZE: ClassVar[int] = 8

    # Non-Galaxy hardware constants
    # 11x10 full grid -> 10x10 (100 cores) for SDPA after reserving the last column for CCL.
    # NOTE: this is a Blackhole quiet-box grid and is returned for non-Galaxy hosts of either
    # architecture. It has not been checked against a Wormhole T3000, whose compute grid is smaller;
    # left as-is rather than guessed at, since the Galaxy path above is the one measured here.
    NON_GALAXY_GRID: ClassVar[Tuple[int, int]] = (11, 10)  # (cols, rows)

    # Instance fields (set by detect())
    is_galaxy: bool
    num_devices: int
    sp_size: int  # Sequence parallel size (devices per ring)
    tp_size: int  # Tensor parallel size (number of rings)
    grid_cols: int
    grid_rows: int

    # Computed fields (set by __post_init__)
    arch_type: str = field(init=False)
    total_cores: int = field(init=False)
    ccl_column: int = field(init=False)
    sdpa_cols: int = field(init=False)
    sdpa_cores: int = field(init=False)

    def __post_init__(self):
        self.arch_type = "galaxy_4x8" if self.is_galaxy else f"single_ring_{self.num_devices}x1"
        self.total_cores = self.grid_cols * self.grid_rows
        self.ccl_column = self.grid_cols - 1
        self.sdpa_cols = self.ccl_column
        self.sdpa_cores = self.sdpa_cols * self.grid_rows

    @classmethod
    def detect(cls) -> "MeshConfig":
        """Detect hardware and create config with all values resolved."""
        num_devices = _detect_devices_without_opening()
        is_galaxy = num_devices == cls.GALAXY_DEVICE_COUNT
        if is_galaxy:
            # `ttnn.get_arch_name()` reads the cluster descriptor and does not open or lock a
            # device, which is what this whole detection path exists to avoid -- verified by
            # calling it while all 32 devices were held by another profiled run.
            import ttnn

            grid = cls.GALAXY_GRID if "blackhole" in ttnn.get_arch_name() else cls.WH_GALAXY_GRID
        else:
            grid = cls.NON_GALAXY_GRID
        return cls(
            is_galaxy=is_galaxy,
            num_devices=num_devices,
            sp_size=cls.GALAXY_SP_SIZE if is_galaxy else num_devices,
            tp_size=cls.GALAXY_TP_SIZE if is_galaxy else 1,
            grid_cols=grid[0],
            grid_rows=grid[1],
        )


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


def compute_cores_used(seqlen, q_chunk_size, compute_cores, num_heads, ring_size=1, batch_size=1):
    """
    Compute number of cores actually used for ring joint attention based on parallelization scheme.

    Parallelization hierarchy (from sdpa_program_factory.cpp):
    1. batch_parallel_factor = min(B, num_cores)       — always 1 for B=1
    2. nh_parallel_factor = min(num_cores, num_heads)
    3. q_parallel_factor = min(num_cores / nh, q_num_chunks)

    Args:
        seqlen: Total (global) sequence length.
        q_chunk_size: Q chunk size.
        compute_cores: Available compute cores.
        num_heads: Attention heads on this device.
        ring_size: Sequence-parallel ring size (1 for single-chip).
        batch_size: Batch size (default 1).
    """
    local_seq_len = seqlen // ring_size
    q_num_chunks = math.ceil(local_seq_len / q_chunk_size)

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
        local_seqlen: Per-device sequence length (= total_seqlen / ring_size).
        total_seqlen: Global sequence length across all ring devices.
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


# ============================================================================
# Cross-device balancing utilities
# ============================================================================


def create_balanced_chunk_order(sp_size: int) -> List[int]:
    """Create balanced chunk order for sequence reordering.

    For sp_size=4, creates 2*4=8 chunks with order: [0, 7, 1, 6, 2, 5, 3, 4]
    This interleaves chunks from start and end to balance causal attention workload
    across devices (early chunks have less work, late chunks have more).
    """
    num_chunks = 2 * sp_size
    balanced_order = []

    left = 0
    right = num_chunks - 1

    for i in range(num_chunks):
        if i % 2 == 0:
            balanced_order.append(left)
            left += 1
        else:
            balanced_order.append(right)
            right -= 1

    return balanced_order


def reorder_tensor_chunks(tensor: torch.Tensor, chunk_order: List[int], seq_dim: int = 2) -> torch.Tensor:
    """Reorder tensor chunks along sequence dimension according to chunk_order.

    Used to prepare Q/K/V tensors for balanced causal attention before sending to device.
    """
    seq_len = tensor.shape[seq_dim]
    num_chunks = len(chunk_order)
    chunk_size = seq_len // num_chunks

    # Split into chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        if seq_dim == 2:
            chunks.append(tensor[:, :, start:end, :])
        else:
            raise NotImplementedError(f"Reordering for seq_dim={seq_dim} not implemented")

    # Reorder chunks according to chunk_order
    reordered_chunks = [chunks[i] for i in chunk_order]

    # Concatenate reordered chunks
    return torch.cat(reordered_chunks, dim=seq_dim)


def reverse_reorder_tensor_chunks(tensor: torch.Tensor, chunk_order: List[int], seq_dim: int = 2) -> torch.Tensor:
    """Reverse the chunk reordering to restore original order.

    Used to convert device output back to original sequence order for comparison.
    """
    # Create inverse permutation
    inverse_order = [0] * len(chunk_order)
    for new_pos, orig_pos in enumerate(chunk_order):
        inverse_order[orig_pos] = new_pos

    return reorder_tensor_chunks(tensor, inverse_order, seq_dim)
