#!/usr/bin/env python3
"""
Ring Iteration 0 Execution Trace for SDPA Kernel

Traces what each core processes (q_id, k_id, v_id) during ring_iter=0.
Based on mla_100k test configuration from:
  tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py

Output: CSV files where rows = cores (x,y), columns = timestamps.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import csv
import math
import os


@dataclass
class SDPAConfig:
    """Configuration matching mla_100k test parameters."""

    # Model params
    batch: int = 1
    nhq: int = 29  # Number of Q heads (non-Galaxy)
    nhk: int = 1  # Number of K heads (MLA: K is shared)
    nhv: int = 29  # Number of V heads

    # Sequence params
    seq_len: int = 3200
    q_chunk_size: int = 160
    k_chunk_size: int = 160

    # Grid params (Blackhole non-Galaxy)
    grid_cols: int = 11
    grid_rows: int = 10

    # Algorithm flags
    is_causal: bool = True
    is_balanced: bool = True
    enable_zigzag: bool = True

    @property
    def ccl_column(self) -> int:
        return self.grid_cols - 1

    @property
    def sdpa_cols(self) -> int:
        return self.ccl_column

    @property
    def num_cores(self) -> int:
        return self.sdpa_cols * self.grid_rows

    @property
    def num_q_chunks(self) -> int:
        return self.seq_len // self.q_chunk_size

    @property
    def num_k_chunks(self) -> int:
        return self.seq_len // self.k_chunk_size

    @property
    def half_sequence(self) -> int:
        """Boundary between light (< half) and heavy (>= half) Q chunks."""
        return self.num_q_chunks // 2


@dataclass
class QWork:
    """Single Q chunk work item."""

    batch: int
    head: int
    q_chunk: int  # Remapped (zigzag) index
    q_iter: int  # Original iteration index


@dataclass
class CoreWork:
    """Work assigned to a single core."""

    core_x: int
    core_y: int
    work_items: List[QWork]

    @property
    def core_idx(self) -> int:
        return self.core_y * 10 + self.core_x  # Column-major layout


def zigzag_remap(q_iter: int, num_q_chunks: int) -> int:
    """
    Remap iteration index to zigzag pattern.
    Even positions -> forward from start
    Odd positions -> backward from end
    """
    if q_iter % 2 == 0:
        return q_iter // 2
    else:
        return num_q_chunks - 1 - q_iter // 2


def is_light_q(q_chunk: int, half_sequence: int) -> bool:
    """Light Q chunks are in the first half of sequence (early original)."""
    return q_chunk < half_sequence


def compute_work_distribution(cfg: SDPAConfig) -> List[CoreWork]:
    """
    Distribute Q chunks across cores using zigzag balancing.
    Returns per-core work assignments.
    """
    cores = []
    for y in range(cfg.grid_rows):
        for x in range(cfg.sdpa_cols):
            cores.append(CoreWork(core_x=x, core_y=y, work_items=[]))

    # Total Q work items
    total_q_items = cfg.batch * cfg.nhq * cfg.num_q_chunks

    if cfg.enable_zigzag:
        # Pair-based distribution
        total_pairs = total_q_items // 2
        base_pairs_per_core = total_pairs // cfg.num_cores
        extra_pairs = total_pairs % cfg.num_cores

        # Assign pairs to cores
        item_idx = 0
        for core_idx in range(cfg.num_cores):
            num_pairs = base_pairs_per_core + (1 if core_idx < extra_pairs else 0)
            num_items = num_pairs * 2

            for _ in range(num_items):
                if item_idx >= total_q_items:
                    break

                # Flatten item_idx to (batch, head, q_iter)
                b = item_idx // (cfg.nhq * cfg.num_q_chunks)
                remainder = item_idx % (cfg.nhq * cfg.num_q_chunks)
                h = remainder // cfg.num_q_chunks
                q_iter = remainder % cfg.num_q_chunks

                # Apply zigzag remap
                q_chunk = zigzag_remap(q_iter, cfg.num_q_chunks)

                cores[core_idx].work_items.append(QWork(batch=b, head=h, q_chunk=q_chunk, q_iter=q_iter))
                item_idx += 1
    else:
        # Simple round-robin distribution
        for item_idx in range(total_q_items):
            core_idx = item_idx % cfg.num_cores
            b = item_idx // (cfg.nhq * cfg.num_q_chunks)
            remainder = item_idx % (cfg.nhq * cfg.num_q_chunks)
            h = remainder // cfg.num_q_chunks
            q_iter = remainder % cfg.num_q_chunks
            q_chunk = q_iter

            cores[core_idx].work_items.append(QWork(batch=b, head=h, q_chunk=q_chunk, q_iter=q_iter))

    return cores


def get_kv_range_for_q(
    q_chunk: int,
    cfg: SDPAConfig,
    ring_iter: int = 0,
) -> Tuple[int, int, bool]:
    """
    Returns (k_start, k_end, is_light) for a Q chunk during ring_iter 0.

    For ring_iter 0 (causal, balanced):
    - Light Q: only need first half of KV (k_chunk 0 to half-1)
    - Heavy Q: need up to causal boundary
    """
    half = cfg.half_sequence
    q_is_light = is_light_q(q_chunk, half)

    if q_is_light and ring_iter == 0:
        # Phase 3 optimization: light Q only needs first half of KV
        # Causal boundary within light region: q_chunk + 1
        k_end = min(q_chunk + 1, half)
    else:
        # Heavy Q: process up to causal boundary (q_chunk + 1 in local indexing)
        # For heavy Q, q_chunk >= half, so causal boundary = q_chunk + 1
        k_end = q_chunk + 1

    return 0, k_end, q_is_light


def format_id(head: int, chunk: int) -> str:
    """Format (head_id, chunk_id) tuple."""
    return f"({head},{chunk})"


def write_work_distribution_csv(cfg: SDPAConfig, filename: str = "ring_iter0_work_distribution.csv") -> None:
    """
    Write CSV showing per-core work distribution.
    Each row = one (core, q_iter) pair with all details.
    """
    cores = compute_work_distribution(cfg)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "core_idx",
                "x",
                "y",
                "q_iter",
                "head",
                "q_chunk",
                "light_heavy",
                "k_start",
                "k_end",
                "compute_ops",
                "discard_ops",
            ]
        )

        for core_idx, core in enumerate(cores):
            for q_iter, w in enumerate(core.work_items):
                k_start, k_end, is_light = get_kv_range_for_q(w.q_chunk, cfg, ring_iter=0)
                tag = "L" if is_light else "H"
                compute_ops = k_end - k_start
                discard_ops = cfg.num_k_chunks - compute_ops

                writer.writerow(
                    [
                        core_idx,
                        core.core_x,
                        core.core_y,
                        q_iter,
                        w.head,
                        w.q_chunk,
                        tag,
                        k_start,
                        k_end,
                        compute_ops,
                        discard_ops,
                    ]
                )

    print(f"Written: {filename}")


def write_flattened_timestamp_csv(cfg: SDPAConfig, filename: str = "ring_iter0_timestamps.csv") -> None:
    """
    Write CSV with columns as flattened timestamps (q_iter, k_chunk).
    Rows = cores with (x,y) coords.
    Each cell shows q_id, k_id, v_id being processed.
    """
    cores = compute_work_distribution(cfg)
    max_q_iters = max(len(c.work_items) for c in cores) if cores else 0

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Header row
        header = ["core_idx", "x", "y"]
        for q_i in range(max_q_iters):
            for k_i in range(cfg.num_k_chunks):
                header.append(f"q{q_i}_k{k_i:02d}")
        writer.writerow(header)

        # Data rows
        for core_idx, core in enumerate(cores):
            row = [core_idx, core.core_x, core.core_y]

            for q_i in range(max_q_iters):
                if q_i >= len(core.work_items):
                    for k_i in range(cfg.num_k_chunks):
                        row.append("-")
                    continue

                w = core.work_items[q_i]
                k_start, k_end, is_light = get_kv_range_for_q(w.q_chunk, cfg, ring_iter=0)

                for k_i in range(cfg.num_k_chunks):
                    if k_i < k_end:
                        # Format: Q<q_chunk>_K<k_chunk>_V<k_chunk>
                        # In MLA: K and V share the same indexing (both from compressed latent)
                        row.append(f"Q{w.q_chunk}_K{k_i}_V{k_i}")
                    else:
                        row.append("D")
            writer.writerow(row)

    print(f"Written: {filename}")


def write_compact_grid_csv(cfg: SDPAConfig, selected_q_iter: int = 0, filename: str = None) -> None:
    """
    Write CSV grid for a single q_iter showing compute vs discard across all cores.
    Columns = k_chunk, Rows = cores.
    'C' = compute, 'D' = discard, '-' = no work.
    """
    if filename is None:
        filename = f"ring_iter0_q{selected_q_iter}_grid.csv"

    cores = compute_work_distribution(cfg)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Header row
        header = ["core_idx", "x", "y", "q_chunk", "light_heavy"]
        for k in range(cfg.num_k_chunks):
            header.append(f"K{k:02d}")
        writer.writerow(header)

        # Data rows
        for core_idx, core in enumerate(cores):
            if selected_q_iter >= len(core.work_items):
                row = [core_idx, core.core_x, core.core_y, "-", "-"]
                row.extend(["-"] * cfg.num_k_chunks)
                writer.writerow(row)
                continue

            w = core.work_items[selected_q_iter]
            k_start, k_end, is_light = get_kv_range_for_q(w.q_chunk, cfg, ring_iter=0)
            tag = "L" if is_light else "H"

            row = [core_idx, core.core_x, core.core_y, w.q_chunk, tag]
            for k in range(cfg.num_k_chunks):
                if k < k_end:
                    row.append("C")
                else:
                    row.append("D")
            writer.writerow(row)

    print(f"Written: {filename}")


def write_summary_csv(cfg: SDPAConfig, filename: str = "ring_iter0_summary.csv") -> None:
    """Write summary statistics to CSV."""
    cores = compute_work_distribution(cfg)
    total_computes = 0
    total_discards = 0

    for core in cores:
        for w in core.work_items:
            k_start, k_end, is_light = get_kv_range_for_q(w.q_chunk, cfg, ring_iter=0)
            computes = k_end - k_start
            discards = cfg.num_k_chunks - computes
            total_computes += computes
            total_discards += discards

    total_ops = total_computes + total_discards

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["seq_len", cfg.seq_len])
        writer.writerow(["q_chunk_size", cfg.q_chunk_size])
        writer.writerow(["k_chunk_size", cfg.k_chunk_size])
        writer.writerow(["num_q_chunks", cfg.num_q_chunks])
        writer.writerow(["num_k_chunks", cfg.num_k_chunks])
        writer.writerow(["half_sequence", cfg.half_sequence])
        writer.writerow(["batch", cfg.batch])
        writer.writerow(["nhq", cfg.nhq])
        writer.writerow(["nhk", cfg.nhk])
        writer.writerow(["grid_cols", cfg.sdpa_cols])
        writer.writerow(["grid_rows", cfg.grid_rows])
        writer.writerow(["num_cores", cfg.num_cores])
        writer.writerow(["total_compute_ops", total_computes])
        writer.writerow(["total_discard_ops", total_discards])
        writer.writerow(["compute_fraction", f"{total_computes / total_ops * 100:.1f}%"])
        writer.writerow(["discard_fraction", f"{total_discards / total_ops * 100:.1f}%"])

    print(f"Written: {filename}")


def main():
    # mla_100k configuration
    cfg = SDPAConfig(
        batch=1,
        nhq=29,
        nhk=1,
        nhv=29,
        seq_len=3200,
        q_chunk_size=160,
        k_chunk_size=160,
        grid_cols=11,
        grid_rows=10,
        is_causal=True,
        is_balanced=True,
        enable_zigzag=True,
    )

    print("Ring Iteration 0 Execution Trace")
    print(f"Config: mla_100k (non-Galaxy Blackhole)")
    print(f"  seq_len={cfg.seq_len}, q_chunk_size={cfg.q_chunk_size}")
    print(f"  num_q_chunks={cfg.num_q_chunks}, half_sequence={cfg.half_sequence}")
    print(f"  grid={cfg.sdpa_cols}x{cfg.grid_rows}, num_cores={cfg.num_cores}")
    print()

    # Write all CSV files
    write_work_distribution_csv(cfg)
    write_compact_grid_csv(cfg, selected_q_iter=0)
    write_compact_grid_csv(cfg, selected_q_iter=1)
    write_flattened_timestamp_csv(cfg)
    write_summary_csv(cfg)

    # Print summary
    cores = compute_work_distribution(cfg)
    total_computes = 0
    total_discards = 0

    for core in cores:
        for w in core.work_items:
            k_start, k_end, is_light = get_kv_range_for_q(w.q_chunk, cfg, ring_iter=0)
            total_computes += k_end - k_start
            total_discards += cfg.num_k_chunks - (k_end - k_start)

    total_ops = total_computes + total_discards
    print(f"\nSummary:")
    print(f"  Compute: {total_computes} ops ({total_computes / total_ops * 100:.1f}%)")
    print(f"  Discard: {total_discards} ops ({total_discards / total_ops * 100:.1f}%)")


if __name__ == "__main__":
    main()
