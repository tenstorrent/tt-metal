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
) -> Tuple[int, int, bool, bool]:
    """
    Returns (k_start, k_end, is_light, is_reverse) for a Q chunk during ring_iter 0.

    New algorithm (no discards):
    - Light Q: process K in forward order (K0, K1, ..., K_causal)
    - Heavy Q: process K in reverse order (K_causal, K_causal-1, ..., K0)

    This eliminates discards by interleaving light/heavy Q with opposite K directions.
    """
    half = cfg.half_sequence
    q_is_light = is_light_q(q_chunk, half)

    # Causal boundary: can only attend to K[0:q_chunk+1]
    k_end = q_chunk + 1

    # Light Q: forward K order, Heavy Q: reverse K order
    is_reverse = not q_is_light

    return 0, k_end, q_is_light, is_reverse


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
                "direction",
                "k_start",
                "k_end",
                "compute_ops",
            ]
        )

        for core_idx, core in enumerate(cores):
            for q_iter, w in enumerate(core.work_items):
                k_start, k_end, is_light, is_reverse = get_kv_range_for_q(w.q_chunk, cfg, ring_iter=0)
                tag = "L" if is_light else "H"
                direction = "reverse" if is_reverse else "forward"
                compute_ops = k_end - k_start

                writer.writerow(
                    [
                        core_idx,
                        core.core_x,
                        core.core_y,
                        q_iter,
                        w.head,
                        w.q_chunk,
                        tag,
                        direction,
                        k_start,
                        k_end,
                        compute_ops,
                    ]
                )

    print(f"Written: {filename}")


def write_flattened_timestamp_csv(cfg: SDPAConfig, filename: str = "ring_iter0_timestamps.csv") -> None:
    """
    Write CSV with columns as flattened timestamps (t0, t1, t2, ...).
    Rows = cores with (x,y) coords.
    Each cell shows q_id, k_id, v_id being processed.

    No padding - each column is a real operation in time order.
    All cells contain actual compute operations (no '-' or 'D').

    Bottom row shows distinct K_V pairs per column (should be at most 2).
    """
    cores = compute_work_distribution(cfg)

    # First pass: compute total operations per core and build ops matrix
    max_ops = 0
    all_ops = []  # List of lists: all_ops[core_idx] = [op0, op1, ...]

    for core in cores:
        ops = []
        for w in core.work_items:
            k_start, k_end, is_light, is_reverse = get_kv_range_for_q(w.q_chunk, cfg, ring_iter=0)

            # Generate K indices in the correct order
            if is_reverse:
                k_indices = list(range(k_end - 1, k_start - 1, -1))
            else:
                k_indices = list(range(k_start, k_end))

            for k_i in k_indices:
                ops.append((w.q_chunk, k_i))  # Store as tuple for analysis

        all_ops.append(ops)
        max_ops = max(max_ops, len(ops))

    # Compute distinct K_V per column
    distinct_kv_per_col = []
    for t in range(max_ops):
        kv_set = set()
        for core_ops in all_ops:
            if t < len(core_ops):
                q_chunk, k_i = core_ops[t]
                kv_set.add(f"K{k_i}_V{k_i}")
        distinct_kv_per_col.append(sorted(kv_set))

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Header row - flat timeline
        header = ["core_idx", "x", "y"]
        for t in range(max_ops):
            header.append(f"t{t:03d}")
        writer.writerow(header)

        # Data rows
        for core_idx, core in enumerate(cores):
            row = [core_idx, core.core_x, core.core_y]

            for t in range(max_ops):
                if t < len(all_ops[core_idx]):
                    q_chunk, k_i = all_ops[core_idx][t]
                    row.append(f"Q{q_chunk}_K{k_i}_V{k_i}")
                else:
                    row.append("")  # Core finished early

            writer.writerow(row)

        # Bottom row: distinct K_V per column
        summary_row = ["DISTINCT_KV", "", ""]
        for t in range(max_ops):
            kv_list = distinct_kv_per_col[t]
            summary_row.append(" | ".join(kv_list))
        writer.writerow(summary_row)

        # Ideal K_V row: shows the expected pattern with fake index 20
        # Pattern repeats every 21 timestamps:
        # t=0: K0_V0 | K20_V20, t=1: K1_V1 | K19_V19, ..., t=20: K20_V20 | K0_V0, t=21: repeats
        ideal_row = ["IDEAL_KV", "", ""]
        cycle_len = 21  # 0 to 20 inclusive
        for t in range(max_ops):
            t_in_cycle = t % cycle_len
            k_fwd = t_in_cycle
            k_bwd = 20 - t_in_cycle
            ideal_row.append(f"K{k_fwd}_V{k_fwd} | K{k_bwd}_V{k_bwd}")
        writer.writerow(ideal_row)

    # Print stats about distinct K_V counts
    kv_counts = [len(kv) for kv in distinct_kv_per_col]
    max_distinct = max(kv_counts) if kv_counts else 0
    print(f"Written: {filename}")
    print(f"  Max distinct K_V per column: {max_distinct}")


def write_compact_grid_csv(cfg: SDPAConfig, selected_q_iter: int = 0, filename: str = None) -> None:
    """
    Write CSV grid for a single q_iter showing K processing order across all cores.
    Columns = k_chunk, Rows = cores.
    Cell value = sequence number (1, 2, 3, ...) showing processing order.
    '-' = not processed (outside causal boundary).
    """
    if filename is None:
        filename = f"ring_iter0_q{selected_q_iter}_grid.csv"

    cores = compute_work_distribution(cfg)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Header row
        header = ["core_idx", "x", "y", "q_chunk", "light_heavy", "direction"]
        for k in range(cfg.num_k_chunks):
            header.append(f"K{k:02d}")
        writer.writerow(header)

        # Data rows
        for core_idx, core in enumerate(cores):
            if selected_q_iter >= len(core.work_items):
                row = [core_idx, core.core_x, core.core_y, "-", "-", "-"]
                row.extend(["-"] * cfg.num_k_chunks)
                writer.writerow(row)
                continue

            w = core.work_items[selected_q_iter]
            k_start, k_end, is_light, is_reverse = get_kv_range_for_q(w.q_chunk, cfg, ring_iter=0)
            tag = "L" if is_light else "H"
            direction = "<-" if is_reverse else "->"

            # Build sequence numbers for each K position
            if is_reverse:
                # Heavy Q: reverse order (k_end-1 down to k_start)
                k_indices = list(range(k_end - 1, k_start - 1, -1))
            else:
                # Light Q: forward order (k_start to k_end-1)
                k_indices = list(range(k_start, k_end))

            # Map K position to sequence number
            k_to_seq = {k_idx: seq + 1 for seq, k_idx in enumerate(k_indices)}

            row = [core_idx, core.core_x, core.core_y, w.q_chunk, tag, direction]
            for k in range(cfg.num_k_chunks):
                if k in k_to_seq:
                    row.append(k_to_seq[k])
                else:
                    row.append("-")
            writer.writerow(row)

    print(f"Written: {filename}")


def write_summary_csv(cfg: SDPAConfig, filename: str = "ring_iter0_summary.csv") -> None:
    """Write summary statistics to CSV."""
    cores = compute_work_distribution(cfg)
    total_computes = 0

    for core in cores:
        for w in core.work_items:
            k_start, k_end, is_light, is_reverse = get_kv_range_for_q(w.q_chunk, cfg, ring_iter=0)
            computes = k_end - k_start
            total_computes += computes

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
        writer.writerow(["algorithm", "reverse_k_for_heavy_q"])
        writer.writerow(["discard_ops", 0])

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

    print("Ring Iteration 0 Execution Trace (Reverse-K Algorithm)")
    print("Algorithm: Light Q -> forward K, Heavy Q -> reverse K")
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

    for core in cores:
        for w in core.work_items:
            k_start, k_end, is_light, is_reverse = get_kv_range_for_q(w.q_chunk, cfg, ring_iter=0)
            total_computes += k_end - k_start

    print(f"\nSummary:")
    print(f"  Total compute ops: {total_computes}")
    print(f"  Discards: 0 (all K/V reads are used)")


if __name__ == "__main__":
    main()
