# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
The MoE forward path was:
    gate (moe_grouped_topk → uint16 indices)
      → to_layout(TILE) → typecast(int32) → to_layout(ROW_MAJOR)
      → dispatch (expects int32 indices)

The typecast op produced byte-identical int32 outputs for byte-identical uint16
inputs when the profiler was OFF, but produced a permuted 24-token cluster
(~0.07% of elements) when the profiler was ON. The shuffled indices fed bad
routing into dispatch and combine, eventually deadlocking the fabric router.

This test runs the typecast once and compares the result to a torch
reference. The right shape + representative values for whoever owns the
typecast kernel to extend with a profiler-enabled variant.

Shape / values are taken straight from the failing config:
    indices.shape   = (1, 1, 3840, 8)   # one chip's view (sharded)
    indices.dtype   = uint16
    values          = top-k expert IDs in [0, 256)

Run with:
    pytest tests/ttnn/unit_tests/operations/eltwise/test_typecast_uint16_int32.py
    PASSED expected

    python -m tracy -r -p -v -m pytest tests/ttnn/unit_tests/operations/eltwise/test_typecast_uint16_int32.py
    FAILED expected

"""

import torch
import ttnn


def _build_indices_tensor(device, seed=0):
    """Generate a uint16 tensor with the same shape + value distribution as the
    deepseek MoE gate output that exposed the bug."""
    torch.manual_seed(seed)
    # Shape: (1, 1, 3840, 8) — single chip's view of the sharded indices.
    # 3840 = isl_7k68 / tp_factor = 7680 / 2.
    # 8 = num_experts_per_tok (top-k).
    shape = (1, 1, 3840, 8)
    # Value range matches actual gate output: expert IDs in [0, 256).
    values = torch.randint(0, 256, shape, dtype=torch.int32).to(torch.int16)
    return ttnn.from_torch(
        values,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _typecast_uint16_to_int32(tt_indices):
    """The exact chain that tt_moe.py uses"""
    x = ttnn.to_layout(tt_indices, ttnn.TILE_LAYOUT)
    x = ttnn.typecast(x, ttnn.int32)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    return x


def test_typecast_uint16_int32_determinism(device):
    """Run uint16→int32 typecast once and assert the output matches the torch
    reference exactly. Without the profiler enabled this passes; with the
    profiler enabled the original deepseek MoE workflow produced shuffled
    int32 values, breaking dispatch."""
    tt_indices = _build_indices_tensor(device)
    indices_host = ttnn.to_torch(tt_indices)

    # Reference: torch.to(int32) is trivially deterministic.
    ref_int32 = indices_host.to(torch.int32)

    out_tt = _typecast_uint16_to_int32(tt_indices)
    out_host = ttnn.to_torch(out_tt)
    ttnn.deallocate(out_tt)

    if not torch.equal(out_host, ref_int32):
        import os
        from collections import Counter

        mismatch = out_host != ref_int32
        n_diff = int(mismatch.sum().item())
        diff_idx = mismatch.nonzero(as_tuple=False)  # shape (n_diff, 4): (b, c, row, col)

        TILE_H, TILE_W = 32, 32
        rows = diff_idx[:, 2]
        cols = diff_idx[:, 3]
        tile_rows = (rows // TILE_H).tolist()
        tile_cols = (cols // TILE_W).tolist()
        row_in_tile = (rows % TILE_H).tolist()
        col_in_tile = (cols % TILE_W).tolist()

        # Histograms — expose tile / face structure if any.
        tile_pos_hist = Counter(zip(tile_rows, tile_cols))
        row_in_tile_hist = Counter(row_in_tile)
        col_in_tile_hist = Counter(col_in_tile)
        unique_rows = sorted(set(rows.tolist()))

        # Group consecutive rows into clusters so "tokens 3528..3551" pattern is
        # obvious instead of buried in 24 individual row numbers.
        clusters = []
        if unique_rows:
            cur_start = unique_rows[0]
            cur_end = unique_rows[0]
            for r in unique_rows[1:]:
                if r == cur_end + 1:
                    cur_end = r
                else:
                    clusters.append((cur_start, cur_end))
                    cur_start = r
                    cur_end = r
            clusters.append((cur_start, cur_end))

        # Persist every diff to a file so nothing is truncated by stdout limits.
        dump_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "typecast_44928_diffs.txt",
        )
        diff_positions_all = diff_idx.tolist()
        with open(dump_path, "w") as f:
            f.write(f"# total diffs: {n_diff}/{out_host.numel()}\n")
            f.write(f"# unique rows: {len(unique_rows)} → clusters: {clusters}\n")
            f.write("# (b, c, row, col, ttnn_val, torch_val, tile_row, tile_col, row_in_tile, col_in_tile)\n")
            for pos in diff_positions_all:
                b, c, r, col = pos
                f.write(
                    f"{b} {c} {r} {col} "
                    f"{int(out_host[b, c, r, col].item())} {int(ref_int32[b, c, r, col].item())} "
                    f"{r // TILE_H} {col // TILE_W} {r % TILE_H} {col % TILE_W}\n"
                )

        # Build a short, pattern-focused assertion message.
        lines = [
            f"typecast disagrees with torch.to(int32) reference: "
            f"{n_diff}/{out_host.numel()} elements differ ({100 * n_diff / out_host.numel():.4f}%)",
            f"unique mismatching rows ({len(unique_rows)}): {unique_rows[:64]}{' ...' if len(unique_rows) > 64 else ''}",
            f"contiguous row clusters: {clusters}",
            f"row-in-tile histogram (which 0..31 row within a 32-row tile): " f"{sorted(row_in_tile_hist.items())}",
            f"col-in-tile histogram: {sorted(col_in_tile_hist.items())}",
            f"(tile_row, tile_col) histogram (top 20 by count): " f"{tile_pos_hist.most_common(20)}",
            f"first 300 divergences (position → ttnn vs torch):",
        ]
        for pos in diff_positions_all[:300]:
            b, c, r, col = pos
            ttnn_val = int(out_host[b, c, r, col].item())
            torch_val = int(ref_int32[b, c, r, col].item())
            lines.append(
                f"  ({b},{c},{r},{col}) tile=({r // TILE_H},{col // TILE_W}) "
                f"inTile=({r % TILE_H},{col % TILE_W}): ttnn={ttnn_val} torch={torch_val}"
            )
        lines.append(f"full per-diff dump written to: {dump_path}")
        raise AssertionError("\n".join(lines))
