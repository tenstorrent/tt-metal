# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The lane engine's decode frame: the host shapes of one decode step and what each shards to per column.

``prepare_decode_inputs_host`` builds the step for the lane-major batch (``lanes x rows_per_lane`` rows): the hidden
rows shard over the lane axis on dim 2, the RoPE positions (``[1, batch]`` uint32, the embedding lookup's layout) on
dim 1, the cache positions (``[batch]`` int32: the paged cache update and the decode SDPA read their batch from
dim 0) and the page table (``[batch, blocks]`` int32) on dim 0. A unit that hands the layer the frame must shard the
same shapes: a 2-D ``[1, rows]`` tensor as the cache positions reads as ONE update index for a 32-row column
("Number of update_idxs (1) should match batch size (32)", window 5, 2026-10-24). Leaf module, host-only.
"""


def lane_decode_frame_shapes(lanes, rows_per_lane, blocks_per_user, hidden):
    """``name -> (host shape, per-column shape, lane-shard dim)`` of the decode frame's tensors."""
    batch = int(lanes) * int(rows_per_lane)
    return {
        "hidden": ((1, 1, batch, int(hidden)), (1, 1, int(rows_per_lane), int(hidden)), 2),
        "rope_positions": ((1, batch), (1, int(rows_per_lane)), 1),
        "cache_positions": ((batch,), (int(rows_per_lane),), 0),
        "page_table": ((batch, int(blocks_per_user)), (int(rows_per_lane), int(blocks_per_user)), 0),
    }


def check_lane_decode_frame(lanes, rows_per_lane, **host_shapes):
    """Raise ValueError when a host tensor of the frame (``hidden``, ``rope_positions``, ``cache_positions``,
    ``page_table``: the shapes handed to ``from_torch``) would not shard to ``rows_per_lane`` rows per column, or
    has the wrong rank for the op that reads it. Returns the per-column shapes."""
    expected = lane_decode_frame_shapes(lanes, rows_per_lane, host_shapes.get("page_table", (0, 0))[-1], 0)
    out = {}
    for name, shape in host_shapes.items():
        if name not in expected:
            raise ValueError(f"unknown decode-frame tensor {name!r}")
        host, _column, dim = expected[name]
        shape = tuple(int(x) for x in shape)
        if len(shape) != len(host):
            raise ValueError(
                f"{name}: rank {len(shape)} {shape}, the frame needs rank {len(host)} "
                f"(the lane shard runs over dim {dim}; the paged cache update reads its batch from dim 0)"
            )
        if shape[dim] != host[dim]:
            raise ValueError(
                f"{name}: {shape[dim]} rows on the lane-shard dim {dim}, the frame has {host[dim]} "
                f"({lanes} lanes x {rows_per_lane} rows)"
            )
        column = list(shape)
        column[dim] = int(rows_per_lane)
        out[name] = tuple(column)
    rows = {name: shape[expected[name][2]] for name, shape in out.items()}
    if len(set(rows.values())) > 1:
        raise ValueError(f"the frame's tensors disagree on rows per column: {rows}")
    return out
