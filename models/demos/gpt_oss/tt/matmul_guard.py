# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Guard against the sparse/1D matmul reload defect described in PR #51514.

The compute kernel bmm_large_block_zm_fused_bias_activation.cpp splits the
reduction over K into chunks when in0_block_w does not cover the whole depth.
It spills the running total to the cb_intermed0 circular buffer in L1 after
each chunk, then reloads it with reload_from_cb_to_dst before the next chunk.

That reload restores only the first tile of a multi-tile subblock correctly.
Later tiles are double counted. With out_subblock_w = 2, every second output
tile is wrong (45 of 90 for gpt-oss gate_up). With out_subblock_w = 3, two of
every three are wrong (60 of 90). Nothing crashes and nothing warns: the wrong
values feed the next layer, grow, and reach infinity a few decode steps later,
so the model writes nonsense.

The defect needs all three of these at the same time:

  1. out_subblock_w > 1        the subblock spans more than one tile, so a
                               later tile exists for the reload to corrupt
  2. num_blocks_w_dim > 1      per_core_N / out_block_w; the core loops over
                               more than one w-block
  3. num_blocks_inner_dim > 1  Kt / in0_block_w; K is split into chunks, so
                               the spill and reload path is active

Remove any one condition and the result is correct. Two ways to do that:

  * set in0_block_w equal to the reduction depth in tiles (Kt), which makes
    num_blocks_inner_dim = 1, so spill stays false and the reload never runs
  * set out_block_w equal to per_core_N, which makes num_blocks_w_dim = 1

A second and separate defect: the destination register file holds 8 tiles, but
only 4 when fp32_dest_acc_en is set. out_subblock_h * out_subblock_w above that
limit also corrupts the output, so that bound is checked here as well.
"""

import math


def check_matmul_program_config(
    name: str,
    Kt: int,
    in0_block_w: int,
    per_core_N: int,
    out_block_w: int,
    out_subblock_w: int,
    out_subblock_h: int = 1,
    fp32_dest_acc_en: bool = False,
) -> None:
    """Raise ValueError if this program config hits the PR #51514 defect.

    All arguments are in tiles. Kt is the reduction depth, so K / 32.
    """
    if in0_block_w <= 0 or out_block_w <= 0 or out_subblock_w <= 0:
        raise ValueError(
            f"{name}: in0_block_w ({in0_block_w}), out_block_w ({out_block_w}) and "
            f"out_subblock_w ({out_subblock_w}) must all be at least 1"
        )

    num_blocks_inner_dim = math.ceil(Kt / in0_block_w)
    num_blocks_w_dim = math.ceil(per_core_N / out_block_w)

    if out_subblock_w > 1 and num_blocks_w_dim > 1 and num_blocks_inner_dim > 1:
        raise ValueError(
            f"{name}: this matmul program config hits the reload defect in "
            f"PR #51514 and would silently produce wrong values.\n"
            f"  out_subblock_w       = {out_subblock_w} (must be 1, or remove one condition below)\n"
            f"  num_blocks_w_dim     = {num_blocks_w_dim} (per_core_N {per_core_N} / out_block_w {out_block_w})\n"
            f"  num_blocks_inner_dim = {num_blocks_inner_dim} (Kt {Kt} / in0_block_w {in0_block_w})\n"
            f"All three are above 1 at the same time, so the running total is "
            f"spilled to L1 and reloaded, and the reload restores only the first "
            f"tile of each subblock. Every tile at an index that is not a multiple "
            f"of out_subblock_w gets the wrong value.\n"
            f"Fix by any one of:\n"
            f"  set in0_block_w = {Kt} (the full reduction depth) so no spill happens\n"
            f"  set out_block_w = {per_core_N} (per_core_N) so the w loop runs once\n"
            f"  set out_subblock_w = 1"
        )

    dest_tiles = 4 if fp32_dest_acc_en else 8
    if out_subblock_h * out_subblock_w > dest_tiles:
        raise ValueError(
            f"{name}: out_subblock_h ({out_subblock_h}) times out_subblock_w "
            f"({out_subblock_w}) is {out_subblock_h * out_subblock_w}, above the "
            f"{dest_tiles} tiles the destination register file holds"
            f"{' with fp32_dest_acc_en set' if fp32_dest_acc_en else ''}. "
            f"The output would be corrupted."
        )
