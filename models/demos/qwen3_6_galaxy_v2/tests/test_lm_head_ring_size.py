# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LM-head ring-matmul config (shape) pin.

The qwen3.6 LM-head logits projection runs as a 1D ``gather_in0`` ring matmul
whose width (``LM_HEAD_RING_SIZE``) splits the per-col vocab across that many
cores.  The size was hard-coded to 24; it is now parametric / dynamic (via
``QWEN36_LM_HEAD_RING_SIZE``) so it can use the wider post-60->110 grid.

This test pins the ring-shape MATH for the supported sizes (24 default + 72
wide) so a future grid/vocab change cannot silently produce an invalid ring
(tile-misaligned shards, or ``num_blocks_total != num_cores`` which the matmul
factory asserts on at runtime).  The end-to-end PCC / coherence of the actual
on-device ring output is covered by the ISL-128 (token 248068) and 128k demo
gates — both validated identical at ring 24 and 72.

Pure host-side (opens a mesh only to build the model config); no fabric / CCL.
"""


import pytest

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.model_config import num_to_coregrid


@pytest.mark.parametrize("ring_size", [24, 72])
def test_lm_head_ring_shape_math(ring_size, monkeypatch):
    monkeypatch.setenv("QWEN36_LM_HEAD_RING_SIZE", str(ring_size))
    # Build the config against the live mesh (BH galaxy → compute grid (12,10)).
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    try:
        args = TtQwen36ModelArgs(mesh)
        mc = args.model_config

        tile = args.tile_size
        per_col_vocab = args.padded_vocab_size // args.cluster_shape[1]  # 62208

        # --- vocab divides the ring cleanly (zero pad for 24 & 72) ---
        per_col_padded = args.lm_head_shape[1]
        assert per_col_padded % (ring_size * tile) == 0
        assert per_col_padded == per_col_vocab, (
            f"ring={ring_size} introduced vocab padding "
            f"({per_col_padded} != {per_col_vocab}); pick a divisor of "
            f"{per_col_vocab // tile} tiles"
        )

        # --- input/output ring shards are tile-aligned on exactly ring_size cores ---
        in_ss = mc["SHARDED_LM_HEAD_INPUT_32_RING_MEMCFG"].shard_spec
        out_ss = mc["LM_HEAD_OUT_RING_MEMCFG"].shard_spec
        assert in_ss.grid.num_cores() == ring_size
        assert out_ss.grid.num_cores() == ring_size
        assert in_ss.shape[1] % tile == 0, f"input shard width {in_ss.shape[1]} not tile-aligned"
        assert out_ss.shape[1] % tile == 0, f"output shard width {out_ss.shape[1]} not tile-aligned"
        assert out_ss.shape[1] == per_col_padded // ring_size

        # --- ring matmul grid: must be a valid rectangular num_to_coregrid ---
        grid = num_to_coregrid(ring_size)
        assert grid is not None, f"ring_size {ring_size} has no valid rectangular compute grid"
        assert grid.x * grid.y == ring_size

        # --- ring topology constraint: num_blocks_total == num_cores ---
        # M=32 (1 tile) → num_blocks_y=1; num_blocks_x = vocab_tiles / per_core_N.
        pc = mc["LM_HEAD_TG_RING_PROGCFG"]
        vocab_tiles = per_col_padded // tile
        num_blocks_x = vocab_tiles // pc.per_core_N
        assert (
            num_blocks_x * 1 == ring_size
        ), f"ring topology broken: num_blocks_total {num_blocks_x} != num_cores {ring_size}"
        assert (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y) == (grid.x, grid.y)
    finally:
        ttnn.close_mesh_device(mesh)
