# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.create_sharded_memory_config`` (host-side config builder).

Model call sites (all build the sharded MemoryConfigs the model runs with):
  * modules/rmsnorm/rmsnorm_1d.py:502   decode_memory_config   — WIDTH shard (tile_padded_batch_rows, dim//num_cores)
  * modules/attention/attention_1d.py:1704  decode_residual_memcfg — WIDTH shard (tile_padded_batch_rows, dim//num_cores)
  * modules/mlp/mlp_1d.py:880 / :917    decode_input_memcfg / decode_residual_memcfg — WIDTH shard
  * models/llama32_1b/model.py:589      lm_input_memcfg        — WIDTH shard
  * modules/lm_head/lm_head_1d.py:263   input_memcfg           — WIDTH shard
  * modules/rope/rope_1d.py:387         cos_sin_shard_mem_config — HEIGHT shard (TILE, head_dim)
  * modules/rope/rope_1d.py:378         decode_trans_mat_mem_config — HEIGHT shard (TILE, TILE)

``create_sharded_memory_config`` runs entirely on host (no device compute); it
returns a ``MemoryConfig``. This test asserts the returned config is sharded with
the expected layout, then round-trips a tensor into it via ``to_memory_config``
(interleaved DRAM -> this sharded config -> host) and checks the values survive.

NOTE: the model derives the core grid from ``dim`` at runtime (``_dram_shard_core_grid``,
``_compute_norm_core_grid``). To keep these emulator tests self-contained they use a
single-core grid, which makes the passed shape the exact per-core shard shape.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U

# Single-core grid: with use_height_and_width_as_shard_shape=True the passed
# shape IS the per-core shard shape, so a (1,1,H,W) tensor maps exactly.
_GRID = ttnn.CoreGrid(y=1, x=1)

# (id, shape, strategy, expected_layout)
_CONFIGS = [
    # WIDTH-sharded residual / rmsnorm / mlp / lm_head activations: (32, dim)
    pytest.param(
        (U.TILE, U.DIM),
        ttnn.ShardStrategy.WIDTH,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        id="width-residual-32x2048",  # rmsnorm_1d.py:502, attention_1d.py:1704, mlp_1d.py:917, model.py:589
    ),
    # WIDTH-sharded MLP2 input: (32, intermediate-ish); use DIM-family width here.
    pytest.param(
        (U.TILE, U.KV_DIM),
        ttnn.ShardStrategy.WIDTH,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        id="width-512",  # mlp_1d.py:907 style width shard
    ),
    # HEIGHT-sharded rope cos/sin: (TILE, head_dim)
    pytest.param(
        (U.TILE, U.HEAD_DIM),
        ttnn.ShardStrategy.HEIGHT,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        id="height-cos-sin-32x64",  # rope_1d.py:387
    ),
    # HEIGHT-sharded rope transform matrix: (TILE, TILE)
    pytest.param(
        (U.TILE, U.TILE),
        ttnn.ShardStrategy.HEIGHT,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        id="height-trans-mat-32x32",  # rope_1d.py:378
    ),
]


@U.with_default_mesh()
@pytest.mark.parametrize("shape, strategy, expected_layout", _CONFIGS)
def test_create_sharded_memory_config(ttnn_mesh_device, reset_seeds, shape, strategy, expected_layout):
    mesh = ttnn_mesh_device

    memcfg = ttnn.create_sharded_memory_config(
        shape,
        _GRID,
        strategy,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Host assertions: it built a valid *sharded* config with the expected layout.
    assert memcfg.is_sharded(), f"expected a sharded MemoryConfig, got {memcfg}"
    assert memcfg.memory_layout == expected_layout, f"expected {expected_layout}, got {memcfg.memory_layout}"

    # Round-trip: a tensor can be moved into this config and read back unchanged.
    h, w = shape
    x_torch = U.torch_rand((1, 1, h, w))
    x = U.to_tt(x_torch, mesh)  # interleaved DRAM
    x_sharded = ttnn.experimental.quasar.to_memory_config(x, memcfg)
    U.assert_pcc(x_torch, x_sharded, pcc=0.999, mesh_device=mesh)
