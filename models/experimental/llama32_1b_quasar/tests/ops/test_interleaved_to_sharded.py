# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.interleaved_to_sharded``.

Model call sites:
  * modules/rope/rope_1d.py:189,190  cos/sin -> cfg.cos_sin_shard_mem_config  (HEIGHT shard (TILE, head_dim), rope_1d.py:387)
  * modules/attention/attention_1d.py:518,519  k_fill/v_fill -> cfg.prefill_kv_memcfg(seq_len)
  * models/llama32_1b/model.py:913,928  x -> lm_head_memcfg  (WIDTH shard (tile_padded_batch_rows, dim//num_cores), lm_head_1d.py:263)

``interleaved_to_sharded`` only re-lays-out data, so moving an interleaved tensor
into a sharded config and reading it back must preserve values. The sharded
configs are (re)built with ``create_sharded_memory_config`` from the real shard
shapes / strategies used at the call sites.

NOTE: single-core grid so the passed shape is the exact per-core shard shape.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

_GRID = ttnn.CoreGrid(y=1, x=1)


def _sharded_cfg(h, w, strategy):
    return ttnn.create_sharded_memory_config(
        (h, w),
        _GRID,
        strategy,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# (id, shape, shard_h, shard_w, strategy)
_CASES = [
    # rope cos/sin: HEIGHT-sharded (TILE, head_dim). rope_1d.py:189-190, cfg at :387
    pytest.param((1, 1, U.TILE, U.HEAD_DIM), U.TILE, U.HEAD_DIM, ttnn.ShardStrategy.HEIGHT, id="rope-cos-sin-32x64"),
    # lm_head / residual activations: WIDTH-sharded (tile_padded_batch_rows, dim). model.py:913,928
    pytest.param((1, 1, U.TILE, U.DIM), U.TILE, U.DIM, ttnn.ShardStrategy.WIDTH, id="lm-head-input-32x2048"),
    # attention kv-fill family width (kv_dim). attention_1d.py:518-519
    pytest.param((1, 1, U.TILE, U.KV_DIM), U.TILE, U.KV_DIM, ttnn.ShardStrategy.WIDTH, id="kv-width-32x512"),
]


@U.with_default_mesh()
@pytest.mark.parametrize("shape, shard_h, shard_w, strategy", _CASES)
def test_interleaved_to_sharded(ttnn_mesh_device, reset_seeds, shape, shard_h, shard_w, strategy):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)  # interleaved DRAM

    memcfg = _sharded_cfg(shard_h, shard_w, strategy)
    out = ttnn.interleaved_to_sharded(x, memcfg)

    assert out.is_sharded(), "expected a sharded output"
    U.assert_pcc(x_torch, out, pcc=0.999, mesh_device=mesh)
