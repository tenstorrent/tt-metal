# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.sharded_to_interleaved``.

Model call sites:
  * modules/rmsnorm/rmsnorm_1d.py:216   x = ttnn.sharded_to_interleaved(x)
  * modules/mlp/mlp_1d.py:445           w2_out -> ttnn.L1_MEMORY_CONFIG
  * modules/attention/attention_1d.py:1099,1116  xqkv/output -> ttnn.L1_MEMORY_CONFIG
  * modules/lm_head/lm_head_1d.py:141   output -> cfg.output_memcfg
  * utility_functions.py:203            tt_tensors_device -> interleaved

``sharded_to_interleaved`` is the inverse of ``interleaved_to_sharded``. This test
does the full round-trip: interleaved -> shard -> unshard, and asserts the result
equals the original (values are preserved by pure layout conversions).

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


# (id, shape, strategy) — shard shape is (shape[-2], shape[-1]).
_CASES = [
    # rmsnorm decode output: WIDTH-sharded (tile_padded_batch_rows, dim). rmsnorm_1d.py:216, cfg at :502
    pytest.param((1, 1, U.TILE, U.DIM), ttnn.ShardStrategy.WIDTH, id="rmsnorm-decode-32x2048"),
    # attention xqkv / dense output width family. attention_1d.py:1099,1116
    pytest.param((1, 1, U.TILE, U.KV_DIM), ttnn.ShardStrategy.WIDTH, id="attn-width-32x512"),
    # HEIGHT-sharded round trip (mirrors rope-style height shards). rope_1d.py:387
    pytest.param((1, 1, U.TILE, U.HEAD_DIM), ttnn.ShardStrategy.HEIGHT, id="height-32x64"),
]


@U.with_default_mesh()
@pytest.mark.parametrize("shape, strategy", _CASES)
def test_sharded_to_interleaved(ttnn_mesh_device, reset_seeds, shape, strategy):
    mesh = ttnn_mesh_device

    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)  # interleaved DRAM

    memcfg = _sharded_cfg(shape[-2], shape[-1], strategy)
    x_sharded = ttnn.interleaved_to_sharded(x, memcfg)
    out = ttnn.sharded_to_interleaved(x_sharded, ttnn.L1_MEMORY_CONFIG)  # mlp_1d.py:445, attention_1d.py:1099

    assert not out.is_sharded(), "expected an interleaved output"
    U.assert_pcc(x_torch, out, pcc=0.999, mesh_device=mesh)
