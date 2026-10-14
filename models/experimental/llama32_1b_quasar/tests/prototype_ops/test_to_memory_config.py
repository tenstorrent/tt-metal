# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.to_memory_config`` (value-preserving layout/placement move).

Model call sites (representative):
  * DRAM <-> L1 interleaved:
      - modules/attention/attention_1d.py:698,703  q/k_heads_pre_rot -> L1
      - modules/attention/attention_1d.py:1153      output -> DRAM
      - modules/mlp/mlp_1d.py:237                    w1_out -> DRAM
      - models/llama32_1b/model.py:915,930           x -> DRAM
  * interleaved -> WIDTH-sharded (residual / mlp / rmsnorm activations):
      - models/llama32_1b/model.py:128,870,967       attn_out/x -> decode_residual_memcfg
      - modules/rmsnorm/rmsnorm_1d.py:204            x -> decode_memory_config
      - modules/mlp/mlp_1d.py:265,292                w2_in / w2_out_reduced -> sharded
      - modules/attention/attention_1d.py:970,1206   dense_out -> decode_residual_memcfg

``to_memory_config`` only relocates data, so a round-trip (DRAM interleaved ->
target config -> host) must preserve values. Both the placement moves (DRAM/L1)
and the interleaved->sharded moves are covered.
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U

# Single-core width-shard grid; passed shape is the exact per-core shard shape.
_GRID = ttnn.CoreGrid(y=1, x=1)


def _width_sharded(h, w):
    return ttnn.create_sharded_memory_config(
        (h, w),
        _GRID,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# Placement moves: DRAM <-> L1 interleaved. Shapes from residual/activation stream.
@U.with_default_mesh()
@pytest.mark.parametrize(
    "target_memcfg",
    [
        pytest.param(ttnn.L1_MEMORY_CONFIG, id="dram-to-l1"),  # attention_1d.py:698, model.py:915
        pytest.param(ttnn.DRAM_MEMORY_CONFIG, id="dram-to-dram"),  # attention_1d.py:1153, model.py:930
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, U.TILE, U.DIM), id="decode-32x2048"),
        pytest.param((1, 1, U.PREFILL_SEQ_LENS[0], U.DIM), id=f"prefill-seq{U.PREFILL_SEQ_LENS[0]}"),
    ],
)
def test_to_memory_config_placement(ttnn_mesh_device, reset_seeds, target_memcfg, shape):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)  # interleaved DRAM
    out = ttnn.experimental.quasar.to_memory_config(x, target_memcfg)
    U.assert_pcc(x_torch, out, pcc=0.999, mesh_device=mesh)


# interleaved -> WIDTH-sharded move (the decode_residual / rmsnorm / mlp pattern).
@U.with_default_mesh()
@pytest.mark.parametrize(
    "width",
    [
        pytest.param(U.DIM, id="residual-2048"),  # model.py:128,967 ; rmsnorm_1d.py:204 (dim=2048)
        pytest.param(U.KV_DIM, id="width-512"),  # mlp2-input-style width shard (mlp_1d.py:265)
    ],
)
def test_to_memory_config_interleaved_to_sharded(ttnn_mesh_device, reset_seeds, width):
    mesh = ttnn_mesh_device
    shape = (1, 1, U.TILE, width)  # decode: tile_padded_batch_rows=32
    x_torch = U.torch_rand(shape)
    x = U.to_tt(x_torch, mesh)  # interleaved DRAM
    out = ttnn.experimental.quasar.to_memory_config(x, _width_sharded(U.TILE, width))
    assert out.is_sharded(), "expected a sharded output"
    U.assert_pcc(x_torch, out, pcc=0.999, mesh_device=mesh)
