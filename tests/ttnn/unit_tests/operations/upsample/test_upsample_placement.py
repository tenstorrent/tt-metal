# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Placement rules of the specification that the generic golden runner cannot exercise.

R5: the output lives in the input's buffer, interleaved (a DRAM input gives a DRAM result; the L1 case is in
    test_upsample_threads.py::test_int_and_pair_factor_agree_and_l1_stays_l1).
R5/R7 (Refinement 0): the trace's one sharded call, [1, 4, 4, 64] bfloat16 row-major height-sharded in L1
    over two cores (shard [8, 64]) with memory_config=x.memory_config(), gives a result height-sharded over
    the same cores with shard [32, 64]; any other memory_config is refused by name.
R6: a sharding the op does not support (width) is refused by name through the `sharding` axis
    (`eval.memory.memory_axes`).
"""

import pytest
import torch

import ttnn
from ttnn.operations._op_contract import UnsupportedAxisValue
from ttnn.operations.upsample import upsample


def test_dram_input_gives_dram_interleaved_output(device):
    torch.manual_seed(3)
    t = torch.randn(1, 20, 20, 512, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    y = upsample(x, (2, 2))
    assert y.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert y.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert torch.equal(ttnn.to_torch(y), t.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2))


def _sharded(layout, shard_shape, cores_x):
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores_x - 1, 0))}),
        list(shard_shape),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(layout, ttnn.BufferType.L1, shard_spec)


def _model_input(device, memory_config):
    t = torch.randn(1, 4, 4, 64, dtype=torch.bfloat16)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config)
    return t, x


@pytest.mark.parametrize("pass_memory_config", [True, False], ids=["memory_config_input", "memory_config_none"])
def test_model_sharded_call(device, pass_memory_config):
    """YOLOv8's sharded upsample (test_upsample_small_grid.py::test_upsample_sharded_small_grid):
    upsample(x, (2, 2), memory_config=x.memory_config()) of a [1, 4, 4, 64] tensor height-sharded over two
    cores (shard [8, 64]) is [1, 8, 8, 64], height-sharded in L1 over the same two cores, shard [32, 64] (R5,
    R7); leaving memory_config out gives the same result."""
    torch.manual_seed(5)
    in_memory = _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (8, 64), cores_x=2)
    t, x = _model_input(device, in_memory)
    y = upsample(x, (2, 2), memory_config=x.memory_config()) if pass_memory_config else upsample(x, (2, 2))
    assert list(y.shape) == [1, 8, 8, 64]
    out_memory = y.memory_config()
    assert out_memory.buffer_type == ttnn.BufferType.L1
    assert out_memory.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert out_memory.shard_spec.grid == in_memory.shard_spec.grid
    assert list(out_memory.shard_spec.shape) == [32, 64]
    assert out_memory.shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    assert torch.equal(ttnn.to_torch(y), t.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2))


def test_other_memory_config_refused_by_name(device):
    """R7: any memory_config but the input's own reads as memory_argument="other" and is refused by name."""
    _, x = _model_input(device, _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (8, 64), cores_x=2))
    with pytest.raises(UnsupportedAxisValue, match="memory_argument='other'"):
        upsample(x, (2, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)


def test_width_sharded_input_refused_by_name(device):
    """R6: a sharding the op does not support (width) is refused by name through the `sharding` axis."""
    _, x = _model_input(device, _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, (16, 32), cores_x=2))
    with pytest.raises(UnsupportedAxisValue, match="sharding='width'"):
        upsample(x, (2, 2))
