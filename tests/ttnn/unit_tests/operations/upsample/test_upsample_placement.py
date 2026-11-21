# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Placement rules of the specification that the generic golden runner cannot exercise.

R5: the output lives in the input's buffer, interleaved (a DRAM input gives a DRAM result; the L1 case is in
    test_upsample_threads.py::test_int_and_pair_factor_agree_and_l1_stays_l1).
R6: a sharded input is refused by name through the `sharding` axis (`eval.memory.memory_axes`). The spec's
    review check says "a sharded input is a test the op carries itself": this is that test. The shape is the
    trace's one sharded call, [1, 4, 4, 64] bfloat16 row-major height-sharded in L1, shard [8, 64].
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


def test_height_sharded_input_refused_by_name(device):
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        [8, 64],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    x = ttnn.from_torch(
        torch.randn(1, 4, 4, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded,
    )
    with pytest.raises(UnsupportedAxisValue, match="sharding='height'"):
        upsample(x, (2, 2))
