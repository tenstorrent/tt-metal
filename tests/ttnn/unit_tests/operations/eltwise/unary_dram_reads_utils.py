# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Unary ops on DRAM height-sharded, width-sharded and interleaved TILE tensors must give the default read path's output
# bit for bit in every read flow: bursts, one page in flight and the work queue. The reference runs the same op on DRAM
# interleaved on a few cores, below the cores-per-bank gate and off the whole grid, so it always takes the default path.
#
# Shared by test_unary_dram_reads.py and test_unary_dram_reads_large.py. The large file holds the cases sized at 600 or
# more pages per core over the whole grid, so the simulator legs can skip it and still run the rest.

import contextlib

import torch
import ttnn

from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_equal

HEIGHT, WIDTH = ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED

# 16 cores: fewer than 8 per DRAM bank and not the whole grid, so the op takes the default path.
DEFAULT_PATH_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})

# Every op class of the flow rule: light, mid (silu, softplus, sigmoid, fast erf), heavy and unmeasured (atan).
OPS = {
    "identity": ttnn.identity,
    "relu": ttnn.relu,
    "exp": ttnn.exp,
    "log": ttnn.log,
    "tanh": ttnn.tanh,
    "hardswish": ttnn.hardswish,
    "gelu_fast_lut": lambda t, **kw: ttnn.gelu(t, variant=ttnn.GeluVariant.FastLut, **kw),
    "silu": ttnn.silu,
    "softplus": ttnn.softplus,
    "sigmoid": ttnn.sigmoid,
    "erf_fast": lambda t, **kw: ttnn.erf(t, fast_and_approximate_mode=True, **kw),
    "gelu_accurate": lambda t, **kw: ttnn.gelu(t, variant=ttnn.GeluVariant.Accurate, **kw),
    "mish_fast": lambda t, **kw: ttnn.mish(t, fast_and_approximate_mode=True, **kw),
    "elu": ttnn.elu,
    "logit": lambda t, **kw: ttnn.logit(t, eps=1e-6, **kw),
    "log_sigmoid": ttnn.log_sigmoid,
    "atan": ttnn.atan,
}
if is_blackhole():  # softcap is Blackhole only
    OPS["softcap"] = lambda t, **kw: ttnn.softcap(t, 10.0, **kw)


# DRAM sharded over num_shards banks. Height shards are whole rows; width shards are shard_width columns of every row,
# shape[-1] // num_shards by default.
def dram_sharded(shape, layout, num_shards, shard_width=None):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_shards - 1, 0))})
    if layout == HEIGHT:
        shard = [-(-shape[-2] // num_shards // 32) * 32, shape[-1]]
    else:
        shard = [shape[-2], shard_width or shape[-1] // num_shards]
    spec = ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(layout, ttnn.BufferType.DRAM, spec)


def random_input(shape, seed=0):
    torch.manual_seed(seed)
    return (torch.rand(shape) * 0.9 + 0.05).to(torch.bfloat16)


# Rows of a [1, 1, rows, width] tensor with about pages_per_core pages per core; 600 is above the work-queue threshold.
def large_rows(device, width, step, pages_per_core=600):
    grid = device.compute_with_storage_grid_size()
    rows = pages_per_core * grid.x * grid.y * 32 * 32 // width
    return -(-rows // step) * step


# layout is "height", "width" or "interleaved".
def memory_config_for(layout, shape, num_shards=7):
    if layout == "interleaved":
        return ttnn.DRAM_MEMORY_CONFIG
    return dram_sharded(shape, HEIGHT if layout == "height" else WIDTH, num_shards)


# measure wraps the tested op's call, e.g. to count the program-cache entries it adds.
def run_against_default_path(op, a, device, memory_config, dtype=ttnn.bfloat16, measure=contextlib.nullcontext):
    tested = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    reference = ttnn.from_torch(
        a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with measure():
        out = op(tested)
    assert out.memory_config().memory_layout == memory_config.memory_layout
    assert_equal(ttnn.to_torch(op(reference, sub_core_grids=DEFAULT_PATH_GRID)), ttnn.to_torch(out))
