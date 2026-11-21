# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests of add_ that the golden suite cannot make (op_design.md, Work Distribution, "Tests the plan needs"):

1. every NEO of the thread plan ran (a correct result alone does not show it);
2. the op in one device session after another program and before one of tt-metal's own;
3. rule R8: operands built as ResNet-50's test builds them, on grids that hold more rows than the tensor;
4. a borrowed ring above the target page's 1,048,560 B per-ring figure;
plus the compute helper at one thread and at the plan's thread counts.
"""

import pytest
import torch

import ttnn
from ttnn.operations.add_ import add_, default_compute_kernel_config
from ttnn.operations.add_.add__program_artifacts import THREAD_PLAN, create_program_artifacts

RELU = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]
TILE = 32


def _sharded_config(shard_rows, width, cores):
    """Height-sharded L1 config with an explicit shard shape: ResNet-50's test pads the rows to a multiple of
    32 times the core count, so its grid may hold more rows than the tensor."""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))]) if cores <= 8 else \
        ttnn.num_cores_to_corerangeset(cores, ttnn.CoreCoord(8, 4), row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (shard_rows, width), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _operands(device, shape, config, seed=0):
    torch.manual_seed(seed)
    ta = torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)
    tb = torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)
    put = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config)  # noqa: E731
    return ta, tb, put(ta), put(tb)


def _expected(ta, tb, sign=1.0):
    return torch.relu(ta.float() + sign * tb.float()).to(torch.bfloat16).float()


def _check(got, expected):
    diff = (got.float() - expected).abs()
    # 1 bfloat16 ulp of the expected value (the specification's grade), with a floor for values near zero.
    bound = expected.abs() * 2.0**-7 + 1e-6
    assert bool((diff <= bound).all()), f"max diff {diff.max().item()}"


def _check_in_place(result, a, b, tb_before):
    assert result is a, "R4: add_ must return input_tensor_a itself"
    assert torch.equal(ttnn.to_torch(b).float(), tb_before.float()), "R5: input_tensor_b changed"


def test_every_neo_computes_its_own_tiles(device):
    """With tag_threads, odd NEOs subtract: tile k of a core must come from NEO k mod C."""
    shape = (1, 1, 256, 128)  # 8 x 4 tiles, two cores of 4 tile rows: 16 tiles per core
    cores, shard_rows = 2, 128
    ta, tb, a, b = _operands(device, shape, _sharded_config(shard_rows, shape[-1], cores))
    spec, run_args, tensor_args = create_program_artifacts(
        a, b, pack_relu=True, compute_kernel_config=default_compute_kernel_config(), tag_threads=True
    )
    C = THREAD_PLAN[1]
    assert C == 4 and 16 >= 2 * C
    ttnn.generic_op([b, a], spec, run_args, tensor_args)
    got = ttnn.to_torch(a).float()

    col_tiles = shape[-1] // TILE
    shard_row_tiles = shard_rows // TILE
    subtracted = 0
    for rt in range(shape[-2] // TILE):
        for ct in range(col_tiles):
            local = (rt % shard_row_tiles) * col_tiles + ct
            odd = (local % C) & 1
            subtracted += odd
            at = (0, 0, slice(rt * TILE, rt * TILE + TILE), slice(ct * TILE, ct * TILE + TILE))
            _check(got[at], _expected(ta[at], tb[at], -1.0 if odd else 1.0))
    assert subtracted == 16


@pytest.mark.parametrize("plan", [(1, 1, 1), (1, 2, 1), (1, 4, 1)], ids=lambda p: "R%dC%dW%d" % p)
@pytest.mark.parametrize("shape,shard_rows,cores", [((1, 1, 256, 256), 128, 2), ((1, 1, 32, 8192), 32, 1)])
def test_helper_at_each_thread_count(device, plan, shape, shard_rows, cores):
    """The eltwise helper's block form at one thread and at the plan's NEO counts."""
    ta, tb, a, b = _operands(device, shape, _sharded_config(shard_rows, shape[-1], cores))
    spec, run_args, tensor_args = create_program_artifacts(
        a, b, pack_relu=True, compute_kernel_config=default_compute_kernel_config(), thread_plan=plan
    )
    ttnn.generic_op([b, a], spec, run_args, tensor_args)
    _check(ttnn.to_torch(a).float(), _expected(ta, tb))


def test_one_device_session(device):
    """add_ after another program (a first add_), then a tt-metal program after it, in one session."""
    shape = (1, 1, 256, 256)
    config = _sharded_config(128, shape[-1], 2)
    ta, tb, a, b = _operands(device, shape, config)
    first = add_(a, b, activations=RELU)
    second = add_(first, b, activations=RELU)
    expected = torch.relu(_expected(ta, tb) + tb.float()).to(torch.bfloat16).float()
    _check_in_place(second, a, b, tb)
    host = ttnn.to_layout(ttnn.to_memory_config(second, ttnn.DRAM_MEMORY_CONFIG), ttnn.ROW_MAJOR_LAYOUT)
    _check(ttnn.to_torch(host).float(), expected)
    # and once more after tt-metal's programs
    tc, td, c, d = _operands(device, shape, config, seed=1)
    _check(ttnn.to_torch(add_(c, d, activations=RELU)).float(), _expected(tc, td))


@pytest.mark.parametrize(
    "shape,shard_rows,cores",
    [
        ((1, 1, 784, 512), 416, 2),  # 800 padded rows in two shards of 416: 208 + 192 tiles
        ((1, 1, 196, 1024), 128, 2),  # 224 padded rows in two shards of 128: 128 + 96 tiles
        ((1, 1, 196, 1024), 32, 32),  # 7 of 32 shards hold a tile row; 25 are empty
    ],
    ids=["layer2_2cores", "layer3_2cores", "layer3_32cores"],
)
def test_resnet_shard_grids(device, shape, shard_rows, cores):
    """Rule R8: grids holding more rows than the tensor, last shards partly filled or empty."""
    ta, tb, a, b = _operands(device, shape, _sharded_config(shard_rows, shape[-1], cores))
    result = add_(a, b, activations=RELU)
    _check_in_place(result, a, b, tb)
    _check(ttnn.to_torch(result).float(), _expected(ta, tb))


def test_ring_above_one_mib(device):
    """[1, 1, 3136, 256] bfloat16 on one core: a 1,605,632 B borrowed ring, above the page's per-ring figure."""
    shape = (1, 1, 3136, 256)
    ta, tb, a, b = _operands(device, shape, _sharded_config(3136, 256, 1))
    result = add_(a, b, activations=RELU)
    _check_in_place(result, a, b, tb)
    _check(ttnn.to_torch(result).float(), _expected(ta, tb))
