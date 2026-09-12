# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Table-driven negative-path (validation) tests for the ttnn reduction op family.

Every row targets ONE specific TT_FATAL/TT_THROW in the reduction sources and
asserts its message, so a passing row proves that exact check fired — not some
earlier check on the same path (see the cumsum/cumprod failing-cases history,
where a host-tensor prealloc masked every labeled check). Rows are keyed by the
check's file:line in a comment. Checks that are unreachable from the python API
(masked by front-end transforms or earlier checks) are deliberately absent;
so are checks whose triggering input crashes instead of raising (host-tensor
ttnn.ema/std/var — null device deref) and configs that hang (sampling Wt=1,
issue #52348) — those are product bugs, not testable rows.

Positive-path coverage lives in the sibling per-op files; this file is
raise-tests only. Everything is host-side validation (no kernels launch), so
the whole file costs seconds and is sim-safe. Grid-containment rows use core
coordinates far outside every CI arch's compute grid (WH ~8x8, BH ~13x10).
"""

from collections import namedtuple

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device

Case = namedtuple("Case", ["id", "match", "run"])

DRAM = ttnn.DRAM_MEMORY_CONFIG
RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT


def _tile(device, shape=(32, 32), dtype=ttnn.bfloat16, torch_dtype=torch.float32, memory_config=None):
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch_dtype), dtype=dtype, layout=TILE, device=device, memory_config=memory_config
    )


def _rm(device, shape=(32, 32), dtype=ttnn.bfloat16, torch_dtype=torch.float32, memory_config=None):
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch_dtype), dtype=dtype, layout=RM, device=device, memory_config=memory_config
    )


def _grid(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _height_shard_cfg(x0, y0, x1, y1, shard_shape, buffer_type=ttnn.BufferType.L1):
    spec = ttnn.ShardSpec(_grid(x0, y0, x1, y1), shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type, spec)


def _run_cases(device, expect_error, case):
    with expect_error(RuntimeError, case.match):
        case.run(device)


# ---------------------------------------------------------------------------
# Generic reduce: ttnn.sum / mean / max / min
# (generic_reductions.cpp, reduction_common.cpp, device/reduce_op*.cpp,
#  device/common.cpp)
# ---------------------------------------------------------------------------

GENERIC_CASES = [
    # reduction_common.cpp:65 — dim out of range
    Case("dim_oor", "at least 0 and less than rank", lambda d: ttnn.sum(_tile(d), dim=2)),
    # reduction_common.cpp:76 — duplicate axis in dim list
    Case("dim_duplicate", "duplicate axis after sorting", lambda d: ttnn.sum(_tile(d), dim=[1, -1])),
    # reduction_common.cpp:35 — int64 dim outside int range
    Case("dim_int64_oor", "Dimension must be in the range", lambda d: ttnn.sum(_tile(d), dim=2**40)),
    # generic_reductions.cpp:545 — mean rejects integer inputs
    Case(
        "mean_int32",
        "does not support integer inputs",
        lambda d: ttnn.mean(_tile(d, dtype=ttnn.int32, torch_dtype=torch.int32), dim=-1),
    ),
    # reduction_common.hpp:81 — zero-size reduce dim for max/min
    Case(
        "max_zero_size_dim",
        "to have non-zero size",
        lambda d: ttnn.max(_rm(d, shape=(0, 32)), dim=0),
    ),
    # device/reduce_op.cpp:117 — host input
    Case(
        "sum_host_input",
        "Expected input tensor to be on device",
        lambda d: ttnn.sum(ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=TILE), dim=-1),
    ),
    # device/reduce_op_device_operation.cpp:104 — dtype whitelist
    Case(
        "sum_uint16",
        "are supported for generic reduction",
        lambda d: ttnn.sum(_tile(d, dtype=ttnn.uint16, torch_dtype=torch.int16), dim=-1),
    ),
    # device/common.cpp:274 — sharded output must be L1
    Case(
        "sum_sharded_output_dram",
        "sharded output memory layout",
        lambda d: ttnn.sum(
            _tile(d), dim=-1, memory_config=_height_shard_cfg(0, 0, 0, 0, [32, 32], ttnn.BufferType.DRAM)
        ),
    ),
    # device/common.cpp:280 — sharded input must be L1
    Case(
        "sum_sharded_input_dram",
        "sharded input memory layout",
        lambda d: ttnn.sum(
            _tile(d, memory_config=_height_shard_cfg(0, 0, 0, 0, [32, 32], ttnn.BufferType.DRAM)),
            dim=-1,
            memory_config=DRAM,
        ),
    ),
    # device/reduce_op_device_operation.cpp:123 — empty program grid
    Case(
        "sum_empty_sub_core_grids",
        "Program core grid must not be empty",
        lambda d: ttnn.sum(_tile(d), dim=-1, sub_core_grids=ttnn.CoreRangeSet([])),
    ),
    # device/reduce_op_device_operation.cpp:124 — program grid outside device grid
    Case(
        "sum_sub_core_grids_oob",
        "must be contained in device grid",
        lambda d: ttnn.sum(_tile(d), dim=-1, sub_core_grids=_grid(0, 0, 99, 99)),
    ),
    # device/reduce_op_device_operation.cpp:136 — input shard grid outside program grid
    Case(
        "sum_input_shard_outside_program_grid",
        "must be contained in program core grid",
        lambda d: ttnn.sum(
            _tile(d, memory_config=_height_shard_cfg(0, 0, 0, 0, [32, 32])),
            dim=-1,
            sub_core_grids=_grid(1, 1, 2, 2),
        ),
    ),
    # device/common.cpp:220 — sharded layout without any shard spec
    Case(
        "sum_sharded_output_no_spec",
        "requires either nd_shard_spec or shard_spec",
        lambda d: ttnn.sum(
            _tile(d),
            dim=-1,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1),
        ),
    ),
    # device/common.cpp:236 — block sharding needs a single CoreRange
    Case(
        "sum_block_shard_two_ranges",
        "Block sharding requires a single CoreRange",
        lambda d: ttnn.sum(
            _tile(d, shape=(64, 64)),
            dim=-1,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                            ttnn.CoreRange(ttnn.CoreCoord(2, 2), ttnn.CoreCoord(2, 2)),
                        ]
                    ),
                    [32, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        ),
    ),
    # device/common.cpp:248 — ND_SHARDED without nd_shard_spec
    Case(
        "sum_nd_sharded_no_spec",
        "requires nd_shard_spec to be set",
        lambda d: ttnn.sum(
            _tile(d), dim=-1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.ND_SHARDED, ttnn.BufferType.L1)
        ),
    ),
]


@pytest.mark.parametrize("case", GENERIC_CASES, ids=[c.id for c in GENERIC_CASES])
def test_generic_reduce_validation(device, expect_error, case):
    _run_cases(device, expect_error, case)


# ---------------------------------------------------------------------------
# std / var (welford path)
# ---------------------------------------------------------------------------

STD_VAR_CASES = [
    # welford_reduce_device_operation.cpp:27-31 — dtype whitelist
    Case(
        "std_int32",
        "are supported for Std/Var reduction",
        lambda d: ttnn.std(_tile(d, shape=(32, 64), dtype=ttnn.int32, torch_dtype=torch.int32), dim=-1),
    ),
    # welford_reduce_program_factory.cpp:62-65 — fp32 needs fp32_dest_acc_en
    Case(
        "var_fp32_no_fp32_acc",
        "requires fp32_dest_acc_en=true in the compute kernel",
        lambda d: ttnn.var(
            _tile(d, shape=(32, 64), dtype=ttnn.float32),
            dim=-1,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False
            ),
        ),
    ),
    # device/common.cpp:274 via welford — sharded output must be L1
    Case(
        "std_sharded_output_dram",
        "sharded output memory layout",
        lambda d: ttnn.std(
            _tile(d), dim=-1, memory_config=_height_shard_cfg(0, 0, 0, 0, [32, 32], ttnn.BufferType.DRAM)
        ),
    ),
    # device/common.cpp:280 via welford — sharded input must be L1
    Case(
        "std_sharded_input_dram",
        "sharded input memory layout",
        lambda d: ttnn.std(
            _tile(d, memory_config=_height_shard_cfg(0, 0, 0, 0, [32, 32], ttnn.BufferType.DRAM)),
            dim=-1,
            memory_config=DRAM,
        ),
    ),
    # device/common.cpp:219-223 — sharded output lacks any shard spec
    Case(
        "std_sharded_output_no_spec",
        "requires either nd_shard_spec or shard_spec",
        lambda d: ttnn.std(
            _tile(d),
            dim=-1,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1),
        ),
    ),
    # device/common.cpp:246-251 — ND_SHARDED without nd_shard_spec
    Case(
        "std_nd_sharded_no_spec",
        "ND_SHARDED memory layout requires nd_shard_spec",
        lambda d: ttnn.std(
            _tile(d), dim=-1, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.ND_SHARDED, ttnn.BufferType.L1)
        ),
    ),
    # reduction_common.cpp:65-72 — dim out of range
    Case("std_dim_oor", "Unsupported dim", lambda d: ttnn.std(_tile(d), dim=4)),
    # reduction_common.cpp:76-83 — duplicate dim
    Case("var_dim_duplicate", "duplicate axis after sorting", lambda d: ttnn.var(_tile(d), dim=(1, -1))),
    # reduction_common.cpp:35-40 — int64 dim range
    Case("std_dim_int64_oor", "Dimension must be in the range", lambda d: ttnn.std(_tile(d), dim=2**40)),
]


@pytest.mark.parametrize("case", STD_VAR_CASES, ids=[c.id for c in STD_VAR_CASES])
def test_std_var_validation(device, expect_error, case):
    _run_cases(device, expect_error, case)


# ---------------------------------------------------------------------------
# argmax (argmax.cpp, device/argmax_device_operation.cpp)
# ---------------------------------------------------------------------------


def _argmax_prealloc(device, shape, dtype=ttnn.uint32, layout=RM, memory_config=None):
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch.int32), dtype=dtype, layout=layout, device=device, memory_config=memory_config
    )


ARGMAX_CASES = [
    # argmax.cpp:173 — input on device
    Case(
        "host_input",
        "Input tensor must be on device",
        lambda d: ttnn.argmax(ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16), dim=-1),
    ),
    # argmax.cpp:174 — prealloc on device
    Case(
        "prealloc_host",
        "Preallocated output tensor must be on device",
        lambda d: ttnn.argmax(
            _rm(d, shape=(64, 128)),
            dim=-1,
            keepdim=True,
            output_tensor=ttnn.from_torch(torch.zeros(64, 1, dtype=torch.int32), dtype=ttnn.uint32),
        ),
    ),
    # argmax.cpp:183 — dim out of range (rank > 0)
    Case("dim_oor", "Dimension out of range", lambda d: ttnn.argmax(_rm(d, shape=(32, 64)), dim=2)),
    # argmax.cpp:191 — dim out of range (rank 0)
    Case(
        "dim_oor_rank0",
        "Dimension out of range for scalar tensor",
        lambda d: ttnn.argmax(ttnn.from_torch(torch.zeros(()), dtype=ttnn.bfloat16, device=d), dim=1),
    ),
    # argmax.cpp:148-152 — zero-volume prealloc shape
    Case(
        "zero_volume_prealloc_shape",
        "Preallocated output tensor has incorrect shape",
        lambda d: ttnn.argmax(
            _rm(d, shape=(0, 32)),
            dim=1,
            keepdim=True,
            output_tensor=_argmax_prealloc(d, (0, 2)),
        ),
    ),
    # argmax.cpp:214-218 — rank-0 prealloc shape
    Case(
        "rank0_prealloc_shape",
        "Preallocated output tensor has incorrect shape",
        lambda d: ttnn.argmax(
            ttnn.from_torch(torch.zeros(()), dtype=ttnn.bfloat16, device=d),
            output_tensor=ttnn.zeros([2], dtype=ttnn.uint32, layout=RM, device=d),
        ),
    ),
    # argmax.cpp:222-224 — rank-0 prealloc dtype
    Case(
        "rank0_prealloc_dtype",
        "must be UINT32 for rank 0 input tensor",
        lambda d: ttnn.argmax(
            ttnn.from_torch(torch.zeros(()), dtype=ttnn.bfloat16, device=d),
            output_tensor=ttnn.from_torch(torch.tensor(0, dtype=torch.int32), device=d),
        ),
    ),
    # argmax_device_operation.cpp:63 — zero-size reduce dim
    Case(
        "zero_size_reduce_dim",
        "to have non-zero size",
        lambda d: ttnn.argmax(_rm(d, shape=(32, 0)), dim=1, keepdim=True),
    ),
    # argmax_device_operation.cpp:89-92 — input INTERLEAVED only
    Case(
        "sharded_input",
        "Only INTERLEAVED memory layout is supported for inputs",
        lambda d: ttnn.argmax(
            ttnn.from_torch(
                torch.zeros(64, 128),
                dtype=ttnn.bfloat16,
                layout=RM,
                device=d,
                memory_config=ttnn.create_sharded_memory_config(
                    shape=(32, 128),
                    core_grid=ttnn.CoreGrid(x=1, y=2),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    use_height_and_width_as_shard_shape=True,
                ),
            ),
            dim=-1,
            keepdim=True,
            # explicit interleaved output: the default (inherit the sharded input
            # config) dies building the output TensorSpec before validate runs
            memory_config=DRAM,
        ),
    ),
    # argmax_device_operation.cpp:95-100 — RM dtype whitelist
    Case(
        "rm_dtype_uint8",
        "supported for inputs with ROW_MAJOR layout",
        lambda d: ttnn.argmax(_rm(d, shape=(32, 64), dtype=ttnn.uint8, torch_dtype=torch.uint8), dim=-1),
    ),
    # argmax_device_operation.cpp:102-105 — TILE dtype whitelist
    Case(
        "tile_dtype_int32",
        "supported for inputs with TILE layout",
        lambda d: ttnn.argmax(_tile(d, dtype=ttnn.int32, torch_dtype=torch.int32), dim=-1),
    ),
    # argmax_device_operation.cpp:138-141 — preallocated output must be INTERLEAVED
    # (a spec-less sharded memory_config dies in TensorLayout creation before the
    #  op's :127 output-config check can fire, so the prealloc variant is the
    #  reachable way to pin the outputs-interleaved contract)
    Case(
        "sharded_prealloc_output",
        "Only INTERLEAVED memory layout is supported for outputs",
        lambda d: ttnn.argmax(
            _rm(d, shape=(64, 128)),
            dim=-1,
            keepdim=True,
            output_tensor=ttnn.from_torch(
                torch.zeros(64, 64, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=RM,
                device=d,
                memory_config=_height_shard_cfg(0, 0, 0, 1, [32, 64]),
            ),
        ),
    ),
    # argmax_device_operation.cpp:134-137 — prealloc must be UINT32
    Case(
        "prealloc_int32",
        "Only UINT32 is supported for outputs",
        lambda d: ttnn.argmax(
            _rm(d, shape=(64, 128)),
            dim=-1,
            keepdim=True,
            output_tensor=ttnn.zeros([64, 1], dtype=ttnn.int32, layout=RM, device=d),
        ),
    ),
    # argmax_device_operation.cpp:142-145 — prealloc must be ROW_MAJOR
    Case(
        "prealloc_tile_layout",
        "Output tensor must have ROW_MAJOR layout",
        lambda d: ttnn.argmax(
            _rm(d, shape=(64, 128)),
            dim=-1,
            keepdim=True,
            output_tensor=ttnn.zeros([64, 1], dtype=ttnn.uint32, layout=TILE, device=d),
        ),
    ),
    # argmax_device_operation.cpp:162-167 — RM non-last dim (int dtype dodges the NC/RM-h front-ends)
    Case(
        "rm_non_last_dim",
        "only argmax on the last dim is supported",
        lambda d: ttnn.argmax(_rm(d, shape=(32, 64), dtype=ttnn.int32, torch_dtype=torch.int32), dim=0),
    ),
    # argmax_device_operation.cpp:170 — TILE requires dim
    Case("tile_requires_dim", "dim parameter must be specified", lambda d: ttnn.argmax(_tile(d))),
    # argmax_device_operation.cpp:174-178 — sub_core_grids max 2 ranges
    Case(
        "three_core_ranges",
        "only supports up to 2 core grid ranges",
        lambda d: ttnn.argmax(
            _rm(d, shape=(64, 128)),
            dim=-1,
            sub_core_grids=ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 2), ttnn.CoreCoord(2, 2)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 4), ttnn.CoreCoord(4, 4)),
                ]
            ),
        ),
    ),
    # reduce_op_validation.cpp:76-82 via argmax — sub_core_grids inside device grid
    Case(
        "sub_core_grids_oob",
        "must be contained in device compute grid",
        lambda d: ttnn.argmax(_rm(d, shape=(64, 128)), dim=-1, sub_core_grids=_grid(90, 90, 90, 90)),
    ),
]


@pytest.mark.parametrize("case", ARGMAX_CASES, ids=[c.id for c in ARGMAX_CASES])
def test_argmax_validation(device, expect_error, case):
    _run_cases(device, expect_error, case)


# ---------------------------------------------------------------------------
# topk (topk.cpp, device/topk_device_operation.cpp)
# Five checks are already message-tested in test_topk.py:358-439 (input dtype
# whitelist, prealloc dtypes, fp32+uint16 indices, indices shape, indices
# non-last-dim) and are not repeated here.
# ---------------------------------------------------------------------------


def _topk_input(device, shape=(1, 1, 32, 64), dtype=ttnn.bfloat16):
    return _tile(device, shape=shape, dtype=dtype)


TOPK_CASES = [
    # topk.cpp:202 — input on device
    Case(
        "host_input",
        "Input tensor must be on device",
        lambda d: ttnn.topk(
            ttnn.from_torch(torch.zeros(1, 1, 32, 64), dtype=ttnn.bfloat16, layout=TILE), 32, -1, True, True
        ),
    ),
    # topk.cpp:209-213 — dim out of range
    Case("dim_oor", "Dimension for topk is out of range", lambda d: ttnn.topk(_topk_input(d), 32, 4, True, True)),
    # topk.cpp:222-226 — k > dim size
    Case(
        "k_too_large",
        "K cannot be larger than the dimension size",
        lambda d: ttnn.topk(_topk_input(d), 128, -1, True, True),
    ),
    # topk.cpp:235 — prealloc values on device
    Case(
        "prealloc_values_host",
        "Preallocated output values tensor must be on device",
        lambda d: ttnn.topk(
            _topk_input(d),
            32,
            -1,
            True,
            True,
            output_tensor=(
                ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=TILE),
                ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.int32), dtype=ttnn.uint16, layout=TILE),
            ),
        ),
    ),
    # topk.cpp:236 — prealloc indices on device
    Case(
        "prealloc_indices_host",
        "Preallocated output indices tensor must be on device",
        lambda d: ttnn.topk(
            _topk_input(d),
            32,
            -1,
            True,
            True,
            output_tensor=(
                _tile(d, shape=(1, 1, 32, 32)),
                ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.int32), dtype=ttnn.uint16, layout=TILE),
            ),
        ),
    ),
    # topk.cpp:238-242 — prealloc values shape
    Case(
        "prealloc_values_shape",
        "Preallocated values tensor has incorrect shape",
        lambda d: ttnn.topk(
            _topk_input(d),
            32,
            -1,
            True,
            True,
            output_tensor=(
                _tile(d, shape=(1, 1, 32, 64)),
                ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.int32), dtype=ttnn.uint16, layout=TILE, device=d),
            ),
        ),
    ),
    # topk.cpp:243-247 — prealloc indices shape
    Case(
        "prealloc_indices_shape",
        "Preallocated indices tensor has incorrect shape",
        lambda d: ttnn.topk(
            _topk_input(d),
            32,
            -1,
            True,
            True,
            output_tensor=(
                _tile(d, shape=(1, 1, 32, 32)),
                ttnn.from_torch(torch.zeros(1, 1, 32, 64, dtype=torch.int32), dtype=ttnn.uint16, layout=TILE, device=d),
            ),
        ),
    ),
    # topk_device_operation.cpp:193 — sharded output not supported
    # (the memory_config needs a ShardSpec: a spec-less sharded config throws
    #  "bad optional access" in output-spec creation before validate runs)
    Case(
        "sharded_output_config",
        "Sharded implementation not supported yet",
        lambda d: ttnn.topk(
            _topk_input(d),
            32,
            -1,
            True,
            True,
            memory_config=_height_shard_cfg(0, 0, 0, 0, [32, 32]),
        ),
    ),
    # topk_device_operation.cpp:196 — input must be TILE (tile-aligned RM dodges the pad)
    Case(
        "rm_input",
        "The input must be in tiled format",
        lambda d: ttnn.topk(_rm(d, shape=(1, 1, 32, 64)), 32, -1, True, True),
    ),
    # topk_device_operation.cpp:209-213 — indices_tensor dtype whitelist
    Case(
        "indices_tensor_dtype",
        "Optional input tensor must be UINT16, UINT32, or INT32",
        lambda d: ttnn.topk(
            _topk_input(d, shape=(1, 1, 32, 8192)),
            32,
            -1,
            True,
            True,
            indices_tensor=_tile(d, shape=(1, 1, 32, 8192)),
        ),
    ),
    # reduce_op_validation.cpp:76-82 via topk — sub_core_grids inside device grid
    Case(
        "sub_core_grids_oob",
        "must be contained in device compute grid",
        lambda d: ttnn.topk(_topk_input(d), 32, -1, True, True, sub_core_grids=_grid(0, 0, 90, 90)),
    ),
    # topk_device_operation.cpp:284-287 — multicore needs one core range
    Case(
        "multicore_two_ranges",
        "Only one core range is supported right now",
        lambda d: ttnn.topk(
            _topk_input(d, shape=(1, 1, 32, 8192)),
            32,
            -1,
            True,
            True,
            sub_core_grids=ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 4), ttnn.CoreCoord(6, 6)),
                ]
            ),
        ),
    ),
    # topk_device_operation.cpp:313 — can_run cost model
    Case(
        "k_exceeds_l1",
        "Not enough cores or cache size available to run TopK operation",
        lambda d: ttnn.topk(_topk_input(d, shape=(1, 1, 32, 8192)), 8192, -1, True, True),
    ),
    # topk_utils.cpp:68 via topk_device_operation.cpp:292 — degenerate multicore grid
    Case(
        "multicore_grid_too_small",
        "Core grid must contain at least one core",
        lambda d: ttnn.topk(
            _topk_input(d, shape=(1, 1, 32, 8192)), 32, -1, True, True, sub_core_grids=_grid(0, 0, 0, 0)
        ),
    ),
]


@pytest.mark.parametrize("case", TOPK_CASES, ids=[c.id for c in TOPK_CASES])
def test_topk_validation(device, expect_error, case):
    _run_cases(device, expect_error, case)


# ---------------------------------------------------------------------------
# prod (prod.cpp, device/prod_all_device_operation.cpp,
#       device/prod_nc_device_operation.cpp, prod_nc_op.cpp)
# ---------------------------------------------------------------------------


def _prod_nc_out(device, shape, dtype=ttnn.bfloat16, torch_dtype=torch.float32):
    return ttnn.from_torch(torch.zeros(shape, dtype=torch_dtype), dtype=dtype, layout=TILE, device=device)


PROD_CASES = [
    # prod.cpp:77-81 — dim range (rank 4)
    Case("dim_oor", "Dimension for prod is out of range", lambda d: ttnn.prod(_tile(d, shape=(1, 1, 32, 32)), dim=4)),
    # prod.cpp:77-81 — dim range, rank-0 branch
    Case(
        "dim_oor_rank0",
        "expected to be in range of",
        lambda d: ttnn.prod(ttnn.from_torch(torch.tensor(2.0), dtype=ttnn.bfloat16, layout=TILE, device=d), dim=1),
    ),
    # prod.cpp:116-118 — bfloat4_b with dim rejected (issue #48813)
    Case(
        "bf4_with_dim",
        "does not support bfloat4_b",
        lambda d: ttnn.prod(_tile(d, dtype=ttnn.bfloat4_b), dim=0),
    ),
    # prod_all_device_operation.cpp:24-28 — dtype whitelist (dim=None path)
    Case(
        "prod_all_int32",
        "expected BFLOAT16, FLOAT32, BFLOAT8_B or BFLOAT4_B",
        lambda d: ttnn.prod(_tile(d, dtype=ttnn.int32, torch_dtype=torch.int32)),
    ),
    # prod_nc_device_operation.cpp:34-38 — dtype whitelist via dim path
    Case(
        "prod_dim_int32",
        "expected BFLOAT16, FLOAT32 or BFLOAT8_B",
        lambda d: ttnn.prod(_tile(d, shape=(1, 2, 32, 32), dtype=ttnn.int32, torch_dtype=torch.int32), dim=0),
    ),
    # prod_nc_device_operation.cpp:11 — NC overload dim 0-3
    Case(
        "nc_dim_oor",
        "dim should be 0 - 3",
        lambda d: ttnn.prod(_tile(d, shape=(2, 2, 32, 32)), _prod_nc_out(d, (1, 2, 32, 32)), dims=[4]),
    ),
    # prod_nc_device_operation.cpp:16 — NC overload rank 4
    Case(
        "nc_rank3",
        "rank should be 4",
        lambda d: ttnn.prod(_tile(d, shape=(2, 32, 32)), _prod_nc_out(d, (1, 32, 32)), dims=[0]),
    ),
    # prod_nc_device_operation.cpp:25-32 — NC shape match
    Case(
        "nc_shape_mismatch",
        "Input and output shapes must match at dimension 3",
        lambda d: ttnn.prod(_tile(d, shape=(2, 2, 32, 32)), _prod_nc_out(d, (1, 2, 32, 64)), dims=[0]),
    ),
    # prod_nc_device_operation.cpp:34-38 — NC dtype whitelist excludes bfloat4_b
    Case(
        "nc_bf4",
        "but got DataType::BFLOAT4_B",
        lambda d: ttnn.prod(
            _tile(d, shape=(2, 2, 32, 32), dtype=ttnn.bfloat4_b),
            _prod_nc_out(d, (1, 2, 32, 32), dtype=ttnn.bfloat4_b),
            dims=[0],
        ),
    ),
    # prod_nc_op.cpp:17 — intermediate dim in multi-dim loop
    Case(
        "nc_multi_dim_hw",
        "Unsupported dim 3 for prod nc op",
        lambda d: ttnn.prod(_tile(d, shape=(2, 2, 32, 32)), _prod_nc_out(d, (1, 2, 32, 32)), dims=[0, 3]),
    ),
    # prod_nc_program_factory.cpp:25 — single dim must be 0/1
    Case(
        "nc_dim2",
        "must be either 0 or 1",
        lambda d: ttnn.prod(_tile(d, shape=(2, 2, 32, 32)), _prod_nc_out(d, (2, 2, 1, 32)), dims=[2]),
    ),
    # prod_nc_op.cpp:24-27 — multi-dim loop requires device input
    Case(
        "nc_host_input",
        "Input tensor must be stored on device",
        lambda d: ttnn.prod(
            ttnn.from_torch(torch.zeros(2, 2, 32, 32), dtype=ttnn.bfloat16, layout=RM),
            _prod_nc_out(d, (1, 1, 32, 32)),
            dims=[0, 1],
            memory_config=DRAM,
        ),
    ),
    # device_operation.hpp:455 (framework guard; masks prod_all_device_operation.cpp:13-16)
    Case(
        "host_input_framework_guard",
        "Device Operations expect device tensors as inputs",
        lambda d: ttnn.prod(ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16, layout=TILE)),
    ),
]


@pytest.mark.parametrize("case", PROD_CASES, ids=[c.id for c in PROD_CASES])
def test_prod_validation(device, expect_error, case):
    _run_cases(device, expect_error, case)


# ---------------------------------------------------------------------------
# cumsum / cumprod (accumulation_common.cpp, device/accumulation_device_operation.cpp)
# The dim-range / shape-mismatch / RM-layout checks are message-tested in
# test_cumsum.py / test_cumprod.py failing-cases; these rows cover the rest.
# ---------------------------------------------------------------------------

ACCUMULATION_CASES = [
    # accumulation_common.cpp:76 — prealloc out must be on device
    Case(
        "prealloc_host",
        "Preallocated output tensor must be on device",
        lambda d: ttnn.cumsum(
            _tile(d, shape=(1, 1, 32, 32)), 0, out=ttnn.zeros([1, 1, 32, 32], dtype=ttnn.bfloat16, layout=TILE)
        ),
    ),
    # accumulation_device_operation.cpp:44-46 — sharded input rejected
    Case(
        "sharded_input",
        "do not support sharded input tensors",
        lambda d: ttnn.cumsum(
            _tile(d, shape=(1, 1, 256, 256), memory_config=_height_shard_cfg(0, 0, 0, 3, [64, 256])), 0
        ),
    ),
    # device_operation.hpp:455 (framework guard; masks accumulation :33-37)
    Case(
        "host_input_framework_guard",
        "Device Operations expect device tensors as inputs",
        lambda d: ttnn.cumsum(ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=TILE), 0),
    ),
    # reduce_op_validation.cpp:96 via accumulation — output shard grid within device grid
    Case(
        "output_shard_grid_oob",
        "must be contained in device compute grid",
        lambda d: ttnn.cumsum(
            _tile(d, shape=(1, 1, 32, 32)), 0, memory_config=_height_shard_cfg(63, 63, 63, 63, [32, 32])
        ),
    ),
    # reduce_op_validation.cpp:144-157 via accumulation program factory — shard grid within program grid
    Case(
        "output_shard_outside_program_grid",
        "must be contained in program core grid",
        lambda d: ttnn.cumsum(_tile(d, shape=(1, 1, 32, 32)), 0, memory_config=_height_shard_cfg(5, 5, 5, 5, [32, 32])),
    ),
]


@pytest.mark.parametrize("case", ACCUMULATION_CASES, ids=[c.id for c in ACCUMULATION_CASES])
def test_accumulation_validation(device, expect_error, case):
    _run_cases(device, expect_error, case)


# ---------------------------------------------------------------------------
# ema (accumulation/ema/device/ema_device_operation.cpp)
# NOTE: ttnn.ema on a HOST input segfaults today (null device deref in
# ema.cpp:21) — that check is untestable until fixed and deliberately absent.
# ---------------------------------------------------------------------------


def _ema_input(device, shape=(1, 2, 3, 64), dtype=ttnn.bfloat16):
    return _tile(device, shape=shape, dtype=dtype)


EMA_CASES = [
    # ema_device_operation.cpp:21 — input BFLOAT16 only
    Case(
        "fp32_input",
        "Input tensor must be BFLOAT16",
        lambda d: ttnn.ema(_ema_input(d, dtype=ttnn.float32), 0.5),
    ),
    # ema_device_operation.cpp:28 — input TILE only
    Case("rm_input", "Input tensor must have TILE layout", lambda d: ttnn.ema(_rm(d, shape=(1, 2, 3, 64)), 0.5)),
    # ema_device_operation.cpp:32 — rank 4 required
    Case("rank3", "EMA input must be 4D", lambda d: ttnn.ema(_tile(d, shape=(2, 3, 64)), 0.5)),
    # ema_device_operation.cpp:33 — leading dim must be 1
    Case(
        "leading_dim",
        "EMA expects leading dimension to be 1, got 2",
        lambda d: ttnn.ema(_tile(d, shape=(2, 2, 3, 64)), 0.5),
    ),
    # ema_device_operation.cpp:42 — out BFLOAT16 only
    Case(
        "out_fp32",
        "Output tensor must be BFLOAT16",
        lambda d: ttnn.ema(
            _ema_input(d), 0.5, out=ttnn.zeros([1, 2, 3, 64], dtype=ttnn.float32, layout=TILE, device=d)
        ),
    ),
    # ema_device_operation.cpp:51 — out TILE only
    Case(
        "out_rm",
        "Output tensor must have TILE layout",
        lambda d: ttnn.ema(_ema_input(d), 0.5, out=ttnn.zeros([1, 2, 3, 64], dtype=ttnn.bfloat16, layout=RM, device=d)),
    ),
    # ema_device_operation.cpp:56 — volume match
    Case(
        "out_volume",
        "Input and output must have the same volume",
        lambda d: ttnn.ema(
            _tile(d, shape=(1, 1, 32, 32)),
            0.5,
            out=ttnn.zeros([1, 1, 32, 64], dtype=ttnn.bfloat16, layout=TILE, device=d),
        ),
    ),
    # ema_device_operation.cpp:60 — alpha NaN
    Case("alpha_nan", "EMA alpha must be a valid number, got NaN", lambda d: ttnn.ema(_ema_input(d), float("nan"))),
    # device_operation.hpp:455 (framework guard; masks ema :43-46)
    Case(
        "out_host_framework_guard",
        "Device Operations expect device tensors as inputs",
        lambda d: ttnn.ema(_ema_input(d), 0.5, out=ttnn.zeros([1, 2, 3, 64], dtype=ttnn.bfloat16, layout=TILE)),
    ),
]


@pytest.mark.parametrize("case", EMA_CASES, ids=[c.id for c in EMA_CASES])
def test_ema_validation(device, expect_error, case):
    _run_cases(device, expect_error, case)


# ---------------------------------------------------------------------------
# moe (moe.cpp, device/moe_device_operation.cpp)
# ---------------------------------------------------------------------------


def _moe_masks(device, w=64, k=32, experts=8, top_e=2, expert_layout=TILE, topk_layout=TILE):
    inf = float("-inf")
    expert_mask = torch.zeros(1, 1, 1, w, dtype=torch.bfloat16)
    expert_mask[..., experts:] = inf
    topk_mask = torch.zeros(1, 1, 1, k, dtype=torch.bfloat16)
    topk_mask[..., top_e:] = inf
    return (
        ttnn.from_torch(expert_mask, dtype=ttnn.bfloat16, layout=expert_layout, device=device),
        ttnn.from_torch(topk_mask, dtype=ttnn.bfloat16, layout=topk_layout, device=device),
    )


def _moe(device, input_shape=(1, 1, 32, 64), k=32, input_layout=TILE, mask_w=None, mask_k=None, **mask_kw):
    inp = ttnn.from_torch(
        torch.zeros(input_shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=input_layout, device=device
    )
    expert_mask, topk_mask = _moe_masks(device, w=mask_w or input_shape[-1], k=mask_k or k, **mask_kw)
    return inp, expert_mask, topk_mask


MOE_CASES = [
    # moe.cpp:40 — zero-volume prealloc on device
    Case(
        "zero_volume_prealloc_host",
        "Preallocated output tensor must be on device",
        lambda d: ttnn.moe(
            *_moe(d, input_shape=(1, 1, 0, 64)),
            32,
            output_tensor=ttnn.from_torch(
                torch.zeros(1, 1, 0, 1, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=TILE
            ),
        ),
    ),
    # moe.cpp:42-46 — zero-volume prealloc shape
    Case(
        "zero_volume_prealloc_shape",
        "Preallocated output tensor has incorrect shape",
        lambda d: ttnn.moe(
            *_moe(d, input_shape=(1, 1, 0, 64)),
            32,
            output_tensor=_tile(d, shape=(1, 1, 32, 1)),
        ),
    ),
    # moe_device_operation.cpp:30 — rank 4
    Case(
        "rank3",
        "Input shape must be 4D",
        lambda d: ttnn.moe(_tile(d, shape=(1, 32, 64)), *_moe_masks(d), 32),
    ),
    # moe_device_operation.cpp:31 — k == 32
    Case("k16", "K must be equal to 32", lambda d: ttnn.moe(*_moe(d, k=16, mask_k=16), 16)),
    # moe_device_operation.cpp:33-36 — W >= 64
    Case("w32", "must be a multiple of 64", lambda d: ttnn.moe(*_moe(d, input_shape=(1, 1, 32, 32), mask_w=32), 32)),
    # moe_device_operation.cpp:37-40 — W power of two
    Case("w96", "must be a power of 2", lambda d: ttnn.moe(*_moe(d, input_shape=(1, 1, 32, 96), mask_w=96), 32)),
    # moe_device_operation.cpp:41-44 — N*C*H % 32 (reachable only via RM input)
    Case(
        "h16_rm",
        "must be a multiple of 32",
        lambda d: ttnn.moe(*_moe(d, input_shape=(1, 1, 16, 64), input_layout=RM), 32),
    ),
    # moe_device_operation.cpp:46 — sharded output rejected
    Case(
        "sharded_output",
        "Sharded implementation not supported yet",
        lambda d: ttnn.moe(*_moe(d), 32, memory_config=_height_shard_cfg(0, 0, 0, 0, [32, 32])),
    ),
    # moe_device_operation.cpp:47 — input TILE
    Case("rm_input", "The input must be in tiled format", lambda d: ttnn.moe(*_moe(d, input_layout=RM), 32)),
    # moe_device_operation.cpp:61-66 — topk mask broadcastable, last dim == k
    Case(
        "topk_mask_wrong_width",
        "Topk mask must be row-broadcastable with last dim",
        lambda d: ttnn.moe(*_moe(d, mask_k=64), 32),
    ),
    # moe_device_operation.cpp:67-73 — expert mask broadcastable, last dim == W
    Case(
        "expert_mask_wrong_width",
        "Expert mask must be row-broadcastable with last dim",
        lambda d: ttnn.moe(*_moe(d, mask_w=32), 32),
    ),
    # moe_device_operation.cpp:74 — topk mask padded[-2] == 32 (RM mask)
    Case(
        "topk_mask_rm",
        "Topk shape inner dim must be padded to 32",
        lambda d: ttnn.moe(*_moe(d, topk_layout=RM), 32),
    ),
    # moe_device_operation.cpp:75 — expert mask padded[-2] == 32 (RM mask)
    Case(
        "expert_mask_rm",
        "Expert shape inner dim must be padded to 32",
        lambda d: ttnn.moe(*_moe(d, expert_layout=RM), 32),
    ),
]


@pytest.mark.parametrize("case", MOE_CASES, ids=[c.id for c in MOE_CASES])
def test_moe_validation(device, expect_error, case):
    _run_cases(device, expect_error, case)


# ---------------------------------------------------------------------------
# sampling (device/sampling_device_operation.cpp)
# Baseline: W=64 (Wt=2) deliberately clear of the W=32/Wt=1 hang (#52348).
# INT32 indices/k keep every row valid on both WH and BH.
# ---------------------------------------------------------------------------


def _sampling_args(
    device,
    shape=(1, 1, 32, 64),
    users=32,
    values_dtype=ttnn.bfloat16,
    values_layout=TILE,
    values_mc=None,
    indices_dtype=ttnn.int32,
    indices_layout=RM,
    indices_shape=None,
    indices_mc=None,
    k_dtype=ttnn.int32,
    k_layout=RM,
    k_len=None,
    p_dtype=ttnn.bfloat16,
    p_layout=RM,
    p_len=None,
    temp_dtype=ttnn.bfloat16,
    temp_layout=RM,
    temp_len=None,
):
    values = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=values_dtype,
        layout=values_layout,
        device=device,
        memory_config=values_mc,
    )
    ishape = indices_shape or shape
    indices = ttnn.from_torch(
        torch.arange(ishape[-1], dtype=torch.int32).expand(ishape).contiguous(),
        dtype=indices_dtype,
        layout=indices_layout,
        device=device,
        memory_config=indices_mc,
    )
    k = ttnn.from_torch(
        torch.full((k_len or users,), 10, dtype=torch.int32), dtype=k_dtype, layout=k_layout, device=device
    )
    p = ttnn.from_torch(
        torch.zeros(p_len or users, dtype=torch.bfloat16), dtype=p_dtype, layout=p_layout, device=device
    )
    temp = ttnn.from_torch(
        torch.ones(temp_len or users, dtype=torch.bfloat16), dtype=temp_dtype, layout=temp_layout, device=device
    )
    return values, indices, k, p, temp


def _sampling(device, sub_core_grids=None, output_tensor=None, **kw):
    args = _sampling_args(device, **kw)
    return ttnn.sampling(*args, seed=42, sub_core_grids=sub_core_grids, output_tensor=output_tensor)


def _sampling_prealloc(device, shape=(1, 1, 1, 32), dtype=ttnn.int32, layout=RM, memory_config=None):
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch.int32), dtype=dtype, layout=layout, device=device, memory_config=memory_config
    )


SAMPLING_CASES = [
    # sampling_device_operation.cpp:36-38
    Case(
        "values_sharded",
        "Only INTERLEAVED memory layout is supported for inputs",
        lambda d: _sampling(d, values_mc=_height_shard_cfg(0, 0, 0, 0, [32, 64])),
    ),
    # :40
    Case("values_fp32", "Only BFLOAT16 is supported for inputs", lambda d: _sampling(d, values_dtype=ttnn.float32)),
    # :41
    Case("values_rm", "Only TILE_LAYOUT is supported for inputs", lambda d: _sampling(d, values_layout=RM)),
    # :47-49
    Case(
        "indices_uint16",
        "dtypes are supported for input indices",
        lambda d: _sampling(d, indices_dtype=ttnn.uint16),
    ),
    # :55
    Case("indices_tile", "Only ROW_MAJOR is supported for input indices", lambda d: _sampling(d, indices_layout=TILE)),
    # :57-59
    Case(
        "shape_mismatch",
        "Input values and indices must have the same shape",
        lambda d: _sampling(d, indices_shape=(1, 1, 32, 128)),
    ),
    # :61
    Case("rank3", "must be rank-4", lambda d: _sampling(d, shape=(1, 32, 64), indices_shape=(1, 32, 64))),
    # :66-72
    Case(
        "batch2",
        "Sampling requires input dims 0 and 1",
        lambda d: _sampling(d, shape=(2, 1, 32, 64), indices_shape=(2, 1, 32, 64)),
    ),
    # :75-78
    Case(
        "users64",
        "Sampling currently supports between 1 and 32 users",
        lambda d: _sampling(d, shape=(1, 1, 64, 64), indices_shape=(1, 1, 64, 64)),
    ),
    # :79-82
    Case(
        "w48",
        "must be non-zero and divisible by 32",
        lambda d: _sampling(d, shape=(1, 1, 32, 48), indices_shape=(1, 1, 32, 48)),
    ),
    # :88-93 (#44558)
    Case(
        "wt3_not_pow2",
        "must yield a power-of-2 number of tiles",
        lambda d: _sampling(d, shape=(1, 1, 32, 96), indices_shape=(1, 1, 32, 96)),
    ),
    # :95-100 → reduce_op_validation.cpp:76-82
    Case(
        "sub_core_grids_oob",
        "must be contained in device compute grid",
        lambda d: _sampling(d, sub_core_grids=_grid(0, 0, 15, 15)),
    ),
    # :105-109
    Case(
        "grid_fewer_than_users",
        "Subcore grid must supply at least num_users",
        lambda d: _sampling(d, sub_core_grids=_grid(0, 0, 3, 3)),
    ),
    # :112-115
    Case(
        "prealloc_bf16",
        "dtypes are supported for outputs",
        lambda d: _sampling(
            d,
            output_tensor=ttnn.from_torch(
                torch.zeros(1, 1, 1, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=RM, device=d
            ),
        ),
    ),
    # :121-123
    Case(
        "prealloc_sharded",
        "Only INTERLEAVED memory layout is supported for outputs",
        lambda d: _sampling(
            d, output_tensor=_sampling_prealloc(d, memory_config=_height_shard_cfg(0, 0, 0, 0, [1, 32]))
        ),
    ),
    # :127-130
    Case(
        "prealloc_rank3",
        "Sampling preallocated output must be rank-4",
        lambda d: _sampling(d, output_tensor=_sampling_prealloc(d, shape=(1, 1, 32))),
    ),
    # :131-139
    Case(
        "prealloc_shape",
        "Sampling preallocated output logical shape must be",
        lambda d: _sampling(d, output_tensor=_sampling_prealloc(d, shape=(1, 1, 1, 64))),
    ),
    # :143-145
    Case("k_bf16", "dtypes are supported for k", lambda d: _sampling(d, k_dtype=ttnn.bfloat16)),
    # :150
    Case("p_fp32", "Only BFLOAT16 dtypes are supported for p", lambda d: _sampling(d, p_dtype=ttnn.float32)),
    # :151
    Case("temp_fp32", "Only BFLOAT16 dtypes are supported for temp", lambda d: _sampling(d, temp_dtype=ttnn.float32)),
    # :152
    Case("k_tile", "Only ROW_MAJOR layout is supported for k", lambda d: _sampling(d, k_layout=TILE)),
    # :153
    Case("p_tile", "Only ROW_MAJOR layout is supported for p", lambda d: _sampling(d, p_layout=TILE)),
    # :154
    Case("temp_tile", "Only ROW_MAJOR layout is supported for temp", lambda d: _sampling(d, temp_layout=TILE)),
    # :157
    Case("k_len16", "k must have shape", lambda d: _sampling(d, k_len=16)),
    # :158
    Case("p_len16", "p must have shape", lambda d: _sampling(d, p_len=16)),
    # :159
    Case("temp_len16", "temp must have shape", lambda d: _sampling(d, temp_len=16)),
]


@pytest.mark.parametrize("case", SAMPLING_CASES, ids=[c.id for c in SAMPLING_CASES])
def test_sampling_validation(device, expect_error, case):
    _run_cases(device, expect_error, case)


# ---------------------------------------------------------------------------
# manual_seed (device/manual_seed_operation.cpp)
# The tensor-seeds + scalar-user_ids row is already message-tested in
# test_manual_seed.py.
# ---------------------------------------------------------------------------


def _u32_rm(device, values, shape=None):
    t = torch.tensor(values, dtype=torch.int32)
    if shape is not None:
        t = t.reshape(shape)
    return ttnn.from_torch(t, dtype=ttnn.uint32, layout=RM, device=device)


def _i32_rm(device, values):
    return ttnn.from_torch(torch.tensor(values, dtype=torch.int32), dtype=ttnn.int32, layout=RM, device=device)


MANUAL_SEED_CASES = [
    # manual_seed_operation.cpp:125 — device required for scalar seed
    Case("scalar_seed_no_device", "Device must be provided when seeds is a", lambda d: ttnn.manual_seed(42)),
    # :92-94 — scalar user_id range
    Case(
        "user_id_oob",
        "User IDs scalar must be in the range",
        lambda d: ttnn.manual_seed(42, device=d, user_ids=32),
    ),
    # :65 — seeds tensor dtype
    Case(
        "seeds_int32",
        "Seeds tensor must be of type UINT32",
        lambda d: ttnn.manual_seed(_i32_rm(d, [4, 5, 6]), user_ids=_u32_rm(d, [0, 1, 2])),
    ),
    # :66 — seeds tensor layout
    Case(
        "seeds_tile",
        "Seeds tensor must have ROW_MAJOR layout",
        lambda d: ttnn.manual_seed(
            ttnn.from_torch(torch.tensor([4, 5, 6], dtype=torch.int32), dtype=ttnn.uint32, layout=TILE, device=d),
            user_ids=_u32_rm(d, [0, 1, 2]),
        ),
    ),
    # :70 — user_ids tensor dtype
    Case(
        "user_ids_int32",
        "User IDs tensor must be of type UINT32",
        lambda d: ttnn.manual_seed(_u32_rm(d, [4, 5, 6]), user_ids=_i32_rm(d, [0, 1, 2])),
    ),
    # :71 — user_ids tensor layout
    Case(
        "user_ids_tile",
        "User IDs tensor must have ROW_MAJOR layout",
        lambda d: ttnn.manual_seed(
            _u32_rm(d, [4, 5, 6]),
            user_ids=ttnn.from_torch(
                torch.tensor([0, 1, 2], dtype=torch.int32), dtype=ttnn.uint32, layout=TILE, device=d
            ),
        ),
    ),
    # :72-77 — same volume
    Case(
        "volume_mismatch",
        "must have the same number of elements",
        lambda d: ttnn.manual_seed(_u32_rm(d, [4, 5, 6]), user_ids=_u32_rm(d, [0, 1, 2, 3])),
    ),
    # :78-80 — same shape
    Case(
        "shape_mismatch",
        "must have the same shape",
        lambda d: ttnn.manual_seed(_u32_rm(d, [4, 5, 6], shape=(1, 3)), user_ids=_u32_rm(d, [0, 1, 2], shape=(3, 1))),
    ),
    # :81-82 — rank 1
    Case(
        "rank2",
        "must be 1-dimensional",
        lambda d: ttnn.manual_seed(_u32_rm(d, [4, 5, 6], shape=(1, 3)), user_ids=_u32_rm(d, [0, 1, 2], shape=(1, 3))),
    ),
]


@pytest.mark.parametrize("case", MANUAL_SEED_CASES, ids=[c.id for c in MANUAL_SEED_CASES])
def test_manual_seed_validation(device, expect_error, case):
    _run_cases(device, expect_error, case)


# ---------------------------------------------------------------------------
# Shared validators (reduce_op_validation.cpp) via consumers no other group
# reaches: ND shard grid containment, tile-aligned shard checks.
# ---------------------------------------------------------------------------

SHARED_VALIDATOR_CASES = [
    # reduce_op_validation.cpp:101-103 — ND shard grid within device grid (via cumsum)
    Case(
        "nd_shard_grid_oob",
        "ND shard grid .+ must be contained in device compute grid",
        lambda d: ttnn.cumsum(
            _tile(d, shape=(1, 1, 32, 32)),
            0,
            memory_config=ttnn.MemoryConfig(
                ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape([32, 32]), _grid(63, 63, 63, 63))
            ),
        ),
    ),
    # reduce_op_validation.cpp:34-41 — tile-height-aligned shard (via prod_nc RM-sharded input)
    Case(
        "prod_nc_shard_not_tile_aligned",
        "must be tile-height-aligned",
        lambda d: ttnn.prod(
            ttnn.from_torch(
                torch.zeros(2, 1, 32, 32, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=RM,
                device=d,
                memory_config=_height_shard_cfg(0, 0, 3, 0, [16, 32]),
            ),
            ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=RM, device=d),
            dims=[0],
        ),
    ),
]


@pytest.mark.parametrize("case", SHARED_VALIDATOR_CASES, ids=[c.id for c in SHARED_VALIDATOR_CASES])
def test_shared_validator_checks(device, expect_error, case):
    _run_cases(device, expect_error, case)
