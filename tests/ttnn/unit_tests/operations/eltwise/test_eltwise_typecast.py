# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)

mem_configs = [
    ttnn.DRAM_MEMORY_CONFIG,
    ttnn.L1_MEMORY_CONFIG,
]

TILE_HEIGHT = 32
TILE_WIDTH = 32

cpu_layout = ttnn.Layout.ROW_MAJOR
npu_layout = ttnn.Layout.TILE


@pytest.mark.parametrize(
    "pt_input_dtype, tt_input_dtype, tt_output_dtype",
    (
        (torch.bfloat16, ttnn.bfloat16, ttnn.float32),
        (torch.float32, ttnn.float32, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.int32),
        (torch.int, ttnn.int32, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.uint16),
        (torch.int, ttnn.uint16, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.uint32),
        (torch.int, ttnn.uint32, ttnn.bfloat16),
        (torch.float32, ttnn.float32, ttnn.int32),
        (torch.int, ttnn.int32, ttnn.float32),
        (torch.float32, ttnn.float32, ttnn.uint16),
        (torch.int, ttnn.uint16, ttnn.float32),
        (torch.float32, ttnn.float32, ttnn.uint32),
        (torch.int, ttnn.uint32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.int32),
        (torch.int, ttnn.int32, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.uint16),
        (torch.int, ttnn.uint16, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.uint32),
        (torch.int, ttnn.uint32, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.float32),
        (torch.float32, ttnn.float32, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.int32),
        (torch.int, ttnn.int32, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.uint16),
        (torch.int, ttnn.uint16, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.uint32),
        (torch.int, ttnn.uint32, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat16, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.float32),
        (torch.float32, ttnn.float32, ttnn.bfloat4_b),
        (torch.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b),
        (torch.int, ttnn.uint16, ttnn.uint32),
        (torch.int, ttnn.uint16, ttnn.int32),
        (torch.int, ttnn.int32, ttnn.uint16),
        (torch.int, ttnn.uint32, ttnn.uint16),
    ),
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 1, 32, 32]],  # Single core
        [[1, 1, 320, 320]],  # multi core
        [[1, 3, 320, 320]],  # multi core
    ],
)
@pytest.mark.parametrize(
    "input_mem_config",
    mem_configs,
)
@pytest.mark.parametrize(
    "dst_mem_config",
    mem_configs,
)
class TestTypecast:
    def test_run_eltwise_typecast_op(
        self,
        tt_output_dtype,
        pt_input_dtype,
        tt_input_dtype,
        input_shapes,
        input_mem_config,
        dst_mem_config,
        device,
        function_level_defaults,
    ):
        if tt_input_dtype == tt_output_dtype:
            pytest.skip("Same I/O data types. Skip.")
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=0, high=100), pt_input_dtype)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["tt_input_dtype"] = [tt_input_dtype]
        test_args["tt_output_dtype"] = [tt_output_dtype]
        test_args["input_mem_config"] = [input_mem_config]
        test_args.update({"output_mem_config": dst_mem_config})
        comparison_func = comparison_funcs.comp_pcc
        if tt_input_dtype == ttnn.bfloat4_b or tt_output_dtype == ttnn.bfloat4_b:
            comparison_func = partial(comparison_funcs.comp_pcc, pcc=0.98)

        run_single_pytorch_test(
            "eltwise-typecast",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )


@pytest.mark.skip("Issue #17237: Does not work with new mantissa rounding")
def test_typecast_bf16_to_bfp8_b(device):
    torch.manual_seed(0)
    shape = [32, 32]

    # bf16 --> bfp8_b by cpu.
    torch_bf16 = torch.randn(shape, dtype=torch.bfloat16)
    bfp8_b_by_cpu = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # bf16 --> bfp8_b by npu
    tt_bf16 = ttnn.Tensor(torch_bf16, ttnn.bfloat16).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_bf16, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed


def print_mismatches(cpu, npu, num_max_print):
    different_indices = (cpu != npu).nonzero(as_tuple=True)
    count = 0
    for idx in zip(*different_indices):
        count = count + 1
        print(f"idx={idx} cpu={cpu[idx]} npu={npu[idx]}")
        if count > num_max_print:
            break


@pytest.mark.skip("Issue #17237: Does not work with new mantissa rounding")
@pytest.mark.parametrize("seed", [0, 2, 4, 6, 8])
@pytest.mark.parametrize("scale", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("bias", [0, 1, 2, 4, 8, 16, 32, 64, 128])
def test_typecast_bf16_to_bfp8_b_various_input(seed, scale, bias, device):
    torch.manual_seed(seed)
    shape = [1024, 1024]

    bias = bias
    low = bias - scale
    high = bias + scale
    torch_bf16 = random_tensor = torch.empty(shape).uniform_(low, high).to(torch.bfloat16)

    random_signs = torch.randint(0, 2, shape) * 2 - 1
    torch_bf16 = torch_bf16 * random_signs

    # bf16 --> bfp8_b by cpu.
    bfp8_b_by_cpu = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # bf16 --> bfp8_b by npu
    tt_bf16 = ttnn.Tensor(torch_bf16, ttnn.bfloat16).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_bf16, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    if not passed:
        print_mismatches(cpu_version, npu_version, 16)
    assert passed


@pytest.mark.skip("Issue #17237: Does not work with new mantissa rounding")
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("scale", [4])
@pytest.mark.parametrize("bias", [2])
# NaN becomes -Inf when converted to bfloat8_b format, skip testing
@pytest.mark.parametrize("insert_inf, insert_nan", [[True, False]])  # , [False, True], [True, True]])
def test_typecast_bf16_to_bfp8_b_with_inf_nan(seed, scale, bias, insert_inf, insert_nan, device):
    torch.manual_seed(seed)
    shape = [1024, 1024]

    bias = bias
    low = bias - scale
    high = bias + scale

    torch_bf16 = random_tensor = torch.empty(shape).uniform_(low, high).to(torch.bfloat16)
    if insert_inf:
        num_inf = torch_bf16.numel() // 8  # 16 elements are pcked into
        inf_indices = torch.randint(0, torch_bf16.numel(), (num_inf,))
        torch_bf16.view(-1)[inf_indices] = float("inf")
    if insert_nan:
        num_nan = torch_bf16.numel() // 8
        nan_indices = torch.randint(0, torch_bf16.numel(), (num_nan,))
        torch_bf16.view(-1)[nan_indices] = float("nan")
    random_signs = torch.randint(0, 2, shape) * 2 - 1
    torch_bf16 = torch_bf16 * random_signs

    # bf16 --> bfp8_b by cpu.
    bfp8_b_by_cpu = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # bf16 --> bfp8_b by npu
    tt_bf16 = ttnn.Tensor(torch_bf16, ttnn.bfloat16).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_bf16, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    if not passed:
        print_mismatches(cpu_version, npu_version, 16)
    assert passed


def test_typecast_bfp8_b_to_bf16(device):
    torch.manual_seed(0)
    shape = [1024, 1024]

    # bfp8_b --> bf16 by cpu.
    torch_bf16 = torch.randn(shape, dtype=torch.bfloat16)
    bfp8_b = ttnn.Tensor(torch_bf16, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b.to(cpu_layout).to_torch()

    # bfp8_b --> bf16 by npu.
    bf16_by_npu = ttnn.typecast(bfp8_b.to(device), ttnn.bfloat16)
    npu_version = bf16_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed


def test_typecast_fp32_to_bfp8_b(device):
    torch.manual_seed(0)
    shape = [32, 32]

    # fp32 --> bfp8_b by cpu.
    torch_fp32 = torch.randn(shape, dtype=torch.float32)
    bfp8_b_by_cpu = ttnn.Tensor(torch_fp32, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b_by_cpu.to(cpu_layout).to_torch()

    # fp32 --> bfp8_b by npu
    tt_fp32 = ttnn.Tensor(torch_fp32, ttnn.float32).to(npu_layout).to(device)
    bfp8_b_by_npu = ttnn.typecast(tt_fp32, ttnn.bfloat8_b)
    npu_version = bfp8_b_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed


def test_typecast_bfp8_b_to_fp32(device):
    torch.manual_seed(0)
    shape = [1024, 1024]

    # bfp8_b --> fp32 by cpu.
    torch_fp32 = torch.randn(shape, dtype=torch.float32)
    bfp8_b = ttnn.Tensor(torch_fp32, ttnn.bfloat8_b).to(npu_layout)
    cpu_version = bfp8_b.to(cpu_layout).to_torch()

    # bfp8_b --> fp32 by npu.
    fp32_by_npu = ttnn.typecast(bfp8_b.to(device), ttnn.float32)
    npu_version = fp32_by_npu.cpu().to(cpu_layout).to_torch()

    passed = torch.equal(cpu_version, npu_version)
    # print(cpu_version[0, 0:16])
    # print(npu_version[0, 0:16])
    assert passed


@pytest.mark.parametrize(
    "shard_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
def test_typecast_legacy_sharded(device, shard_layout):
    torch.manual_seed(0)
    shape = [1, 1, 128, 128]

    if shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shard_shape = [64, 64]
    elif shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        shard_shape = [32, 128]
    else:
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        shard_shape = [128, 32]

    shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1, shard_spec)

    torch_input = torch.randint(0, 100, shape, dtype=torch.int32).float()
    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    output_tensor = ttnn.typecast(input_tensor, dtype=ttnn.int32)
    assert output_tensor.dtype == ttnn.int32

    result = ttnn.to_torch(output_tensor)
    expected = torch_input.int()
    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    "tensor_shape, nd_shard_shape, shard_grid",
    [
        (
            [1, 1, 128, 128],
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        ([4, 128, 128], [2, 64, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})),
        (
            [2, 2, 128, 128],
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (
            [3, 128, 128],
            [2, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),  # uneven: 3 % 2 != 0
        (
            [5, 4, 160, 160],
            [2, 3, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))}),
        ),  # uneven sharding in all dims
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
def test_typecast_nd_sharded_int(device, tensor_shape, nd_shard_shape, shard_grid, shard_orientation, layout):
    torch.manual_seed(0)

    nd_shard_spec = ttnn.NdShardSpec(shard_shape=nd_shard_shape, grid=shard_grid, orientation=shard_orientation)
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch_input = torch.randint(0, 100, tensor_shape, dtype=torch.int32).float()
    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.float32, layout=layout, device=device, memory_config=mem_config
    )
    output_tensor = ttnn.typecast(input_tensor, dtype=ttnn.int32)
    assert output_tensor.dtype == ttnn.int32

    result = ttnn.to_torch(output_tensor)
    expected = torch_input.int()
    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    "pt_input_dtype, tt_input_dtype, tt_output_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.float32),
    ],
)
@pytest.mark.parametrize(
    "tensor_shape, nd_shard_shape, shard_grid",
    [
        (
            [1, 1, 128, 128],
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (
            [4, 128, 128],
            [2, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
        (
            [3, 128, 128],
            [2, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),  # uneven in dim 0
        (
            [5, 3, 96, 160],
            [3, 2, 64, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),  # uneven in all dims: 5%3, 3%2, 96%64, 160%96
    ],
)
def test_typecast_nd_sharded_float(
    device, pt_input_dtype, tt_input_dtype, tt_output_dtype, tensor_shape, nd_shard_shape, shard_grid
):
    torch.manual_seed(0)

    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=nd_shard_shape, grid=shard_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch_input = torch.randn(tensor_shape, dtype=pt_input_dtype)

    # Build CPU-side reference: run the same input -> tt_input_dtype -> tt_output_dtype path on host
    # so that any precision loss from the input format is reflected in the expected output.
    cpu_tensor = ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT)
    cpu_reference = ttnn.to_torch(cpu_tensor)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    output_tensor = ttnn.typecast(input_tensor, dtype=tt_output_dtype)
    assert output_tensor.dtype == tt_output_dtype

    npu_result = ttnn.to_torch(output_tensor)

    # Compare in a common dtype so shapes/dtypes align
    common_dtype = torch.float32
    assert torch.equal(cpu_reference.to(common_dtype), npu_result.to(common_dtype))


# Tests that legacy-sharded inputs violating TypecastShardedProgramFactory preconditions
# fall back to TypecastProgramFactory correctly.
@pytest.mark.parametrize(
    "pt_input_dtype, tt_input_dtype, tt_output_dtype",
    [
        # Violates legacy sharded program factory condition: tile_size(input) != tile_size(output)
        # bfloat16 tile = 2048 bytes, float32 tile = 4096 bytes
        (torch.bfloat16, ttnn.bfloat16, ttnn.float32),
        # bfloat16 tile = 2048 bytes, bfp8_b tile = 1088 bytes
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16),
        (torch.bfloat16, ttnn.bfloat8_b, ttnn.float32),
    ],
)
@pytest.mark.parametrize(
    "shard_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
def test_typecast_legacy_sharded_tile_size_mismatch(
    device, pt_input_dtype, tt_input_dtype, tt_output_dtype, shard_layout
):
    """Violates is_valid_for_sharded_optimized_typecast condition 2: tile_size(input) != tile_size(output)."""
    torch.manual_seed(0)
    shape = [1, 1, 128, 128]

    if shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        shard_shape = [64, 64]
    elif shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        shard_shape = [32, 128]
    else:
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        shard_shape = [128, 32]

    shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1, shard_spec)

    torch_input = torch.randn(shape, dtype=pt_input_dtype)

    cpu_input = ttnn.from_torch(torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT)
    cpu_output = ttnn.typecast(cpu_input, dtype=tt_output_dtype)
    cpu_reference = ttnn.to_torch(cpu_output)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=tt_input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    output_tensor = ttnn.typecast(input_tensor, dtype=tt_output_dtype)
    assert output_tensor.dtype == tt_output_dtype

    npu_result = ttnn.to_torch(output_tensor)
    common_dtype = torch.float32
    assert torch.equal(cpu_reference.to(common_dtype), npu_result.to(common_dtype))


@pytest.mark.parametrize(
    "shard_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
)
def test_typecast_legacy_sharded_dram_buffer(device, shard_layout):
    """Violates is_valid_for_sharded_optimized_typecast condition 3: input buffer is in DRAM, not L1."""
    torch.manual_seed(0)
    shape = [1, 1, 128, 128]

    if shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        shard_shape = [32, 128]
    else:
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
        shard_shape = [128, 32]

    shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.DRAM, shard_spec)

    torch_input = torch.randint(0, 100, shape, dtype=torch.int32).float()

    cpu_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    cpu_output = ttnn.typecast(cpu_input, dtype=ttnn.int32)
    cpu_reference = ttnn.to_torch(cpu_output)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    output_tensor = ttnn.typecast(input_tensor, dtype=ttnn.int32)
    assert output_tensor.dtype == ttnn.int32

    npu_result = ttnn.to_torch(output_tensor)
    expected = torch_input.int()
    assert torch.equal(npu_result, expected)


def test_typecast_legacy_sharded_shard_size_not_tile_aligned(device):
    """Violates is_valid_for_sharded_optimized_typecast condition 5: shard_size_in_bytes % tile_size != 0.

    Uses ROW_MAJOR WIDTH_SHARDED with shard = [64, 8] and float32 (4 bytes).
    8 * 4 = 32 bytes, which IS a multiple of L1 alignment (16 bytes) — condition 4 satisfied.
    64 * 8 * 4 = 2048 bytes, tile_size(Float32) = 4096 — 2048 % 4096 != 0 — condition 5 violated.
    Uses float32 -> int32 (same tile size) so only condition 5 is violated.
    """
    torch.manual_seed(0)
    shape = [1, 1, 64, 32]

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    shard_shape = [64, 8]  # 4 cores, width 32 / 4 = 8

    shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    torch_input = torch.randint(0, 100, shape, dtype=torch.int32).float()

    cpu_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cpu_output = ttnn.typecast(cpu_input, dtype=ttnn.int32)
    cpu_reference = ttnn.to_torch(cpu_output)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )
    output_tensor = ttnn.typecast(input_tensor, dtype=ttnn.int32)
    assert output_tensor.dtype == ttnn.int32

    npu_result = ttnn.to_torch(output_tensor)
    expected = torch_input.int()
    assert torch.equal(npu_result, expected)
