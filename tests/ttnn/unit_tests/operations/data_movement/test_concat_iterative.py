# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

# ==================================================
# 1-1. Small Tensor (shape [1,1], interleaved)
# ==================================================


@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("num_inputs", [2, 32, 64, 128, 256, 512, 1024, 2048])
def test_concat_small_tensors_1(device, tensor_layout, dim, num_inputs):
    input_tensor = torch.tensor([[1.0]], dtype=torch.bfloat16)
    memory_layout = ttnn.TensorMemoryLayout.INTERLEAVED

    memory_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.DRAM)

    tt_tensor = ttnn.from_torch(
        input_tensor, dtype=ttnn.bfloat16, layout=tensor_layout, memory_config=memory_config, device=device
    )

    tt_concat_inputs = []
    for i in range(num_inputs):
        tt_concat_inputs.append(tt_tensor)

    tt_concat_output = ttnn.concat(tt_concat_inputs, dim=dim)

    print(
        f"Test small_tensors: Success for {tensor_layout}, #tensor = {num_inputs}, dim={dim}, out_shape={tt_concat_output.shape}"
    )


# ==================================================
# 1-2. Small Tensor ([64,63~65] or [64,127~129])
# ==================================================


@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("num_inputs", [2, 32, 64, 128, 256, 512])
@pytest.mark.parametrize(
    "tid, input_shapes",
    [
        (1, ((64, 64), (64, 128))),
        (2, ((64, 63), (64, 127))),
        (3, ((64, 65), (64, 127))),
        (4, ((65, 63), (65, 129))),
        (5, ((65, 65), (65, 129))),
    ],
)
def test_concat_small_tensors_2(device, tensor_layout, dim, num_inputs, input_shapes, tid):
    memory_layout = ttnn.TensorMemoryLayout.INTERLEAVED
    memory_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.DRAM)

    tensor_shape = []
    if dim == -1:
        tensor_shape = input_shapes
    else:
        tensor_shape = [s[::-1] for s in input_shapes]

    torch_input_tensor1 = torch.full(tensor_shape[0], -1.0, dtype=torch.bfloat16)
    torch_input_tensor2 = torch.full(tensor_shape[1], 1.0, dtype=torch.bfloat16)

    tt_input_tensor1 = ttnn.from_torch(
        torch_input_tensor1, dtype=ttnn.bfloat16, layout=tensor_layout, memory_config=memory_config, device=device
    )
    tt_input_tensor2 = ttnn.from_torch(
        torch_input_tensor2, dtype=ttnn.bfloat16, layout=tensor_layout, memory_config=memory_config, device=device
    )

    tt_concat_inputs = []
    for i in range(num_inputs):
        if i % 2:
            tt_concat_inputs.append(tt_input_tensor1)
        else:
            tt_concat_inputs.append(tt_input_tensor2)

    tt_concat_output = ttnn.concat(tt_concat_inputs, dim=dim)
    print(
        f"Test small_tensors ({tid}): Success for {tensor_layout}, #tensor = {num_inputs}, dim={dim}, out_shape={tt_concat_output.shape}"
    )


# ============================================================
# 2. Mid Tensor
#  - input : Each input tensor can be loaded in one core's L1)
# ============================================================

## (1) runtime error: 512, dim=-1, ROW_MAJOR_LAYOUT,


@pytest.mark.parametrize("input_mem_config", ["interleaved_dram"])
@pytest.mark.parametrize("dim", [-2, -1])
@pytest.mark.parametrize("num_inputs", [2, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("input_shape", [(512, 1024)])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_concat_mid_tensor_1(device, input_mem_config, tensor_layout, dim, num_inputs, input_shape):
    core_grid = ttnn.CoreGrid(y=2, x=2)

    # Special case: do not test if output_tensor.page >= Core L1 memory
    if tensor_layout == ttnn.ROW_MAJOR_LAYOUT and num_inputs >= 500 and dim == -1:
        pytest.skip(
            "Skipping test: output tensor page size exceeds Core L1 memory for ROW_MAJOR_LAYOUT, num_inputs >= 500, dim = -1"
        )

    shape_align = None
    if input_mem_config == "interleaved_l1" or input_mem_config == "interleaved_dram":
        shape_align = input_shape
    elif tensor_layout == ttnn.ROW_MAJOR_LAYOUT:
        shape_align = (2 * ((dim + 1) // 2) for dim in input_shape)
        shape_align = tuple(shape_align)
    else:
        shape_align = (64 * ((dim + 63) // 64) for dim in input_shape)
        shape_align = tuple(shape_align)

    if dim == -1:
        out_shape_align = (shape_align[0], shape_align[1] * num_inputs)
    else:
        out_shape_align = (shape_align[0] * num_inputs, shape_align[1])

    output_memory_config = None
    if input_mem_config == "interleaved_l1":
        memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    elif input_mem_config == "sharded_width":
        memory_config = ttnn.create_sharded_memory_config(
            shape=(shape_align[0], shape_align[1] // 4), core_grid=core_grid, strategy=ttnn.ShardStrategy.WIDTH
        )
        output_memory_config = ttnn.create_sharded_memory_config(
            shape=(out_shape_align[0], out_shape_align[1] // 4), core_grid=core_grid, strategy=ttnn.ShardStrategy.WIDTH
        )
    elif input_mem_config == "sharded_height":
        memory_config = ttnn.create_sharded_memory_config(
            shape=(shape_align[0] // 4, shape_align[1]), core_grid=core_grid, strategy=ttnn.ShardStrategy.HEIGHT
        )
        output_memory_config = ttnn.create_sharded_memory_config(
            shape=(out_shape_align[0] // 4, out_shape_align[1]), core_grid=core_grid, strategy=ttnn.ShardStrategy.HEIGHT
        )
    elif input_mem_config == "sharded_block":
        memory_config = ttnn.create_sharded_memory_config(
            shape=(shape_align[0] // 2, shape_align[1] // 2), core_grid=core_grid, strategy=ttnn.ShardStrategy.BLOCK
        )
        output_memory_config = ttnn.create_sharded_memory_config(
            shape=(out_shape_align[0] // 2, out_shape_align[1] // 2),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
        )
    else:
        memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    torch_input_tensor1 = torch.full(input_shape, -1.0, dtype=torch.bfloat16)
    torch_input_tensor2 = torch.full(input_shape, 1.0, dtype=torch.bfloat16)
    tt_input_tensor1 = ttnn.from_torch(
        torch_input_tensor1, dtype=ttnn.bfloat16, layout=tensor_layout, memory_config=memory_config, device=device
    )
    tt_input_tensor2 = ttnn.from_torch(
        torch_input_tensor2, dtype=ttnn.bfloat16, layout=tensor_layout, memory_config=memory_config, device=device
    )

    tt_concat_inputs = []
    for i in range(num_inputs):
        if i % 2:
            tt_concat_inputs.append(tt_input_tensor1)
        else:
            tt_concat_inputs.append(tt_input_tensor2)

    if input_mem_config == "interleaved_l1" or input_mem_config == "interleaved_dram":
        tt_concat_output = ttnn.concat(tt_concat_inputs, dim=dim)
    else:
        tt_concat_output = ttnn.concat(tt_concat_inputs, dim=dim, memory_config=output_memory_config)

    print(
        f"Test2: [{tensor_layout},{input_mem_config}] num_inp: {num_inputs}, input_shape : {input_shape}, shape_align : {shape_align}, out_shape = {tt_concat_output.shape}"
    )


# ============================================================
# 3. Large Tensor - Automatic allocation # Chip
# ============================================================


@pytest.mark.parametrize("num_inputs", [2, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("tensor_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "dim, input_shapes", [(-1, ((1, 1, 1, 5000), (1, 1, 1, 33))), (-2, ((1, 1, 5000, 1), (1, 1, 33, 1)))]
)
def test_concat_lg_tensor_1(device, tensor_layout, num_inputs, dim, input_shapes):
    memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    if dim == -1 and num_inputs >= 256:
        pytest.skip("Skipping test: dim == -1 and num_inputs >= 256 is not supported.")

    torch_input_tensor1 = torch.full(input_shapes[0], -1.0, dtype=torch.bfloat16)
    torch_input_tensor2 = torch.full(input_shapes[1], 1.0, dtype=torch.bfloat16)
    tt_input_tensor1 = ttnn.from_torch(
        torch_input_tensor1, dtype=ttnn.bfloat16, layout=tensor_layout, memory_config=memory_config, device=device
    )
    tt_input_tensor2 = ttnn.from_torch(
        torch_input_tensor2, dtype=ttnn.bfloat16, layout=tensor_layout, memory_config=memory_config, device=device
    )

    tt_concat_inputs = []
    for i in range(num_inputs):
        if i % 2:
            tt_concat_inputs.append(tt_input_tensor1)
        else:
            tt_concat_inputs.append(tt_input_tensor2)

    tt_concat_output = ttnn.concat(tt_concat_inputs, dim=dim)

    print(
        f"Test3: [{tensor_layout}] num_inp: {num_inputs}, input_shape : {input_shapes[0]}, {input_shapes[1]}, out_shape = {tt_concat_output.shape}"
    )
