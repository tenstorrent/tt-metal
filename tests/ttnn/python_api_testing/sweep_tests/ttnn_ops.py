# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import tt_lib
from models.helper_funcs import Linear as tt_Linear


def torch_to_ttnn(x, device, layout, input_mem_config, dtype):
    input_tensor = ttnn.from_torch(x)
    input_tensor = ttnn.to_layout(input_tensor, layout)
    input_tensor = ttnn.to_device(input_tensor, device, memory_config=input_mem_config)

    # assert dtype == ttnn.bfloat16 , "ttnn sweeeps for now support only BFLOAT16 dtype"
    return input_tensor


def setup_ttnn_tensor(x, device, layout, input_mem_config, dtype):
    input_tensor = ttnn.from_torch(x)
    if layout == tt_lib.tensor.Layout.TILE:
        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    elif layout == tt_lib.tensor.Layout.ROW_MAJOR:
        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    else:
        assert False, "Unknown layout passed"

    assert dtype == tt_lib.tensor.DataType.BFLOAT16, "ttnn sweeeps for now support only BFLOAT16 dtype"

    if input_mem_config.buffer_type == tt_lib.tensor.BufferType.DRAM:
        input_tensor = ttnn.to_device(input_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    elif input_mem_config.buffer_type == tt_lib.tensor.BufferType.L1:
        input_tensor = ttnn.to_device(input_tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        assert False, "Unknown memory passed"

    return input_tensor


def ttnn_tensor_to_torch(x, output_mem_config):
    output_tensor = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    # assert output_mem_config == tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM)
    return ttnn.to_torch(output_tensor)


def eltwise_add(
    x,
    y,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.add(t0, t1, alpha=scalar)
    return ttnn_tensor_to_torch(t2, output_mem_config)


def eltwise_exp(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.exp(t0)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def permute(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    permute_dims,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.permute(t0, permute_dims)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def reshape(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    reshape_dims,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.reshape(t0, reshape_dims)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def gelu(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.gelu(t0)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def eltwise_sub(
    x,
    y,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.sub(t0, t1, alpha=scalar)
    return ttnn_tensor_to_torch(t2, output_mem_config)


def embeddings(x, y, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    x = x.int()
    x = torch.clamp(x, min=0, max=y.shape[0] - 1)

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[1])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.embedding(t0, t1, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t2, output_mem_config)


def eltwise_tanh(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.tanh(t0)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def softmax(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    # t2 = ttnn.add(t0, t1, alpha=scalar)
    t1 = ttnn.softmax(t0, dim=-1)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def mul(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.mul(t0, t1)
    return ttnn_tensor_to_torch(t2, output_mem_config)


def linear(x, weight, bias=None, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    # tensor preprocessing
    if bias is not None:
        bias = bias.repeat(1, 1, 32, 1)

    weight = torch.transpose(weight, 2, 3)
    batch_size = x.shape[0]
    num_cores_x = 12

    # ttnn setup
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    tt_weight = setup_ttnn_tensor(weight, device, layout[1], input_mem_config[1], dtype[1])

    if bias is not None:
        tt_bias = setup_ttnn_tensor(bias, device, layout[2], input_mem_config[2], dtype[2])
    else:
        tt_bias = None

    t1 = ttnn.linear(t0, tt_weight, bias=tt_bias, dtype=ttnn.bfloat16, core_grid=(batch_size, num_cores_x))
    return ttnn_tensor_to_torch(t1, output_mem_config)


def eltwise_softmax_in_place(x, *args, device, dtype, layout, input_mem_config, output_mem_config, **kwargs):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])

    t1 = ttnn.softmax(t0, -1)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def matmul(
    x,
    y,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = t0 @ t1  # ttnn.matmul(t0, t1)
    return ttnn_tensor_to_torch(t2, output_mem_config)


def layernorm(
    x,
    y,
    z,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])

    t3 = ttnn.layer_norm(t0, weight=t1, bias=t2)

    return ttnn_tensor_to_torch(t3, output_mem_config)


def layernorm_residual(
    x,
    y,
    z,
    w,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    t2 = setup_ttnn_tensor(z, device, layout[2], input_mem_config[2], dtype[2])
    t3 = setup_ttnn_tensor(w, device, layout[3], input_mem_config[3], dtype[3])

    t4 = ttnn.layer_norm(t0, residual_input_tensor=t1, weight=t2, bias=t3)

    return ttnn_tensor_to_torch(t4, output_mem_config)


def layernorm_noweights(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.layer_norm(t0)

    return ttnn_tensor_to_torch(t1, output_mem_config)
