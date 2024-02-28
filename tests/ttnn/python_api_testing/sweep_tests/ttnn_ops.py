# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import tt_lib
from models.helper_funcs import Linear as tt_Linear


def layout_to_ttnn(layout):
    if layout == tt_lib.tensor.Layout.TILE:
        return ttnn.TILE_LAYOUT

    elif layout == tt_lib.tensor.Layout.ROW_MAJOR:
        return ttnn.ROW_MAJOR_LAYOUT

    else:
        assert False, "Unknown layout passed"


def dtype_to_ttnn(dtype):
    if dtype == tt_lib.tensor.DataType.BFLOAT16:
        return ttnn.bfloat16

    elif dtype == tt_lib.tensor.DataType.BFLOAT8_B:
        return ttnn.bfloat8_b

    else:
        assert False, "Unknown dtype passed"


def memory_config_to_ttnn(mem_config):
    if mem_config is None:
        return None

    if mem_config.buffer_type == tt_lib.tensor.BufferType.DRAM:
        return ttnn.DRAM_MEMORY_CONFIG

    elif mem_config.buffer_type == tt_lib.tensor.BufferType.L1:
        return ttnn.L1_MEMORY_CONFIG

    else:
        assert False, "Unknown memory passed"


def setup_ttnn_tensor(x, device, layout, input_mem_config, dtype):
    input_tensor = ttnn.from_torch(
        x,
        dtype=dtype_to_ttnn(dtype),
        layout=layout_to_ttnn(layout),
        device=device if input_mem_config is not None else None,
        memory_config=memory_config_to_ttnn(input_mem_config),
    )

    return input_tensor


def ttnn_tensor_to_torch(x, output_mem_config=None):
    output_tensor = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    # assert output_mem_config == tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM)
    return ttnn.to_torch(output_tensor)


def ones(
    x,
    *args,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = ttnn.ones(
        x.shape,
        device=device if input_mem_config[0] is not None else None,
        dtype=dtype_to_ttnn(dtype[0]),
        layout=layout_to_ttnn(layout[0]),
        memory_config=memory_config_to_ttnn(output_mem_config),
    )

    return ttnn_tensor_to_torch(t0)


def ones_like(
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
    t1 = ttnn.ones_like(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def full(
    x,
    *args,
    scalar,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = ttnn.full(
        x.shape,
        fill_value=scalar,
        device=device if input_mem_config[0] is not None else None,
        dtype=dtype_to_ttnn(dtype[0]),
        layout=layout_to_ttnn(layout[0]),
        memory_config=memory_config_to_ttnn(output_mem_config),
    )

    return ttnn_tensor_to_torch(t0)


def eltwise_hardswish(
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
    t1 = ttnn.hardswish(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_hardtanh(
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
    t1 = ttnn.hardtanh(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_heaviside(
    x,
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
    t1 = ttnn.heaviside(t0, scalar, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_hypot(
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
    t2 = ttnn.hypot(t0, t1, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t2)


def eltwise_i0(
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
    t1 = ttnn.i0(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_isfinite(
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
    t1 = ttnn.isfinite(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_isinf(
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
    t1 = ttnn.isinf(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_isnan(
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
    t1 = ttnn.isnan(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_isneginf(
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
    t1 = ttnn.isneginf(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_isposinf(
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
    t1 = ttnn.isposinf(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_leaky_relu(
    x,
    *args,
    negative_slope,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.leaky_relu(t0, negative_slope, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_lerp(
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
    t2 = ttnn.lerp(t0, t1, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t2)


def eltwise_lgamma(
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
    t1 = ttnn.lgamma(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_log(
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
    t1 = ttnn.log(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_log10(
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
    t1 = ttnn.log10(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_log1p(
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
    t1 = ttnn.log1p(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_log2(
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
    t1 = ttnn.log2(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_log_sigmoid(
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
    t1 = ttnn.log_sigmoid(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_logit(
    x,
    *args,
    eps,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.logit(t0, eps, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_mish(
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
    t1 = ttnn.mish(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_multigammaln(
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
    t1 = ttnn.multigammaln(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_neg(
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
    t1 = ttnn.neg(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_prelu(
    x,
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
    t1 = ttnn.prelu(t0, scalar, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_relu(
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
    t1 = ttnn.relu(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_logical_not(
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
    t1 = ttnn.logical_not(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_add(
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

    t2 = ttnn.add(t0, t1)
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


def eltwise_softmax(
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
    t1 = ttnn.softmax(t0, dim=-1)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def eltwise_softplus(
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
    t1 = ttnn.softplus(t0)
    return ttnn_tensor_to_torch(t1, output_mem_config)


def mul(
    x,
    y,
    *args,
    scalar=0,
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

    t1 = ttnn.linear(
        t0,
        tt_weight,
        bias=tt_bias,
        dtype=ttnn.bfloat16,  # core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x)
    )
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


def attention_softmax_nomask(
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

    t2 = ttnn.transformer.attention_softmax(t0, head_size=None, attention_mask=None)

    return ttnn_tensor_to_torch(t2, output_mem_config)


def attention_softmax(
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
    y[y <= 0.50] = 0
    y[y > 0.50] = 1

    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    if scalar < 0:
        scalar = -scalar

    t2 = ttnn.transformer.attention_softmax(t0, head_size=scalar, attention_mask=t1)

    return ttnn_tensor_to_torch(t2, output_mem_config)


def rmsnorm(
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
    t1 = setup_ttnn_tensor(y, device, layout[0], input_mem_config[0], dtype[0])

    t2 = ttnn.rms_norm(t0, t1)

    return ttnn_tensor_to_torch(t2, output_mem_config)


def transformer_concatenate_heads(
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
    t1 = ttnn.transformer.concatenate_heads(t0, memory_config=output_mem_config)

    return ttnn_tensor_to_torch(t1, output_mem_config)


def full_like(
    x,
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
    t1 = ttnn.full_like(t0, fill_value=scalar, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_relu6(
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
    t1 = ttnn.relu6(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_rsqrt(
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
    t1 = ttnn.rsqrt(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_sigmoid(
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
    t1 = ttnn.sigmoid(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_sign(
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
    t1 = ttnn.sign(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_silu(
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
    t1 = ttnn.silu(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_sin(
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
    t1 = ttnn.sin(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_sinh(
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
    t1 = ttnn.sinh(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_softshrink(
    x,
    *args,
    _lambda,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.softshrink(t0, _lambda, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_softsign(
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
    t1 = ttnn.softsign(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_tan(
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
    t1 = ttnn.tan(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_tanh(
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
    t1 = ttnn.tanh(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_swish(
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
    t1 = ttnn.swish(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_signbit(
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
    t1 = ttnn.signbit(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_rad2deg(
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
    t1 = ttnn.rad2deg(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_polygamma(
    x,
    *args,
    k,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.polygamma(t0, k, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_recip(
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
    t1 = ttnn.reciprocal(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_sqrt(
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
    t1 = ttnn.sqrt(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_square(
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
    t1 = ttnn.square(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def tril(
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
    t1 = ttnn.tril(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def triu(
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
    t1 = ttnn.triu(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_tanhshrink(
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
    t1 = ttnn.tanhshrink(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_threshold(
    x,
    *args,
    threshold,
    value,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = ttnn.threshold(t0, threshold, value, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_eqz(
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
    t1 = ttnn.eqz(t0, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t1)


def eltwise_gt(
    x,
    y,
    *args,
    scalar=0,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    if scalar == 0:
        t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    else:
        t1 = scalar

    t2 = ttnn.gt(t0, t1, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t2)


def eltwise_gte(
    x,
    y,
    *args,
    scalar=0,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    if scalar == 0:
        t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    else:
        t1 = scalar

    t2 = ttnn.gt(t0, t1, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t2)


def eltwise_lt(
    x,
    y,
    *args,
    scalar=0,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    if scalar == 0:
        t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    else:
        t1 = scalar

    t2 = ttnn.lt(t0, t1, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t2)


def eltwise_lte(
    x,
    y,
    *args,
    scalar=0,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    if scalar == 0:
        t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    else:
        t1 = scalar

    t2 = ttnn.lte(t0, t1, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t2)


def eltwise_eq(
    x,
    y,
    *args,
    scalar=0,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    if scalar == 0:
        t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    else:
        t1 = scalar

    t2 = ttnn.eq(t0, t1, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t2)


def eltwise_ne(
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
    if scalar == 0:
        t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])
    else:
        t1 = scalar

    t2 = ttnn.ne(t0, t1, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t2)


def eltwise_isclose(
    x,
    y,
    *args,
    rtol,
    atol,
    device,
    dtype,
    layout,
    input_mem_config,
    output_mem_config,
    **kwargs,
):
    t0 = setup_ttnn_tensor(x, device, layout[0], input_mem_config[0], dtype[0])
    t1 = setup_ttnn_tensor(y, device, layout[1], input_mem_config[1], dtype[1])

    t2 = ttnn.isclose(t0, t1, rtol, atol, memory_config=memory_config_to_ttnn(output_mem_config))

    return ttnn_tensor_to_torch(t2)
