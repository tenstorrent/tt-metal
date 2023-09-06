import torch
import tt_lib as ttl
from tests.tt_eager.python_api_testing.sweep_tests.common import is_wormhole_b0
from tests.models.helper_funcs import Linear as tt_Linear

from itertools import product


def setup_host_and_device(func):
    def wrap(*args, pcie_slot, **kwargs):
        ARCH = is_wormhole_b0() and ttl.device.Arch.WORMHOLE_B0 or ttl.device.Arch.GRAYSKULL
        device = ttl.device.CreateDevice(ARCH, pcie_slot)
        ttl.device.InitializeDevice(device)
        ttl.device.SetDefaultDevice(device)
        try:
            output = func(*args, device=device, **kwargs)
        finally:
            ttl.device.CloseDevice(device)

        return output

    return wrap


################################################
################## Helper-Funcs ################
################################################

def make_mem_config(beffer_type):
    if beffer_type == None:
        return None

    return ttl.tensor.MemoryConfig(True, beffer_type)


def tensor_to_device(x, device, beffer_type):
    if beffer_type == None:
        return x

    return x.to(device, ttl.tensor.MemoryConfig(True, beffer_type))


@setup_host_and_device
def linear(x, weight, bias=None, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    tt_bias = None

    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    tt_weight = ttl.tensor.Tensor(
        weight.reshape(-1).tolist(),
        weight.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if bias is not None:
        tt_bias = ttl.tensor.Tensor(
            bias.reshape(-1).tolist(),
            bias.shape,
            dtype[2],
            ttl.tensor.Layout.ROW_MAJOR,
        )
        tt_bias = tt_bias.to(layout[2])

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    tt_weight = tt_weight.to(layout[1])
    tt_weight = tensor_to_device(tt_weight, device, buffer_type[0])

    if bias is not None:
        tt_bias = tensor_to_device(tt_bias, device, buffer_type[0])

    _, __, out_features, in_features = tt_weight.shape()
    tt_linear = tt_Linear(in_features, out_features, tt_weight, tt_bias)

    t1 = tt_linear(t0)
    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


################################################
#################### TT-DNN ####################
################################################
@setup_host_and_device
def move(
    x,
    *args,
    device,
    dtype,
    layout,
    buffer_type,
    output_mem_config,
    **kwargs,
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.move(t0, output_mem_config=output_mem_config)
    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_erf(x, *args, fast_and_appx, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.erf(t0, fast_and_appx, output_mem_config=output_mem_config)
    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_erfc(x, *args, fast_and_appx, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.erfc(t0, fast_and_appx, output_mem_config=output_mem_config)
    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_logical_not(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.logical_not(t0, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_threshold(x, *args, threshold, value, device, dtype, layout, buffer_type, output_mem_config, **kwargs
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.threshold(t0, threshold, value, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_hardtanh(
    x, *args, low, high, device, dtype, layout, buffer_type, output_mem_config, **kwargs
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.hardtanh(t0, low, high, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_leaky_relu(
    x, *args, negative_slope, device, dtype, layout, buffer_type, output_mem_config, **kwargs
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.leaky_relu(t0, negative_slope, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_bias_gelu(x, *args, bias, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.bias_gelu(t0, bias, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_hardshrink(x, *args, _lambda, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.hardshrink(t0, _lambda, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_softshrink(x, *args, _lambda, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.softshrink(t0, _lambda, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_elu(x, *args, alpha, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.elu(t0, alpha, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_gelu(x, *args, fast_and_appx, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.gelu(t0, fast_and_appx, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output

@setup_host_and_device
def eltwise_softmax_in_place(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.operations.primary.softmax_in_place(t0)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output



@setup_host_and_device
def eltwise_scale_mask_softmax_in_place(x, y, scale, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.operations.primary.transformers.scale_mask_softmax_in_place(t0, scale, t1)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output

@setup_host_and_device
def eltwise_rsqrt(x, *args, fast_and_appx, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.rsqrt(t0, fast_and_appx, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_relu_min(x, *args, lower_limit, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.relu_min(t0, lower_limit, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_relu_max(x, *args, upper_limit, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.relu_max(t0, upper_limit, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


# stats ops
@setup_host_and_device
def std_hw(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.std_hw(t0, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    output = output.max(2, True)[0].max(3, True)[0]

    return output


@setup_host_and_device
def var_hw(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.var_hw(t0, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    output = output.max(2, True)[0].max(3, True)[0]

    return output


@setup_host_and_device
def mean_hw(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.mean_hw(t0, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    output = output.max(2, True)[0].max(3, True)[0]

    return output


@setup_host_and_device
def normalize_hw(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.normalize_hw(t0, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_polyval(x, *args, coeffs, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.polyval(t0, coeffs, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_mac(x, y, z, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.Tensor(
        z.reshape(-1).tolist(),
        z.shape,
        dtype[2],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t2 = t2.to(layout[2])
    t2 = tensor_to_device(t2, device, buffer_type[2])

    t3 = ttl.tensor.mac(t0, t1, t2, output_mem_config=output_mem_config)

    output = t3.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_addcmul(x, y, z, *args, scalar, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.Tensor(
        z.reshape(-1).tolist(),
        z.shape,
        dtype[2],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t2 = t2.to(layout[2])
    t2 = tensor_to_device(t2, device, buffer_type[2])

    t3 = ttl.tensor.addcmul(t0, t1, t2, scalar, output_mem_config=output_mem_config)

    output = t3.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_addcdiv(x, y, z, *args, scalar, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.Tensor(
        z.reshape(-1).tolist(),
        z.shape,
        dtype[2],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t2 = t2.to(layout[2])
    t2 = tensor_to_device(t2, device, buffer_type[2])

    t3 = ttl.tensor.addcdiv(t0, t1, t2, scalar, output_mem_config)

    output = t3.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_lerp_binary(
    x, y, *args, weight, device, dtype, layout, buffer_type, output_mem_config, **kwargs
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.lerp(t0, t1, weight, output_mem_config=output_mem_config)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def conv(x, y, conv_params, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.conv(t0, t1, conv_params, 0, 0, 0, 0, 0, conv_params[0])

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output

@setup_host_and_device
def layernorm(x, y, z, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    if layout[1] == ttl.tensor.Layout.TILE:
        y = torch.nn.functional.pad(y, (0, 0, 0, 32 - y.shape[2]))

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    if layout[2] == ttl.tensor.Layout.TILE:
        z = torch.nn.functional.pad(z, (0, 0, 0, 32 - z.shape[2]))

    t2 = ttl.tensor.Tensor(
        z.reshape(-1).tolist(),
        z.shape,
        dtype[2],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t2 = t2.to(layout[2])
    t2 = tensor_to_device(t2, device, buffer_type[2])

    t3 = ttl.operations.primary.layernorm(t0, 1e-5, t1, t2, output_mem_config=output_mem_config)

    output = t3.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def layernorm_noweights(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t3 = ttl.operations.primary.layernorm(t0, 1e-5, None, None, output_mem_config=output_mem_config)

    output = t3.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def add_layernorm_noweights(x, y, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t4 = ttl.operations.primary.add_layernorm(t0, t1, 1e-5, None, None, output_mem_config=output_mem_config)

    output = t4.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def add_layernorm(x, y, z, w, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])


    if layout[2] == ttl.tensor.Layout.TILE:
        z = torch.nn.functional.pad(z, (0, 0, 0, 32 - z.shape[2]))

    t2 = ttl.tensor.Tensor(
        z.reshape(-1).tolist(),
        z.shape,
        dtype[2],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t2 = t2.to(layout[2])
    t2 = tensor_to_device(t2, device, buffer_type[2])

    if layout[3] == ttl.tensor.Layout.TILE:
        w = torch.nn.functional.pad(w, (0, 0, 0, 32 - w.shape[2]))

    t3 = ttl.tensor.Tensor(
        w.reshape(-1).tolist(),
        w.shape,
        dtype[3],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t3 = t3.to(layout[3])
    t3 = tensor_to_device(t3, device, buffer_type[3])

    t4 = ttl.operations.primary.add_layernorm(t0, t1, 1e-5, t2, t3, output_mem_config=output_mem_config)

    output = t4.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output

@setup_host_and_device
def eltwise_lerp_ternary(x, y, z, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.Tensor(
        z.reshape(-1).tolist(),
        z.shape,
        dtype[2],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t2 = t2.to(layout[2])
    t2 = tensor_to_device(t2, device, buffer_type[2])

    t3 = ttl.tensor.lerp(t0, t1, t2, output_mem_config=output_mem_config)

    output = t3.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_subalpha(x, y, *args, alpha, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.subalpha(t0, t1, alpha, output_mem_config=output_mem_config)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def eltwise_heaviside(x, *args, scalar, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.heaviside(t0, scalar, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def full_like(x, *args, scalar, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.full_like(t0, scalar, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def ones(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t1 = ttl.tensor.ones(
        x.shape,
        layout=layout[0],
        device=device if buffer_type[0] is not None else None,
        output_mem_config = output_mem_config
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def zeros(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t1 = ttl.tensor.zeros(
        x.shape,
        layout=layout[0],
        device=device if buffer_type[0] is not None else None,
        output_mem_config = output_mem_config
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def full(x, *args, scalar, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t1 = ttl.tensor.full(
        x.shape,
        scalar,
        layout=layout[0],
        device=device if buffer_type[0] is not None else None,
        output_mem_config = output_mem_config
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def fill_rm(x, *args, hOnes, wOnes, val_hi, val_lo, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    # Layout must be row mayor
    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.fill_rm(
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
        hOnes,
        wOnes,
        t0,
        val_hi,
        val_lo,
        output_mem_config=output_mem_config
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def fill_ones_rm(x, *args, hOnes, wOnes, device, dtype, layout, buffer_type, output_mem_config, **kwargs):

    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    # Layout must be row mayor
    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.fill_ones_rm(
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
        hOnes,
        wOnes,
        t0,
        output_mem_config=output_mem_config
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def arange(x, *args, start, end, step=1, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t1 = ttl.tensor.arange(
        start,
        end,
        step,
        device=device if buffer_type[0] is not None else None,
        output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def clip(x, *args, low, high, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.clip(t0, low, high, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def where(x, y, z, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )
    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )
    t2 = ttl.tensor.Tensor(
        z.reshape(-1).tolist(),
        z.shape,
        dtype[2],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t1 = t1.to(layout[1])
    t2 = t2.to(layout[2])

    t0 = tensor_to_device(t0, device, buffer_type[0])
    t1 = tensor_to_device(t1, device, buffer_type[1])
    t2 = tensor_to_device(t2, device, buffer_type[2])

    t3 = ttl.tensor.where(t0, t1, t2, output_mem_config=output_mem_config)

    output = t3.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_div_unary(x, *args, scalar, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.div_unary(t0, scalar, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_mul_unary(x, *args, scalar, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.mul_unary(t0, scalar, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_sub_unary(x, *args, scalar, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.sub_unary(t0, scalar, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_add_unary(x, *args, scalar, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.add_unary(t0, scalar, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def matmul(x, y, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.matmul(t0, t1, output_mem_config=output_mem_config)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def outer(x, y, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.outer(t0, t1, output_mem_config=output_mem_config)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def bmm(x, y, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.Tensor(
        y.reshape(-1).tolist(),
        y.shape,
        dtype[1],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[0])

    t2 = ttl.tensor.bmm(t0, t1, output_mem_config=output_mem_config)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def bcast_add_h(x, y, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype[0])

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 0, 0, 32 - y.shape[2]))

    t1 = ttl.tensor.Tensor(t1, dtype[1])
    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.H, output_mem_config=output_mem_config)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def bcast_add_w(x, y, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype[0])

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE or (
        buffer_type[1] and layout == ttl.tensor.Layout.ROW_MAJOR
    ):
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))

    t1 = ttl.tensor.Tensor(t1, dtype[1])
    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.W, output_mem_config=output_mem_config)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def bcast_add_hw(x, y, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype[0])

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3], 0, 32 - y.shape[2]))
    elif buffer_type[1] and layout[1] == ttl.tensor.Layout.ROW_MAJOR:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))

    t1 = ttl.tensor.Tensor(t1, dtype[1])
    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.HW, output_mem_config=output_mem_config)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return output


@setup_host_and_device
def bcast_sub_h(x, y, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype[0])

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 0, 0, 32 - y.shape[2]))
    t1 = ttl.tensor.Tensor(t1, dtype[1])
    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.H, output_mem_config=output_mem_config)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def bcast_sub_w(x, y, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype[0])

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE or (
        buffer_type[1] and layout == ttl.tensor.Layout.ROW_MAJOR
    ):
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))
    t1 = ttl.tensor.Tensor(t1, dtype[1])
    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.W, output_mem_config=output_mem_config)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def bcast_sub_hw(x, y, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype[0])

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3], 0, 32 - y.shape[2]))
    elif buffer_type[1] and layout[1] == ttl.tensor.Layout.ROW_MAJOR:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))

    t1 = ttl.tensor.Tensor(t1, dtype[1])
    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.SUB, ttl.tensor.BcastOpDim.HW, output_mem_config=output_mem_config)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def bcast_mul_h(x, y, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype[0])

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 0, 0, 32 - y.shape[2]))

    t1 = ttl.tensor.Tensor(t1, dtype[1])
    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.H, output_mem_config=output_mem_config)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def bcast_mul_w(x, y, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype[0])
    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE or (
        buffer_type[1] and layout[1] == ttl.tensor.Layout.ROW_MAJOR
    ):
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))

    t1 = ttl.tensor.Tensor(t1, dtype[1])
    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.W, output_mem_config=output_mem_config)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def bcast_mul_hw(x, y, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(x, dtype[0])

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = y
    if layout[1] == ttl.tensor.Layout.TILE:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3], 0, 32 - y.shape[2]))
    elif buffer_type[1] and layout[1] == ttl.tensor.Layout.ROW_MAJOR:
        t1 = torch.nn.functional.pad(y, (0, 32 - y.shape[3]))

    t1 = ttl.tensor.Tensor(t1, dtype[1])
    t1 = t1.to(layout[1])
    t1 = tensor_to_device(t1, device, buffer_type[1])

    t2 = ttl.tensor.bcast(t0, t1, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.HW, output_mem_config=output_mem_config)

    output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def reduce_sum_h(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.reduce(
        t0, ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.H, 1.0, output_mem_config=output_mem_config
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    # Slice out the 0 values from reduction
    output = output[..., :1, :]
    return output


@setup_host_and_device
def reduce_sum_w(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.reduce(
        t0, ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.W, 1.0, output_mem_config=output_mem_config
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    # Slice out the 0 values from reduction
    output = output[..., :, :1]
    return output


@setup_host_and_device
def reduce_sum_hw(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.reduce(
        t0, ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.HW, 1.0, output_mem_config=output_mem_config
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]
    return output


@setup_host_and_device
def reduce_max_h(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.reduce(
        t0, ttl.tensor.ReduceOpMath.MAX, ttl.tensor.ReduceOpDim.H, 1.0, output_mem_config=output_mem_config
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    # Slice out the 0 values from reduction
    output = output[..., :1, :]
    return output


@setup_host_and_device
def reduce_max_w(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.reduce(
        t0, ttl.tensor.ReduceOpMath.MAX, ttl.tensor.ReduceOpDim.W, 1.0, output_mem_config=output_mem_config
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    # Slice out the 0 values from reduction
    output = output[..., :1]
    return output


@setup_host_and_device
def reduce_max_hw(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.reduce(
        t0, ttl.tensor.ReduceOpMath.MAX, ttl.tensor.ReduceOpDim.HW, 1.0, output_mem_config=output_mem_config
    )

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    # Slice out the 0 values from reduction
    output = output[..., :1, :1]
    return output


@setup_host_and_device
def transpose_wh(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.transpose(t0, output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def transpose_hc(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.transpose_hc(t0, output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def transpose_cn(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.transpose_cn(t0, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def transpose_nh(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.transpose(t0, 0, 2, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def transpose_nw(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.transpose(t0, 0, 3, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def transpose_cw(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.transpose(t0, 1, 3, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def sum(x, *args, dim, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    assert dim >= 0 and dim <= 3
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.sum(t0, dim, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    if dim == 2:
        output = output[:, :, :1, :]
    elif dim == 3:
        output = output[:, :, :, :1]
    return output


@setup_host_and_device
def permute(x, *args, device, dtype, layout, buffer_type, output_mem_config, permute_dims, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.permute(t0, *permute_dims, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def reshape(x, *args, device, dtype, layout, buffer_type, output_mem_config, reshape_dims, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.reshape(t0, *reshape_dims, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def split_last_dim_two_chunks_tiled(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    print(f"tt input.shape {x.shape}")

    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.split_last_dim_two_chunks_tiled(t0, output_mem_config=output_mem_config)

    output0 = t1[0].cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    output1 = t1[1].cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

    return [output0, output1]


@setup_host_and_device
def tilize(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.tilize(t0, output_mem_config=output_mem_config)
    output = t1.cpu().to_torch()

    return output


@setup_host_and_device
def tilize_with_zero_padding(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.tilize_with_zero_padding(t0, output_mem_config=output_mem_config)

    output = t1.cpu().to_torch()
    return output


@setup_host_and_device
def untilize(x, *args, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.TILE,
    )

    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.untilize(t0, output_mem_config=output_mem_config)

    output = t1.cpu().to_torch()
    return output


@setup_host_and_device
def tilize_with_val_padding(
    x,
    *args,
    device,
    dtype,
    layout,
    buffer_type,
    output_mem_config,
    output_tensor_shape,
    input_tensor_start,
    pad_value,
    **kwargs,
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.tilize_with_val_padding(
        t0, output_tensor_shape, input_tensor_start, pad_value, output_mem_config=output_mem_config
    )

    output = t1.cpu().to_torch()
    return output


@setup_host_and_device
def untilize_with_unpadding(
    x,
    *args,
    device,
    dtype,
    layout,
    buffer_type,
    output_mem_config,
    output_tensor_start,
    output_tensor_end,
    **kwargs,
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.TILE,
    )

    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.untilize_with_unpadding(t0, output_tensor_start, output_tensor_end, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def pad(
    x,
    *args,
    device,
    dtype,
    layout,
    buffer_type,
    output_mem_config,
    output_tensor_shape,
    input_tensor_start,
    pad_value,
    **kwargs,
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.pad(t0, output_tensor_shape, input_tensor_start, pad_value, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def unpad(
    x,
    *args,
    device,
    dtype,
    layout,
    buffer_type,
    output_mem_config,
    output_tensor_start,
    output_tensor_end,
    **kwargs,
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.unpad(t0, output_tensor_start, output_tensor_end, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


@setup_host_and_device
def eltwise_power(x, *args, exponent, device, dtype, layout, buffer_type, output_mem_config, **kwargs):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = ttl.tensor.power(t0, exponent, output_mem_config=output_mem_config)

    output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    return output


def make_eltwise_unary_op(ttl_tensor_unop):
    @setup_host_and_device
    def eltwise_unary_op(
        x,
        *args,
        device,
        dtype,
        layout,
        buffer_type,
        output_mem_config=ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        **kwargs,
    ):
        t0 = ttl.tensor.Tensor(x, dtype[0])
        t0 = t0.to(layout[0])
        t0 = tensor_to_device(t0, device, buffer_type[0])

        t1 = ttl_tensor_unop(t0, output_mem_config=output_mem_config)
        output = t1.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        return output

    return eltwise_unary_op

#eltwise_softmax_in_place = make_eltwise_unary_op(ttl.tensor.softmax_in_place)
eltwise_cos = make_eltwise_unary_op(ttl.tensor.cos)
eltwise_sin = make_eltwise_unary_op(ttl.tensor.sin)
eltwise_acos = make_eltwise_unary_op(ttl.tensor.acos)
eltwise_asin = make_eltwise_unary_op(ttl.tensor.asin)
eltwise_atan = make_eltwise_unary_op(ttl.tensor.atan)
eltwise_atanh = make_eltwise_unary_op(ttl.tensor.atanh)
eltwise_cosh = make_eltwise_unary_op(ttl.tensor.cosh)
eltwise_sinh = make_eltwise_unary_op(ttl.tensor.sinh)
eltwise_tanh = make_eltwise_unary_op(ttl.tensor.tanh)
eltwise_asinh = make_eltwise_unary_op(ttl.tensor.asinh)
eltwise_acosh = make_eltwise_unary_op(ttl.tensor.acosh)
eltwise_tanhshrink = make_eltwise_unary_op(ttl.tensor.tanhshrink)
eltwise_softsign = make_eltwise_unary_op(ttl.tensor.softsign)
eltwise_relu = make_eltwise_unary_op(ttl.tensor.relu)
eltwise_relu6 = make_eltwise_unary_op(ttl.tensor.relu6)
eltwise_sqrt = make_eltwise_unary_op(ttl.tensor.sqrt)
eltwise_cbrt = make_eltwise_unary_op(ttl.tensor.cbrt)
eltwise_rad2deg = make_eltwise_unary_op(ttl.tensor.rad2deg)
eltwise_deg2rad = make_eltwise_unary_op(ttl.tensor.deg2rad)
eltwise_sign = make_eltwise_unary_op(ttl.tensor.sign)
eltwise_signbit = make_eltwise_unary_op(ttl.tensor.signbit)
eltwise_abs = make_eltwise_unary_op(ttl.tensor.abs)
eltwise_exp = make_eltwise_unary_op(ttl.tensor.exp)
eltwise_exp2 = make_eltwise_unary_op(ttl.tensor.exp2)
eltwise_expm1 = make_eltwise_unary_op(ttl.tensor.expm1)
eltwise_neg = make_eltwise_unary_op(ttl.tensor.neg)
eltwise_recip = make_eltwise_unary_op(ttl.tensor.recip)
eltwise_sigmoid = make_eltwise_unary_op(ttl.tensor.sigmoid)
eltwise_log_sigmoid = make_eltwise_unary_op(ttl.tensor.log_sigmoid)
eltwise_log = make_eltwise_unary_op(ttl.tensor.log)
eltwise_log2 = make_eltwise_unary_op(ttl.tensor.log2)
eltwise_log10 = make_eltwise_unary_op(ttl.tensor.log10)
eltwise_swish = make_eltwise_unary_op(ttl.tensor.swish)
eltwise_add1 = make_eltwise_unary_op(ttl.tensor.add1)
eltwise_log1p = make_eltwise_unary_op(ttl.tensor.log1p)
eltwise_softplus = make_eltwise_unary_op(ttl.tensor.softplus)
eltwise_mish = make_eltwise_unary_op(ttl.tensor.mish)
eltwise_hardswish = make_eltwise_unary_op(ttl.tensor.hardswish)
eltwise_hardsigmoid = make_eltwise_unary_op(ttl.tensor.hardsigmoid)
eltwise_silu = make_eltwise_unary_op(ttl.tensor.silu)
eltwise_square = make_eltwise_unary_op(ttl.tensor.square)
eltwise_ltz = make_eltwise_unary_op(ttl.tensor.ltz)
eltwise_gtz = make_eltwise_unary_op(ttl.tensor.gtz)
eltwise_lez = make_eltwise_unary_op(ttl.tensor.lez)
eltwise_gez = make_eltwise_unary_op(ttl.tensor.gez)
eltwise_nez = make_eltwise_unary_op(ttl.tensor.nez)
eltwise_eqz = make_eltwise_unary_op(ttl.tensor.eqz)
zeros_like = make_eltwise_unary_op(ttl.tensor.zeros_like)
ones_like = make_eltwise_unary_op(ttl.tensor.ones_like)


def make_eltwise_binary_op(ttl_tensor_binop):
    @setup_host_and_device
    def eltwise_binary_op(
        x,
        y,
        *args,
        device,
        dtype,
        layout,
        buffer_type,
        output_mem_config,
        **kwargs,
    ):
        t0 = ttl.tensor.Tensor(x, dtype[0])
        t0 = t0.to(layout[0])
        t0 = tensor_to_device(t0, device, buffer_type[0])

        t1 = ttl.tensor.Tensor(y, dtype[1])
        t1 = t1.to(layout[1])
        t1 = tensor_to_device(t1, device, buffer_type[1])

        t2 = ttl_tensor_binop(t0, t1, output_mem_config=output_mem_config)
        output = t2.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()

        return output

    return eltwise_binary_op


eltwise_add = make_eltwise_binary_op(ttl.tensor.add)
eltwise_sub = make_eltwise_binary_op(ttl.tensor.sub)
eltwise_mul = make_eltwise_binary_op(ttl.tensor.mul)
eltwise_squared_difference = make_eltwise_binary_op(ttl.tensor.squared_difference)
eltwise_hypot = make_eltwise_binary_op(ttl.tensor.hypot)
eltwise_atan2 = make_eltwise_binary_op(ttl.tensor.atan2)
eltwise_min = make_eltwise_binary_op(ttl.tensor.min)
eltwise_max = make_eltwise_binary_op(ttl.tensor.max)
eltwise_ne = make_eltwise_binary_op(ttl.tensor.ne)
eltwise_eq = make_eltwise_binary_op(ttl.tensor.eq)
eltwise_gt = make_eltwise_binary_op(ttl.tensor.gt)
eltwise_lt = make_eltwise_binary_op(ttl.tensor.lt)
eltwise_gte = make_eltwise_binary_op(ttl.tensor.gte)
eltwise_lte = make_eltwise_binary_op(ttl.tensor.lte)
eltwise_xlogy = make_eltwise_binary_op(ttl.tensor.xlogy)
eltwise_ldexp = make_eltwise_binary_op(ttl.tensor.ldexp)
eltwise_logaddexp = make_eltwise_binary_op(ttl.tensor.logaddexp)
eltwise_logaddexp2 = make_eltwise_binary_op(ttl.tensor.logaddexp2)


################################################
#################### Tensor ####################
################################################

@setup_host_and_device
def datacopy(
    x,
    *args,
    device,
    dtype,
    layout,
    buffer_type,
    output_mem_config,
    **kwargs,
):
    device_tensor = (
        ttl.tensor.Tensor(x, dtype[0])
        .to(layout[0])
        .to(device, output_mem_config)
    )

    host_tensor = device_tensor.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    output = host_tensor.to_torch()

    return output


@setup_host_and_device
def tensor_pad(
    x,
    *args,
    device,
    dtype,
    layout,
    buffer_type,
    output_mem_config,
    output_tensor_shape,
    input_tensor_start,
    pad_value,
    **kwargs
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = t0.pad(output_tensor_shape, input_tensor_start, pad_value)
    output = t1.to_torch()

    return output


@setup_host_and_device
def tensor_unpad(
    x,
    *args,
    device,
    dtype,
    layout,
    buffer_type,
    output_mem_config,
    output_tensor_start,
    output_tensor_end,
    **kwargs
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = t0.unpad(output_tensor_start, output_tensor_end)
    output = t1.to_torch()

    return output


@setup_host_and_device
def pad_to_tile(
    x,
    *args,
    device,
    dtype,
    layout,
    buffer_type,
    output_mem_config,
    pad_value,
    **kwargs
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = t0.pad_to_tile(pad_value)
    output = t1.to_torch()

    return output


@setup_host_and_device
def unpad_from_tile(
    x,
    *args,
    device,
    dtype,
    layout,
    buffer_type,
    output_mem_config,
    output_tensor_shape,
    **kwargs
):
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        dtype[0],
        ttl.tensor.Layout.ROW_MAJOR,
    )

    t0 = t0.to(layout[0])
    t0 = tensor_to_device(t0, device, buffer_type[0])

    t1 = t0.unpad_from_tile(output_tensor_shape)
    output = t1.to_torch()

    return output
