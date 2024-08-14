# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor_rm, tt2torch_tensor
import torch
import ttnn


def Linear(
    in_features: int,
    out_features: int,
    weight: ttnn.Tensor,
    bias: Optional[ttnn.Tensor] = None,
    output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
    device=None,
):
    """
    Returns a function that performs a Linear operation with optional bias.

    ``weight`` must be tt_tensor.
    """
    assert weight.get_legacy_shape() == [1, 1, out_features, in_features], "weight does not have the expected shape"

    if bias is not None:
        assert bias.get_legacy_shape()[-1] == out_features, "bias does not have the expected shape"

    weight = weight
    bias = bias
    weight = tt_to_torch_tensor(weight)
    weight_T = weight.transpose(2, 3)
    weight_T = torch_to_tt_tensor_rm(weight_T, device, put_on_device=False)

    def linear_(activation):
        assert activation.get_legacy_shape()[-1] == in_features, "activation tensor do not have the expected shape"
        output = ttnn.matmul(activation, weight_T, output_mem_config)

        if bias is not None:
            output_plus_bias = ttnn.experimental.tensor.bcast(
                output,
                bias,
                ttnn.experimental.tensor.BcastOpMath.ADD,
                ttnn.experimental.tensor.BcastOpDim.H,
                output_mem_config,
            )
            return output_plus_bias

        return output

    return linear_


def format_tensor(x, target_layout, device, output_mem_config, pad_value=0.0):
    if x.get_layout() == target_layout:
        return x
    if x.get_layout() == ttnn.ROW_MAJOR_LAYOUT and target_layout == ttnn.TILE_LAYOUT:
        x_padded_shape = ttnn.experimental.tensor.pad_to_tile_shape(x.get_legacy_shape(), False, False, True, True)
        if x.get_legacy_shape() != x_padded_shape:
            return ttnn.experimental.tensor.format_output_tensor(
                x, device, x_padded_shape, pad_value, target_layout, output_mem_config
            )
        else:
            return ttnn.tilize(x, memory_config=output_mem_config, use_multicore=True)
    elif x.get_layout() == ttnn.TILE_LAYOUT and target_layout == ttnn.ROW_MAJOR_LAYOUT:
        if x.get_legacy_shape() != x.shape_without_padding():
            return ttnn.experimental.tensor.format_output_tensor(
                x, x.shape_without_padding(), device, target_layout, output_mem_config
            )
        else:
            return ttnn.untilize(x, memory_config=output_mem_config, use_multicore=True)
    else:
        assert False


def unpad_from_zero(x, desired_shape):
    if x.get_legacy_shape()[-1] == desired_shape[-1] and x.get_legacy_shape()[-2] == desired_shape[-2]:
        x = tt2torch_tensor(x)
    else:
        x = x.cpu()
        if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            x = x.to(ttnn.ROW_MAJOR_LAYOUT)
        x = x.unpad(
            (0, 0, 0, 0), (desired_shape[0] - 1, desired_shape[1] - 1, desired_shape[2] - 1, desired_shape[3] - 1)
        )
        x = x.to_torch().to(torch.float)
    return x


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x_shape, x_ndim) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x_ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x_shape[1], x_shape[-1]), (
        freqs_cis.shape,
        (x_shape[1], x_shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x_shape)]
    return freqs_cis.view(*shape)


def get_freqs_cis(freqs_cis: torch.Tensor, query_shape, key_shape, device=None, mem_config=None):
    freqs_cis = _reshape_for_broadcast(freqs_cis, query_shape, 4)

    freq_real = torch_to_tt_tensor_rm(freqs_cis.real, device)
    freq_img = torch_to_tt_tensor_rm(freqs_cis.imag, device)
    freqs_cis = ttnn.complex_tensor(freq_real, freq_img)

    freq_real.deallocate()
    freq_img.deallocate()

    BCH = ttnn.experimental.tensor.BcastOpDim.HW
    BCMUL = ttnn.experimental.tensor.BcastOpMath.MUL

    t_one_xq = ttnn.ones(query_shape, memory_config=mem_config)
    t_one_xq = ttnn.permute(t_one_xq, (3, 1, 2, 0), memory_config=mem_config)

    freqs_real = ttnn.permute(freqs_cis.real, (3, 1, 2, 0), memory_config=mem_config)
    freqs_imag = ttnn.permute(freqs_cis.imag, (3, 1, 2, 0), memory_config=mem_config)

    bcast_freq_re_xq = ttnn.experimental.tensor.bcast(t_one_xq, freqs_real, BCMUL, BCH, output_mem_config=mem_config)
    bcast_freq_im_xq = ttnn.experimental.tensor.bcast(t_one_xq, freqs_imag, BCMUL, BCH, output_mem_config=mem_config)
    bcast_freq_re_xq = ttnn.permute(bcast_freq_re_xq, (3, 1, 2, 0), memory_config=mem_config)
    bcast_freq_im_xq = ttnn.permute(bcast_freq_im_xq, (3, 1, 2, 0), memory_config=mem_config)
    t_one_xq.deallocate()

    bcast_freq_xq = ttnn.complex_tensor(bcast_freq_re_xq, bcast_freq_im_xq)

    bcast_freq_re_xq.deallocate()
    bcast_freq_im_xq.deallocate()

    t_one_xk = ttnn.ones(key_shape, memory_config=mem_config)
    t_one_xk = ttnn.permute(t_one_xk, (3, 1, 2, 0), memory_config=mem_config)

    bcast_freq_re_xk = ttnn.experimental.tensor.bcast(t_one_xk, freqs_real, BCMUL, BCH, output_mem_config=mem_config)
    bcast_freq_im_xk = ttnn.experimental.tensor.bcast(t_one_xk, freqs_imag, BCMUL, BCH, output_mem_config=mem_config)
    bcast_freq_re_xk = ttnn.permute(bcast_freq_re_xk, (3, 1, 2, 0), memory_config=mem_config)
    bcast_freq_im_xk = ttnn.permute(bcast_freq_im_xk, (3, 1, 2, 0), memory_config=mem_config)

    bcast_freq_xk = ttnn.complex_tensor(bcast_freq_re_xk, bcast_freq_im_xk)

    t_one_xk.deallocate()
    bcast_freq_re_xk.deallocate()
    bcast_freq_im_xk.deallocate()
    freqs_cis.deallocate()
    freqs_real.deallocate()
    freqs_imag.deallocate()

    return bcast_freq_xq, bcast_freq_xk
