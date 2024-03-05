# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import tt_lib
from typing import Optional
from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor_rm, tt2torch_tensor
import torch


def Linear(
    in_features: int,
    out_features: int,
    weight: tt_lib.tensor.Tensor,
    bias: Optional[tt_lib.tensor.Tensor] = None,
    output_mem_config=tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
    ),
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
        output = tt_lib.tensor.matmul(activation, weight_T, output_mem_config)

        if bias is not None:
            output_plus_bias = tt_lib.tensor.bcast(
                output, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H, output_mem_config
            )
            return output_plus_bias

        return output

    return linear_


def format_tensor(x, target_layout, device, output_mem_config, pad_value=0.0):
    if x.get_layout() == target_layout:
        return x
    if x.get_layout() == tt_lib.tensor.Layout.ROW_MAJOR and target_layout == tt_lib.tensor.Layout.TILE:
        x_padded_shape = tt_lib.tensor.pad_to_tile_shape(x.get_legacy_shape(), False, False, True, True)
        if x.get_legacy_shape() != x_padded_shape:
            return tt_lib.tensor.format_input_tensor(
                x, device, x_padded_shape, pad_value, target_layout, output_mem_config
            )
        else:
            return tt_lib.tensor.tilize(x, output_mem_config, use_multicore=True)
    elif x.get_layout() == tt_lib.tensor.Layout.TILE and target_layout == tt_lib.tensor.Layout.ROW_MAJOR:
        if x.get_legacy_shape() != x.shape_without_padding():
            return tt_lib.tensor.format_output_tensor(
                x, x.shape_without_padding(), device, target_layout, output_mem_config
            )
        else:
            return tt_lib.tensor.untilize(x, output_mem_config, use_multicore=True)
    else:
        assert False


def unpad_from_zero(x, desired_shape):
    if x.get_legacy_shape()[-1] == desired_shape[-1] and x.get_legacy_shape()[-2] == desired_shape[-2]:
        x = tt2torch_tensor(x)
    else:
        x = x.cpu()
        if x.get_layout() != tt_lib.tensor.Layout.ROW_MAJOR:
            x = x.to(tt_lib.tensor.Layout.ROW_MAJOR)
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
    freqs_cis = tt_lib.tensor.complex_tensor(freq_real, freq_img)

    freq_real.deallocate()
    freq_img.deallocate()

    BCH = tt_lib.tensor.BcastOpDim.HW
    BCMUL = tt_lib.tensor.BcastOpMath.MUL

    t_one_xq = tt_lib.tensor.ones(query_shape, output_mem_config=mem_config)
    t_one_xq = tt_lib.tensor.permute(t_one_xq, (3, 1, 2, 0), output_mem_config=mem_config)

    freqs_real = tt_lib.tensor.permute(freqs_cis.real, (3, 1, 2, 0), output_mem_config=mem_config)
    freqs_imag = tt_lib.tensor.permute(freqs_cis.imag, (3, 1, 2, 0), output_mem_config=mem_config)

    bcast_freq_re_xq = tt_lib.tensor.bcast(t_one_xq, freqs_real, BCMUL, BCH, output_mem_config=mem_config)
    bcast_freq_im_xq = tt_lib.tensor.bcast(t_one_xq, freqs_imag, BCMUL, BCH, output_mem_config=mem_config)
    bcast_freq_re_xq = tt_lib.tensor.permute(bcast_freq_re_xq, (3, 1, 2, 0), output_mem_config=mem_config)
    bcast_freq_im_xq = tt_lib.tensor.permute(bcast_freq_im_xq, (3, 1, 2, 0), output_mem_config=mem_config)
    t_one_xq.deallocate()

    bcast_freq_xq = tt_lib.tensor.complex_tensor(bcast_freq_re_xq, bcast_freq_im_xq)

    bcast_freq_re_xq.deallocate()
    bcast_freq_im_xq.deallocate()

    t_one_xk = tt_lib.tensor.ones(key_shape, output_mem_config=mem_config)
    t_one_xk = tt_lib.tensor.permute(t_one_xk, (3, 1, 2, 0), output_mem_config=mem_config)

    bcast_freq_re_xk = tt_lib.tensor.bcast(t_one_xk, freqs_real, BCMUL, BCH, output_mem_config=mem_config)
    bcast_freq_im_xk = tt_lib.tensor.bcast(t_one_xk, freqs_imag, BCMUL, BCH, output_mem_config=mem_config)
    bcast_freq_re_xk = tt_lib.tensor.permute(bcast_freq_re_xk, (3, 1, 2, 0), output_mem_config=mem_config)
    bcast_freq_im_xk = tt_lib.tensor.permute(bcast_freq_im_xk, (3, 1, 2, 0), output_mem_config=mem_config)

    bcast_freq_xk = tt_lib.tensor.complex_tensor(bcast_freq_re_xk, bcast_freq_im_xk)

    t_one_xk.deallocate()
    bcast_freq_re_xk.deallocate()
    bcast_freq_im_xk.deallocate()
    freqs_cis.deallocate()
    freqs_real.deallocate()
    freqs_imag.deallocate()

    return bcast_freq_xq, bcast_freq_xk
