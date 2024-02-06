# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import tt_lib
import torch
from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor_rm


def _reshape_for_broadcast(freqs_cis, x_shape, x_dim):
    ndim = x_dim
    assert 1 < ndim
    assert freqs_cis.shape == (x_shape[1], x_shape[-1]), (
        freqs_cis.shape,
        (x_shape[1], x_shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x_shape)]
    return freqs_cis.view(*shape)


def get_freqs_cis(freqs_cis, query_shape, key_shape, device, mem_config):
    freqs_cis = _reshape_for_broadcast(freqs_cis, query_shape, 4)

    freq_real = torch_to_tt_tensor_rm(freqs_cis.real, device)
    freq_img = torch_to_tt_tensor_rm(freqs_cis.imag, device)
    freqs_cis = tt_lib.tensor.complex_tensor(freq_real, freq_img)

    freq_real.deallocate()
    freq_img.deallocate()

    t_one_xq = torch.ones(query_shape)
    t_one_xq = ttnn.from_torch(t_one_xq, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    freqs_real = ttnn.to_layout(ttnn.Tensor(freqs_cis.real), layout=ttnn.TILE_LAYOUT)

    freqs_imag = ttnn.to_layout(ttnn.Tensor(freqs_cis.imag), layout=ttnn.TILE_LAYOUT)

    freqs_real = ttnn.pad(
        freqs_real, padding=((0, t_one_xq.shape[-0] - freqs_real.shape[0]), (0, 0), (0, 0), (0, 0)), value=1
    )
    bcast_freq_re_xq = t_one_xq * freqs_real
    freqs_imag = ttnn.pad(
        freqs_imag, padding=((0, t_one_xq.shape[-0] - freqs_imag.shape[0]), (0, 0), (0, 0), (0, 0)), value=1
    )

    bcast_freq_im_xq = t_one_xq * freqs_imag
    bcast_freq_xq = tt_lib.tensor.complex_tensor(bcast_freq_re_xq.value, bcast_freq_im_xq.value)

    t_one_xk = torch.ones(key_shape)
    t_one_xk = ttnn.from_torch(t_one_xk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    bcast_freq_re_xk = ttnn.to_layout(t_one_xk * freqs_real, layout=ttnn.ROW_MAJOR_LAYOUT)
    bcast_freq_im_xk = ttnn.to_layout(t_one_xk * freqs_imag, layout=ttnn.ROW_MAJOR_LAYOUT)

    bcast_freq_xk = tt_lib.tensor.complex_tensor(bcast_freq_re_xk.value, bcast_freq_im_xk.value)

    freqs_cis.deallocate()

    return bcast_freq_xq, bcast_freq_xk
