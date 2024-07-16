# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys

import torch

import tt_lib as ttl
from models.utility_functions import print_diff_argmax
import pytest
import ttnn
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal, comp_allclose
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import (
    data_gen_with_range,
)
from models.utility_functions import (
    skip_for_wormhole_b0,
)
from models.utility_functions import is_wormhole_b0
from functools import partial
from math import pi


class Complex:
    def __init__(self, input_shape: torch.Size = None, re=None, im=None):
        if input_shape:
            val = 1.0 + torch.arange(0, input_shape.numel()).reshape(input_shape).bfloat16()
            self._cplx = val[:, :, :, : input_shape[-1] // 2] + val[:, :, :, input_shape[-1] // 2 :] * 1j
        else:
            self._cplx = re + im * 1j

    def reset(self, val: torch.Tensor):
        self._cplx = val

    def is_imag(self):
        return self.real == 0.0

    def is_real(self):
        return self.imag == 0.0

    @property
    def angle(self):
        return torch.angle(self._cplx)

    @property
    def real(self):
        return self._cplx.real

    @property
    def imag(self):
        return self._cplx.imag

    @property
    def metal(self):
        return torch.cat([self.real, self.imag], -1)

    ## operations
    def abs(self):
        return (self.real**2 + self.imag**2).sqrt()

    def conj(self) -> "Complex":
        self._cplx = self._cplx.conj()
        return self

    def recip(self):
        self._cplx = 1.0 / self._cplx
        return self

    def add(self, that: "Complex"):
        self._cplx += that._cplx
        return self

    def sub(self, that: "Complex"):
        self._cplx -= that._cplx
        return self

    def __mul__(self, scale):
        self._cplx *= scale
        return self

    def mul(self, that: "Complex"):
        self._cplx *= that._cplx
        return self

    def div(self, that: "Complex"):
        self._cplx /= that._cplx
        return self


def random_complex_tensor(shape, real_range=(-100, 100), imag_range=(-100, 100)):
    torch.manual_seed(213919)
    real_part = (real_range[1] - real_range[0]) * torch.rand(shape) + real_range[0]
    imag_part = (imag_range[1] - imag_range[0]) * torch.rand(shape) + imag_range[0]
    return torch.complex(real_part, imag_part)


def convert_to_torch_tensor(tt_dev):
    for i in range(len(tt_dev)):
        tt_dev_r = tt_dev[i].real.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        tt_dev_i = tt_dev[i].imag.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        tt_dev[i] = Complex(re=tt_dev_r, im=tt_dev_i).metal
    return tt_dev


# backward tests for type 2 complex tensor


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("hw", ((32, 64), (320, 384)))
def test_level2_conj_bw(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (-90, 90), (-70, 70))
    in_data.requires_grad = True

    input_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(in_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(in_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )

    grad_data = random_complex_tensor(input_shape, (-50, 50), (-60, 60))
    grad_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(grad_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(grad_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )
    tt_dev = ttl.tensor.conj_bw(grad_tensor, input_tensor, memcfg)
    in_data.retain_grad()

    tt_dev = convert_to_torch_tensor(tt_dev)

    pyt_y = torch.conj(in_data)

    pyt_y.backward(gradient=grad_data)

    grad_res_real = torch.real(in_data.grad)
    grad_res_imag = torch.imag(in_data.grad)

    tt_cpu = [torch.cat((grad_res_real, grad_res_imag), dim=-1)]

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        else:
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        logger.info(output)
        assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("hw", ((32, 64), (320, 384)))
def test_level2_real_bw(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (-90, 90), (-70, 70))
    in_data.requires_grad = True

    input_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(in_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(in_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )

    grad_data = random_complex_tensor(input_shape, (-50, 50), (-60, 60))
    grad_data = grad_data.real
    grad_tensor = ttl.tensor.Tensor(
        ttl.tensor.Tensor(grad_data, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )
    tt_dev = ttl.tensor.real_bw(grad_tensor, input_tensor, memcfg)
    in_data.retain_grad()

    tt_dev = convert_to_torch_tensor(tt_dev)

    pyt_y = torch.real(in_data)

    pyt_y.backward(gradient=grad_data)

    grad_res_real = torch.real(in_data.grad)
    grad_res_imag = torch.imag(in_data.grad)

    tt_cpu = [torch.cat((grad_res_real, grad_res_imag), dim=-1)]

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        else:
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        logger.info(output)
        assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("hw", ((32, 64), (320, 384)))
@pytest.mark.parametrize("alpha", [0.0, -5.0, 3.5])
def test_level2_complex_add_bw(bs, hw, alpha, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (-90, 90), (-70, 70))
    in_data.requires_grad = True

    other_data = random_complex_tensor(input_shape, (-20, 90), (-30, 100))
    other_data.requires_grad = True

    input_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(in_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(in_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )
    other_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(other_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(other_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )

    grad_data = random_complex_tensor(input_shape, (-50, 50), (-60, 60))
    grad_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(grad_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(grad_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )
    tt_dev = ttl.tensor.complex_add_bw(grad_tensor, input_tensor, other_tensor, alpha, memcfg)
    in_data.retain_grad()

    tt_dev = convert_to_torch_tensor(tt_dev)

    pyt_y = torch.add(in_data, other_data, alpha=alpha)

    pyt_y.backward(gradient=grad_data)

    grad_in_real = torch.real(in_data.grad)
    grad_in_imag = torch.imag(in_data.grad)
    grad_other_real = torch.real(other_data.grad)
    grad_other_imag = torch.imag(other_data.grad)

    tt_cpu = [torch.cat((grad_in_real, grad_in_imag), dim=-1), torch.cat((grad_other_real, grad_other_imag), dim=-1)]

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        else:
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        logger.info(output)
        assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("hw", ((32, 64), (320, 384)))
@pytest.mark.parametrize("alpha", [-5.0, 1.0, 3.5])
def test_level2_complex_sub_bw(bs, hw, alpha, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (-90, 90), (-70, 70))
    in_data.requires_grad = True

    other_data = random_complex_tensor(input_shape, (-20, 90), (-30, 100))
    other_data.requires_grad = True

    input_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(in_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(in_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )
    other_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(other_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(other_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )

    grad_data = random_complex_tensor(input_shape, (-50, 50), (-60, 60))
    grad_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(grad_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(grad_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )
    tt_dev = ttl.tensor.complex_sub_bw(grad_tensor, input_tensor, other_tensor, alpha, memcfg)
    in_data.retain_grad()

    tt_dev = convert_to_torch_tensor(tt_dev)

    pyt_y = torch.sub(in_data, other_data, alpha=alpha)

    pyt_y.backward(gradient=grad_data)

    grad_in_real = torch.real(in_data.grad)
    grad_in_imag = torch.imag(in_data.grad)
    grad_other_real = torch.real(other_data.grad)
    grad_other_imag = torch.imag(other_data.grad)

    tt_cpu = [torch.cat((grad_in_real, grad_in_imag), dim=-1), torch.cat((grad_other_real, grad_other_imag), dim=-1)]

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        else:
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        logger.info(output)
        assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("hw", ((32, 64), (320, 384)))
def test_level2_complex_mul_bw(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (-90, 90), (-70, 70))
    in_data.requires_grad = True

    other_data = random_complex_tensor(input_shape, (-20, 90), (-30, 100))
    other_data.requires_grad = True

    input_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(in_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(in_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )
    other_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(other_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(other_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )

    grad_data = random_complex_tensor(input_shape, (-50, 50), (-60, 60))
    grad_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(grad_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(grad_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )
    tt_dev = ttl.tensor.complex_mul_bw(grad_tensor, input_tensor, other_tensor, memcfg)
    in_data.retain_grad()

    tt_dev = convert_to_torch_tensor(tt_dev)

    pyt_y = torch.mul(in_data, other_data)

    pyt_y.backward(gradient=grad_data)

    grad_in_real = torch.real(in_data.grad)
    grad_in_imag = torch.imag(in_data.grad)
    grad_other_real = torch.real(other_data.grad)
    grad_other_imag = torch.imag(other_data.grad)

    tt_cpu = [torch.cat((grad_in_real, grad_in_imag), dim=-1), torch.cat((grad_other_real, grad_other_imag), dim=-1)]

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        else:
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        logger.info(output)
        assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("hw", ((32, 64), (320, 384)))
def test_level2_complex_div_bw(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (-90, 90), (-70, 70))
    in_data.requires_grad = True

    other_data = random_complex_tensor(input_shape, (-20, 90), (-30, 100))
    other_data.requires_grad = True

    input_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(in_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(in_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )
    other_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(other_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(other_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )

    grad_data = random_complex_tensor(input_shape, (-50, 50), (-60, 60))
    grad_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(grad_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(grad_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )
    tt_dev = ttl.tensor.complex_div_bw(grad_tensor, input_tensor, other_tensor, memcfg)
    in_data.retain_grad()

    tt_dev = convert_to_torch_tensor(tt_dev)

    pyt_y = torch.div(in_data, other_data)

    pyt_y.backward(gradient=grad_data)

    grad_in_real = torch.real(in_data.grad)
    grad_in_imag = torch.imag(in_data.grad)
    grad_other_real = torch.real(other_data.grad)
    grad_other_imag = torch.imag(other_data.grad)

    tt_cpu = [torch.cat((grad_in_real, grad_in_imag), dim=-1), torch.cat((grad_other_real, grad_other_imag), dim=-1)]

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        else:
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        logger.info(output)
        assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2)))
@pytest.mark.parametrize("hw", ((32, 64), (320, 384)))
@skip_for_wormhole_b0()
def test_level2_complex_div_bw_other_zero(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (-90, 90), (-70, 70))
    in_data.requires_grad = True

    other_data = random_complex_tensor(input_shape, (0, 0), (0, 0))
    other_data.requires_grad = True

    input_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(in_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(in_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )
    other_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(other_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(other_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )

    grad_data = random_complex_tensor(input_shape, (-50, 50), (-60, 60))
    grad_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(grad_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(grad_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )
    tt_dev = ttl.tensor.complex_div_bw(grad_tensor, input_tensor, other_tensor, memcfg)
    in_data.retain_grad()

    tt_dev = convert_to_torch_tensor(tt_dev)

    pyt_y = torch.div(in_data, other_data)

    pyt_y.backward(gradient=grad_data)

    grad_in_real = torch.real(in_data.grad)
    grad_in_imag = torch.imag(in_data.grad)
    grad_other_real = torch.real(other_data.grad)
    grad_other_imag = torch.imag(other_data.grad)

    tt_cpu = [torch.cat((grad_in_real, grad_in_imag), dim=-1), torch.cat((grad_other_real, grad_other_imag), dim=-1)]

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        else:
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        logger.info(output)
        assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("hw", ((32, 64), (320, 384)))
def test_level2_abs_bw(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (-90, 90), (-70, 70))
    in_data.requires_grad = True

    input_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(in_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(in_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )

    grad_data, grad_tensor = data_gen_with_range(input_shape, -50, 40, device)

    tt_dev = ttl.tensor.complex_abs_bw(grad_tensor, input_tensor, memcfg)
    in_data.retain_grad()

    tt_dev = convert_to_torch_tensor(tt_dev)

    pyt_y = torch.abs(in_data)

    pyt_y.backward(gradient=grad_data)

    grad_res_real = torch.real(in_data.grad)
    grad_res_imag = torch.imag(in_data.grad)

    tt_cpu = [torch.cat((grad_res_real, grad_res_imag), dim=-1)]

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        else:
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        logger.info(output)
        assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2)))
@pytest.mark.parametrize("hw", ((32, 64), (320, 384)))
def test_level2_abs_bw_inp_zero(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (0, 0), (0, 0))
    in_data.requires_grad = True

    input_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(in_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(in_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )

    grad_data, grad_tensor = data_gen_with_range(input_shape, -50, 80, device)

    tt_dev = ttl.tensor.complex_abs_bw(grad_tensor, input_tensor, memcfg)
    in_data.retain_grad()

    tt_dev = convert_to_torch_tensor(tt_dev)

    pyt_y = torch.abs(in_data)

    pyt_y.backward(gradient=grad_data)

    grad_res_real = torch.real(in_data.grad)
    grad_res_imag = torch.imag(in_data.grad)

    tt_cpu = [torch.cat((grad_res_real, grad_res_imag), dim=-1)]

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        else:
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        logger.info(output)
        assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("hw", ((32, 64), (320, 384)))
def test_level2_recip_bw(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (-90, 90), (-70, 70))
    in_data.requires_grad = True

    input_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(in_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(in_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )

    grad_data = random_complex_tensor(input_shape, (-50, 50), (-60, 60))
    grad_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(grad_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(grad_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )
    tt_dev = ttl.tensor.complex_recip_bw(grad_tensor, input_tensor, memcfg)
    in_data.retain_grad()

    tt_dev = convert_to_torch_tensor(tt_dev)

    pyt_y = torch.reciprocal(in_data)

    pyt_y.backward(gradient=grad_data)

    grad_res_real = torch.real(in_data.grad)
    grad_res_imag = torch.imag(in_data.grad)

    tt_cpu = [torch.cat((grad_res_real, grad_res_imag), dim=-1)]

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        else:
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        logger.info(output)
        assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2)))
@pytest.mark.parametrize("hw", ((32, 64), (320, 384)))
@skip_for_wormhole_b0()
def test_level2_recip_bw_inp_zero(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (0, 0), (0, 0))
    in_data.requires_grad = True

    input_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(in_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(in_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )

    grad_data = random_complex_tensor(input_shape, (-50, 50), (-60, 60))
    grad_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(grad_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(grad_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )
    tt_dev = ttl.tensor.complex_recip_bw(grad_tensor, input_tensor, memcfg)
    in_data.retain_grad()

    tt_dev = convert_to_torch_tensor(tt_dev)

    pyt_y = torch.reciprocal(in_data)

    pyt_y.backward(gradient=grad_data)

    grad_res_real = torch.real(in_data.grad)
    grad_res_imag = torch.imag(in_data.grad)

    tt_cpu = [torch.cat((grad_res_real, grad_res_imag), dim=-1)]

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        else:
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        logger.info(output)
        assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttl.tensor.DataType.BFLOAT16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("hw", ((32, 64), (320, 384)))
def test_level2_angle_bw(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (-90, 90), (-70, 70))
    in_data.requires_grad = True

    input_tensor = ttl.tensor.complex_tensor(
        ttl.tensor.Tensor(in_data.real, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
        ttl.tensor.Tensor(in_data.imag, dtype).to(ttl.tensor.Layout.TILE).to(device, memcfg),
    )

    grad_data, grad_tensor = data_gen_with_range(input_shape, -50, 40, device)

    tt_dev = ttl.tensor.angle_bw(grad_tensor, input_tensor, True, memcfg)
    in_data.retain_grad()

    tt_dev = convert_to_torch_tensor(tt_dev)

    pyt_y = torch.angle(in_data)

    pyt_y.backward(gradient=grad_data)

    grad_res_real = torch.real(in_data.grad)
    grad_res_imag = torch.imag(in_data.grad)

    tt_cpu = [torch.cat((grad_res_real, grad_res_imag), dim=-1)]

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        else:
            passing, output = comp_pcc(tt_cpu[i], tt_dev[i])
        logger.info(output)
        assert passing
