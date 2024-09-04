# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys

import torch

import ttnn
import pytest
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal, comp_allclose
from models.utility_functions import is_wormhole_b0


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


@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
def test_level2_real(bs, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], 32, 64])
    # check real
    x = Complex(input_shape)
    xtt = ttnn.complex_tensor(
        ttnn.Tensor(x.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(x.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )
    tt_dev = ttnn.real(xtt, memory_config=memcfg)
    tt_dev = tt_dev.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_cpu = x.real
    passing, output = comp_equal(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
def test_level2_imag(bs, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], 32, 64])
    # check imag
    x = Complex(input_shape)
    xtt = ttnn.complex_tensor(
        ttnn.Tensor(x.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(x.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )
    tt_dev = ttnn.imag(xtt, memory_config=memcfg)
    tt_dev = tt_dev.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_cpu = x.imag
    passing, output = comp_equal(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
def test_level2_abs(bs, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], 32, 64])
    # check abs
    x = Complex(input_shape)
    xtt = ttnn.complex_tensor(
        ttnn.Tensor(x.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(x.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )
    tt_dev = ttnn.abs(xtt, memory_config=memcfg)
    tt_dev = tt_dev.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_cpu = x.abs().real
    if is_wormhole_b0():
        passing, output = comp_pcc(tt_cpu, tt_dev, pcc=0.8)
    else:
        passing, output = comp_pcc(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
def test_level2_abs(bs, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], 32, 64])
    # check abs
    x = Complex(input_shape)
    xtt = ttnn.complex_tensor(
        ttnn.Tensor(x.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(x.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )
    tt_dev = ttnn.abs(xtt, memory_config=memcfg)
    tt_dev = tt_dev.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_cpu = x.abs().real
    if is_wormhole_b0():
        passing, output = comp_pcc(tt_cpu, tt_dev, pcc=0.8)
    else:
        passing, output = comp_pcc(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
def test_level2_conj(bs, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], 32, 64])
    # check abs
    x = Complex(input_shape)
    xtt = ttnn.complex_tensor(
        ttnn.Tensor(x.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(x.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )
    tt_dev = ttnn.conj(xtt, memory_config=memcfg)
    tt_dev_r = tt_dev.real.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_dev_i = tt_dev.imag.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_dev = Complex(re=tt_dev_r, im=tt_dev_i).metal
    tt_cpu = x.conj().metal
    if is_wormhole_b0():
        passing, output = comp_pcc(tt_cpu, tt_dev, pcc=0.8)
    else:
        passing, output = comp_pcc(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
def test_level2_recip(bs, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], 32, 64])
    # check abs
    x = Complex(input_shape)
    x = x.div(x * 0.5)
    xtt = ttnn.complex_tensor(
        ttnn.Tensor(x.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(x.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )
    tt_dev = ttnn.reciprocal(xtt, memory_config=memcfg)
    tt_dev_r = tt_dev.real.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_dev_i = tt_dev.imag.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_dev = Complex(re=tt_dev_r, im=tt_dev_i).metal
    tt_cpu = x.recip().metal

    if is_wormhole_b0():
        pass  # pytest.skip("[RECIP]: skip assertion for this test on WH B0")

    passing, output = comp_pcc(tt_cpu, tt_dev, pcc=0.96)
    logger.info(output)
    assert passing


@pytest.mark.skip(reason="This test is failing because ttnn.add doesn't support complex tensors")
@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
def test_level2_add(bs, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], 32, 64])
    # check add
    x = Complex(input_shape)
    y = Complex(input_shape) * -0.5

    xtt = ttnn.complex_tensor(
        ttnn.Tensor(x.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(x.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )
    ytt = ttnn.complex_tensor(
        ttnn.Tensor(y.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(y.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )

    tt_dev = ttnn.add(xtt, ytt, memory_config=memcfg)
    tt_dev_r = tt_dev.real.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_dev_i = tt_dev.imag.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_dev = Complex(re=tt_dev_r, im=tt_dev_i).metal
    tt_cpu = x.add(y).metal

    passing, output = comp_pcc(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.skip(reason="This test is failing because ttnn.sub doesn't support complex tensors")
@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
def test_level2_sub(bs, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], 32, 64])
    # check add
    x = Complex(input_shape)
    y = Complex(input_shape) * -0.5

    xtt = ttnn.complex_tensor(
        ttnn.Tensor(x.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(x.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )
    ytt = ttnn.complex_tensor(
        ttnn.Tensor(y.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(y.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )

    tt_dev = ttnn.subtract(xtt, ytt, memory_config=memcfg)
    tt_dev_r = tt_dev.real.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_dev_i = tt_dev.imag.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_dev = Complex(re=tt_dev_r, im=tt_dev_i).metal

    tt_cpu = x.sub(y).metal

    passing, output = comp_pcc(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.skip(reason="This test is failing because ttnn.mul doesn't support complex tensors")
@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
def test_level2_mul(bs, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], 32, 64])
    # check add
    x = Complex(input_shape)
    y = Complex(input_shape) * -0.5

    xtt = ttnn.complex_tensor(
        ttnn.Tensor(x.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(x.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )
    ytt = ttnn.complex_tensor(
        ttnn.Tensor(y.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(y.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )

    tt_dev = ttnn.multiply(xtt, ytt, memory_config=memcfg)
    tt_dev_r = tt_dev.real.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_dev_i = tt_dev.imag.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_dev = Complex(re=tt_dev_r, im=tt_dev_i).metal

    tt_cpu = x.mul(y).metal

    passing, output = comp_pcc(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.skip(reason="This test is failing because ttnn.div doesn't support complex tensors")
@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
def test_level2_div(bs, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], 32, 64])
    # check add
    x = Complex(input_shape) * 0.5
    y = Complex(input_shape) * 1

    xtt = ttnn.complex_tensor(
        ttnn.Tensor(x.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(x.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )
    ytt = ttnn.complex_tensor(
        ttnn.Tensor(y.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(y.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )

    tt_dev = ttnn.divide(xtt, xtt, memory_config=memcfg)
    tt_dev_r = tt_dev.real.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_dev_i = tt_dev.imag.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_dev = Complex(re=tt_dev_r, im=tt_dev_i).metal

    tt_cpu = x.div(y).metal

    passing, output = comp_pcc(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
def test_level2_is_real(bs, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], 32, 64])
    # check abs
    x = Complex(input_shape)
    xtt = ttnn.complex_tensor(
        ttnn.Tensor(x.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(0 * x.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )
    tt_dev = ttnn.is_real(xtt, memory_config=memcfg)
    tt_dev = tt_dev.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_cpu = torch.ones(x.real.shape)
    if is_wormhole_b0():
        passing, output = comp_pcc(tt_cpu, tt_dev, pcc=0.8)
    else:
        passing, output = comp_pcc(tt_cpu, tt_dev)
    xtt.deallocate()
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
def test_level2_is_imag(bs, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], 32, 64])
    # check abs
    x = Complex(input_shape)
    xtt = ttnn.complex_tensor(
        ttnn.Tensor(0 * x.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(x.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )
    tt_dev = ttnn.is_imag(xtt, memory_config=memcfg)
    tt_dev = tt_dev.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_cpu = torch.ones(x.imag.shape)
    if is_wormhole_b0():
        passing, output = comp_pcc(tt_cpu, tt_dev, pcc=0.8)
    else:
        passing, output = comp_pcc(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
def test_level2_angle(bs, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], 32, 64])
    # check imag
    x = Complex(input_shape)
    xtt = ttnn.complex_tensor(
        ttnn.Tensor(x.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(x.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )
    tt_dev = ttnn.angle(xtt, memory_config=memcfg)
    tt_dev = tt_dev.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    x_real = torch.tensor(x.real, dtype=torch.bfloat16)
    x_imag = torch.tensor(x.imag, dtype=torch.bfloat16)
    x_torch = torch.complex(x_real.float(), x_imag.float())
    tt_cpu = torch.angle(x_torch).to(torch.bfloat16)
    passing, output = comp_pcc(tt_cpu, tt_dev)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
def test_level2_polar(bs, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], 32, 32])
    # check polar function

    # we set real = abs = 1 on unit circle
    # we set imag = angle theta
    x = Complex(None, re=torch.ones(input_shape), im=torch.rand(input_shape))

    xtt = ttnn.complex_tensor(
        ttnn.Tensor(x.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(x.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )
    tt_dev = ttnn.polar(xtt, memory_config=memcfg)
    tt_dev_real = tt_dev.real.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_dev_imag = tt_dev.imag.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_cpu = torch.polar(x.real, x.imag)
    tt_cpu_real = tt_cpu.real.to(torch.bfloat16).to(float)
    tt_cpu_imag = tt_cpu.imag.to(torch.bfloat16).to(float)

    real_passing, real_output = comp_allclose(tt_cpu_real, tt_dev_real, 0.0125, 1)
    logger.info(real_output)
    imag_passing, imag_output = comp_allclose(tt_cpu_imag, tt_dev_imag, 0.0125, 1)
    logger.info(imag_output)
    assert real_passing and imag_passing
