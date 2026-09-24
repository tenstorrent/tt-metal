# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys

import torch

import ttnn
import pytest
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal, comp_allclose
from models.common.utility_functions import is_wormhole_b0


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
    passing, output = comp_pcc(tt_cpu, tt_dev, 0.98)
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


def _complex_tensor_in(shape, device, real_memcfg, imag_memcfg):
    x = Complex(shape)
    real = ttnn.Tensor(x.real, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, real_memcfg)
    imag = ttnn.Tensor(x.imag, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, imag_memcfg)
    return ttnn.complex_tensor(real, imag), real, imag


@pytest.mark.parametrize("op, component", ((ttnn.real, "real"), (ttnn.imag, "imag")), ids=["real", "imag"])
@pytest.mark.parametrize(
    "request_tag", ("DRAM", "sharded", "matching"), ids=["explicit_DRAM", "explicit_sharded", "explicit_matching"]
)
def test_real_imag_honours_memory_config(op, component, request_tag, device):
    """real and imag place the returned component where the caller asks.

    Both used to accept `memory_config` and return the component untouched, so an explicit
    request was dropped. The component is placed in L1 and a different config is requested;
    with matching configs the requested and inherited values coincide and the defect is
    invisible. `explicit_matching` pins the other half of the contract: a request the
    component already satisfies must stay a view of it rather than become a copy.
    """
    shape = torch.Size([1, 1, 32, 64])
    complex_tensor, real, imag = _complex_tensor_in(shape, device, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
    expected_component = real if component == "real" else imag

    if request_tag == "DRAM":
        requested_memcfg = ttnn.DRAM_MEMORY_CONFIG
    elif request_tag == "sharded":
        # Complex() splits the last dim, so a component is half as wide as `shape`.
        requested_memcfg = ttnn.create_sharded_memory_config(
            expected_component.shape,
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    else:
        requested_memcfg = ttnn.L1_MEMORY_CONFIG

    output = op(complex_tensor, memory_config=requested_memcfg)

    assert (
        output.memory_config() == requested_memcfg
    ), f"requested {request_tag}: expected {requested_memcfg} but landed in {output.memory_config()}"

    # Placement is all this argument controls, so the data must be the component's own.
    passing, message = comp_equal(ttnn.to_torch(expected_component), ttnn.to_torch(output))
    logger.info(message)
    assert passing

    if request_tag == "matching":
        assert (
            output.buffer_address() == expected_component.buffer_address()
        ), "a request the component already satisfies must stay a view, not copy"


@pytest.mark.parametrize("op, component", ((ttnn.real, "real"), (ttnn.imag, "imag")), ids=["real", "imag"])
def test_real_imag_unset_memory_config_follows_own_component(op, component, device):
    """An unset memory_config leaves the component exactly where it is.

    The two components are independent tensors and are placed in opposite spaces here, so a
    default taken from the wrong one is visible: `imag` resolved its unset config from the
    *real* component, which is harmless only while the config is ignored.
    """
    shape = torch.Size([1, 1, 32, 64])
    complex_tensor, real, imag = _complex_tensor_in(shape, device, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
    expected_component = real if component == "real" else imag

    output = op(complex_tensor)

    assert (
        output.memory_config() == expected_component.memory_config()
    ), f"{component} with no memory_config: expected {expected_component.memory_config()} but landed in {output.memory_config()}"
    assert output.buffer_address() == expected_component.buffer_address(), "the default path must stay a view"


@pytest.mark.parametrize("op, component", ((ttnn.real, "real"), (ttnn.imag, "imag")), ids=["real", "imag"])
def test_real_imag_host_component_is_returned_as_is(op, component):
    """A host component is handed back untouched whatever is requested.

    Both ops accept a host ComplexTensor, and a host tensor has no memory config to satisfy;
    relocating one would need a device op, which fails on `is_device_tensor`.

    Both requests are covered: an explicit one, and an unset one, which resolves the default
    from the component's own config and so runs that lookup on a host tensor too.
    """
    shape = torch.Size([1, 1, 32, 64])
    x = Complex(shape)
    real = ttnn.Tensor(x.real, ttnn.bfloat16).to(ttnn.TILE_LAYOUT)
    imag = ttnn.Tensor(x.imag, ttnn.bfloat16).to(ttnn.TILE_LAYOUT)
    expected_component = real if component == "real" else imag
    complex_tensor = ttnn.complex_tensor(real, imag)

    for requested_memcfg in (ttnn.DRAM_MEMORY_CONFIG, None):
        output = op(complex_tensor) if requested_memcfg is None else op(complex_tensor, memory_config=requested_memcfg)

        assert output.storage_type() == ttnn.StorageType.HOST, f"requested {requested_memcfg}"
        passing, message = comp_equal(ttnn.to_torch(expected_component), ttnn.to_torch(output))
        logger.info(message)
        assert passing
