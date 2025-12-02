from functools import partial
from functools import reduce
from loguru import logger
from models.common.utility_functions import comp_pcc
from models.common.utility_functions import is_wormhole_b0
from models.common.utility_functions import torch_random
from pathlib import Path
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal, comp_allclose
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp
import math
import pytest
import sys
import torch
import ttnn

# Test file to reproduce PCC ERROR. The required modules will be exported according to branch.


@pytest.mark.parametrize(
    "shape",
    [
        # [1, 1, 80, 80],
        [1, 1, 32, 32],
    ],
)
def test_add_fp32_input_activ(device, shape):
    x_torch = torch.ones(shape, dtype=torch.float32) * 2
    y_torch = torch.ones(shape, dtype=torch.float32) * 4
    z_torch = torch.square(torch.nn.functional.silu(x_torch) + y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn.add(
        x_tt,
        y_tt,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        activations=[ttnn.UnaryOpType.SQUARE],
        use_legacy=None,
    )
    tt_out = ttnn.to_torch(z_tt_add)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.9999
    assert status


input_bcast_shape_pairs = [
    # ((1, 1, 1, 1), (1, 1, 80, 80)),
    ((1, 1, 1, 1), (1, 1, 32, 32)),
]


@pytest.mark.parametrize("shape_and_broadcast_spec", input_bcast_shape_pairs)
def test_broadcast_to_bf8_b(device, shape_and_broadcast_spec):
    # pytest.skip("bfloat8 has issues.")
    shape, broadcast_shape = shape_and_broadcast_spec
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b
    )(shape)
    torch_result = torch_input_tensor.broadcast_to(broadcast_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)

    out_tt = ttnn.experimental.broadcast_to(input_tensor, ttnn.Shape(broadcast_shape))
    output = ttnn.to_torch(out_tt)

    assert (
        output.shape == torch_result.shape
    ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

    assert_with_pcc(torch_result, output, 0.9999)


class Complex:
    def __init__(self, input_shape: torch.Size = None, re=None, im=None):
        if input_shape:
            # the first method is needed for the abs test to fail.
            # val = 1.0 + torch.arange(0, input_shape.numel()).reshape(input_shape).bfloat16()
            # self._cplx = val[:, :, :, : input_shape[-1] // 2] + val[:, :, :, input_shape[-1] // 2 :] * 1j
            self._cplx = torch.ones(input_shape) + torch.zeros(input_shape) * 1j
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


"""
@pytest.mark.parametrize(
    "memcfg",
    (
        # ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    # ids=["out_DRAM", "out_L1"],
    ids=["out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
# @pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
# @pytest.mark.parametrize("bs", ((1, 1),))
@pytest.mark.parametrize("bs", ((1, 2),))
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
"""


@pytest.mark.parametrize(
    "memcfg",
    (
        # ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    # ids=["out_DRAM", "out_L1"],
    ids=["out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
# @pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("bs", ((1, 1),))
def test_level2_recip(bs, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], 32, 32])
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


"""
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
"""
