# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import pytest
import ttnn
from loguru import logger
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal, comp_allclose

from models.common.utility_functions import is_wormhole_b0
from tests.ttnn.nightly.unit_tests.operations.eltwise.complex.utility_funcs import (
    convert_complex_to_torch_tensor,
    random_complex_tensor,
)


@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.float32,)))
@pytest.mark.parametrize("bs", ((1, 1),))
@pytest.mark.parametrize("hw", ((32, 32),))
def test_conj(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (-90, 90), (-70, 70))

    input_tensor = ttnn.complex_tensor(
        ttnn.Tensor(in_data.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(in_data.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )

    tt_dev = ttnn.conj(input_tensor, memory_config=memcfg)

    tt_to_torch = convert_complex_to_torch_tensor(tt_dev)

    golden_function = ttnn.get_golden_function(ttnn.conj)
    golden_tensor = golden_function(in_data)

    passing, output = comp_pcc(golden_tensor, tt_to_torch)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "request_tag",
    ("DRAM", "sharded", "matching", None),
    ids=["explicit_DRAM", "explicit_sharded", "explicit_matching", "unset"],
)
def test_conj_memory_config_applies_to_the_imaginary_component(request_tag, device):
    """conj computes only the imaginary component, so memory_config places that component.

    The real component of the result is the input's own, returned as a view, and stays where that
    component already is, which is what the op's docstring states. This test pins the two
    together: if conj is ever changed to place both components, it fails and the note has to be
    rewritten with it.

    The explicit requests are for somewhere the components are not, so a config that is applied
    is distinguishable from one that is not; with matching configs the requested and inherited
    values coincide and the assertion holds either way.

    The unset case needs the two components in different spaces. conj resolves an unset config
    from the *real* component (`complex_unary.cpp`), so only when the imaginary component sits
    elsewhere does the placement of the computed component say which one it came from. That
    default is today's behaviour and is pinned here, not endorsed: see #56954.
    """
    input_shape = torch.Size([1, 1, 32, 32])
    in_data = random_complex_tensor(input_shape, (-90, 90), (-70, 70))
    real_memcfg = ttnn.DRAM_MEMORY_CONFIG if request_tag is None else ttnn.L1_MEMORY_CONFIG
    real = ttnn.Tensor(in_data.real, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, real_memcfg)
    imag = ttnn.Tensor(in_data.imag, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, ttnn.L1_MEMORY_CONFIG)
    input_tensor = ttnn.complex_tensor(real, imag)

    if request_tag == "DRAM":
        requested_memcfg = ttnn.DRAM_MEMORY_CONFIG
    elif request_tag == "sharded":
        requested_memcfg = ttnn.create_sharded_memory_config(
            input_shape,
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    elif request_tag == "matching":
        requested_memcfg = ttnn.L1_MEMORY_CONFIG
    else:
        requested_memcfg = None

    output = (
        ttnn.conj(input_tensor) if requested_memcfg is None else ttnn.conj(input_tensor, memory_config=requested_memcfg)
    )

    # The computed component lands where the caller asked; an unset request follows the real component,
    # which here is in DRAM while the imaginary component it is computed from is in L1.
    expected_imag_memcfg = real.memory_config() if requested_memcfg is None else requested_memcfg
    assert output.imag.memory_config() == expected_imag_memcfg, (
        f"requested {request_tag}: imaginary component expected {expected_imag_memcfg} "
        f"but landed in {output.imag.memory_config()}"
    )

    # The real component is the input's own: same config, same buffer, no copy.
    assert (
        output.real.memory_config() == real.memory_config()
    ), f"requested {request_tag}: real component moved to {output.real.memory_config()}"
    assert output.real.buffer_address() == real.buffer_address(), "the real component must stay a view of the input"

    real_passing, real_message = comp_equal(ttnn.to_torch(real), ttnn.to_torch(output.real))
    imag_passing, imag_message = comp_equal(-ttnn.to_torch(imag), ttnn.to_torch(output.imag))
    logger.info(real_message)
    logger.info(imag_message)
    assert real_passing and imag_passing
