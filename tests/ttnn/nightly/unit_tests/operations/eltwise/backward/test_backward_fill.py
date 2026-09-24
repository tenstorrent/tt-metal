# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.common.utility_functions import is_wormhole_b0, is_blackhole
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_all_close,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
# Pytorch Reference
# - name: fill.Scalar(Tensor self, Scalar value) -> Tensor
#   self: zeros_like(grad)
#   result: at::fill(self_t, 0)
def test_bw_fill(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)

    tt_output_tensor_on_device = ttnn.fill_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.fill_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_all_close(tt_output_tensor_on_device, golden_tensor, atol=0, rtol=0)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_fill_opt_tensor(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)

    _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)
    input_grad = ttnn.to_memory_config(input_grad, ttnn.L1_MEMORY_CONFIG)
    cq_id = 0
    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.fill_bw(grad_tensor, input_tensor, input_grad=input_grad, queue_id=cq_id)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    golden_function = ttnn.get_golden_function(ttnn.fill_bw)
    golden_tensor = golden_function(grad_data, in_data)

    tt_output_tensor_on_device = [input_grad]
    comp_pass = compare_all_close(tt_output_tensor_on_device, golden_tensor, atol=0, rtol=0)
    assert comp_pass


@pytest.mark.parametrize("input_shapes", ((torch.Size([1, 1, 32, 32])),))
def test_bw_fill_honours_memory_config(input_shapes, device):
    """The gradient fill_bw allocates itself must land in the requested memory config.

    Regression: both `zeros_like` calls passed `std::nullopt`, so the result inherited
    grad's config — the tensor is created from grad — and an explicitly requested config
    was ignored. `output_memory_config` was computed one line above and never read: an
    unused local rather than an unused parameter, so nothing warned.

    grad and input are both placed in L1 with DRAM requested, so the requested config
    differs from the inherited one; with matching configs the defect is invisible.
    See test_bw_fill_unset_memory_config_follows_grad for the unset-config half.
    """
    _, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    _, grad_tensor = data_gen_with_range(input_shapes, -50, 50, device)

    input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG)
    grad_tensor = ttnn.to_memory_config(grad_tensor, ttnn.L1_MEMORY_CONFIG)

    result = ttnn.fill_bw(grad_tensor, input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    assert (
        result[0].memory_config() == ttnn.DRAM_MEMORY_CONFIG
    ), f"requested {ttnn.DRAM_MEMORY_CONFIG} but landed in {result[0].memory_config()}"


@pytest.mark.parametrize("input_shapes", ((torch.Size([1, 1, 32, 32])),))
def test_bw_fill_preallocated_output_wins_over_memory_config(input_shapes, device):
    """A caller-supplied input_grad keeps its own config and is written in place."""
    _, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    _, grad_tensor = data_gen_with_range(input_shapes, -50, 50, device)
    preallocated = ttnn.full(input_shapes, 7.0, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)

    result = ttnn.fill_bw(grad_tensor, input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG, input_grad=preallocated)

    assert result[0].memory_config() == ttnn.L1_MEMORY_CONFIG, "preallocated output should win over memory_config"
    assert torch.all(ttnn.to_torch(preallocated) == 0.0), "preallocated tensor was not written"


@pytest.mark.parametrize("input_shapes", ((torch.Size([1, 1, 32, 32])),))
@pytest.mark.parametrize(
    "grad_memcfg, input_memcfg",
    (
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ),
    ids=["grad_L1_input_DRAM", "grad_DRAM_input_L1"],
)
def test_bw_fill_unset_memory_config_follows_grad(input_shapes, device, grad_memcfg, input_memcfg):
    """With no memory_config, the result must follow grad's placement, not input's.

    fill_bw builds its output from `grad`, so an unset config has always inherited grad's
    config. The obvious fix -- passing the `output_mem_config.value_or(input.memory_config())`
    local that sat unused above -- would silently change that default whenever grad and input
    live in different spaces. grad and input are deliberately placed in opposite spaces here;
    with both in the same space this cannot discriminate.
    """
    _, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    _, grad_tensor = data_gen_with_range(input_shapes, -50, 50, device)

    grad_tensor = ttnn.to_memory_config(grad_tensor, grad_memcfg)
    input_tensor = ttnn.to_memory_config(input_tensor, input_memcfg)

    result = ttnn.fill_bw(grad_tensor, input_tensor)

    assert (
        result[0].memory_config() == grad_memcfg
    ), f"unset memory_config should follow grad ({grad_memcfg}) but landed in {result[0].memory_config()}"


# Zero-gradient backward ops that build their output from grad, same shape as fill_bw.
_ZERO_GRAD_OPS = {"floor": ttnn.floor_bw, "round": ttnn.round_bw, "ceil": ttnn.ceil_bw}


@pytest.mark.parametrize("input_shapes", ((torch.Size([1, 1, 32, 32])),))
@pytest.mark.parametrize("op", sorted(_ZERO_GRAD_OPS))
def test_bw_zero_gradient_ops_honour_memory_config(input_shapes, device, op):
    """floor_bw, round_bw and ceil_bw are the same shape as fill_bw and dropped the config too.

    Each is a bare `zeros_like(grad)` with `output_mem_config` commented out in the
    signature, which is exactly how the fill_bw case stayed hidden.
    """
    _, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    _, grad_tensor = data_gen_with_range(input_shapes, -50, 50, device)

    grad_tensor = ttnn.to_memory_config(grad_tensor, ttnn.L1_MEMORY_CONFIG)
    input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG)

    result = _ZERO_GRAD_OPS[op](grad_tensor, input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    assert (
        result[0].memory_config() == ttnn.DRAM_MEMORY_CONFIG
    ), f"{op}_bw requested DRAM but landed in {result[0].memory_config()}"


@pytest.mark.parametrize("input_shapes", ((torch.Size([1, 1, 32, 32])),))
@pytest.mark.parametrize("op", sorted(_ZERO_GRAD_OPS))
def test_bw_zero_gradient_ops_unset_memory_config_follows_grad(input_shapes, device, op):
    """With no memory_config these must keep inheriting grad's placement."""
    _, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    _, grad_tensor = data_gen_with_range(input_shapes, -50, 50, device)

    grad_tensor = ttnn.to_memory_config(grad_tensor, ttnn.L1_MEMORY_CONFIG)
    input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    result = _ZERO_GRAD_OPS[op](grad_tensor, input_tensor)

    assert result[0].memory_config() == ttnn.L1_MEMORY_CONFIG, f"{op}_bw unset config should follow grad"
