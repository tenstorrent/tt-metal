# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
    data_gen_with_range_dtype,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_add(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.add_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.add_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_add_bf8b(input_shapes, device):
    in_data, input_tensor = data_gen_with_range_dtype(input_shapes, -100, 100, device, True, False, ttnn.bfloat8_b)
    other_data, other_tensor = data_gen_with_range_dtype(input_shapes, -100, 100, device, True, False, ttnn.bfloat8_b)
    grad_data, grad_tensor = data_gen_with_range_dtype(input_shapes, -100, 100, device, False, False, ttnn.bfloat8_b)

    tt_output_tensor_on_device = ttnn.add_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.add_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True]])
def test_bw_add_with_opt_output(input_shapes, device, are_required_outputs):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -90, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -70, 90, device)
    input_grad = None
    other_grad = None

    if are_required_outputs[0]:
        _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)
    if are_required_outputs[1]:
        _, other_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.add_bw(
        grad_tensor,
        input_tensor,
        other_tensor,
        are_required_outputs=are_required_outputs,
        input_grad=input_grad,
        other_grad=other_grad,
        queue_id=cq_id,
    )
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))
    tt_output_tensor_on_device = [input_grad, other_grad]

    golden_function = ttnn.get_golden_function(ttnn.add_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

    status = True
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            status = status & compare_pcc([tt_output_tensor_on_device[i]], [golden_tensor[i]])
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("scalar", [1.0, 0.5, 0.035])
def test_bw_add_scalar(input_shapes, scalar, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.add_bw(grad_tensor, input_tensor, scalar)

    golden_function = ttnn.get_golden_function(ttnn.add_bw)
    golden_tensor = golden_function(grad_data, in_data, scalar)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


# Tensor-tensor ops whose gradients are allocated by preallocated_tensors_check.
# concat_bw is covered separately: it needs a differently shaped grad tensor.
_TT_GRAD_OPS = {
    "add": lambda g, a, b, req, mc: ttnn.add_bw(g, a, b, are_required_outputs=req, memory_config=mc),
    "addalpha": lambda g, a, b, req, mc: ttnn.addalpha_bw(g, a, b, 1.0, are_required_outputs=req, memory_config=mc),
    "subalpha": lambda g, a, b, req, mc: ttnn.subalpha_bw(g, a, b, 1.0, are_required_outputs=req, memory_config=mc),
    "rsub": lambda g, a, b, req, mc: ttnn.rsub_bw(g, a, b, are_required_outputs=req, memory_config=mc),
    "div": lambda g, a, b, req, mc: ttnn.div_bw(g, a, b, are_required_outputs=req, memory_config=mc),
    "mul": lambda g, a, b, req, mc: ttnn.mul_bw(g, a, b, are_required_outputs=req, memory_config=mc),
    "assign": lambda g, a, b, req, mc: ttnn.assign_bw(g, a, b, are_required_outputs=req, memory_config=mc),
}


@pytest.mark.parametrize("input_shapes", ((torch.Size([1, 1, 32, 32])),))
@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True]])
@pytest.mark.parametrize("op", sorted(_TT_GRAD_OPS))
def test_bw_helper_allocated_grads_honour_memory_config(input_shapes, device, are_required_outputs, op):
    """Gradients the op allocates itself must land in the requested memory config.

    Regression for `preallocated_tensors_check`, which allocated via `empty_like`
    without a memory config, so the gradients inherited the input's config instead.
    Existing coverage either omits `memory_config` or supplies preallocated outputs,
    so the requested and inherited configs never diverge there.

    Inputs are placed in L1 and DRAM is requested: with DRAM inputs the inherited and
    requested configs coincide and the defect is invisible.
    """
    _, input_tensor_a = data_gen_with_range(input_shapes, 1, 100, device, True)
    _, input_tensor_b = data_gen_with_range(input_shapes, 1, 100, device, True)
    _, grad_tensor = data_gen_with_range(input_shapes, -50, 50, device)

    input_tensor_a = ttnn.to_memory_config(input_tensor_a, ttnn.L1_MEMORY_CONFIG)
    input_tensor_b = ttnn.to_memory_config(input_tensor_b, ttnn.L1_MEMORY_CONFIG)
    grad_tensor = ttnn.to_memory_config(grad_tensor, ttnn.L1_MEMORY_CONFIG)

    # no preallocated outputs, so the op allocates them through the helper
    result = _TT_GRAD_OPS[op](
        grad_tensor, input_tensor_a, input_tensor_b, are_required_outputs, ttnn.DRAM_MEMORY_CONFIG
    )

    for i, required in enumerate(are_required_outputs):
        if not required:
            continue
        assert result[i] is not None
        # compare the whole config, not just buffer_type: a wrong memory_layout or shard
        # spec is the same class of bug, and empty_like is handed the full config
        assert (
            result[i].memory_config() == ttnn.DRAM_MEMORY_CONFIG
        ), f"{op}_bw grad {i} was requested in {ttnn.DRAM_MEMORY_CONFIG} but landed in {result[i].memory_config()}"


@pytest.mark.parametrize("input_shapes", ((torch.Size([1, 1, 32, 32])),))
@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True]])
def test_bw_concat_helper_allocated_grads_honour_memory_config(input_shapes, device, are_required_outputs):
    """concat_bw reaches its grads through ttnn.slice, a different write path from assign."""
    _, input_tensor_a = data_gen_with_range(input_shapes, -100, 100, device, True)
    _, input_tensor_b = data_gen_with_range(input_shapes, -100, 100, device, True)
    _, grad_tensor = data_gen_with_range(torch.Size([1, 1, 64, 32]), -50, 50, device)

    input_tensor_a = ttnn.to_memory_config(input_tensor_a, ttnn.L1_MEMORY_CONFIG)
    input_tensor_b = ttnn.to_memory_config(input_tensor_b, ttnn.L1_MEMORY_CONFIG)
    grad_tensor = ttnn.to_memory_config(grad_tensor, ttnn.L1_MEMORY_CONFIG)

    result = ttnn.concat_bw(
        grad_tensor,
        input_tensor_a,
        input_tensor_b,
        0,
        are_required_outputs=are_required_outputs,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for i, required in enumerate(are_required_outputs):
        if required:
            assert result[i].memory_config() == ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize("input_shapes", ((torch.Size([1, 1, 32, 32])),))
@pytest.mark.parametrize("op", ["add", "sub", "div", "mul"])
def test_bw_scalar_overload_grads_honour_memory_config(input_shapes, device, op):
    """The tensor-scalar overloads allocate their own gradient and must honour memory_config.

    add_bw and sub_bw scalar previously ignored output_mem_config outright (it was
    commented out in the signature); div_bw and mul_bw scalar took it for the compute but
    allocated with a bare empty_like. Existing scalar coverage passes no memory_config, so
    inherited and requested never diverged.

    Inputs must be in L1 with DRAM requested: with DRAM inputs the two coincide and the
    defect is unobservable.
    """
    _, input_tensor = data_gen_with_range(input_shapes, 1, 100, device, True)
    _, grad_tensor = data_gen_with_range(input_shapes, -50, 50, device)

    input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG)
    grad_tensor = ttnn.to_memory_config(grad_tensor, ttnn.L1_MEMORY_CONFIG)

    fn = {"add": ttnn.add_bw, "sub": ttnn.sub_bw, "div": ttnn.div_bw, "mul": ttnn.mul_bw}[op]
    result = fn(grad_tensor, input_tensor, 2.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    assert (
        result[0].memory_config() == ttnn.DRAM_MEMORY_CONFIG
    ), f"{op}_bw scalar overload was requested in DRAM but landed in {result[0].memory_config()}"
