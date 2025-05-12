# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)


def data_gen_with_range_batch_norm(
    input_shapes,
    low,
    high,
    device,
    is_input=False,
    required_grad=False,
    testing_dtype="bfloat16",
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    assert high > low, "Incorrect range provided"
    torch.manual_seed(213919)
    channels = input_shapes[1]
    size = input_shapes if is_input else channels
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)
    pt_tensor = torch.rand(size, requires_grad=required_grad, dtype=torch_dtype) * (high - low) + low
    reshaped_tensor = pt_tensor
    if not is_input:
        reshaped_tensor = pt_tensor.view(1, channels, 1, 1)
    tt_tensor = ttnn.from_torch(
        reshaped_tensor,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn_dtype,
        memory_config=memory_config,
    )
    return pt_tensor, tt_tensor


def data_gen_pt_tt(input_shapes, device, required_grad=False):
    torch.manual_seed(213919)
    pt_tensor = torch.randn(input_shapes, requires_grad=required_grad).bfloat16()
    tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    return pt_tensor, tt_tensor


def data_gen_with_range(input_shapes, low, high, device, required_grad=False, is_row_major=False, seed=213919):
    assert high > low, "Incorrect range provided"
    torch.manual_seed(seed)
    pt_tensor = torch.rand(input_shapes, requires_grad=required_grad).bfloat16() * (high - low) + low
    if is_row_major:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16).to(ttnn.ROW_MAJOR_LAYOUT).to(device)
    else:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    return pt_tensor, tt_tensor


def data_gen_with_range_dtype(
    input_shapes, low, high, device, required_grad=False, is_row_major=False, ttnn_dtype=ttnn.bfloat16
):
    assert high > low, "Incorrect range provided"
    torch.manual_seed(213919)
    pt_tensor = torch.rand(input_shapes, requires_grad=required_grad).bfloat16() * (high - low) + low
    if is_row_major:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn_dtype).to(ttnn.ROW_MAJOR_LAYOUT).to(device)
    else:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn_dtype).to(ttnn.TILE_LAYOUT).to(device)
    return pt_tensor, tt_tensor


def data_gen_with_range_int(input_shapes, low, high, device, required_grad=False, is_row_major=False):
    assert high > low, "Incorrect range provided"
    torch.manual_seed(213919)
    pt_tensor = torch.randint(low, high, input_shapes, dtype=torch.int32, requires_grad=required_grad)

    if is_row_major:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.float32).to(ttnn.ROW_MAJOR_LAYOUT).to(device)
    else:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.float32).to(ttnn.TILE_LAYOUT).to(device)

    return pt_tensor, tt_tensor


def data_gen_with_val(input_shapes, device, required_grad=False, val=1, is_row_major=False):
    pt_tensor = (torch.ones(input_shapes, requires_grad=required_grad) * val).bfloat16()
    if is_row_major:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16).to(ttnn.ROW_MAJOR_LAYOUT).to(device)
    else:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    return pt_tensor, tt_tensor


def data_gen_pt_tt_prod(input_shapes, device, all_dimensions=True, dim=0, required_grad=False):
    torch.manual_seed(213919)
    pt_tensor_temp = torch.zeros(input_shapes, requires_grad=required_grad).bfloat16()
    shape_Required = torch.Size(
        [
            input_shapes[0] if (dim != 0 and dim != -4) else 1,
            input_shapes[1] if (dim != 1 and dim != -3) else 1,
            input_shapes[2] if (dim != 2 and dim != -2) else 1,
            input_shapes[3] if (dim != 3 and dim != -1) else 1,
        ]
    )
    if all_dimensions == False and (dim == 1 or dim == 0 or dim == -4 or dim == -3):
        pt_tensor = torch.randn(shape_Required, requires_grad=required_grad).bfloat16()
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
        return pt_tensor, tt_tensor
    elif all_dimensions == False:
        pt_tensor = torch.randn(shape_Required, requires_grad=required_grad).bfloat16()
        if dim == 3 or dim == -1:
            pt_tensor_temp[:, :, :, :1] = pt_tensor
        elif dim == 2 or dim == -2:
            pt_tensor_temp[:, :, :1, :] = pt_tensor
    else:
        shape_Required = torch.Size([])
        pt_tensor = torch.randn(shape_Required, requires_grad=required_grad).bfloat16()
        pt_tensor_temp[:1, :1, :1, :1] = pt_tensor
    tt_tensor = ttnn.Tensor(pt_tensor_temp, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    return pt_tensor, tt_tensor


def compare_results(tt_tensor, golden_tensor, pcc=0.99):
    status = True
    for i in range(len(tt_tensor)):
        tt_out_tensor = tt_tensor[i].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        pt_out_tensor = golden_tensor[i]
        comp_pass, comp_out = comparison_funcs.comp_pcc(pt_out_tensor, tt_out_tensor, pcc=pcc)
        comp_all, _ = comparison_funcs.comp_allclose(pt_out_tensor, tt_out_tensor, atol=4, rtol=1e-1)
        logger.debug(comp_pass)
        logger.debug(comp_all)
        logger.debug(comp_out)
        status = status & (comp_pass | comp_all)
    return status


def compare_results_batch_norm(tt_tensor, golden_tensor, pcc=0.99, stats=False):
    status = True
    for i in range(len(tt_tensor)):
        tt_out_tensor = tt_tensor[i]
        pt_out_tensor = golden_tensor[i]
        comp_pass, comp_out = comparison_funcs.comp_pcc(pt_out_tensor, tt_out_tensor, pcc=pcc)
        comp_all, comp_out_res = comparison_funcs.comp_allclose(pt_out_tensor, tt_out_tensor, atol=4, rtol=1e-1)
        logger.debug(comp_pass)
        logger.debug(comp_all)
        logger.debug(comp_out)
        logger.debug(comp_out_res)
        if stats:
            status = status & comp_all
        else:
            status = status & comp_pass & comp_all

    return status


def compare_pcc(tt_tensor, golden_tensor, pcc=0.99):
    status = True
    for i in range(len(tt_tensor)):
        tt_out_tensor = tt_tensor[i].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        pt_out_tensor = golden_tensor[i]
        comp_pass, comp_out = comparison_funcs.comp_pcc(pt_out_tensor, tt_out_tensor, pcc=pcc)
        logger.debug(comp_pass)
        logger.debug(comp_out)
        status = status & comp_pass
    return status


def compare_equal(tt_tensor, golden_tensor):
    status = True
    for i in range(len(tt_tensor)):
        tt_out_tensor = tt_tensor[i].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        pt_out_tensor = golden_tensor[i]
        comp_pass, comp_out = comparison_funcs.comp_equal(pt_out_tensor, tt_out_tensor)
        logger.debug(comp_pass)
        logger.debug(comp_out)
        status = status & comp_pass
    return status


def compare_all_close(tt_tensor, golden_tensor, atol=4, rtol=1e-1):
    status = True
    for i in range(len(tt_tensor)):
        tt_out_tensor = tt_tensor[i].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        pt_out_tensor = golden_tensor[i]
        comp_all, comp_out = comparison_funcs.comp_allclose(pt_out_tensor, tt_out_tensor, atol=atol, rtol=rtol)
        logger.debug(comp_all)
        logger.debug(comp_out)
        status = status & comp_all
    return status
