# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)


def data_gen_pt_tt(input_shapes, device, required_grad=False):
    torch.manual_seed(213919)
    pt_tensor = torch.randn(input_shapes, requires_grad=required_grad).bfloat16()
    tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    return pt_tensor, tt_tensor


def data_gen_with_range(input_shapes, low, high, device, required_grad=False, is_row_major=False):
    assert high > low, "Incorrect range provided"
    torch.manual_seed(213919)
    pt_tensor = torch.rand(input_shapes, requires_grad=required_grad).bfloat16() * (high - low) + low
    if is_row_major:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16).to(ttnn.ROW_MAJOR_LAYOUT).to(device)
    else:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    return pt_tensor, tt_tensor


def data_gen_with_val(input_shapes, device, required_grad=False, val=1, is_row_major=False):
    pt_tensor = (torch.ones(input_shapes, requires_grad=required_grad) * val).bfloat16()
    if is_row_major:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16).to(ttnn.ROW_MAJOR_LAYOUT).to(device)
    else:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    return pt_tensor, tt_tensor


def data_gen_pt_tt_prod(input_shapes, device, all_dimensions, dim, required_grad=False):
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
        shape_Required = torch.Size([1, 1, 1, 1])
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
