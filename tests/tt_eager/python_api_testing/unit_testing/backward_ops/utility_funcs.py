# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import tt_lib
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)


def data_gen_pt_tt(input_shapes, device, required_grad=False):
    torch.manual_seed(213919)
    pt_tensor = torch.randn(input_shapes, requires_grad=required_grad).bfloat16()
    tt_tensor = (
        tt_lib.tensor.Tensor(pt_tensor, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    return pt_tensor, tt_tensor


def compare_results(tt_tensor, golden_tensor, pcc=0.99):
    status = True
    for i in range(len(tt_tensor)):
        tt_out_tensor = tt_tensor[i].cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
        pt_out_tensor = golden_tensor[i]
        comp_pass, comp_out = comparison_funcs.comp_pcc(pt_out_tensor, tt_out_tensor, pcc=pcc)
        comp_all, _ = comparison_funcs.comp_allclose(pt_out_tensor, tt_out_tensor, atol=4, rtol=1e-1)
        logger.debug(comp_pass)
        logger.debug(comp_all)
        logger.debug(comp_out)
        status = status & (comp_pass | comp_all)
    return status
