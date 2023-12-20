# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import tt_lib
from loguru import logger


def bw_data_gen(input_shapes, device, required_grad=False):
    torch.manual_seed(1235)
    pt_tensor = torch.randn(input_shapes, requires_grad=required_grad).bfloat16()
    tt_tensor = (
        tt_lib.tensor.Tensor(pt_tensor, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    return pt_tensor, tt_tensor


def compare_results(tt_tensor, golden_tensor, comparison_funcs):
    status = True
    for i in range(len(tt_tensor)):
        tt_out_tensor = tt_tensor[i].cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
        pt_out_tensor = golden_tensor[i]
        comp_pass, comp_out = comparison_funcs(pt_out_tensor, tt_out_tensor)
        logger.info(comp_pass)
        logger.info(comp_out)
        status = status & comp_pass
    return status
