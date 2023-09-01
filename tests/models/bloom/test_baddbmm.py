# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
import pytest

from models.utility_functions import (
    comp_allclose,
    comp_pcc,
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

from loguru import logger
from models.bloom.tt.baddbmm import TtBaddbmm


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_baddbmm(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    input = torch.randn(16, 62, 62)
    batch1 = torch.randn(16, 62, 64)
    batch2 = torch.randn(16, 64, 62)

    tt_input = torch_to_tt_tensor_rm(input, device)
    tt_batch1 = torch_to_tt_tensor_rm(batch1, device)
    tt_batch2 = torch_to_tt_tensor_rm(batch2, device)

    tt_baddbmm = TtBaddbmm(device)

    pt_out = torch.baddbmm(input, batch1, batch2)
    pt_out = pt_out.unsqueeze(0)

    tt_out = tt_baddbmm(tt_input, tt_batch1, tt_batch2)
    tt_out_converted = tt_to_torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, pcc)

    logger.info(comp_allclose(pt_out, tt_out_converted))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("baddbmm: Passed!")
    else:
        logger.warning("baddbmm: Failed!")

    assert does_pass, f"baddbmm output does not meet PCC requirement {pcc}."
