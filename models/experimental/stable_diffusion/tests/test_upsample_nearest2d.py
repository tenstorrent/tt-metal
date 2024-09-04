# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn.functional as F
import torch
from loguru import logger


from models.utility_functions import torch_to_tt_tensor, tt_to_torch_tensor
from models.utility_functions import (
    comp_pcc,
    comp_allclose_and_pcc,
)
from models.experimental.stable_diffusion.tt.upsample_nearest2d import TtUpsampleNearest2d


def test_run_upsample_nearest_inference(device):
    # synthesize the input
    input_shape = [1, 1, 32, 32]
    input = torch.randn(input_shape)

    torch_output = F.interpolate(input, scale_factor=2.0, mode="nearest")

    tt_input = torch_to_tt_tensor(input, device)
    tt_up = TtUpsampleNearest2d(scale_factor=2.0)
    tt_out = tt_up(tt_input)
    tt_output = tt_to_torch_tensor(tt_out)

    passing = comp_pcc(torch_output, tt_output)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output))

    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")
