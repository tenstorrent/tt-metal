# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from pathlib import Path
import os

from transformers import GPT2LMHeadModel

from loguru import logger
import models.experimental.nanogpt.tt.nanogpt_block as nanogpt_block
from models.experimental.nanogpt.nanogpt_utils import get_tt_cache_path, store_weights

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_allclose,
    comp_pcc,
    skip_for_wormhole_b0,
)


@skip_for_wormhole_b0()
@pytest.mark.skip(reason="Test is hanging gs, see issue #7534")
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
@pytest.mark.parametrize(
    "pcc",
    ((0.99,),),
)
def test_nanogpt_block(device, pcc, dtype, reset_seeds):
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    config = model_hf.config
    model_hf.eval()
    block = 0
    base_address = f"transformer.h.{block}"

    test_in = torch.rand(1, 60, 768)
    pt_block = model_hf.transformer.h[block]
    pt_out = pt_block.forward(test_in)

    tt_test_in = torch_to_tt_tensor_rm(test_in, device)
    model_version = "gpt2"
    tt_cache_path = get_tt_cache_path(model_version)

    if (
        tt_cache_path == (str(Path(f"models/experimental/nanogpt/datasets/{model_version}")) + "/")
        and len(os.listdir(f"models/experimental/nanogpt/datasets/{model_version}")) < 320
    ):
        store_weights(model_version=model_version, file_name=tt_cache_path, dtype=dtype, base_address=base_address)

    tt_block = nanogpt_block.TtBlock(config, base_address, device, tt_cache_path, dtype)
    tt_block.eval()

    tt_out = tt_block.forward(tt_test_in)

    tt_out_converted = tt_to_torch_tensor(tt_out).squeeze(0)

    does_pass, pcc_message = comp_pcc(pt_out[0], tt_out_converted, pcc)

    logger.info(comp_allclose(pt_out[0], tt_out_converted))
    logger.info(pcc_message)

    if does_pass:
        logger.info("nanogpt_block: Passed!")
    else:
        logger.warning("nanogpt_block: Failed!")

    assert does_pass
