# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.experimental.nanogpt.nanogpt_utils import get_tt_cache_path, store_weights
from pathlib import Path
import os

from transformers import GPT2LMHeadModel

from loguru import logger
import models.experimental.nanogpt.tt.nanogpt_mlp as nanogpt_mlp


from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_allclose,
    comp_pcc,
    skip_for_wormhole_b0,
)


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
@pytest.mark.parametrize(
    "pcc",
    ((0.99,),),
)
@skip_for_wormhole_b0()
def test_nanogpt_mlp(device, pcc, dtype, reset_seeds):
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    config = model_hf.config
    model_hf.eval()
    block = 0
    base_address = f"transformer.h.{block}.mlp"

    test_in = torch.rand(1, 43, 768)
    tt_test_in = torch_to_tt_tensor_rm(test_in, device)
    model_version = "gpt2"
    tt_cache_path = get_tt_cache_path(model_version)

    if (
        tt_cache_path == (str(Path(f"models/experimental/nanogpt/datasets/{model_version}")) + "/")
        and len(os.listdir(f"models/experimental/nanogpt/datasets/{model_version}")) < 320
    ):
        store_weights(model_version=model_version, file_name=tt_cache_path, dtype=dtype, base_address=base_address)

    tt_mlp = nanogpt_mlp.TtMLP(base_address, config, device, tt_cache_path, dtype)

    tt_out = tt_mlp.forward(tt_test_in)

    pt_mlp = model_hf.transformer.h[block].mlp
    pt_out = pt_mlp.forward(test_in)

    tt_out_converted = tt_to_torch_tensor(tt_out).squeeze(0)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, pcc)

    logger.info(comp_allclose(pt_out, tt_out_converted))
    logger.info(pcc_message)

    if does_pass:
        logger.info("nanogpt_mlp: Passed!")
    else:
        logger.warning("nanogpt_mlp: Failed!")

    assert does_pass
