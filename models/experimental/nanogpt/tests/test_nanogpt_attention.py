# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import os
from pathlib import Path

from transformers import GPT2LMHeadModel

from loguru import logger
import models.experimental.nanogpt.tt.nanogpt_attention as nanogpt_attention
from models.experimental.nanogpt.nanogpt_utils import get_tt_cache_path, store_weights

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_allclose,
    comp_pcc,
    is_wormhole_b0,
    is_blackhole,
)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.skip(reason="Test is hanging gs, see issue #7534")
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
@pytest.mark.parametrize(
    "pcc",
    ((0.99,),),
)
def test_nanogpt_attn(device, pcc, dtype, reset_seeds):
    # Prepare input
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    config = model_hf.config
    model_hf.eval()
    block = 0
    base_address = f"transformer.h.{block}.attn"

    test_in = torch.rand(1, 60, 768)
    pt_attn = model_hf.transformer.h[block].attn
    pt_out = pt_attn.forward(test_in)

    model_version = "gpt2"
    tt_cache_path = get_tt_cache_path(model_version)

    if (
        tt_cache_path == (str(Path(f"models/experimental/nanogpt/datasets/{model_version}")) + "/")
        and len(os.listdir(f"models/experimental/nanogpt/datasets/{model_version}")) < 320
    ):
        store_weights(model_version=model_version, file_name=tt_cache_path, dtype=dtype, base_address=base_address)

    tt_test_in = torch_to_tt_tensor_rm(test_in, device)
    tt_attn = nanogpt_attention.TtCausalSelfAttention(config, base_address, device, tt_cache_path, dtype)

    tt_out = tt_attn.forward(tt_test_in)

    tt_out_converted = tt_to_torch_tensor(tt_out).squeeze(0)

    does_pass, pcc_message = comp_pcc(pt_out[0], tt_out_converted, pcc)

    logger.info(comp_allclose(pt_out[0], tt_out_converted))
    logger.info(pcc_message)

    if does_pass:
        logger.info("nanogpt_attention: Passed!")
    else:
        logger.warning("nanogpt_attention: Failed!")

    assert does_pass
