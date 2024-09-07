# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_allclose,
    comp_pcc,
)
from models.experimental.swin.tt.swin_self_attention import (
    TtSwinSelfAttention,
)
from transformers import SwinModel


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_swin_self_attention_inference(device, pcc, reset_seeds):
    SELF_ATTN_LAYER_INDEX = 0
    base_address = f"encoder.layers.{SELF_ATTN_LAYER_INDEX}.blocks.{SELF_ATTN_LAYER_INDEX}.attention.self"

    model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    # Torch swinselfattention
    torch_model = model.encoder.layers[SELF_ATTN_LAYER_INDEX].blocks[SELF_ATTN_LAYER_INDEX].attention.self

    # Tt swinselfattention
    num_heads, window_size, dim = 3, 7, 96
    tt_model = TtSwinSelfAttention(
        config=model.config,
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        base_address=base_address,
        device=device,
        state_dict=model.state_dict(),
    )

    # Run torch model
    hidden_states = torch.rand(64, 49, 96)
    attention_mask = torch.ones(64, 49, 49)

    torch_output = torch_model(hidden_states, attention_mask)

    # Run tt model
    hidden_states = torch.unsqueeze(hidden_states, 0)
    tt_hidden_states = torch_to_tt_tensor_rm(hidden_states, device)

    attention_mask = torch.unsqueeze(attention_mask, 0)
    tt_attention_mask = torch_to_tt_tensor_rm(attention_mask, device)

    tt_output = tt_model(tt_hidden_states, tt_attention_mask)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output[0])
    tt_output_torch = tt_output_torch.squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output[0], tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output[0], tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("SwinSelfAttention Passed!")
    else:
        logger.warning("SwinSelfAttention Failed!")

    assert does_pass
