# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import json

from models.mistral.tt.mistral_transformer_block import TtTransformerBlock
from models.mistral.tt.mistral_configuration import TtModelArgs
from tests.models.mistral.demo.model import TransformerBlock
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_mistral_attention_inference(pcc, model_location_generator, device):
    mistral_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    state_dict = torch.load(mistral_path / "consolidated.00.pth")
    base_address = f""
    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))

    state_dict = {k[9:]: v for k, v in state_dict.items() if (k.startswith("layers.0."))}

    model_args.max_batch_size = 1
    reference_model = TransformerBlock(args=model_args)
    reference_model.load_state_dict(state_dict)

    tt_model = TtTransformerBlock(
        args=model_args,
        state_dict=state_dict,
        device=device,
        base_address=base_address,
    )

    input = torch.randn(1, 11, 4096)
    freqs_cis = torch.randn(11, 64)
    positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    mask = torch.randn(11, 11)

    reference_output = reference_model(input, freqs_cis, positions, mask=mask)

    tt_input = torch_to_tt_tensor_rm(input, device)
    tt_freqs_cis = torch_to_tt_tensor_rm(freqs_cis, device)
    tt_mask = torch_to_tt_tensor_rm(mask, device, put_on_device=False)
    tt_position = torch_to_tt_tensor_rm(positions, device, put_on_device=False)
    tt_output = tt_model(tt_input, tt_freqs_cis, tt_position, mask)
    tt_output_torch = tt_to_torch_tensor(tt_output).squeeze(0)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mistral_Transformer_Block Passed!")
    else:
        logger.warning("Mistral_Transformer_Block Failed!")

    assert passing, f"Mistral_Transformer_Block output does not meet PCC requirement {pcc}."
