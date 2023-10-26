# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import json

from models.mistral.tt.mistral_transformer import TtTransformer
from models.mistral.tt.mistral_configuration import TtModelArgs
from tests.models.mistral.demo.model import Transformer
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.569994959501121),),
)
def test_mistral_attention_inference(pcc, model_location_generator, device):
    mistral_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    state_dict = torch.load(mistral_path / "consolidated.00.pth")
    base_address = f""
    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))

    state_dict = {
        k[0:]: v
        for k, v in state_dict.items()
        if (
            k.startswith("layers.0.")
            or k.startswith("tok_embeddings")
            or k.startswith("norm.weight")
            or k.startswith("output.weight")
        )
    }

    model_args.max_batch_size = 1
    model_args.n_layers = 1
    reference_model = Transformer(args=model_args)
    reference_model.load_state_dict(state_dict)

    tt_model = TtTransformer(
        args=model_args,
        state_dict=state_dict,
        device=device,
        base_address=base_address,
    )

    input = torch.tensor([[1, 851, 349, 264, 7324, 2245, 354, 2692, 7487, 12068, 28705]])

    positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    reference_output = reference_model(input, positions)

    tt_output = tt_model(input, positions)
    tt_output_torch = tt_to_torch_tensor(tt_output).squeeze(0)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mistral_Transformer Passed!")
    else:
        logger.warning("Mistral_Transformer Failed!")

    assert passing, f"Mistral_Transformer output does not meet PCC requirement {pcc}."
