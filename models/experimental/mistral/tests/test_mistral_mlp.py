# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import json
from pathlib import Path
from models.experimental.mistral.tt.model_config import TtModelArgs, get_model_config
from models.experimental.mistral.tt.mistral_mlp import TtMistralMLP
from models.experimental.mistral.reference.model import FeedForward
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "model_config",
    ("BFLOAT16-DRAM", "BFLOAT8-DRAM"),
)
@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_mistral_mlp_inference(pcc, model_config, model_location_generator, device):
    dtype = model_config.split("-")[0]
    mistral_path = Path(model_location_generator("mistral-7B-v0.1", model_subdir="mistral"))
    state_dict = torch.load(mistral_path / "consolidated.00.pth")
    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))

    state_dict = {k[22:]: v for k, v in state_dict.items() if (k.startswith("layers.0.feed_forward"))}
    base_address = f""
    model_args.max_batch_size = 1
    model_args.WEIGHTS_DTYPE = dtype
    reference_model = FeedForward(args=model_args)
    reference_model.load_state_dict(state_dict)

    tt_model = TtMistralMLP(
        device=device,
        state_dict=state_dict,
        base_address=base_address,
        model_config=get_model_config(model_config),
    )
    input = torch.rand(1, 32, 4096)
    reference_output = reference_model(input)

    tt_input = torch2tt_tensor(input, device)

    tt_output = tt_model(tt_input)
    tt_output_torch = tt2torch_tensor(tt_output).squeeze(0)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mistral_Feed_Forward Passed!")
    else:
        logger.warning("Mistral_feed_Forward Failed!")

    assert passing, f"Mistral_Feed_forward output does not meet PCC requirement {pcc}: {pcc_message}."
