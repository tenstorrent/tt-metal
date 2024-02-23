# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import json
from models.experimental.mistral.tt.model_config import TtModelArgs, get_model_config
from models.experimental.mistral.tt.mistral_rms_norm import TtRMSNorm
from models.experimental.mistral.reference.model import RMSNorm
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "model_config",
    ("BFLOAT16-DRAM", "BFLOAT8-DRAM"),
)
def test_mistral_rms_norm_inference(model_config, model_location_generator, device):
    dtype = model_config.split("-")[0]
    mistral_path = model_location_generator("mistral-7B-v0.1", model_subdir="mistral")
    state_dict = torch.load(mistral_path / "consolidated.00.pth")
    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))

    state_dict = {k[24:]: v for k, v in state_dict.items() if (k.startswith("layers.0.attention_norm."))}
    base_address = f""  # f"layers.0.attention_norm."
    reference_model = RMSNorm(dim=model_args.dim)
    reference_model.load_state_dict(state_dict)

    tt_model = TtRMSNorm(
        device=device,
        base_address=base_address,
        state_dict=state_dict,
        model_config=get_model_config(model_config),
    )
    input = torch.rand(1, 32, 4096)
    reference_output = reference_model(input)

    tt_input = torch_to_tt_tensor_rm(input, device, put_on_device=False)

    tt_output = tt_model(tt_input)
    tt_output_torch = tt_to_torch_tensor(tt_output).squeeze(0)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mistral_rms_norm Passed!")
    else:
        logger.warning("Mistral_rms_norm Failed!")

    assert passing, f"Mistral_rms_norm output does not meet PCC requirement {0.99}."
