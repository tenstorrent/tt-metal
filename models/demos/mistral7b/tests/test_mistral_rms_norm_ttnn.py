# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import json
from pathlib import Path
import ttnn
from models.demos.mistral7b.tt.model_config_ttnn import TtModelArgs, get_model_config
from models.demos.mistral7b.tt.mistral_rms_norm_ttnn import TtRMSNorm
from models.demos.mistral7b.reference.model import RMSNorm
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
    if dtype == "BFLOAT8":
        dtype = ttnn.bfloat8_b
    elif dtype == "BFLOAT16":
        dtype = ttnn.bfloat16
    model_config = get_model_config(model_config)

    mistral_path = Path(model_location_generator(model_config["DEFAULT_CACHE_PATH"], model_subdir="mistral"))
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
        model_config=model_config,
    )
    input = torch.rand(1, 32, 4096)
    reference_output = reference_model(input)

    tt_input = ttnn.from_torch(
        input, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT
    )  # , device, put_on_device=False)

    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(0)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mistral_rms_norm Passed!")
    else:
        logger.warning("Mistral_rms_norm Failed!")

    assert passing, f"Mistral_rms_norm output does not meet PCC requirement {0.99}."
