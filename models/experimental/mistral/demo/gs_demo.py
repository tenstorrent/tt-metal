# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import json
from pathlib import Path
from models.experimental.mistral.tt.mistral_transformer import TtTransformer
from models.experimental.mistral.mistral_utils import generate
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.experimental.mistral.reference.tokenizer import Tokenizer


@pytest.mark.parametrize(
    "batch_size",
    (
        (1),
        (8),
        (32),
    ),
)
def test_gs_demo_single_input_inference(batch_size, model_location_generator, device, get_tt_cache_path):
    prompts = batch_size * [
        "A man is sitting on a roof ",
    ]

    mistral_path = model_location_generator("mistral-7B-v0.1", model_subdir="mistral")
    tokenizer = Tokenizer(str(Path(mistral_path) / "tokenizer.model"))
    state_dict = torch.load(mistral_path / "consolidated.00.pth")
    base_address = f""
    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))

    model_args.max_batch_size = batch_size
    model_args.n_layers = 32

    tt_cache_path = str(get_tt_cache_path("mistral-7B-v0.1", model_subdir="mistral")) + "/"

    tt_model = TtTransformer(
        args=model_args,
        device=device,
        base_address=base_address,
        tt_cache_path=tt_cache_path,
    )

    tt_output = generate(
        prompts,
        tt_model,
        tokenizer,
        max_tokens=35,
    )

    logger.info("Input Prompt")
    logger.info(prompts)
    logger.info("Mistral Model output")
    logger.info(tt_output)
