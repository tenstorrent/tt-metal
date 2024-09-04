# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger

import ttnn
from models.demos.t3000.mixtral8x7b.tt.mixtral_embedding import TtMixtralEmbedding
from models.demos.t3000.mixtral8x7b.reference.tokenizer import Tokenizer
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


def test_mixtral_embedding(device, use_program_cache, reset_seeds):
    device.enable_async(True)

    dtype = ttnn.bfloat16

    model_args = TtModelArgs(device)
    state_dict = model_args.load_state_dict()
    tokenizer = Tokenizer(model_args.tokenizer_path)

    reference_emb = Emb()
    reference_emb.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    tt_emb = TtMixtralEmbedding(
        device=device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=dtype,
    )

    prompts = ["Joy"] * 32
    pt_input = torch.tensor([tokenizer.encode(prompt) for prompt in prompts])
    reference_output = reference_emb(pt_input)
    logger.info(f"reference_output: {reference_output.shape}")

    tt_input = ttnn.from_torch(pt_input, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_output = tt_emb(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)
    logger.info(f"tt_output_torch: {tt_output_torch.shape}")

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mixtral_embedding Passed!")
    else:
        logger.warning("Mixtral_embedding Failed!")

    assert passing, f"Mixtral_embedding output does not meet PCC requirement {0.99}."
