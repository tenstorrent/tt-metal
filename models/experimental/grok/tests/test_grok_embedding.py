# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger

# Set Grok flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["GROK_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"
    os.environ["GROK_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Grok/Grok-1/"

import ttnn
from models.experimental.grok.tt.grok_embedding import TtGrokEmbedding
from models.experimental.grok.reference.tokenizer import Tokenizer
from models.experimental.grok.tt.model_config import TtModelArgs
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


def test_grok_embedding(device, use_program_cache, reset_seeds):
    for device in t3k_mesh_device.get_device_ids():
        t3k_mesh_device.get_device(device).enable_async(True)
    dtype = ttnn.bfloat16

    model_args = TtModelArgs(device, dummy_weights=os.getenv("CI") == "true")
    state_dict = model_args.load_state_dict()
    tokenizer = Tokenizer(model_args.tokenizer_path)

    reference_emb = Emb()
    reference_emb.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    tt_emb = TtGrokEmbedding(
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
        logger.info("Grok_embedding Passed!")
    else:
        logger.warning("Grok_embedding Failed!")

    assert passing, f"Grok_embedding output does not meet PCC requirement {0.99}."
