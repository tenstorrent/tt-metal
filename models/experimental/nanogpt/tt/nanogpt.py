# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import GPT2LMHeadModel
from models.experimental.nanogpt.tt.nanogpt_model import TtGPT


def _nanogpt(config, state_dict, device):
    return TtGPT(
        config=config,
        state_dict=state_dict,
        device=device,
    )


def nanogpt_model(device) -> TtGPT:
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    state_dict = model.state_dict()
    config = model.config
    model = _nanogpt(config, state_dict, device)
    return model
