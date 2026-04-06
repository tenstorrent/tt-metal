# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import GPT2LMHeadModel
from models.experimental.nanogpt.tt.nanogpt_model import TtGPT


def _nanogpt(config, device, tt_cache_path, dtype):
    return TtGPT(
        config=config,
        device=device,
        tt_cache_path=tt_cache_path,
        dtype=dtype,
    )


def nanogpt_model(device, dtype) -> TtGPT:
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    config = model.config
    tt_cache_path = "/mnt/MLPerf/tt_dnn-models/tt/NanoGPT/"
    model = _nanogpt(config, device, tt_cache_path, dtype)
    return model
