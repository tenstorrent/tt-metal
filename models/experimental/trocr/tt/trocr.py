# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.trocr.trocr_generate_utils import GenerationMixin

from transformers import VisionEncoderDecoderModel


def _trocr(model, config, state_dict, device):
    return GenerationMixin(model=model, device=device, config=config, state_dict=state_dict)


def trocr_causal_llm(device) -> GenerationMixin:
    model_name = "microsoft/trocr-base-handwritten"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    config = model.decoder.config
    state_dict = model.state_dict()
    model = _trocr(model, config, state_dict, device)
    return model
