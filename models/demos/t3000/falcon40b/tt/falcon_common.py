# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

MODEL_VERSION = "tiiuae/falcon-40b-instruct"


class PytorchFalconCausalLM(torch.nn.Module):
    def __init__(self, hf_reference_model, num_layers=None):
        super().__init__()
        self.model = hf_reference_model

        if num_layers is None:
            pass
        else:
            self.model.transformer.h = self.model.transformer.h[:num_layers]

        # Disable dropout
        self.model.eval()

    def forward(self, input_ids, past_key_values, use_cache):
        # this method is returning the logits
        result = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=False,
        )
        return result
