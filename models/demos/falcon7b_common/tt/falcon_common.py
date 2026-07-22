# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from transformers.cache_utils import DynamicCache

MODEL_VERSION = "tiiuae/falcon-7b-instruct"


def build_past_key_values_cache(past_key_values):
    """Convert legacy per-layer (key, value) tuples into a transformers Cache.

    transformers 5.x dropped legacy-tuple KV-cache support (and DynamicCache's
    from_legacy_cache); the HF model now requires a Cache object. The
    DynamicCache constructor takes the per-layer (k, v) tuples directly; fall
    back to from_legacy_cache for transformers <5.x.
    """
    if past_key_values is None or isinstance(past_key_values, DynamicCache):
        return past_key_values
    past_key_values = tuple(past_key_values)
    if len(past_key_values) == 0:
        return None
    if hasattr(DynamicCache, "from_legacy_cache"):  # transformers < 5.x
        return DynamicCache.from_legacy_cache(past_key_values)
    return DynamicCache(past_key_values)  # transformers >= 5.x


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
            past_key_values=build_past_key_values_cache(past_key_values),
            use_cache=use_cache,
            return_dict=False,
        )
        return result
