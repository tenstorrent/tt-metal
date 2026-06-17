# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HF text-only reference wrapper for Mistral3 multimodal models."""

import torch
from transformers import DynamicCache


class MistralTextReferenceWrapper:
    """Runs language_model + lm_head for text-only prefill/decode reference.

    ``reference_transformer(wrap=False)`` swaps ``model.model`` to the text
    language_model backbone. The top-level ``Mistral3ForConditionalGeneration.forward``
    still expects ``image_hidden_states`` on that output, so text-only PCC tests
    must call the backbone and lm_head directly.
    """

    def __init__(self, model):
        self.model = model
        self.past_key_values = DynamicCache()

    def forward(self, inputs_embeds, start_pos, mode="decode"):
        position_ids = torch.tensor(
            [list(range(start_pos, start_pos + inputs_embeds.shape[1]))] * inputs_embeds.shape[0]
        )
        outputs = self.model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=self.past_key_values,
            return_dict=True,
            output_hidden_states=True,
        )
        self.past_key_values = outputs.past_key_values
        logits = self.model.lm_head(outputs.last_hidden_state)
        if mode == "decode":
            return logits
        return outputs.hidden_states[-2]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def eval(self):
        self.model.eval()


def load_reference_text_model(model_args, *, load_checkpoint=True):
    model = model_args.reference_transformer(wrap=False, load_checkpoint=load_checkpoint)
    return MistralTextReferenceWrapper(model)
