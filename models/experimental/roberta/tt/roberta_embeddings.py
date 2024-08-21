# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch


class TtEmbeddings(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return


class PytorchEmbeddings(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.embeddings = hugging_face_reference_model.embeddings

        # Disable dropout
        self.eval()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        return self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
