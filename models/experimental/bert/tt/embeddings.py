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
        self.embeddings = hugging_face_reference_model.bert.embeddings

        # Disable dropout
        self.eval()

    def forward(self, input_ids, token_type_ids=None):
        return self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)


def run_embeddings_inference():
    return
