# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.common.lightweightmodule import LightweightModule


class TtEmbeddings(LightweightModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return


class PytorchEmbeddings(LightweightModule):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.embeddings = hugging_face_reference_model.model.embed_tokens

        # Disable dropout
        self.eval()

    def forward(self, x):
        return self.embeddings(x)
