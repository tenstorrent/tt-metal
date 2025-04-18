# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_transformers.tt.model import Transformer as BaseTransformer
from models.demos.glm4.tt.decoder import Glm4TransformerBlock
from models.demos.glm4.tt.model_config import Glm4ModelArgs  # needed for type hints/checks


class Glm4Transformer(BaseTransformer):
    """
    glm4 specific transformer model, inheriting from base Transformer
    instantiates glm4 specific components like Glm4TransformerBlock
    """

    def __init__(self, mesh_device, state_dict, args: Glm4ModelArgs, layers, cache_path=None, expert_group=None):
        # note: signature might need adjustment based on BaseTransformer.__init__
        super().__init__(mesh_device, state_dict, args, layers, cache_path)

        # override self.layers to use Glm4TransformerBlock
        # the base init likely creates layers using BaseTransformerBlock
        # we need to recreate them here using Glm4TransformerBlock

        # clear layers created by base init
        self.layers = ttnn.ModuleList()

        # instantiate layers using the Glm4TransformerBlock
        for layer_i in layers:
            self.layers.append(
                Glm4TransformerBlock(
                    mesh_device,
                    state_dict,
                    args,
                    layer_i,
                    cache_path,
                )
            )
        # ensure other components like norm, output etc are correctly initialized or inherited

    # forward method likely remains the same as BaseTransformer
    # unless GLM-4 has specific changes in how blocks are called
    # def forward(self, tokens: ttnn.Tensor, start_pos: int, freqs_cis: ttnn.Tensor = None):
    #     return super().forward(tokens, start_pos, freqs_cis)
