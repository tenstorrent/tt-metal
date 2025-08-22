# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
This file implements the Vision Transformer submodule specific for the Mistral-Small-3.1-24B-Instruct-2503 model.
This pipeline iterates over the pixtral image blocks to generate the image embeddings.
"""

from tqdm import tqdm

from models.common.lightweightmodule import LightweightModule
from models.experimental.mistral_24b.tt.vision_pixtral_image_block import TtPixtralImageTransformerBlock


class TtPixtralTransformer(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
        layers,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl

        block_key = "layers"
        self.resblocks = [
            TtPixtralImageTransformerBlock(
                mesh_device=mesh_device,
                tt_ccl=self.tt_ccl,
                state_dict=state_dict,
                state_dict_prefix=f"{state_dict_prefix}{block_key}.{i}.",
                weight_cache_path=weight_cache_path,
                dtype=dtype,
                configuration=configuration,
            )
            for i in tqdm(range(layers), desc=f"Loading vision transformer layers")
        ]

    def forward(self, x, return_intermediate=None, position_embeddings=None):
        """
        Different from reference impl in that if return_intermediates, it returns
        a list of intermediate tensors rather than a stack of intermediates.
        Outer code will have to be aware and handle this correctly.
        """
        out = []
        for idx, r in enumerate(self.resblocks):
            if return_intermediate is not None and idx in return_intermediate:
                out.append(x)
            x = r(x, position_embeddings=position_embeddings)
        if return_intermediate is not None:
            return x, out
        return x
