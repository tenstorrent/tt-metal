# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch

import ttnn
from models.utility_functions import (
    nearest_32,
)
from models.common.lightweightmodule import LightweightModule
from models.demos.wormhole.llama31_8b_N300.tt.llama_image_block import TtLlamaImageTransformerBlock


class TtLlamaImageTransformer(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
        layers,
        gated=False,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.gated = gated

        self.resblocks = [
            TtLlamaImageTransformerBlock(
                mesh_device=mesh_device,
                state_dict=state_dict,
                state_dict_prefix=f"{state_dict_prefix}resblocks.{i}.",
                weight_cache_path=weight_cache_path,
                dtype=dtype,
                configuration=configuration,
                gated=gated,
            )
            for i in range(layers)
        ]

    def forward(self, x, return_intermediate=None, mask=None):
        """
        Different from reference impl in that if return_intermediates, it returns
        a list of intermediate tensors rather than a stack of intermediates.
        Outer code will have to be aware and handle this correctly.
        """
        seq_len = x.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"

        out = []
        for idx, r in enumerate(self.resblocks):
            if return_intermediate is not None and idx in return_intermediate:
                out.append(x)
            x = r(x, mask=mask)
        if return_intermediate is not None:
            return x, out
        return x
