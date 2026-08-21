"""
The stack of encoder blocks in the Janus-Pro-7B vision tower.

HF reference: `vision_model.encoder` -- every block sees the same shape, so the whole stack is
one loop over identical layers.
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tqdm import tqdm

from models.common.lightweightmodule import LightweightModule
from models.experimental.janus_pro.tt.janus_pro_image_block import TtJanusProImageTransformerBlock


class TtJanusProImageTransformer(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        tt_ccl,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
        layers,
    ):
        super().__init__()

        self.resblocks = [
            TtJanusProImageTransformerBlock(
                mesh_device=mesh_device,
                state_dict=state_dict,
                tt_ccl=tt_ccl,
                state_dict_prefix=f"{state_dict_prefix}layers.{i}.",
                weight_cache_path=weight_cache_path,
                dtype=dtype,
                configuration=configuration,
                residual_dtype=(ttnn.bfloat8_b if i >= configuration.VISION_BFP8_RESIDUAL_FROM_LAYER else None),
            )
            for i in tqdm(range(layers), desc=f"Loading {layers} vision transformer blocks")
        ]

    def forward(self, x, mask=None):
        seq_len = x.shape[-2]
        assert seq_len % 32 == 0 and seq_len > 0, "Seqlen must be divisible by 32"

        for r in self.resblocks:
            x = r(x, mask=mask)
        return x
