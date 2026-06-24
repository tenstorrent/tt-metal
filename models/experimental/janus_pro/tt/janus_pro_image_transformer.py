"""
This is the Entire ImageTransformer for Janus-Pro-7B.
We have adapted the TtJanusProImageTransformer from TtLlamaImageTransformer
with changes incorporating the JanusImageAttention and JanusImageFeedForward
"""
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

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
        block_key="resblocks",
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device

        self.resblocks = [
            TtJanusProImageTransformerBlock(
                mesh_device=mesh_device,
                state_dict=state_dict,
                tt_ccl=tt_ccl,
                state_dict_prefix=f"{state_dict_prefix}{block_key}.{i}.",
                weight_cache_path=weight_cache_path,
                dtype=dtype,
                configuration=configuration,
            )
            for i in tqdm(range(layers), desc=f"Loading vision transformer layers")
        ]

    def forward(self, x, mask=None):
        seq_len = x.shape[-2]
        assert seq_len % 32 == 0 and seq_len > 0, "Seqlen must be divisible by 32"

        for r in self.resblocks:
            x = r(x, mask=mask)
        return x
