"""
This is the Entire ImageTransformer for Gemma-3-4b-it.
We have adapted the TtGemmaImageTransformerBlock from TtLlamaImageTransformerBlock
with changes incorporating the GemmaImageAttention and GemmaImageFeedForward
"""
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tqdm import tqdm

from models.common.lightweightmodule import LightweightModule
from models.demos.gemma3.tt.gemma_image_block import TtGemmaImageTransformerBlock


class TtGemmaImageTransformer(LightweightModule):
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
        gated=False,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.gated = gated

        self.resblocks = [
            TtGemmaImageTransformerBlock(
                mesh_device=mesh_device,
                state_dict=state_dict,
                tt_ccl=tt_ccl,
                state_dict_prefix=f"{state_dict_prefix}{block_key}.{i}.",
                weight_cache_path=weight_cache_path,
                dtype=dtype,
                configuration=configuration,
                gated=gated,
            )
            for i in tqdm(range(layers), desc=f"Loading vision transformer layers")
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
