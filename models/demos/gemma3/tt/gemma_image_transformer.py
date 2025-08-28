"""
This is the Entire ImageTransformer for Gemma-3-4b-it.
Uses simplified pattern that mirrors llama structure but with gemma blocks
Only differences: gemma blocks and sequence length validation (128 vs 32)
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

        # Store the same attributes as llama transformer for compatibility
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.gated = gated

        # Create gemma blocks (same pattern as llama but with gemma blocks)
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
        Forward pass - same logic as llama but with gemma-specific seq_len validation.
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
