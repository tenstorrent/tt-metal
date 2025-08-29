"""
This is the FeedForward submodule for vision block in Gemma-3-4b-it
"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_transformers.tt.multimodal.llama_image_mlp import TtLlamaImageFeedForward


def create_gemma_mlp_forward(args):
    """
    Creates gemma-specific forward function with dynamic MAX_MM_SEQ_LEN and stability optimizations (optimizations currently do not work for gemma3)
    """
    def gemma_forward(llama_mlp, x: ttnn.Tensor) -> ttnn.Tensor:
        seq_len = x.shape[-2]
        
        # Gemma-3 specific: dynamic MAX_MM_SEQ_LEN and disable optimizations for stability
        original_max_seq = llama_mlp.args.VISION_MAX_MM_SEQ
        llama_mlp.args.VISION_MAX_MM_SEQ = seq_len
        
        # Enable disable_mlp_optimizations for stability
        original_disable = getattr(llama_mlp.args, 'disable_mlp_optimizations', False)
        llama_mlp.args.disable_mlp_optimizations = True
        
        try:
            return llama_mlp.forward(x)
        finally:
            llama_mlp.args.VISION_MAX_MM_SEQ = original_max_seq
            llama_mlp.args.disable_mlp_optimizations = original_disable
    
    return gemma_forward


class TtGemmaImageFeedForward:    
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
    ):
        # Create the underlying llama MLP (handles all weight initialization)
        self.llama_mlp = TtLlamaImageFeedForward(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=args,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )
        
        # Create gemma-specific forward function
        self.gemma_forward = create_gemma_mlp_forward(args)
    
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.gemma_forward(self.llama_mlp, x)
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.forward(x)
