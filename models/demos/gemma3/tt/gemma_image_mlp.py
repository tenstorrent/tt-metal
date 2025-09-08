# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field

import ttnn
from models.tt_transformers.tt.multimodal.llama_image_mlp import TtLlamaImageFeedForward


class GemmaMLPConfig(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True,
    }
    num_devices: int
    vision_dim: int
    vision_mlp_ratio: float = 4.0
    compute_kernel_config_hifi2: ttnn.WormholeComputeKernelConfig = Field(
        default_factory=lambda: ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    )
    compute_kernel_config_hifi4: ttnn.WormholeComputeKernelConfig = Field(
        default_factory=lambda: ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    )
    dummy_weights: bool = False
    # Gemma uses dynamic MAX_MM_SEQ_LEN - this will be overridden at runtime
    VISION_MAX_MM_SEQ: int = 32

    @property
    def vision_mlp_dim(self):
        return int(self.vision_dim * self.vision_mlp_ratio)

    def get_model_config(self):
        # Gemma-specific: Return None for program configs to disable optimizations
        # This makes llama_image_mlp use core_grid=ttnn.CoreGrid(y=8, x=8) and program_config=None
        return {
            "IMAGE_MLP_FC_PROGCFG": lambda seq_len, max_seq: None,
            "IMAGE_MLP_PROJ_PROGCFG": lambda seq_len, max_seq: None,
        }


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
        # For backward compatibility: if args is not a GemmaMLPConfig, create one
        if not isinstance(args, GemmaMLPConfig):
            # Extract necessary parameters from the args object
            self.mlp_config = GemmaMLPConfig(
                num_devices=mesh_device.get_num_devices() if mesh_device else 0,
                vision_dim=getattr(args, "vision_dim", 1152),  # Default vision dim
                vision_mlp_ratio=getattr(args, "vision_mlp_ratio", 4.0),  # Default ratio
                dummy_weights=(weight_cache_path is None),
            )
        else:
            self.mlp_config = args

        # Create the underlying llama MLP - just like siglip does
        self.llama_mlp = TtLlamaImageFeedForward(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=self.mlp_config,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        seq_len = x.shape[-2]

        # Gemma-3 specific: use dynamic MAX_MM_SEQ_LEN
        # Temporarily set VISION_MAX_MM_SEQ to current sequence length
        original_max_seq = self.llama_mlp.args.VISION_MAX_MM_SEQ
        self.llama_mlp.args.VISION_MAX_MM_SEQ = seq_len

        try:
            return self.llama_mlp.forward(x)
        finally:
            self.llama_mlp.args.VISION_MAX_MM_SEQ = original_max_seq

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.forward(x)
