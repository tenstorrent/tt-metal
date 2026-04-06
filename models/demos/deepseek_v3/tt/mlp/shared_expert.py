# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.mlp.mlp_dequant import MLPDequant
from models.demos.deepseek_v3.utils.run_config import RunDecodeConfig, RunPrefillConfig


class SharedExpert(MLPDequant):  # The only difference with the regular Dequantized MLP is the intermediate layer size
    """Shared Expert layer for Mixture-of-Experts (MoE) models."""

    @classmethod
    def _get_model_dims_from_cfg(cls, hf_config: PretrainedConfig) -> tuple[int, int]:
        dim = hf_config.hidden_size
        hidden_dim = hf_config.moe_intermediate_size
        return dim, hidden_dim

    @classmethod
    def forward_prefill(
        cls, x: ttnn.Tensor, cfg: RunPrefillConfig, handle_tensor_parallel: bool = False
    ) -> ttnn.Tensor:
        """
        SharedExpert forward for prefill mode.

        Args:
            x: Input tensor
            cfg: Configuration
            handle_tensor_parallel: If True, use the parent class method with CCLs (for standalone testing).
                                   If False, skip CCLs (when called from MoEDecoderBlock2D).
        """
        if handle_tensor_parallel:
            # For standalone testing - use parent's forward_prefill which includes CCLs
            return super().forward_prefill(x, cfg)
        else:
            # When called from MoEDecoderBlock2D - skip CCLs
            # Convert to expected memory config if needed (in case all_gather outputs different memory config)
            if "input_memory_config" in cfg and x.memory_config() != cfg["input_memory_config"]:
                x = ttnn.to_memory_config(x, cfg["input_memory_config"])
            return cls._forward_compute_only(x, cfg, mode="prefill")

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig, handle_tensor_parallel: bool = False) -> ttnn.Tensor:
        """
        SharedExpert forward for decode mode.

        Args:
            x: Input tensor
            cfg: Configuration
            handle_tensor_parallel: If True, use the parent class method with CCLs (for standalone testing).
                                   If False, skip CCLs (when called from MoEDecoderBlock2D).
        """
        if handle_tensor_parallel:
            # For standalone testing - use parent's forward_decode which includes CCLs
            return super().forward_decode(x, cfg)
        else:
            # When called from MoEDecoderBlock2D - skip CCLs
            # Convert to expected memory config if needed (MoE's all_gather outputs INTERLEAVED, we need WIDTH_SHARDED)
            if "input_memory_config" in cfg and x.memory_config() != cfg["input_memory_config"]:
                x = ttnn.to_memory_config(x, cfg["input_memory_config"])
            return cls._forward_compute_only(x, cfg, mode="decode")
