# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from transformers.configuration_utils import PretrainedConfig

from models.demos.deepseek_v3.tt.mlp_1d_dequant import MLP1DDequant


class SharedExpert(MLP1DDequant):  # The only difference with the regular Dequantized MLP is the intermediate layer size
    """Shared Expert layer for Mixture-of-Experts (MoE) models."""

    @classmethod
    def _get_model_dims_from_cfg(cls, hf_config: PretrainedConfig) -> tuple[int, int]:
        dim = hf_config.hidden_size
        hidden_dim = hf_config.moe_intermediate_size
        return dim, hidden_dim
