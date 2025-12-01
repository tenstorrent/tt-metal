# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn
from models.demos.bge_large_en.ttnn.ttnn_bge_ffn import TtnnBGEFeedForwardModel
from models.demos.bge_large_en.ttnn.ttnn_bge_self_attention import TtnnBGESelfAttention


class TtnnBGEEncoderLayer:
    """
    Single BGE encoder layer following metal_BERT_large_11 pattern exactly.
    Structure: MHA (OP1-OP6) -> Self-output (OP7) -> LayerNorm (OP8) -> FFN (OP9-OP10) -> LayerNorm (OP11)
    """

    def __init__(self, layer_params, config, layer_idx):
        self.config = config
        self.layer_idx = layer_idx
        self.device = None  # Will be set during forward pass

        # MHA sub-graph (OP1-OP6)
        self.mha = TtnnBGESelfAttention(layer_params.attention.self, config)

        # Self-output weights and bias (OP7)
        self.attention_output_weight = layer_params.attention.output.dense.weight
        self.attention_output_bias = layer_params.attention.output.dense.bias

        # MHA LayerNorm weights (OP8)
        self.mha_gamma = layer_params.attention.output.LayerNorm.weight
        self.mha_beta = layer_params.attention.output.LayerNorm.bias

        # FFN sub-graph (OP9-OP10) - following metal_BERT pattern
        self.ffn = TtnnBGEFeedForwardModel(layer_params)

        # FFN LayerNorm weights (OP11)
        self.ffn_gamma = layer_params.output.LayerNorm.weight
        self.ffn_beta = layer_params.output.LayerNorm.bias

        self.layer_norm_eps = config.layer_norm_eps

    def op7_self_output(self, mha_res):
        """
        OP7: Self-output linear transformation (equivalent to metal_BERT OP7)
        """
        # Calculate per_core_M dynamically
        *batch_sizes, height, width = mha_res.shape
        batch_size = 1
        for bs in batch_sizes:
            batch_size *= bs
        M_tiles = (batch_size * height) // 32
        core_grid_y = 8
        per_core_M = M_tiles // core_grid_y
        if per_core_M == 0:
            per_core_M = 1

        dynamic_self_out_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            out_subblock_h=2,
            out_subblock_w=4,
            per_core_M=per_core_M,
            per_core_N=4,
            transpose_mcast=False,
            fused_activation=None,
        )

        mha_out = ttnn.linear(
            mha_res,
            self.attention_output_weight,
            bias=self.attention_output_bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=dynamic_self_out_program_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
            ),
            dtype=ttnn.bfloat8_b,
        )
        return mha_out

    def op8_add_layernorm(self, activation, mha_out):
        """
        OP8: Add residual and apply LayerNorm after MHA (equivalent to metal_BERT OP8)
        """
        # Calculate block_h dynamically
        *batch_sizes, height, width = mha_out.shape
        batch_size = 1
        for bs in batch_sizes:
            batch_size *= bs
        M_tiles = (batch_size * height) // 32
        core_grid_y = 8
        block_h = M_tiles // core_grid_y
        if block_h == 0:
            block_h = 1

        dynamic_layernorm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            subblock_w=4,
            block_h=block_h,
            block_w=4,
            inplace=True,
            legacy_reduction=True,
            legacy_rsqrt=True,
        )

        # Reshard activation to match mha_out memory config
        activation = ttnn.reshard(activation, mha_out.memory_config())

        mha_out_add_and_norm = ttnn.layer_norm(
            activation,
            residual_input_tensor=mha_out,
            epsilon=self.layer_norm_eps,
            weight=self.mha_gamma,
            bias=self.mha_beta,
            program_config=dynamic_layernorm_program_config,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
            ),
        )
        return mha_out_add_and_norm

    def op11_add_layernorm(self, mha_out_add_and_norm, ffn_out):
        """
        OP11: Add residual and apply LayerNorm after FFN (equivalent to metal_BERT OP11)
        """
        # Calculate block_h dynamically
        *batch_sizes, height, width = ffn_out.shape
        batch_size = 1
        for bs in batch_sizes:
            batch_size *= bs
        M_tiles = (batch_size * height) // 32
        core_grid_y = 8
        block_h = M_tiles // core_grid_y
        if block_h == 0:
            block_h = 1

        dynamic_layernorm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            subblock_w=4,
            block_h=block_h,
            block_w=4,
            inplace=True,
            legacy_reduction=True,
            legacy_rsqrt=True,
        )

        # Convert ffn_out from DRAM to sharded for layer norm
        # FFN output is in DRAM, need to convert to sharded
        ffn_out_sharded = ttnn.to_memory_config(ffn_out, mha_out_add_and_norm.memory_config())

        # Reshard mha_out_add_and_norm to match ffn_out_sharded memory config
        mha_out_add_and_norm = ttnn.reshard(mha_out_add_and_norm, ffn_out_sharded.memory_config())

        ffn_out_add_and_norm = ttnn.layer_norm(
            mha_out_add_and_norm,
            residual_input_tensor=ffn_out_sharded,
            epsilon=self.layer_norm_eps,
            weight=self.ffn_gamma,
            bias=self.ffn_beta,
            program_config=dynamic_layernorm_program_config,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
            ),
        )
        return ffn_out_add_and_norm

    def __call__(
        self, activation: ttnn.Tensor, attention_mask: Optional[ttnn.Tensor] = None, device=None
    ) -> ttnn.Tensor:
        """
        Forward pass following metal_BERT_large_11 pattern exactly:
        MHA (OP1-OP6) -> Self-output (OP7) -> LayerNorm (OP8) -> FFN (OP9-OP10) -> LayerNorm (OP11)
        """
        # MHA - OP1 - OP6
        mha_res = self.mha(activation, attention_mask, device=device)
        # Don't deallocate activation here since it is used by more ops

        # Self-output - OP7
        mha_out = self.op7_self_output(mha_res)
        mha_res.deallocate()

        # LayerNorm - OP8
        mha_out_add_and_norm = self.op8_add_layernorm(activation, mha_out)
        activation.deallocate()
        mha_out.deallocate()

        # FFN - OP9 - OP10
        # FFN handles OP9 and OP10 internally, returns FF2 output in DRAM
        ffn_out = self.ffn(mha_out_add_and_norm)

        # LayerNorm - OP11
        ffn_out_add_and_norm = self.op11_add_layernorm(mha_out_add_and_norm, ffn_out)
        mha_out_add_and_norm.deallocate()
        ffn_out.deallocate()  # Deallocate DRAM tensor after conversion (matches metal_BERT pattern)

        return ffn_out_add_and_norm


class TtnnBGEEncoder:
    """
    BGE Encoder following metal_BERT_large_11 pattern exactly.
    Structure matches metal_BERT: MHA -> OP7 -> OP8 -> FFN -> OP11
    """

    def __init__(self, parameters, config):
        self.config = config
        self.layers = {}

        for i in range(config.num_hidden_layers):
            layer_params = parameters.layer[i]
            self.layers[i] = TtnnBGEEncoderLayer(layer_params, config, i)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        device=None,
    ):
        """
        Forward pass through all encoder layers.
        Pattern matches metal_BERT_large_11 exactly.
        """
        # Convert interleaved hidden_states to sharded if needed
        if not hidden_states.is_sharded():
            try:
                *batch_sizes, height, width = hidden_states.shape
                batch_size = 1
                for bs in batch_sizes:
                    batch_size *= bs
                core_grid = device.compute_with_storage_grid_size() if device else ttnn.CoreGrid(y=8, x=8)

                hidden_states = ttnn.to_memory_config(
                    hidden_states,
                    memory_config=ttnn.create_sharded_memory_config(
                        hidden_states.shape,
                        core_grid=core_grid,
                        strategy=ttnn.ShardStrategy.BLOCK,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )
            except RuntimeError as e:
                raise RuntimeError(f"Failed to convert interleaved hidden_states to sharded for encoder: {e}") from e

        # Convert attention_mask to interleaved if needed
        if attention_mask.is_sharded():
            attention_mask_interleaved = ttnn.sharded_to_interleaved(attention_mask, ttnn.L1_MEMORY_CONFIG)
            attention_mask_interleaved = ttnn.to_layout(attention_mask_interleaved, ttnn.TILE_LAYOUT)
            ttnn.deallocate(attention_mask)
        else:
            attention_mask_interleaved = attention_mask

        # Process through all encoder layers
        for i in range(len(self.layers)):
            hidden_states = self.layers[i](hidden_states, attention_mask_interleaved, device=device)

        return hidden_states
