# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from ...layers.linear import ColParallelLinear
from ...parallel.config import EncoderParallelManager
from ...utils.substate import substate


# TODO: merge attention param and instance classes


class EncoderAttention:
    def __init__(self, config, mesh_device=None, ccl_manager=None, parallel_config=None):
        self.config = config
        # Store references to managers and device
        self.ccl_manager = ccl_manager
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config

        self.num_heads = config.num_attention_heads
        self.embed_dim = config.hidden_size
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.tensor_parallel_factor = self.parallel_config.tensor_parallel.factor
        logger.debug(f"tensor_parallel_factor: {self.tensor_parallel_factor}")
        self.num_local_heads = self.num_heads // self.tensor_parallel_factor

        logger.debug(
            f"INIT DEBUG: num_heads={self.num_heads}, tp_factor={self.tensor_parallel_factor}, num_local_heads={self.num_local_heads}"
        )

        self.attention_dropout = config.attention_dropout

        self.q_proj = ColParallelLinear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=True,
            mesh_device=self.mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config else 0,
        )
        self.k_proj = ColParallelLinear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=True,
            mesh_device=self.mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config else 0,
        )
        self.v_proj = ColParallelLinear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=True,
            mesh_device=self.mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config else 0,
        )

        self.out_proj = ColParallelLinear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=True,
            mesh_device=self.mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config else 0,
        )

    def load_state_dict(self, state_dict):
        self.q_proj.load_state_dict(substate(state_dict, "q_proj"))
        self.k_proj.load_state_dict(substate(state_dict, "k_proj"))
        self.v_proj.load_state_dict(substate(state_dict, "v_proj"))
        self.out_proj.load_state_dict(substate(state_dict, "out_proj"))

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        causal_attention_mask: ttnn.Tensor = None,
        parallel_manager: EncoderParallelManager = None,
        ccl_manager=None,
        parallel_config=None,
    ) -> ttnn.Tensor:
        """
        input is replicated
        q, k, v are head parallel
        SDPA executes head-parallel
        output is replicated
        """

        parallel_manager = ccl_manager

        orig_shape = hidden_states.shape
        batch_size, seq_length, _ = hidden_states.shape
        logger.debug(f"Input shape: {hidden_states.shape}")

        q = self.q_proj(hidden_states)  # (batch, seq_len, local_hidden_size)
        logger.debug(f"Q projection done, shape: {q.shape}")

        k = self.k_proj(hidden_states)  # (batch, seq_len, local_hidden_size)
        logger.debug(f"K projection done, shape: {k.shape}")

        v = self.v_proj(hidden_states)  # (batch, seq_len, local_hidden_size)
        logger.debug(f"V projection done, shape: {v.shape}")

        q = q * self.scale

        # debug
        logger.debug(f"DEBUG attention shapes:")
        logger.debug(f"  hidden_states.shape: {hidden_states.shape}")
        logger.debug(f"  self.num_heads: {self.num_heads}")
        logger.debug(f"  self.head_dim: {self.head_dim}")
        logger.debug(f"  self.tensor_parallel_factor: {self.tensor_parallel_factor}")

        q = ttnn.reshape(q, (batch_size, seq_length, self.num_local_heads, self.head_dim))
        k = ttnn.reshape(k, (batch_size, seq_length, self.num_local_heads, self.head_dim))
        v = ttnn.reshape(v, (batch_size, seq_length, self.num_local_heads, self.head_dim))

        logger.debug("Reshaping Q, K, V to multihead format...")
        logger.debug(f"Q reshaped to: {q.shape}")
        logger.debug(f"K reshaped to: {k.shape}")
        logger.debug(f"V reshaped to: {v.shape}")

        # transpose to [batch_size, num_heads, seq_length, head_dim]
        q = ttnn.transpose(q, 1, 2)
        k = ttnn.transpose(k, 1, 2)
        v = ttnn.transpose(v, 1, 2)

        logger.debug("Transposing Q, K, V to [batch_size, num_heads, seq_length, head_dim]...")
        logger.debug(f"Q transposed to: {q.shape}")
        logger.debug(f"K transposed to: {k.shape}")
        logger.debug(f"V transposed to: {v.shape}")

        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
        logger.debug(f"Attention scores computed, shape: {scores.shape}")

        if causal_attention_mask is not None:
            logger.debug("Applying causal attention mask...")
            scores = scores + causal_attention_mask
        else:
            logger.debug("No attention mask provided")

        attn_weights = ttnn.softmax(scores, dim=-1)
        logger.debug(f"Softmax computed, shape: {attn_weights.shape}")

        # TODO: Add dropout when ttnn.dropout is available
        # attn_weights = ttnn.dropout(attn_weights, self.config.attention_dropout)

        attn_output = ttnn.matmul(attn_weights, v)  # (batch, num_local_heads, seq_len, head_dim)
        logger.debug(f"Attention output computed, shape: {attn_output.shape}")

        attn_output = ttnn.transpose(attn_output, 1, 2)
        logger.debug(f"Attention output transposed to: {attn_output.shape}")

        attn_output = ttnn.reshape(
            attn_output, (1, batch_size, seq_length, self.embed_dim // self.tensor_parallel_factor)
        )
        logger.debug(f"Attention output reshaped to: {attn_output.shape}")

        orig_shape = list(attn_output.shape)

        attn_output = ttnn.experimental.all_gather_async(
            input_tensor=attn_output,
            dim=len(attn_output.shape) - 1,
            cluster_axis=parallel_config.tensor_parallel.mesh_axis,  # 1d
            mesh_device=parallel_manager.mesh_device,
            topology=parallel_manager.topology,
            multi_device_global_semaphore=parallel_manager.get_ag_ping_pong_semaphore(),
            num_links=parallel_manager.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logger.debug(f"First all_gather completed, shape: {attn_output.shape}")

        dense_out = self.out_proj(attn_output)
        logger.debug(f"Output projection computed, shape: {dense_out.shape}")

        logger.debug("Starting second all_gather_async...")
        dense_out = ttnn.experimental.all_gather_async(
            input_tensor=dense_out,
            dim=len(dense_out.shape) - 1,
            cluster_axis=parallel_config.tensor_parallel.mesh_axis,  # 1d
            mesh_device=parallel_manager.mesh_device,
            topology=parallel_manager.topology,
            multi_device_global_semaphore=parallel_manager.get_rs_ping_pong_semaphore(),
            num_links=parallel_manager.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logger.debug(f"Second all_gather completed, shape: {dense_out.shape}")

        dense_out_shape = list(dense_out.shape)
        dense_out_shape[2] = orig_shape[2]
        dense_out = ttnn.reshape(dense_out, tuple(dense_out_shape), dense_out.shape)

        return ttnn.reshape(dense_out, tuple(dense_out.shape)[1:])
