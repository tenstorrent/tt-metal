# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from ...utils.tensor import bf16_tensor
from ...utils.substate import substate, indexed_substates
from ...parallel.manager import CCLManager
from ...parallel.config import EncoderParallelConfig
from ...layers.feedforward import ColParallelLinear
import math
from ...layers.normalization import RMSNorm


# default values from sd35
class T5Config:
    def __init__(
        self,
        vocab_size: int = 32128,
        embed_dim: int = 4096,
        ff_dim: int = 10240,
        kv_dim: int = 64,
        num_heads: int = 64,
        num_hidden_layers: int = 24,
        max_prompt_length: int = 256,
        layer_norm_eps: float = 1e-06,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_prompt_length = max_prompt_length
        self.layer_norm_eps = layer_norm_eps
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance


class T5Encoder:
    def __init__(
        self,
        config: T5Config,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.token_embeddings = RelativeTextEmbeddings(config, self.mesh_device, self.ccl_manager, self.parallel_config)
        self.encoder = T5Stack(config, self.mesh_device, self.ccl_manager, self.parallel_config)
        self.final_layer_norm = RMSNorm(  # final layer norm
            embedding_dim=self.config.embed_dim,
            norm_eps=self.config.layer_norm_eps,
            bias=False,
            mesh_device=self.mesh_device,
        )

    def load_state_dict(self, state_dict):
        self.token_embeddings.load_state_dict(state_dict)
        self.encoder.load_state_dict(substate(state_dict, "encoder"))
        self.final_layer_norm.load_state_dict(substate(state_dict, "encoder.final_layer_norm"))

    def __call__(self, prompt: ttnn.Tensor, device: ttnn.Device) -> ttnn.Tensor:
        embeddings, position_bias = self.token_embeddings(prompt, device)
        hidden_states = self.encoder(embeddings, position_bias)
        # Apply final layer norm to last hidden state

        output = self.final_layer_norm(hidden_states[-1])  # Shape [batch, seq_len, embed_dim]
        hidden_states.append(output)
        return hidden_states  # Return normalized final hidden state
        # TODO: return the list of all hidden states with normalized final hidden state as last element


class T5Stack:
    """
    all encoder layers
    """

    def __init__(
        self,
        config: T5Config,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.layers = [
            T5EncoderLayer(self.config, self.mesh_device, self.ccl_manager, self.parallel_config)
            for _ in range(self.config.num_hidden_layers)
        ]

    def load_state_dict(self, state_dict):
        """
        confirm: each encoder layer's weights are replicated across all devices
        """
        # TODO: check if this is correct
        # logger.info("starting to load T5Stack state dictionary")
        layer_states = indexed_substates(state_dict, "block")
        # logger.debug(f"extracted {len(layer_states)} layer states from state dictionary")

        for idx, (layer, layer_state) in enumerate(zip(self.layers, layer_states)):
            # logger.info(f"loading layer {idx} of {len(self.layers)}")
            # logger.debug(f"layer {idx} state keys: {list(layer_state.keys())}")
            # logger.debug(f"layer {idx} state shapes: {[(k, v.shape if hasattr(v, 'shape') else len(v)) for k, v in layer_state.items()]}")
            layer.load_state_dict(layer_state)

        # logger.info("completed loading T5Stack state dictionary")

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        position_bias: ttnn.Tensor,
    ) -> ttnn.Tensor:
        all_hidden_states = []  # list of hidden states from each layer
        all_hidden_states.append(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_bias=position_bias)
            all_hidden_states.append(hidden_states)
        return all_hidden_states


class T5FF:
    def __init__(
        self,
        config: T5Config,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.layer_norm = RMSNorm(
            embedding_dim=self.config.embed_dim,
            norm_eps=self.config.layer_norm_eps,
            bias=False,
            mesh_device=self.mesh_device,
        )

        self.dense_gated_dense = T5DenseGatedActDense(config, mesh_device, ccl_manager, parallel_config)

    def load_state_dict(self, state_dict):
        self.layer_norm.load_state_dict(substate(state_dict, "layer_norm"))
        self.dense_gated_dense.load_state_dict(substate(state_dict, "DenseReluDense"))

    def __call__(
        self, hidden_states: ttnn.Tensor, ccl_manager: CCLManager, parallel_config: EncoderParallelConfig
    ) -> ttnn.Tensor:
        # breakpoint()
        normalized_hidden_states = self.layer_norm(hidden_states)  # [1, 256, 4096]
        gated_hidden_states = self.dense_gated_dense(normalized_hidden_states)
        return gated_hidden_states + hidden_states  # residual


class T5DenseGatedActDense:
    def __init__(
        self,
        config: T5Config,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.wi0 = ColParallelLinear(
            in_features=self.config.embed_dim,
            out_features=self.config.ff_dim,
            bias=False,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            init=True,
        )
        self.wi1 = ColParallelLinear(
            in_features=self.config.embed_dim,
            out_features=self.config.ff_dim,
            bias=False,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            init=True,
        )
        self.wo = ColParallelLinear(
            in_features=self.config.ff_dim,
            out_features=self.config.embed_dim,
            bias=False,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            init=True,
            shard_dim=-2,
        )

    def load_state_dict(self, state_dict):
        self.wi0.load_state_dict({"weight": state_dict["wi_0.weight"]})
        self.wi1.load_state_dict({"weight": state_dict["wi_1.weight"]})
        self.wo.load_state_dict({"weight": state_dict["wo.weight"]})

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # breakpoint()
        # self.wi1.weight.shape = Shape([4096, 2560])
        # self.wi0.weight.shape = Shape([4096, 2560])
        # self.wo.weight.shape = Shape([2560, 4096])

        gelu = new_gelu_activation(self.wi0(x))  # Shape([1, 256, 2560])
        linear = self.wi1(x)  # Shape([1, 256, 2560])
        x = gelu * linear  # Shape([1, 256, 2560])
        hidden_states = self.wo(x)

        hidden_states_shape = list(hidden_states.shape)
        hidden_states = ttnn.unsqueeze(hidden_states, 0)
        # AllReduce output

        if self.parallel_config.tensor_parallel.factor > 1:
            hidden_states_scattered = ttnn.experimental.reduce_scatter_minimal_async(
                hidden_states,
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_rs_ping_pong_semaphore(),
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )
            # all gather
            hidden_states = ttnn.experimental.all_gather_async(
                hidden_states_scattered,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    hidden_states_scattered.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        hidden_states = ttnn.reshape(hidden_states, hidden_states_shape, hidden_states.shape)
        return hidden_states  # Shape([1, 256, 4096])


def new_gelu_activation(x: ttnn.Tensor) -> ttnn.Tensor:
    c = math.sqrt(2.0 / math.pi)
    y = 0.044715 * ttnn.pow(x, 3) + x
    return 0.5 * x * (1.0 + ttnn.tanh(c * y))


class T5EncoderLayer:
    def __init__(
        self,
        config: T5Config,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.self_attn = T5Attention(config, mesh_device, ccl_manager, parallel_config)
        self.ff = T5FF(config, mesh_device, ccl_manager, parallel_config)

    def load_state_dict(self, state_dict):
        self.self_attn.load_state_dict(substate(state_dict, "layer.0"))
        self.ff.load_state_dict(substate(state_dict, "layer.1"))

    def __call__(self, hidden_states: ttnn.Tensor, position_bias: ttnn.Tensor) -> ttnn.Tensor:
        attn_output = self.self_attn(
            hidden_states,
            position_bias=position_bias,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
        )

        hidden_states_residual1 = attn_output + hidden_states  # residual
        # breakpoint()
        hidden_states_ff = self.ff(
            hidden_states_residual1, ccl_manager=self.ccl_manager, parallel_config=self.parallel_config
        )
        return hidden_states_ff + hidden_states_residual1  # residual


class T5Attention:
    def __init__(
        self,
        config: T5Config,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // self.num_heads

        # weights to be added in load_state_dict, column sharded
        self.q_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=False,  # T5 doesn't use bias in attention
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            init=True,
        )
        self.k_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=False,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            init=True,
        )
        self.v_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=False,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            init=True,
        )
        self.o_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=False,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            init=True,
        )

        self.layer_norm = RMSNorm(
            embedding_dim=self.config.embed_dim,
            norm_eps=self.config.layer_norm_eps,
            bias=False,
            mesh_device=self.mesh_device,
        )

    # TODO: change all load_state_dict method names to load_weights
    def load_state_dict(self, state_dict):
        self.q_proj.load_state_dict(substate(state_dict, "SelfAttention.q"))
        self.k_proj.load_state_dict(substate(state_dict, "SelfAttention.k"))
        self.v_proj.load_state_dict(substate(state_dict, "SelfAttention.v"))
        self.o_proj.load_state_dict(substate(state_dict, "SelfAttention.o"))

        self.layer_norm.load_state_dict(substate(state_dict, "layer_norm"))

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        position_bias: ttnn.Tensor,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> ttnn.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        hidden_states = self.layer_norm(hidden_states)

        # Project input into Q, K, V
        q = self.q_proj(hidden_states)  # [batch_size, seq_len, embed_dim/4] # [4096, 1024]
        k = self.k_proj(hidden_states)  # [batch_size, seq_len, embed_dim/4] # [4096, 1024]
        v = self.v_proj(hidden_states)  # [batch_size, seq_len, embed_dim/4] # [4096, 1024]

        qkv = ttnn.concat([q, k, v], dim=-1)  # [batch_size, seq_len, embed_dim*(3/4)]

        num_devices = self.parallel_config.tensor_parallel.factor
        num_local_heads = self.num_heads // num_devices

        # Split and reshape for multi-head attention:
        # 1. Split qkv into q, k, v tensors
        # 2. Reshape to add head dimension [batch_size, seq_len, num_heads, head_dim=embed_dim/num_heads]
        # 3. Transpose to:
        #    q [batch_size, num_heads, seq_len, head_dim]
        #    k [batch_size, num_heads, head_dim, seq_len] (since transpose_key=True)
        #    v [batch_size, num_heads, seq_len, head_dim]
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=num_local_heads, transpose_key=True
        )

        # TODO: (idk yet) ? replace scores with scores = ttnn.matmul(q, k) / math.sqrt(self.head_dim)

        scores = ttnn.matmul(q, k)  # attention scores # [batch_size, num_heads, seq_len, seq_len]
        scores = scores + position_bias  # add position bias
        attn_weights = ttnn.softmax(scores, dim=-1)  # attention weights
        # print(attn_weights)
        attn_output = ttnn.matmul(attn_weights, v)  # attention output # [batch_size, num_heads, seq_len, head_dim]
        attn_output = ttnn.transformer.concatenate_heads(
            attn_output
        )  # concat heads # [batch_size, seq_len, num_heads*head_dim=embed_dim]

        # all gather to get attention output on all devices
        if self.parallel_config.tensor_parallel.factor > 1:
            attn_output = ttnn.unsqueeze(attn_output, 0)  # unsqueeze for all gather
            orig_shape = list(attn_output.shape)

            attn_output = ttnn.experimental.all_gather_async(
                attn_output,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    attn_output.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )

        # column sharded. need all gather
        dense_out = self.o_proj(attn_output)  # [batch_size, seq_len, embed_dim/4]

        if self.parallel_config.tensor_parallel.factor > 1:
            dense_out = ttnn.experimental.all_gather_async(
                dense_out,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    dense_out.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                # memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        # breakpoint()
        dense_out_shape = list(dense_out.shape)
        dense_out_shape[2] = orig_shape[2]
        dense_out = ttnn.reshape(dense_out, tuple(dense_out_shape), dense_out.shape)

        return ttnn.reshape(dense_out, tuple(dense_out.shape)[1:])


def _relative_position_bucket(relative_position: torch.Tensor, num_buckets: int, max_distance: int) -> torch.Tensor:
    num_buckets //= 2

    relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
    relative_position = torch.abs(relative_position)

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


class RelativeTextEmbeddings:
    """
    Embeds text tokens and adds *relative* position embeddings.
    """

    def __init__(
        self,
        config: T5Config,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        self.config = config
        self.mesh_device = mesh_device
        self.token_embedding_weights = None
        self.relative_attention_bias_weights = None
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

    # TODO: make sure state_dict keys are correct
    def load_state_dict(self, state_dict):
        # weights are replicated across all devices
        self.token_embedding_weights = bf16_tensor(
            state_dict["encoder.embed_tokens.weight"], device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.relative_attention_bias_weights = bf16_tensor(
            state_dict["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
        )

    def __call__(self, prompt: ttnn.Tensor, device: ttnn.Device) -> ttnn.Tensor:
        # breakpoint()
        input_embeddings = ttnn.embedding(prompt, self.token_embedding_weights, layout=ttnn.TILE_LAYOUT)
        position_bias = _compute_relative_position_bias(
            seq_length=prompt.shape[-1],
            device=device,
            relative_attention_num_buckets=self.config.relative_attention_num_buckets,
            relative_attention_max_distance=self.config.relative_attention_max_distance,
            relative_attention_bias=self.relative_attention_bias_weights,
            parallel_config=self.parallel_config,
        )

        return input_embeddings, position_bias


def _compute_relative_position_bias(
    seq_length: int,
    device: ttnn.Device,
    relative_attention_num_buckets: int,
    relative_attention_max_distance: int,
    relative_attention_bias: ttnn.Tensor,
    parallel_config: EncoderParallelConfig,
) -> ttnn.Tensor:
    context_position = torch.arange(seq_length)[:, None]
    memory_position = torch.arange(seq_length)[None, :]
    relative_position = memory_position - context_position

    relative_position_bucket = _relative_position_bucket(
        relative_position,
        num_buckets=relative_attention_num_buckets,
        max_distance=relative_attention_max_distance,
    )

    relative_attention_bias = ttnn.get_device_tensors(relative_attention_bias)[0]
    torch_relative_attention_bias = ttnn.to_torch(relative_attention_bias)
    output = torch.nn.functional.embedding(relative_position_bucket, torch_relative_attention_bias)
    output = output.permute([2, 0, 1]).unsqueeze(0)
    output = output[:, :, -seq_length:, :]
    # Shard outputs on dim=-3, heads
    shard_dims = [None, None]
    shard_dims[parallel_config.tensor_parallel.mesh_axis] = -3
    return ttnn.from_torch(
        output,
        device=device,
        dtype=relative_attention_bias.get_dtype(),
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=shard_dims),
    )


# t5 stack:
# 1. transformer block
# (one transformer layer)
# 1.1 self attention  (uses position bias / relative attention)
# 1.2 feedforward
# 2. final layer norm
