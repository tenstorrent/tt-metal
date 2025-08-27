# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from ...utils.tensor import bf16_tensor
from ...utils.substate import substate, indexed_substates
from ...parallel.manager import CCLManager
from ...parallel.config import EncoderParallelConfig
from ...layers.linear import ColParallelLinear, RowParallelLinear
import math
from ...layers.normalization import RMSNorm


class T5Config:
    """
    Configuration class to store the configuration of a `T5Encoder` model.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    T5 model used in the Stable Diffusion 3.5 Large model.

    Args:
        vocab_size (`int`, *required*, defaults to 32128)
            The size of the vocabulary
        embed_dim (`int`, *required*, defaults to 4096)
            The dimension of the embeddings
        ff_dim (`int`, *required*, defaults to 10240)
            The dimension of the feedforward layer
        kv_dim (`int`, *required*, defaults to 64)
            The dimension of the key and value vectors
        num_heads (`int`, *required*, defaults to 64)
            The number of attention heads
        num_hidden_layers (`int`, *required*, defaults to 24)
            The number of hidden layers
        max_prompt_length (`int`, *required*, defaults to 256)
            The maximum length of the prompt
        layer_norm_eps (`float`, *required*, defaults to 1e-06)
            The epsilon value for the layer normalization
        relative_attention_num_buckets (`int`, *required*, defaults to 32)
            The number of relative attention buckets
        relative_attention_max_distance (`int`, *required*, defaults to 128)
            The maximum distance for relative attention
    """

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

        output = self.final_layer_norm(hidden_states[-1])
        hidden_states.append(output)
        return hidden_states


class T5Stack:
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
        layer_states = indexed_substates(state_dict, "block")

        for idx, (layer, layer_state) in enumerate(zip(self.layers, layer_states)):
            layer.load_state_dict(layer_state)

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
        normalized_hidden_states = self.layer_norm(hidden_states)
        gated_hidden_states = self.dense_gated_dense(normalized_hidden_states)
        return gated_hidden_states


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
        self.wo = RowParallelLinear(
            in_features=self.config.ff_dim,
            out_features=self.config.embed_dim,
            bias=False,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=self.ccl_manager,
            init=True,
        )

    def load_state_dict(self, state_dict):
        self.wi0.load_state_dict({"weight": state_dict["wi_0.weight"]})
        self.wi1.load_state_dict({"weight": state_dict["wi_1.weight"]})
        self.wo.load_state_dict({"weight": state_dict["wo.weight"]})

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # breakpoint()
        gelu = new_gelu_activation(self.wi0(x))
        linear = self.wi1(x)
        x = gelu * linear
        hidden_states = self.wo(x)
        hidden_states = ttnn.unsqueeze(hidden_states, 0)

        if self.parallel_config.tensor_parallel.factor > 1:
            hidden_states = ttnn.experimental.all_gather_async(
                hidden_states,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    hidden_states.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        hidden_states = ttnn.squeeze(hidden_states, 0)
        return hidden_states


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

        hidden_states_residual1 = attn_output + hidden_states

        hidden_states_ff = self.ff(
            hidden_states_residual1, ccl_manager=self.ccl_manager, parallel_config=self.parallel_config
        )
        return hidden_states_ff + hidden_states_residual1


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

        self.q_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=False,
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

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        qkv = ttnn.concat([q, k, v], dim=-1)

        num_devices = self.parallel_config.tensor_parallel.factor
        num_local_heads = self.num_heads // num_devices

        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=num_local_heads, transpose_key=True
        )

        scores = ttnn.matmul(q, k)
        scores = scores + position_bias
        attn_weights = ttnn.softmax(scores, dim=-1)
        attn_output = ttnn.matmul(attn_weights, v)
        attn_output = ttnn.transformer.concatenate_heads(attn_output)

        if self.parallel_config.tensor_parallel.factor > 1:
            attn_output = ttnn.unsqueeze(attn_output, 0)  # unsqueeze for all gather
            orig_shape = list(attn_output.shape)

            attn_output = ttnn.experimental.all_gather_async(
                attn_output,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    attn_output.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )

        dense_out = self.o_proj(attn_output)

        if self.parallel_config.tensor_parallel.factor > 1:
            dense_out = ttnn.experimental.all_gather_async(
                dense_out,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    dense_out.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )

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
    Implements text token embeddings with relative positional encoding

    Two main components:
    1. Token Embeddings: Converts input tokens into dense vectors
    2. Relative Position Bias: Computes relative positional encodings between tokens

    Args:
        config (`T5Config`)
            Configuration object containing model parameters
        mesh_device (`ttnn.Device`)
            Device for tensor placement and computation
        ccl_manager (`CCLManager`)
            Manager for collective communication operations
        parallel_config (`EncoderParallelConfig`)
            Configuration for parallel processing
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

    def load_state_dict(self, state_dict):
        self.token_embedding_weights = bf16_tensor(
            state_dict["encoder.embed_tokens.weight"], device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.relative_attention_bias_weights = bf16_tensor(
            state_dict["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
        )

    def __call__(self, prompt: ttnn.Tensor, device: ttnn.Device) -> ttnn.Tensor:
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

    shard_dims = [None, None]
    shard_dims[parallel_config.tensor_parallel.mesh_axis] = -3
    return ttnn.from_torch(
        output,
        device=device,
        dtype=relative_attention_bias.get_dtype(),
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=shard_dims),
    )
