# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn

from ...layers.linear import ColParallelLinear, RowParallelLinear
from ...layers.module import Module, ModuleList, Parameter
from ...layers.normalization import RMSNorm
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils.substate import pop_substate, rename_substate


# Make this a dataclass. Also consider using HF config directly.
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
        self.use_relative_position_bias = [True] + [False] * (
            num_hidden_layers - 1
        )  # use bias only for the first layer for original T5


class T5Encoder(Module):
    def __init__(
        self,
        config: T5Config,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.token_embeddings = TokenEmbeddings(config, self.mesh_device)
        self.encoder = T5Stack(config, self.mesh_device, self.ccl_manager, self.parallel_config)
        self.final_layer_norm = T5RMSNorm(  # final layer norm
            embedding_dim=self.config.embed_dim,
            norm_eps=self.config.layer_norm_eps,
            bias=False,
            mesh_device=self.mesh_device,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "encoder.embed_tokens", "token_embeddings")
        rename_substate(state, "encoder.final_layer_norm", "final_layer_norm")
        pop_substate(state, "shared")

    def forward(self, prompt: ttnn.Tensor, *, attention_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        embeddings = self.token_embeddings(prompt)
        hidden_states = self.encoder(embeddings, attention_mask=attention_mask)
        output = self.final_layer_norm(hidden_states[-1])
        hidden_states.append(output)
        return hidden_states


class T5Stack(Module):
    def __init__(
        self,
        config: T5Config,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.layers = ModuleList(
            T5EncoderLayer(
                self.config,
                self.mesh_device,
                self.ccl_manager,
                self.parallel_config,
                self.config.use_relative_position_bias[i],
            )
            for i in range(self.config.num_hidden_layers)
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "block", "layers")

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
    ) -> ttnn.Tensor:
        all_hidden_states = []  # list of hidden states from each layer
        all_hidden_states.append(hidden_states)

        position_bias = None
        if attention_mask is not None:
            attention_mask = (attention_mask - 1.0) * float("inf")

        for layer in self.layers:
            # Precompute position bias to preserve previous behaviour.If not set for this layer, use the previous layer's position bias.
            if layer.self_attn.use_relative_position_bias:
                position_bias = layer.self_attn.relative_attention_bias(hidden_states.shape[-2])  # seq_length

                if attention_mask is not None:
                    position_bias += attention_mask

            if position_bias is None:
                raise ValueError("Position bias cannot be None. Please check if the model is configured correctly.")

            hidden_states = layer(hidden_states, position_bias=position_bias)
            all_hidden_states.append(hidden_states)
        return all_hidden_states


class T5RMSNorm(RMSNorm):
    def __init__(self, embedding_dim, norm_eps=1e-5, norm_elementwise_affine=True, bias=True, mesh_device=None):
        super().__init__(
            embedding_dim=embedding_dim,
            norm_eps=norm_eps,
            norm_elementwise_affine=norm_elementwise_affine,
            bias=bias,
            mesh_device=mesh_device,
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True, fp32_dest_acc_en=True
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return super().forward(x, compute_kernel_config=self.compute_kernel_config)

    def reference(self, x: ttnn.Tensor) -> ttnn.Tensor:
        variance = ttnn.mean(ttnn.pow(x, 2), dim=-1, keepdim=True)
        x_normed = x * ttnn.rsqrt(variance + self.norm_eps)
        return self.weight.data * x_normed


class T5FF(Module):
    def __init__(
        self,
        config: T5Config,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.layer_norm = T5RMSNorm(
            embedding_dim=self.config.embed_dim,
            norm_eps=self.config.layer_norm_eps,
            bias=False,
            mesh_device=self.mesh_device,
        )

        self.dense_gated_dense = T5DenseGatedActDense(config, mesh_device, ccl_manager, parallel_config)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "DenseReluDense", "dense_gated_dense")

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        normalized_hidden_states = self.layer_norm(hidden_states)
        gated_hidden_states = self.dense_gated_dense(normalized_hidden_states)
        return gated_hidden_states + hidden_states


class T5DenseGatedActDense(Module):
    def __init__(
        self,
        config: T5Config,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.wi0 = ColParallelLinear(
            in_features=self.config.embed_dim,
            out_features=self.config.ff_dim,
            bias=False,
            activation_fn="gelu",
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
        )
        self.wi1 = ColParallelLinear(
            in_features=self.config.embed_dim,
            out_features=self.config.ff_dim,
            bias=False,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
        )
        self.wo = RowParallelLinear(
            in_features=self.config.ff_dim,
            out_features=self.config.embed_dim,
            bias=False,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=self.ccl_manager,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "wi_0", "wi0")
        rename_substate(state, "wi_1", "wi1")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # TODO: Consider fusing the wi0 and wi1 calls with a single linear layer.
        gelu = self.wi0(x)
        linear = self.wi1(x)
        x = gelu * linear
        hidden_states = self.wo(x)
        hidden_states = ttnn.unsqueeze(hidden_states, 0)

        if self.parallel_config.tensor_parallel.factor > 1:
            hidden_states = self.ccl_manager.all_gather(
                hidden_states, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis, use_hyperparams=True
            )
        hidden_states = ttnn.squeeze(hidden_states, 0)
        return hidden_states


class T5EncoderLayer(Module):
    def __init__(
        self,
        config: T5Config,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        use_relative_position_bias: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.self_attn = T5Attention(config, mesh_device, ccl_manager, parallel_config, use_relative_position_bias)
        self.ff = T5FF(config, mesh_device, ccl_manager, parallel_config)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "layer.0", "self_attn")
        rename_substate(state, "layer.1", "ff")

    def forward(self, hidden_states: ttnn.Tensor, position_bias: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states_residual1 = self.self_attn(hidden_states, position_bias=position_bias)
        hidden_states_ff = self.ff(hidden_states_residual1)
        return hidden_states_ff


class T5Attention(Module):
    def __init__(
        self,
        config: T5Config,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        use_relative_position_bias: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // self.num_heads
        self.use_relative_position_bias = use_relative_position_bias

        self.q_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=False,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
        )
        self.k_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=False,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
        )
        self.v_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=False,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
        )
        self.o_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=False,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
        )

        self.layer_norm = T5RMSNorm(
            embedding_dim=self.config.embed_dim,
            norm_eps=self.config.layer_norm_eps,
            bias=False,
            mesh_device=self.mesh_device,
        )

        self.relative_attention_bias = (
            RelativePositionEmbeddings(self.config, self.mesh_device, self.ccl_manager, self.parallel_config)
            if self.use_relative_position_bias
            else None
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "SelfAttention.q", "q_proj")
        rename_substate(state, "SelfAttention.k", "k_proj")
        rename_substate(state, "SelfAttention.v", "v_proj")
        rename_substate(state, "SelfAttention.o", "o_proj")
        if self.use_relative_position_bias:
            rename_substate(state, "SelfAttention.relative_attention_bias", "relative_attention_bias")

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        position_bias: ttnn.Tensor,
    ) -> ttnn.Tensor:
        hidden_states_ = hidden_states
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
        attn_weights = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.layer_norm.compute_kernel_config)
        attn_output = ttnn.matmul(attn_weights, v)
        attn_output = ttnn.transformer.concatenate_heads(attn_output)

        attn_output = ttnn.unsqueeze(attn_output, 0)  # unsqueeze for all gather
        orig_shape = list(attn_output.shape)
        if self.parallel_config.tensor_parallel.factor > 1:
            attn_output = self.ccl_manager.all_gather(
                attn_output, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis, use_hyperparams=True
            )

        dense_out = self.o_proj(attn_output)

        if self.parallel_config.tensor_parallel.factor > 1:
            dense_out = self.ccl_manager.all_gather(
                dense_out, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis, use_hyperparams=True
            )

        dense_out_shape = list(dense_out.shape)
        dense_out_shape[2] = orig_shape[2]
        dense_out = ttnn.reshape(dense_out, tuple(dense_out_shape), dense_out.shape)

        return ttnn.reshape(dense_out, tuple(dense_out.shape)[1:]) + hidden_states_


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


# TODO: Replace with Embedding layer from embeddings.py
class TokenEmbeddings(Module):
    def __init__(self, config: T5Config, mesh_device: ttnn.Device):
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.weight = Parameter(
            total_shape=[config.vocab_size, config.embed_dim], layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
        )

    def forward(self, prompt: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.embedding(prompt, self.weight.data, layout=ttnn.TILE_LAYOUT)


class RelativePositionEmbeddings(Module):
    """
    Implements text token embeddings with relative positional encoding

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
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.weight = Parameter(
            total_shape=[config.relative_attention_num_buckets, config.num_heads],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axes=[None, self.parallel_config.tensor_parallel.mesh_axis],
            device=self.mesh_device,
        )

        # If we are using max sequence length. We can just precompute. This seems to be the same for all subset of the max. We can also discard it if we want to save memory.
        self.relative_bias_cache = None

    def forward(self, seq_length: int) -> ttnn.Tensor:
        """
        Get relative position bias for the given sequence length.
        Curently, we compure and cache the relative position bias with the assumption that the sequence length is the same (max seq length) for all the tokens in the prompt.
        If the sequence length is not in the same as cached, we recomopute and update the cache. We can also use slicing to adapt from max cache
        Args:
            seq_length: int
                The sequence length
        Returns:
            ttnn.Tensor
                The relative position bias
        """
        if self.relative_bias_cache is None or self.relative_bias_cache.shape[2] != seq_length:
            position_ids = _compute_relative_position_bias(
                seq_length=seq_length,
                device=self.mesh_device,
                relative_attention_num_buckets=self.config.relative_attention_num_buckets,
                relative_attention_max_distance=self.config.relative_attention_max_distance,
            )
            r = ttnn.embedding(position_ids, self.weight.data, layout=ttnn.TILE_LAYOUT)
            self.relative_bias_cache = ttnn.unsqueeze(ttnn.permute(r, (2, 0, 1)), 0)
        return self.relative_bias_cache


def _compute_relative_position_bias(
    seq_length: int,
    device: ttnn.Device,
    relative_attention_num_buckets: int,
    relative_attention_max_distance: int,
) -> ttnn.Tensor:
    context_position = torch.arange(seq_length)[:, None]
    memory_position = torch.arange(seq_length)[None, :]
    relative_position = memory_position - context_position

    relative_position_bucket = _relative_position_bucket(
        relative_position,
        num_buckets=relative_attention_num_buckets,
        max_distance=relative_attention_max_distance,
    )

    return ttnn.from_torch(
        relative_position_bucket, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
