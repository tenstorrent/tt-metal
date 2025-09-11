# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from ...utils.tensor import bf16_tensor
from ...utils.substate import substate, indexed_substates
from ...parallel.manager import CCLManager
from ...parallel.config import EncoderParallelConfig
from ...layers.feedforward import ColParallelLinear, ParallelFeedForward


class CLIPConfig:
    """
    Configuration class to store the configuration of a `CLIPEncoder` model.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    CLIP text encoder model used in Stable Diffusion 3.5

    Args:
        vocab_size (`int`, *required*, defaults to 49408)
            The size of the token vocabulary
        embed_dim (`int`, *required*, defaults to 768)
            The dimension of the embeddings and hidden states
        ff_dim (`int`, *required*, defaults to 3072)
            The dimension of the feedforward layer
        num_heads (`int`, *required*, defaults to 12)
            The number of attention heads
        num_hidden_layers (`int`, *required*, defaults to 12)
            The number of transformer layers
        max_prompt_length (`int`, *required*, defaults to 77)
            The maximum length of the prompt sequence
        layer_norm_eps (`float`, *required*, defaults to 1e-5)
            The epsilon value for layer normalization
        attention_dropout (`float`, *required*, defaults to 0.0)
            The dropout probability for attention weights
        hidden_act (`str`, *required*, defaults to "quick_gelu")
            The activation function used in the feedforward layers
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_hidden_layers: int = 12,
        max_prompt_length=77,
        layer_norm_eps: float = 1e-05,
        attention_dropout: float = 0.0,
        hidden_act: str = "quick_gelu",
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_prompt_length = max_prompt_length
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act


class CLIPEncoder:
    def __init__(
        self,
        config: CLIPConfig,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        eos_token_id: int,
    ) -> None:
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.embeddings = TextEmbeddings(config, mesh_device)
        self.eos_token_id = eos_token_id
        self.encoder = CLIPStack(config, self.mesh_device, self.ccl_manager, self.parallel_config)

    def load_state_dict(self, state_dict):
        self.embeddings.load_state_dict(substate(state_dict, "text_model.embeddings"))
        self.encoder.load_state_dict(substate(state_dict, "text_model.encoder"))

        self.final_layer_norm = bf16_tensor(
            state_dict["text_model.final_layer_norm.weight"], device=self.mesh_device, layout=ttnn.TILE_LAYOUT
        )
        self.final_layer_norm_bias = bf16_tensor(
            state_dict["text_model.final_layer_norm.bias"], device=self.mesh_device, layout=ttnn.TILE_LAYOUT
        )
        if "text_projection.weight" in state_dict:
            self.text_projection = bf16_tensor(
                state_dict["text_projection.weight"], device=self.mesh_device, layout=ttnn.TILE_LAYOUT
            )

    def __call__(
        self, prompt_tokenized: ttnn.Tensor, mesh_device: ttnn.Device, with_projection: bool = True
    ) -> ttnn.Tensor:
        hidden_states = self.embeddings(prompt_tokenized, mesh_device)

        causal_attention_mask = create_4d_causal_attention_mask(
            prompt_tokenized.shape, mesh_device, dtype=hidden_states.get_dtype()
        )

        encoder_output = self.encoder(
            hidden_states,
            causal_attention_mask,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
        )
        final_hidden_layer = encoder_output[-1]  # final hidden layer
        normalized_final_state = ttnn.layer_norm(  # final layer norm
            final_hidden_layer,
            weight=self.final_layer_norm,
            bias=self.final_layer_norm_bias,
            epsilon=self.config.layer_norm_eps,
        )

        encoder_output.append(normalized_final_state)

        # gather eos
        if self.eos_token_id is None:
            self.eos_token_id = 2

        pooled_output = _gather_eos(normalized_final_state, prompt_tokenized, self.eos_token_id, mesh_device)

        # apply text projection if specified
        if with_projection:
            if self.text_projection is None:
                raise ValueError("projection weights are not loaded")
            text_projection_transposed = ttnn.transpose(self.text_projection, -2, -1)
            projected_output = ttnn.matmul(pooled_output, text_projection_transposed)
            # sequence embedding, pooled embedding with projection
            return encoder_output, projected_output
        else:
            # sequence embedding, pooled embedding without projection
            return encoder_output, pooled_output


def _gather_eos(seq_emb: ttnn.Tensor, input_ids: ttnn.Tensor, eos_token_id: int, device: ttnn.Device) -> ttnn.Tensor:
    ids_t = ttnn.to_torch(ttnn.get_device_tensors(input_ids)[0])

    # from HF: if self.eos_token_id == 2: use argmax, else: search for eos_token_id
    if eos_token_id == 2:
        # use argmax (highest token ID position)
        eos_idx = ids_t.to(dtype=torch.int, device=ids_t.device).argmax(dim=-1)
    else:
        # search for specific eos_token_id
        eos_mask = (ids_t.to(dtype=torch.int, device=ids_t.device) == eos_token_id).int()
        eos_idx = eos_mask.argmax(dim=-1)

    seq_t = ttnn.to_torch(ttnn.get_device_tensors(seq_emb)[0])  # [B, S, H]
    b = torch.arange(seq_t.size(0))
    pooled_t = seq_t[b, eos_idx]  # [B, H]

    return ttnn.from_torch(
        pooled_t,
        dtype=seq_emb.get_dtype(),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


class CLIPStack:
    def __init__(
        self,
        config: CLIPConfig,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        self.config = config
        self.layers = [
            CLIPEncoderLayer(config, mesh_device, ccl_manager, parallel_config) for _ in range(config.num_hidden_layers)
        ]

    def load_state_dict(self, state_dict):
        layer_states = indexed_substates(state_dict, "layers")
        for layer, layer_state in zip(self.layers, layer_states):
            layer.load_state_dict(layer_state)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        causal_attention_mask: ttnn.Tensor,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        output_hidden_states: bool = True,
    ) -> ttnn.Tensor:
        all_hidden_states = []  # list of hidden states from each layer
        all_hidden_states.append(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_attention_mask, ccl_manager, parallel_config)
            all_hidden_states.append(hidden_states)

        return all_hidden_states  # list of hidden states from each layer


class CLIPEncoderLayer:
    def __init__(
        self,
        config: CLIPConfig,
        mesh_device: ttnn.Device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        self.config = config
        self.mesh_device = mesh_device
        self.layer_norm1 = None
        self.layer_norm2 = None
        self.layer_norm_eps = config.layer_norm_eps
        self.self_attn = CLIPAttention(config, mesh_device, ccl_manager, parallel_config)
        self.mlp = ParallelFeedForward(
            dim=config.embed_dim,
            dim_out=config.embed_dim,
            activation_fn=config.hidden_act,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

    def load_state_dict(self, state_dict):
        self.layer_norm1 = bf16_tensor(
            state_dict["layer_norm1.weight"], device=self.mesh_device, layout=ttnn.TILE_LAYOUT
        )
        self.layer_norm1_bias = bf16_tensor(
            state_dict["layer_norm1.bias"], device=self.mesh_device, layout=ttnn.TILE_LAYOUT
        )
        self.layer_norm2 = bf16_tensor(
            state_dict["layer_norm2.weight"], device=self.mesh_device, layout=ttnn.TILE_LAYOUT
        )
        self.layer_norm2_bias = bf16_tensor(
            state_dict["layer_norm2.bias"], device=self.mesh_device, layout=ttnn.TILE_LAYOUT
        )

        self.self_attn.load_state_dict(substate(state_dict, "self_attn"))

        # remap MLP keys from fc1/fc2 to ff1/ff2 format
        mlp_state = substate(state_dict, "mlp")
        remapped_mlp_state = {
            "ff1.weight": mlp_state["fc1.weight"],
            "ff1.bias": mlp_state["fc1.bias"],
            "ff2.weight": mlp_state["fc2.weight"],
            "ff2.bias": mlp_state["fc2.bias"],
        }
        self.mlp.load_state_dict(remapped_mlp_state)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        causal_attention_mask: ttnn.Tensor,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> ttnn.Tensor:
        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states, weight=self.layer_norm1, bias=self.layer_norm1_bias, epsilon=self.layer_norm_eps
        )
        attn_output = self.self_attn(hidden_states, causal_attention_mask)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states, weight=self.layer_norm2, bias=self.layer_norm2_bias, epsilon=self.layer_norm_eps
        )
        mlp_output_fractured = self.mlp(hidden_states)  # fractured on columns
        hidden_states_shape = list(mlp_output_fractured.shape)

        mlp_output_fractured = ttnn.unsqueeze(mlp_output_fractured, 0)

        if self.parallel_config.tensor_parallel.factor > 1:
            mlp_output = ttnn.experimental.all_gather_async(
                mlp_output_fractured,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    mlp_output_fractured.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
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
        else:
            mlp_output = mlp_output_fractured

        mlp_output = ttnn.squeeze(mlp_output, 0)

        hidden_states = residual + mlp_output

        return hidden_states


class CLIPAttention:
    def __init__(
        self,
        config: CLIPConfig,
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
        self.scale = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=True,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            init=True,
        )
        self.k_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=True,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            init=True,
        )
        self.v_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=True,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            init=True,
        )
        self.o_proj = ColParallelLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=True,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            init=True,
        )

    def load_state_dict(self, state_dict):
        self.q_proj.load_state_dict(substate(state_dict, "q_proj"))
        self.k_proj.load_state_dict(substate(state_dict, "k_proj"))
        self.v_proj.load_state_dict(substate(state_dict, "v_proj"))
        self.o_proj.load_state_dict(substate(state_dict, "out_proj"))

    def __call__(self, hidden_states, causal_attention_mask):
        batch_size, seq_length, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q * self.scale

        # reshape for multihead attention
        num_devices = self.parallel_config.tensor_parallel.factor
        num_local_heads = self.num_heads // num_devices

        q = ttnn.reshape(q, (batch_size, seq_length, num_local_heads, self.head_dim))
        k = ttnn.reshape(k, (batch_size, seq_length, num_local_heads, self.head_dim))
        v = ttnn.reshape(v, (batch_size, seq_length, num_local_heads, self.head_dim))

        q = ttnn.transpose(q, 1, 2)
        k = ttnn.transpose(k, 1, 2)
        v = ttnn.transpose(v, 1, 2)

        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))  # [batch_size, num_heads, seq_length, seq_length]

        if causal_attention_mask is not None:
            scores = scores + causal_attention_mask

        attn_weights = ttnn.softmax(scores, dim=-1)

        attn_output = ttnn.matmul(attn_weights, v)

        attn_output = ttnn.transpose(attn_output, 1, 2)  # [batch_size, seq_length, num_heads, head_dim]
        attn_output = ttnn.reshape(attn_output, (1, batch_size, seq_length, self.embed_dim // num_devices))

        orig_shape = list(attn_output.shape)

        if self.parallel_config.tensor_parallel.factor > 1:
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


class TextEmbeddings:
    """
    Implements text token embeddings with absolute positional encoding

    Args:
        config (`CLIPConfig`)
            Configuration object containing model parameters
        mesh_device (`ttnn.Device`)
            Device for tensor placement and computation
        token_embedding (`ttnn.Tensor`)
            Token embeddings
        position_embedding (`ttnn.Tensor`)
            Position embeddings
    """

    def __init__(self, config, mesh_device: ttnn.Device) -> None:
        self.config = config
        self.mesh_device = mesh_device

        self.token_embedding = None
        self.position_embedding = None

    def load_state_dict(self, state_dict):
        self.token_embedding = bf16_tensor(
            state_dict["token_embedding.weight"], device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.position_embedding = bf16_tensor(
            state_dict["position_embedding.weight"], device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    def __call__(self, prompt: ttnn.Tensor, device: ttnn.Device) -> ttnn.Tensor:
        seq_len = prompt.shape[-1]

        if seq_len > self.config.max_prompt_length:
            prompt = prompt[:, : self.config.max_prompt_length]
            seq_len = self.config.max_prompt_length

        input_embeddings = ttnn.embedding(prompt, self.token_embedding, layout=ttnn.TILE_LAYOUT)

        position_ids = torch.arange(seq_len).expand((1, -1))  # shape: (1, seq_len)
        position_ids_ttnn = ttnn.from_torch(position_ids, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
        position_embeddings = ttnn.embedding(position_ids_ttnn, self.position_embedding, layout=ttnn.TILE_LAYOUT)

        return input_embeddings + position_embeddings


# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
def create_4d_causal_attention_mask(
    input_shape: tuple[int, int], device: ttnn.Device, dtype: ttnn.DataType
) -> ttnn.Tensor:
    """Create a 4D causal attention mask for the given input shape."""
    batch_size, tgt_len = input_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len)
    return ttnn.from_torch(mask, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
