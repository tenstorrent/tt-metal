# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from loguru import logger

from ...utils.tensor import bf16_tensor
from ...utils.substate import substate, indexed_substates
from ...parallel.manager import CCLManager
from ...parallel.config import EncoderParallelConfig
from ...layers.feedforward import ColParallelLinear, ParallelFeedForward
from ...utils.padding import PaddingConfig


class CLIPConfig:
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
        self.embeddings = TextEmbeddings(config, mesh_device)
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
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
        self.text_projection = bf16_tensor(
            state_dict["text_projection.weight"], device=self.mesh_device, layout=ttnn.TILE_LAYOUT
        )

    def __call__(
        self, prompt_tokenized: ttnn.Tensor, mesh_device: ttnn.Device, with_projection: bool = True
    ) -> torch.Tensor:
        hidden_states = self.embeddings(prompt_tokenized, mesh_device)

        causal_attention_mask = _create_4d_causal_attention_mask(
            prompt_tokenized.shape, mesh_device, dtype=hidden_states.get_dtype()
        )

        encoder_output = self.encoder(
            hidden_states,
            causal_attention_mask,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
        )

        # breakpoint()
        # final layer norm
        final_hidden_layer = encoder_output[-1]  # final hidden layer
        normalized_final_state = ttnn.layer_norm(
            final_hidden_layer,
            weight=self.final_layer_norm,
            bias=self.final_layer_norm_bias,
            epsilon=self.config.layer_norm_eps,
        )

        encoder_output[-1] = normalized_final_state

        # gather eos
        if self.eos_token_id is None:
            self.eos_token_id = 2

        pooled_output = _gather_eos(self, normalized_final_state, prompt_tokenized, self.eos_token_id, mesh_device)
        text_projection_transposed = ttnn.transpose(self.text_projection, -2, -1)
        projected_output = ttnn.matmul(pooled_output, text_projection_transposed)

        # sequence embedding, pooled embedding
        return encoder_output, projected_output


def _gather_eos(
    self, seq_emb: ttnn.Tensor, input_ids: ttnn.Tensor, eos_token_id: int, device: ttnn.Device
) -> ttnn.Tensor:
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
        """
        each encoder layer's weights are replicated across all devices
        """
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
    ) -> torch.Tensor:
        all_hidden_states = []  # list of hidden states from each layer
        all_hidden_states.append(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_attention_mask, ccl_manager, parallel_config)
            all_hidden_states.append(hidden_states)

        return all_hidden_states


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
        self.padding_config = None
        self.self_attn = EncoderLayerSelfAttention(config, mesh_device, ccl_manager, parallel_config)
        # breakpoint()
        # Configure MLP with padded dimensions for tensor parallelism compatibility
        # TODO: make 16 modular
        target_embed_dim = 16 * (config.embed_dim // config.num_heads)  # 16 * 64 = 1024
        # Calculate padding to get to tiling-compatible dimensions
        # 768 -> 1024 (32 tiles, divisible by 4 devices)
        original_head_dim = config.embed_dim // config.num_heads  # 64
        # TODO: make 16 modular
        target_heads = 16  # 16 * 64 = 1024 (vs original 12 * 64 = 768)

        self.padding_config = PaddingConfig(
            original_heads=config.num_heads,  # 12
            target_heads=target_heads,  # 16
            head_dim=original_head_dim,  # 64
            tensor_parallel_factor=parallel_config.tensor_parallel.factor,  # 4
        )
        logger.info(f"Padding config: {config.embed_dim} -> {self.padding_config.target_dim} for tensor parallelism")

        self.mlp = ParallelFeedForward(
            dim=config.embed_dim,
            dim_out=config.embed_dim,
            activation_fn=config.hidden_act,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            padding_config=self.padding_config,
        )

    def load_state_dict(self, state_dict):
        """
        weights are replicated across all devices
        """
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

        # Pad the layer norm weights and biases using padding utility
        # if self.padding_config.is_padding_needed():
        #     logger.info(f"Padding layer norm weights from {self.layer_norm1.shape} to {self.padding_config.target_dim}")

        #     # Convert to torch tensors for padding, then back to ttnn
        #     ln1_torch = ttnn.to_torch(ttnn.get_device_tensors(self.layer_norm1)[0])
        #     ln1_bias_torch = ttnn.to_torch(ttnn.get_device_tensors(self.layer_norm1_bias)[0])
        #     ln2_torch = ttnn.to_torch(ttnn.get_device_tensors(self.layer_norm2)[0])
        #     ln2_bias_torch = ttnn.to_torch(ttnn.get_device_tensors(self.layer_norm2_bias)[0])

        #     # Apply padding
        #     ln1_padded = pad_bias_tensor(ln1_torch, self.padding_config)
        #     ln1_bias_padded = pad_bias_tensor(ln1_bias_torch, self.padding_config)
        #     ln2_padded = pad_bias_tensor(ln2_torch, self.padding_config)
        #     ln2_bias_padded = pad_bias_tensor(ln2_bias_torch, self.padding_config)

        #     # Convert back to ttnn tensors
        #     self.layer_norm1 = bf16_tensor(ln1_padded, device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        #     self.layer_norm1_bias = bf16_tensor(ln1_bias_padded, device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        #     self.layer_norm2 = bf16_tensor(ln2_padded, device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        #     self.layer_norm2_bias = bf16_tensor(ln2_bias_padded, device=self.mesh_device, layout=ttnn.TILE_LAYOUT)

        self.self_attn.load_state_dict(substate(state_dict, "self_attn"))
        # TODO: Implement MLP loading when self.mlp is not None
        self.mlp.load_state_dict(substate(state_dict, "mlp"))

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        causal_attention_mask: ttnn.Tensor,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ) -> ttnn.Tensor:
        # layer_norm1 weights shape: [768]
        # layer_norm1 bias shape: [768]

        # layer_norm2 weights shape: [768]
        # layer_norm2 bias shape: [768]

        # breakpoint()

        # self attention block
        residual = hidden_states  # Shape([1, 19, 768])

        hidden_states = ttnn.layer_norm(  # Now: [1, 19, 1024]
            hidden_states, weight=self.layer_norm1, bias=self.layer_norm1_bias, epsilon=self.layer_norm_eps
        )
        attn_output = self.self_attn(hidden_states, causal_attention_mask, ccl_manager, parallel_config)
        hidden_states = residual + attn_output

        # mlp block
        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states, weight=self.layer_norm2, bias=self.layer_norm2_bias, epsilon=self.layer_norm_eps
        )

        # breakpoint()
        # target: [1, 19, 1024]
        # Pad hidden_states to match padded layer norm weights
        if self.padding_config.is_padding_needed():
            target_dim = self.padding_config.target_dim  # 1024
            current_dim = hidden_states.shape[-1]  # 768
            padding_needed = target_dim - current_dim  # 256

            logger.debug(f"Padding hidden_states from {current_dim} to {target_dim}")

            # Pad last dimension: [1, 19, 768] -> [1, 19, 1024]
            padding = [(0, 0)] * len(hidden_states.shape)
            padding[-1] = (0, padding_needed)
            hidden_states = ttnn.pad(hidden_states, padding, 0.0)
            residual = ttnn.pad(residual, padding, 0.0)  # Pad residual for later addition

        # breakpoint()
        # might have to pad mlp weights.
        # TODO: convert ff1 and ff2 weights to shape [1024, 1024]
        print(f"Before MLP: hidden_states.shape = {hidden_states.shape}")
        original_shape = hidden_states.shape
        mlp_output_fractured = self.mlp(hidden_states)  # fractured on columns
        print(f"After MLP: mlp_output_fractured.shape = {mlp_output_fractured.shape}")
        hidden_states_shape = list(mlp_output_fractured.shape)

        # Handle 3D tensors for operations that require 4D tensors and dim=3

        needs_reshape = len(original_shape) == 3

        if needs_reshape:
            # Reshape [B, S, E] -> [1, B, S, E] to make it 4D
            mlp_output_fractured = ttnn.unsqueeze(mlp_output_fractured, 0)  # [1, 1, 19, 1024]
            # [1,1,19,192]

        # target: [1, 1, 19, 1024]

        # reduce scatter
        # TODO: "Always | Error, The number of tiles at input tensor dimension 3 should be divisible by ring_size but the number of tiles is 6 and the ring_size is 4 (assert.hpp:107)"
        # each tile contains 32 elements, so we need to pad the tensor to make it divisible by 4
        # ring size is 4 = num_devices
        # dimension 3 is 192, tile size is 32, num_devices is 4
        # 192 / 32 = 6 != 4
        # need 8 times or 4 tiles (divisible by 4)

        # mlp_output_fractured = ttnn.experimental.reduce_scatter_minimal_async(
        #     mlp_output_fractured,
        #     dim=3,
        #     multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(),
        #     num_links=1,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     topology=ccl_manager.topology,
        #     cluster_axis=parallel_config.tensor_parallel.mesh_axis,
        # )
        print(f"After reduce_scatter: mlp_output_fractured.shape = {mlp_output_fractured.shape}")

        # all gather
        mlp_output = ttnn.experimental.all_gather_async(
            mlp_output_fractured,
            dim=3,
            cluster_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=ccl_manager.mesh_device,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        print(f"After all_gather: mlp_output.shape = {mlp_output.shape}")

        # Reshape back to original dimensions if we reshaped earlier
        if needs_reshape:
            # Remove the extra dimension we added: [1, B, S, E] -> [B, S, E]
            mlp_output = ttnn.squeeze(mlp_output, 0)
        else:
            mlp_output = ttnn.reshape(mlp_output, hidden_states_shape)  # [1, 19, 768]
        print(f"Final: mlp_output.shape = {mlp_output.shape}, residual.shape = {residual.shape}")

        hidden_states = residual + mlp_output

        # Unpad back to original dimensions if padding was applied
        if self.padding_config.is_padding_needed():
            original_dim = self.padding_config.original_dim  # 768
            # Slice to remove padding: [1, 19, 1024] -> [1, 19, 768]
            hidden_states = hidden_states[..., :original_dim]
            logger.debug(f"Unpadded hidden_states from {mlp_output.shape} to {hidden_states.shape}")

        logger.info(f"CLIPEncoderLayer completed, final shape: {hidden_states.shape}")

        return hidden_states


class CLIPEncoderLayerMLP:
    def __init__(self, config: CLIPConfig):
        self.config = config


class EncoderLayerSelfAttention:
    """
    input is replicated
    Q, K, V are head/column parallel
    SDPA executes head/column parallel
    output is all-gather
    """

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

        # weights to be added in load_state_dict, column sharded
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

    def __call__(self, hidden_states, causal_attention_mask, ccl_manager, parallel_config):
        """
        input is replicated
        Q, K, V are head-parallel (Each device gets embed_dim // num_devices columns of each weight matrix)
        SDPA executes head-parallel
        output is replicated
        """
        # breakpoint()
        # determine the parallelism status (replicated, shareded, etc) of
        # every weight/activation/etc right before self attn (here) is done

        # hidden_states is replicated. [batch_size, seq_length, embed_dim]
        # Shape([1, 19, 768])

        # causal_attention_mask is replicated. [1, 1, seq_length, seq_length]
        # Shape([1, 1, 19, 19])

        # q_proj weight is column parallel [embed_dim, embed_dim/num_heads = head_dim]
        # q_proj bias Shape([1, 192])

        # k_proj weight Shape([768, 192])
        # v_proj weight Shape([768, 192])
        # o_proj weight Shape([192, 768])

        batch_size, seq_length, _ = hidden_states.shape

        # get q, k, v  matrices
        q = self.q_proj(hidden_states)  # Shape([1, 19, 192])
        k = self.k_proj(hidden_states)  # Shape([1, 19, 192])
        v = self.v_proj(hidden_states)  # Shape([1, 19, 192])

        q = q * self.scale  # Shape([1, 19, 192])

        # reshape for multihead attention
        num_devices = parallel_config.tensor_parallel.factor
        num_local_heads = self.num_heads // num_devices

        q = ttnn.reshape(q, (batch_size, seq_length, num_local_heads, self.head_dim))
        k = ttnn.reshape(k, (batch_size, seq_length, num_local_heads, self.head_dim))
        v = ttnn.reshape(v, (batch_size, seq_length, num_local_heads, self.head_dim))
        # shape([1, 19, 16, 64])

        # transpose to [batch_size, num_heads, seq_length, head_dim]
        q = ttnn.transpose(q, 1, 2)
        k = ttnn.transpose(k, 1, 2)
        v = ttnn.transpose(v, 1, 2)

        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))  # [batch_size, num_heads, seq_length, seq_length]

        if causal_attention_mask is not None:
            scores = scores + causal_attention_mask

        attn_weights = ttnn.softmax(scores, dim=-1)

        # TODO: replace with ttnn.dropout once it's supported
        # attn_weights = ttnn.experimental.dropout(attn_weights, self._attention_dropout)

        attn_output = ttnn.matmul(attn_weights, v)  # head_parallel. [batch_size, num_heads, seq_length, head_dim]

        # transpose back and reshape
        attn_output = ttnn.transpose(attn_output, 1, 2)  # [batch_size, seq_length, num_heads, head_dim]
        attn_output = ttnn.reshape(
            attn_output, (1, batch_size, seq_length, self.embed_dim // num_devices)
        )  # [1, batch_size, seq_length, embed_dim/num_heads]

        orig_shape = list(attn_output.shape)

        # need to gather attn_output across all devices
        attn_output = ttnn.experimental.all_gather_async(  # [1, batch_size, seq_length, full embed_dim] (replicated)
            attn_output,
            dim=len(attn_output.shape) - 1,
            cluster_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=ccl_manager.mesh_device,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Shape([1, 1, 19, 768]) (replicated)
        # breakpoint()
        dense_out = self.o_proj(attn_output)  # o_proj is still head parallel. Shape([1, 1, 19, 192])

        dense_out = ttnn.experimental.all_gather_async(
            dense_out,
            dim=len(dense_out.shape) - 1,
            cluster_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=ccl_manager.mesh_device,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # breakpoint()

        dense_out_shape = list(dense_out.shape)  # [1,1,19,768]
        dense_out_shape[2] = orig_shape[2]
        dense_out = ttnn.reshape(dense_out, tuple(dense_out_shape), dense_out.shape)  # Shape([1, 1, 19, 768])

        return ttnn.reshape(dense_out, tuple(dense_out.shape)[1:])  # [1, 19, 768]


class TextEmbeddings:
    """
    Embeds text tokens and adds position embeddings.

    Args:
        config: Config
        mesh_device: ttnn.Device

    Returns:
        ttnn.Tensor: Token + position embeddings - shape: (batch_size, seq_len, embed_dim)
    """

    def __init__(self, config, mesh_device: ttnn.Device) -> None:
        self.config = config
        self.mesh_device = mesh_device

        # weights to be added in load_state_dict
        self.token_embedding = None
        self.position_embedding = None

    def load_state_dict(self, state_dict):
        # weights are replicated across all devices
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
def _create_4d_causal_attention_mask(
    input_shape: tuple[int, int], device: ttnn.Device, dtype: ttnn.DataType
) -> ttnn.Tensor:
    """Create a 4D causal attention mask for the given input shape."""
    batch_size, tgt_len = input_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len)
    return ttnn.from_torch(mask, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
