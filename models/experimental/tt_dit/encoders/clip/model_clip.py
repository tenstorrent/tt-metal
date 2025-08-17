# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from ...utils.tensor import bf16_tensor
from ...utils.substate import substate


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
    def __init__(self, config: CLIPConfig, mesh_device: ttnn.Device) -> None:
        self.config = config
        self.embeddings = TextEmbeddings(config, mesh_device)
        # self.encoder = CLIPStack(config)

    def load_state_dict(self, state_dict):
        self.embeddings.load_state_dict(substate(state_dict, "text_model.embeddings"))
        # self.encoder.load_state_dict(substate(state_dict, "text_model.encoder"))

    def __call__(self, prompt: ttnn.Tensor, mesh_device: ttnn.Device, with_projection: bool = True) -> torch.Tensor:
        input_embeddings = self.embeddings(prompt, mesh_device)
        return input_embeddings
        # return self.encoder(input_ids, with_projection=with_projection)


class TextEmbeddings:
    """
    Embeds text tokens and adds position embeddings.

    Args:
        config: Config
        mesh_device: ttnn.Device

    Returns:
        ttnn.Tensor: Token + position embeddings - shape: (batch_size, seq_len, embed_dim)
    """

    def __init__(self, config: CLIPConfig, mesh_device: ttnn.Device) -> None:
        self.config = config
        self.mesh_device = mesh_device

        # weights added in load_state_dict
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
