# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import Optional

from models.experimental.bark_small.tt.bark_attention import TtBarkAttention
from models.experimental.bark_small.tt.bark_mlp import TtBarkMLP


class TtBarkTextToSemanticLayer(torch.nn.Module):
    def __init__(
        self,
        config,
        state_dict,
        base_key,
        device,
        mesh_mapper=None,
    ):
        super().__init__()
        self.device = device
        self.mesh_mapper = mesh_mapper

        self.ln_1 = ttnn.GroupNorm(
            num_groups=1,
            num_channels=config.hidden_size,
            weight=ttnn.from_torch(
                state_dict[f"{base_key}.ln_1.weight"],
                device=device,
                mesh_mapper=mesh_mapper,
            ),
            bias=ttnn.from_torch(
                state_dict[f"{base_key}.ln_1.bias"],
                device=device,
                mesh_mapper=mesh_mapper,
            ),
            epsilon=config.layer_norm_epsilon,
        )

        self.attn = TtBarkAttention(
            config=config,
            state_dict=state_dict,
            base_key=f"{base_key}.attn",
            device=device,
            mesh_mapper=mesh_mapper,
            causal=True,
        )

        self.ln_2 = ttnn.GroupNorm(
            num_groups=1,
            num_channels=config.hidden_size,
            weight=ttnn.from_torch(
                state_dict[f"{base_key}.ln_2.weight"],
                device=device,
                mesh_mapper=mesh_mapper,
            ),
            bias=ttnn.from_torch(
                state_dict[f"{base_key}.ln_2.bias"],
                device=device,
                mesh_mapper=mesh_mapper,
            ),
            epsilon=config.layer_norm_epsilon,
        )

        self.mlp = TtBarkMLP(
            config=config,
            state_dict=state_dict,
            base_key=f"{base_key}.mlp",
            device=device,
            mesh_mapper=mesh_mapper,
        )

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)
        hidden_states = ttnn.add(residual, attn_output)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = ttnn.add(residual, feed_forward_hidden_states)

        return hidden_states


class TtBarkTextToSemanticModel(torch.nn.Module):
    def __init__(
        self,
        config,
        state_dict,
        device,
        mesh_mapper=None,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.mesh_mapper = mesh_mapper

        # Token embedding
        self.wte = ttnn.from_torch(
            state_dict["transformer.wte.weight"],
            device=device,
            mesh_mapper=mesh_mapper,
        )

        # Position embedding
        self.wpe = ttnn.from_torch(
            state_dict["transformer.wpe.weight"],
            device=device,
            mesh_mapper=mesh_mapper,
        )

        # Transformer layers
        self.layers = torch.nn.ModuleList(
            [
                TtBarkTextToSemanticLayer(
                    config=config,
                    state_dict=state_dict,
                    base_key=f"transformer.h.{i}",
                    device=device,
                    mesh_mapper=mesh_mapper,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # Layer norm
        self.ln_f = ttnn.GroupNorm(
            num_groups=1,
            num_channels=config.hidden_size,
            weight=ttnn.from_torch(
                state_dict["transformer.ln_f.weight"],
                device=device,
                mesh_mapper=mesh_mapper,
            ),
            bias=ttnn.from_torch(
                state_dict["transformer.ln_f.bias"],
                device=device,
                mesh_mapper=mesh_mapper,
            ),
            epsilon=config.layer_norm_epsilon,
        )

        # LM head
        self.lm_head = ttnn.from_torch(
            state_dict["lm_head.weight"],
            device=device,
            mesh_mapper=mesh_mapper,
        )

    def forward(
        self,
        input_ids: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
    ):
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embeddings = ttnn.embedding(input_ids, self.wte)

        # Position embeddings
        position_ids = ttnn.arange(0, seq_len, 1, device=self.device)
        position_embeddings = ttnn.embedding(position_ids, self.wpe)

        # Add embeddings
        hidden_states = ttnn.add(token_embeddings, position_embeddings)

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Language modeling head
        logits = ttnn.matmul(hidden_states, self.lm_head)

        return logits