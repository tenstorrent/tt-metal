# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Implementation of Bark Fine Model (Coarse-to-Fine, Stage 3).

This is a non-causal GPT model that takes 2 coarse EnCodec codebooks
and generates the remaining 6 fine codebooks (total 8).

Key differences from the causal model:
- Non-causal attention (no causal mask)
- Multiple embedding layers (one per codebook)
- Multiple LM heads (one per predicted codebook)
- Input is sum of codebook embeddings up to current codebook index
- LayerNorm uses bias=True

Reference: HuggingFace BarkFineModel
"""

from typing import Optional

import torch
import ttnn

from models.demos.wormhole.bark.tt.bark_gpt import (
    BarkConfig,
    TtBarkBlock,
    preprocess_layernorm_weight,
    preprocess_linear_weight,
)


class TtBarkFineModel:
    """Bark Fine model: non-causal GPT for generating fine EnCodec codebooks.

    Architecture:
        For each codebook_idx from n_codes_given to n_codes_total:
            input = sum(embedding_i(tokens[:,:,i]) for i in 0..codebook_idx)
            input += position_embedding
            output = transformer_blocks(input)
            output = layernorm(output)
            predicted_codebook = lm_head[codebook_idx - n_codes_given](output)

    Config (Bark Small):
        n_codes_total = 8 (total EnCodec codebooks)
        n_codes_given = 2 (provided by coarse stage)
        So we predict codebooks 2-7 (6 codebooks)
    """

    def __init__(self, device, parameters, config: BarkConfig, n_codes_total=8, n_codes_given=2):
        self.device = device
        self.config = config
        self.n_codes_total = n_codes_total
        self.n_codes_given = n_codes_given

        # One embedding layer per codebook (all 8)
        self.input_embeds_layers = []
        for i in range(n_codes_total):
            emb = torch.nn.Embedding(config.input_vocab_size, config.hidden_size)
            emb.weight = torch.nn.Parameter(parameters["input_embeds_layers"][str(i)]["weight"])
            self.input_embeds_layers.append(emb)

        # Position embedding (shared across codebooks)
        self.position_embeds_layer = torch.nn.Embedding(config.block_size, config.hidden_size)
        self.position_embeds_layer.weight = torch.nn.Parameter(parameters["position_embeds_layer"]["weight"])

        # Transformer blocks (non-causal)
        self.blocks = []
        for i in range(config.num_layers):
            block = TtBarkBlock(device, parameters["layers"][str(i)], config, is_causal=False)
            self.blocks.append(block)

        # Final layer norm (with bias for fine model)
        self.ln_f_weight = preprocess_layernorm_weight(parameters["layernorm_final"]["weight"], device)
        self.ln_f_bias = (
            preprocess_layernorm_weight(parameters["layernorm_final"]["bias"], device)
            if "bias" in parameters["layernorm_final"]
            else None
        )

        # One LM head per predicted codebook (6 heads for codebooks 2-7)
        self.lm_heads = []
        for i in range(n_codes_total - n_codes_given):
            weight = preprocess_linear_weight(parameters["lm_heads"][str(i)]["weight"], device)
            self.lm_heads.append(weight)

    def __call__(
        self,
        codebook_idx: int,
        input_ids: torch.Tensor,
    ) -> ttnn.Tensor:
        """Forward pass for a specific codebook prediction.

        Args:
            codebook_idx: Which codebook to predict (2-7 for fine model)
            input_ids: [batch, seq_len, n_codes_total] - all codebook tokens

        Returns:
            logits: [batch, seq_len, codebook_vocab_size]
        """
        if codebook_idx < self.n_codes_given:
            raise ValueError(
                f"Cannot predict codebook {codebook_idx} - codebooks 0..{self.n_codes_given-1} "
                f"should be predicted by the coarse model"
            )

        # Sum embeddings of codebooks 0..codebook_idx
        # input_ids shape: [batch, seq_len, n_codes_total]
        inputs_embeds = sum(self.input_embeds_layers[i](input_ids[:, :, i]) for i in range(codebook_idx + 1))

        batch_size, seq_len = inputs_embeds.shape[0], inputs_embeds.shape[1]

        # Position embeddings
        position_ids = torch.arange(0, seq_len, dtype=torch.long)
        position_embeds = self.position_embeds_layer(position_ids).unsqueeze(0)

        # Combine
        hidden = inputs_embeds + position_embeds

        # Convert to TTNN [1, batch, seq, hidden]
        if hidden.dim() == 3:
            hidden = hidden.unsqueeze(0)

        tt_hidden = ttnn.from_torch(hidden.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        # Transformer blocks
        for block in self.blocks:
            tt_hidden = block(tt_hidden)

        # Final layer norm
        tt_hidden = ttnn.layer_norm(tt_hidden, epsilon=1e-5, weight=self.ln_f_weight, bias=self.ln_f_bias)

        # LM head for the specific codebook
        lm_head_idx = codebook_idx - self.n_codes_given
        logits = ttnn.linear(tt_hidden, self.lm_heads[lm_head_idx])
        ttnn.deallocate(tt_hidden)

        return logits


def preprocess_fine_model_parameters(model, device):
    """Extract and organize parameters from a HuggingFace BarkFineModel.

    Args:
        model: HuggingFace BarkFineModel
        device: TTNN device

    Returns:
        dict: Organized parameter dictionary for TtBarkFineModel
    """
    state_dict = model.state_dict()
    parameters = {}
    config = model.config

    # Embedding layers (one per codebook, kept as torch tensors)
    parameters["input_embeds_layers"] = {}
    for i in range(config.n_codes_total):
        parameters["input_embeds_layers"][str(i)] = {"weight": state_dict[f"input_embeds_layers.{i}.weight"].clone()}

    # Position embedding
    parameters["position_embeds_layer"] = {"weight": state_dict["position_embeds_layer.weight"].clone()}

    # Transformer layers
    parameters["layers"] = {}
    for i in range(config.num_layers):
        prefix = f"layers.{i}"
        layer_params = {}

        # LayerNorms (fine model has bias=True)
        layer_params["layernorm_1"] = {
            "weight": state_dict[f"{prefix}.layernorm_1.weight"].clone(),
        }
        if f"{prefix}.layernorm_1.bias" in state_dict:
            layer_params["layernorm_1"]["bias"] = state_dict[f"{prefix}.layernorm_1.bias"].clone()

        layer_params["layernorm_2"] = {
            "weight": state_dict[f"{prefix}.layernorm_2.weight"].clone(),
        }
        if f"{prefix}.layernorm_2.bias" in state_dict:
            layer_params["layernorm_2"]["bias"] = state_dict[f"{prefix}.layernorm_2.bias"].clone()

        # Attention (no bias for attention in fine model either)
        layer_params["attn"] = {
            "att_proj": {"weight": state_dict[f"{prefix}.attn.att_proj.weight"].clone()},
            "out_proj": {"weight": state_dict[f"{prefix}.attn.out_proj.weight"].clone()},
        }
        if f"{prefix}.attn.att_proj.bias" in state_dict:
            layer_params["attn"]["att_proj"]["bias"] = state_dict[f"{prefix}.attn.att_proj.bias"].clone()
        if f"{prefix}.attn.out_proj.bias" in state_dict:
            layer_params["attn"]["out_proj"]["bias"] = state_dict[f"{prefix}.attn.out_proj.bias"].clone()

        # MLP
        layer_params["mlp"] = {
            "in_proj": {"weight": state_dict[f"{prefix}.mlp.in_proj.weight"].clone()},
            "out_proj": {"weight": state_dict[f"{prefix}.mlp.out_proj.weight"].clone()},
        }
        if f"{prefix}.mlp.in_proj.bias" in state_dict:
            layer_params["mlp"]["in_proj"]["bias"] = state_dict[f"{prefix}.mlp.in_proj.bias"].clone()
        if f"{prefix}.mlp.out_proj.bias" in state_dict:
            layer_params["mlp"]["out_proj"]["bias"] = state_dict[f"{prefix}.mlp.out_proj.bias"].clone()

        parameters["layers"][str(i)] = layer_params

    # Final layer norm (with bias)
    parameters["layernorm_final"] = {
        "weight": state_dict["layernorm_final.weight"].clone(),
    }
    if "layernorm_final.bias" in state_dict:
        parameters["layernorm_final"]["bias"] = state_dict["layernorm_final.bias"].clone()

    # LM heads (one per predicted codebook)
    parameters["lm_heads"] = {}
    n_predicted = config.n_codes_total - config.n_codes_given
    for i in range(n_predicted):
        parameters["lm_heads"][str(i)] = {"weight": state_dict[f"lm_heads.{i}.weight"].clone()}

    return parameters
