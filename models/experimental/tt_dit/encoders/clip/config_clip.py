# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


# TODO: add module docstring. finalize docstrings/comments for classes and methods


class CLIPTextConfig:
    # TODO: create a PretrainedConfig abstract class like HF? (https://github.com/huggingface/transformers/blob/v4.43.0/src/transformers/models/clip/configuration_clip.py)
    # TODO: confirm desired default vals. currently defaulted to HF SD3.5 clip vals
    # TODO: add example usage to docstrings like HF?

    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        max_prompt_length=77,
        layer_norm_eps: float = 1e-05,
        attention_dropout: float = 0.0,
        hidden_act: str = "quick_gelu",
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_prompt_length = max_prompt_length
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
