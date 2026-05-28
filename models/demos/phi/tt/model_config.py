# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import ttnn

class PhiConfig:
    def __init__(self):
        self.vocab_size = 51200
        self.hidden_size = 2048
        self.num_hidden_layers = 24
        self.num_attention_heads = 32
        self.num_key_value_heads = 32
        self.intermediate_size = 8192
        self.hidden_act = "gelu_new"
        self.layer_norm_eps = 1e-5
        self.max_position_embeddings = 2048
        self.initializer_range = 0.02
        self.tie_word_embeddings = False
        self.rope_theta = 10000.0
        self.partial_rotary_factor = 0.5

    @classmethod
    def from_phi1_5(cls):
        return cls()
