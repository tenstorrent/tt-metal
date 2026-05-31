# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs


class LFMBackbone:
    def __init__(self, device, model_args: ModelArgs, parameters):
        self.device = device
        self.model_args = model_args
        self.transformer = Transformer(
            args=model_args,
            device=device,
            parameters=parameters,
            weight_cache_path=None,
        )

    def __call__(self, input_ids, mode="prefill"):
        return self.transformer(input_ids, mode=mode)

    @staticmethod
    def get_model_args(vocab_size=65536, num_layers=24, hidden_dim=2048, num_heads=16):
        args = ModelArgs()
        args.dim = hidden_dim
        args.n_layers = num_layers
        args.n_heads = num_heads
        args.vocab_size = vocab_size
        args.max_seq_len = 32768
        args.head_dim = hidden_dim // num_heads
        args.n_kv_heads = num_heads
        return args
