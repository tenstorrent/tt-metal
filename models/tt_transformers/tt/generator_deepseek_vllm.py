# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch

from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer


def _pad_tokens(tokens: torch.Tensor, pad_value: int = 0, block_size: int = 32) -> torch.Tensor:
    """
    Pad tokens to the nearest multiple of block_size.

    Args:
        tokens: Input tensor of shape [batch_size, seq_len]
        pad_value: Value to use for padding (default: 0)

    Returns:
        Padded tensor of shape [batch_size, padded_seq_len] where padded_seq_len is multiple of block_size
    """
    batch_size, seq_len = tokens.shape
    # Calculate the nearest multiple of block_size
    padded_len = ((seq_len + block_size - 1) // block_size) * block_size
    if padded_len == seq_len:
        return tokens
    padded_tokens = torch.full((batch_size, padded_len), pad_value, dtype=tokens.dtype, device=tokens.device)
    padded_tokens[:, :seq_len] = tokens
    return padded_tokens


class DeepseekV3ForCausalLM(DeepseekGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, n_layers=None, tt_data_parallel=1
    ):
        model_path = os.getenv("DEEPSEEK_V3_HF_MODEL", "models/demos/deepseek_v3/reference")
        cache_dir = os.getenv("DEEPSEEK_V3_CACHE", "generated/deepseek_v3")
        tokenizer = load_tokenizer(model_path)

        model = cls(
            hf_config=hf_config,
            mesh_device=mesh_device,
            model_path=Path(model_path),
            cache_dir=Path(cache_dir),
            tokenizer=tokenizer,
        )
        return model

    @property
    def cache_path(self):
        return self.cache_dir

    def prefill_forward(self, *args, **kwargs):
        self._prepare_run_configs("prefill")
        tokens = kwargs["tokens"]
        lengths = kwargs["prompt_lens"]
        tokens = _pad_tokens(tokens, self.tokenizer.pad_token_id, block_size=self.mesh_device.shape[1])
        num_of_users = tokens.shape[0]
        last_logits = []
        for user_id in range(num_of_users):
            if lengths[user_id] == 0:
                last_logits.append(torch.zeros(self.hf_config.vocab_size))
                continue
            user_out = self._prefill(tokens[user_id], user_id)
            last_logits.append(user_out.squeeze(0).squeeze(0))  # [1, 1, S, V] -> [S, V]
        last_logits = torch.stack(last_logits)  # [num_of_users, S, V]
        self._cleanup_run_configs("prefill")
        self._prepare_run_configs("decode")
        return last_logits

    def decode_forward(self, *args, **kwargs):
        kwargs["tokens"] = kwargs["tokens"].squeeze(1)
        return_value = (
            self._decode_step(tokens_step=kwargs["tokens"], positions=kwargs["start_pos"])
            .squeeze(0)
            .squeeze(0)
            .unsqueeze(1)
        )  # [1,1,B,V] -> [B, 1, V]
        return return_value

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return [None] * self.hf_config.num_hidden_layers
