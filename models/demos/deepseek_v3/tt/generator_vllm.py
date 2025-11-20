# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from loguru import logger

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
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations: str = None
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
        model._prepare_run_configs("prefill")
        model._prepare_run_configs("decode")
        return model

    @property
    def cache_path(self):
        return self.cache_dir

    def prefill_forward(self, *args, **kwargs):
        logger.info(f"prefill_forward called with args: {args} and kwargs: {kwargs.keys()}")
        tokens = kwargs["tokens"]
        lengths = kwargs["prompt_lens"]
        page_table = kwargs["page_table"]
        logger.info(f"tokens.shape: {tokens.shape}")
        logger.info(f"lengths: {lengths}")
        logger.info(f"page_table.shape: {page_table.shape}")
        tokens = _pad_tokens(tokens, self.tokenizer.pad_token_id, block_size=self.mesh_device.shape[1])
        num_of_users = tokens.shape[0]
        last_logits = []
        for user_id in range(num_of_users):
            if lengths[user_id] == 0:
                logger.info(f"vllm skipping prefill for user_id: {user_id} as prompt length is 0")
                last_logits.append(
                    torch.zeros(tokens.shape[1], self.hf_config.vocab_size, device=tokens.device, dtype=tokens.dtype)
                )
                continue
            user_out = self._prefill(tokens[user_id], user_id)
            logger.info(f"vllm user_out.shape: {user_out.shape}")
            last_logits.append(user_out.squeeze(0).squeeze(0))  # [1, 1, S, V] -> [S, V]
        last_logits = torch.stack(last_logits)  # [num_of_users, S, V]
        logger.info(f"vllm last_logits.shape: {last_logits.shape}")
        return last_logits

    def decode_forward(self, *args, **kwargs):
        logger.info(f"decode_forward called with args: {args} and kwargs: {kwargs.keys()}")
        logger.info(f"trace_mode: {kwargs.get('trace_mode', 'None')}")
        tokens_step = kwargs["tokens"].squeeze(1)
        logger.info(f"tokens_step.shape: {tokens_step.shape}")
        return_value = (
            self._decode_step(tokens_step=tokens_step, positions=kwargs["start_pos"], batch_size_per_row=32)
            .squeeze(0)
            .squeeze(0)
            .unsqueeze(1)
        )  # [1,1,B,V] -> [B, 1, V]
        return return_value

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        logger.info(
            f"allocate_kv_cache called with kv_cache_shape: {kv_cache_shape}, dtype: {dtype}, num_layers: {num_layers}"
        )
        return [None] * num_layers
