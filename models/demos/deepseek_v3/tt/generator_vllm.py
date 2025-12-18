# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from loguru import logger

from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
from models.demos.deepseek_v3.utils.config_dataclass import KvCacheConfig
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer


def _pad_tokens(tokens: torch.Tensor, pad_value: int = 0, block_size: int = USERS_PER_ROW) -> torch.Tensor:
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
        model_path = os.environ.get("DEEPSEEK_V3_HF_MODEL")
        cache_dir = os.environ.get("DEEPSEEK_V3_CACHE")
        if not model_path:
            raise ValueError(
                "DEEPSEEK_V3_HF_MODEL is not set. Set the environment variable or initialize via the demo "
                "entrypoint with an explicit --model-path."
            )
        if not cache_dir:
            raise ValueError(
                "DEEPSEEK_V3_CACHE is not set. Set the environment variable or initialize via the demo "
                "entrypoint with an explicit --cache-dir."
            )
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
        # print args and kwargs
        logger.info(f"prefill_forward: args: {args}")
        logger.info(f"prefill_forward: kwargs: {kwargs.keys()}")
        assert self.model_run_config_prefill is not None, "Model run config prefill is not initialized"
        kwargs.pop("enable_trace", None)
        logger.warning(f"Prefill tracing not supported for DeepseekGenerator")

        tokens = kwargs["tokens"]
        lengths = kwargs["prompt_lens"]
        page_table = kwargs.get("page_table", None)
        kv_cache = kwargs.get("kv_cache", None)
        # Set kv_cache if provided and all entries are valid
        if kv_cache is not None and not any(entry is None for entry in kv_cache):
            logger.info(f"prefill_forward: Setting kv_cache for {len(kv_cache)}")
            self.set_kv_cache(kv_cache)
        else:
            logger.info(f"prefill_forward: kv_cache not updated")

        tokens = _pad_tokens(tokens, self.tokenizer.pad_token_id, block_size=USERS_PER_ROW)
        num_of_users = tokens.shape[0]
        last_logits = []
        for user_id in range(num_of_users):
            if lengths[user_id] == 0:
                logger.info(f"prefill_forward: User {user_id} has no tokens")
                last_logits.append(
                    torch.zeros(tokens.shape[1], self.hf_config.vocab_size, device=tokens.device, dtype=tokens.dtype)
                )
                continue
            logger.info(f"prefill_forward: Running prefill for user {user_id}")
            user_out = self._prefill(tokens[user_id], user_id, page_table)
            last_logits.append(user_out.squeeze(0).squeeze(0))  # [1, 1, S, V] -> [S, V]
        last_logits = torch.stack(last_logits)  # [num_of_users, S, V]

        logger.info(f"prefill_forward: Last logits shape: {last_logits.shape}")
        return last_logits

    def decode_forward(self, *args, **kwargs):
        # print args and kwargs
        logger.info(f"decode_forward: args: {args}")
        logger.info(f"decode_forward: kwargs: {kwargs.keys()}")
        assert self.model_run_config_decode is not None, "Model run config decode is not initialized"

        page_table = kwargs.get("page_table", None)
        kv_cache = kwargs.get("kv_cache", None)
        # Set kv_cache if provided and all entries are valid
        if kv_cache is not None and not any(entry is None for entry in kv_cache):
            logger.info(f"decode_forward: Setting kv_cache for {len(kv_cache)} decoder blocks")
            self.set_kv_cache(kv_cache)
        else:
            logger.info(f"decode_forward: kv_cache not updated")

        tokens_step = kwargs["tokens"].squeeze(1)
        return_value = (
            self._decode_step(
                tokens_step=tokens_step,
                positions=kwargs["start_pos"],
                batch_size_per_row=USERS_PER_ROW,
                page_table=page_table,
            )
            .squeeze(0)
            .squeeze(0)
            .unsqueeze(1)
        )  # [1,1,B,V] -> [B, 1, V]
        return return_value

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        assert (
            num_layers == self.hf_config.num_hidden_layers
        ), f"Number of layers {num_layers} does not match the number of layers in the model {self.hf_config.num_hidden_layers}"
        logger.info(f"allocate_kv_cache: kv cache shape: {kv_cache_shape}")

        kv_cache_config = KvCacheConfig(kv_cache_shape=kv_cache_shape, dtype=dtype)
        self._prepare_run_configs("prefill", kv_cache_override=kv_cache_config)
        self._prepare_run_configs("decode", kv_cache_override=kv_cache_config)

        kv_cache = self.get_kv_cache()
        return kv_cache
