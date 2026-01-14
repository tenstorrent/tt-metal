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
    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": False,
    }

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
        start_pos = kwargs.get("start_pos", None)
        assert (start_pos is None) or all(
            x == 0 for x in start_pos
        ), f"Prefix caching is not supported for DeepseekV3ForCausalLM, got start_pos: {start_pos}"
        assert self.model_run_config_prefill is not None, "Model run config prefill is not initialized"

        kwargs.pop("enable_trace", None)
        logger.warning("Prefill tracing not supported for DeepseekGenerator")
        tokens = kwargs["tokens"]
        lengths = kwargs["prompt_lens"]
        page_table = kwargs.get("page_table", None)
        kv_cache = kwargs.get("kv_cache", None)
        empty_slots = kwargs.get("empty_slots", None)

        if all(length == 0 for length in lengths):
            return torch.zeros(tokens.shape[0], self.hf_config.vocab_size, device=tokens.device, dtype=tokens.dtype)

        # Set kv_cache if provided and all entries are valid
        if kv_cache is not None and not any(entry is None for entry in kv_cache):
            self.set_kv_cache(kv_cache)

        pad_value = self.tokenizer.pad_token_id
        pad_block_size = self.paged_config.block_size if self.paged_config is not None else USERS_PER_ROW
        max_prompt_len = int(max(lengths)) if len(lengths) else 0
        max_padded_len = (
            ((max_prompt_len + pad_block_size - 1) // pad_block_size) * pad_block_size if max_prompt_len > 0 else 0
        )
        num_of_users = tokens.shape[0]
        last_logits = []
        for i in range(num_of_users):
            user_id = empty_slots[i] if empty_slots is not None else i
            prompt_len = int(lengths[i])
            if prompt_len == 0:
                last_logits.append(
                    torch.zeros(max_padded_len, self.hf_config.vocab_size, device=tokens.device, dtype=tokens.dtype)
                )
                continue
            user_tokens = tokens[i, :prompt_len].unsqueeze(0)
            user_tokens = _pad_tokens(user_tokens, pad_value, block_size=pad_block_size).squeeze(0)
            user_out = self._prefill(user_tokens, user_id, page_table, local_user_id=i)
            user_logits = user_out.squeeze(0).squeeze(0)  # [1, 1, S, V] -> [S, V]
            if user_logits.shape[0] > prompt_len:
                user_logits = user_logits[:prompt_len]
            if user_logits.shape[0] < max_padded_len:
                pad_len = max_padded_len - user_logits.shape[0]
                pad_logits = user_logits[-1:].expand(pad_len, -1)
                user_logits = torch.cat([user_logits, pad_logits], dim=0)
            last_logits.append(user_logits)
        last_logits = torch.stack(last_logits)  # [num_of_users, S, V]

        return last_logits

    def decode_forward(self, *args, **kwargs):
        assert self.model_run_config_decode is not None, "Model run config decode is not initialized"

        page_table = kwargs.get("page_table", None)
        kv_cache = kwargs.get("kv_cache", None)
        # Set kv_cache if provided and all entries are valid
        if kv_cache is not None and not any(entry is None for entry in kv_cache):
            self.set_kv_cache(kv_cache)

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

        kv_cache_config = KvCacheConfig(kv_cache_shape=kv_cache_shape, dtype=dtype)
        self._prepare_run_configs("prefill", kv_cache_override=kv_cache_config)
        self._prepare_run_configs("decode", kv_cache_override=kv_cache_config)
        kv_cache = self.get_kv_cache()

        return kv_cache
