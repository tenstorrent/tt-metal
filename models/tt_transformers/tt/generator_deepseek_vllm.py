# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from loguru import logger

from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer


class DeepseekV3ForCausalLM(DeepseekGenerator):
    def __init__(self, *args, **kwargs):
        logger.info("Initializing DeepseekV3ForCausalLM init called")
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, n_layers=None, tt_data_parallel=1
    ):
        logger.info("Initializing DeepseekV3ForCausalLM initialize_vllm_model called")
        model_path = os.getenv("DEEPSEEK_V3_HF_MODEL", "models/demos/deepseek_v3/reference")
        cache_dir = os.getenv("DEEPSEEK_V3_CACHE", "generated/deepseek_v3")
        tokenizer = load_tokenizer(model_path)

        return cls(
            hf_config=hf_config,
            mesh_device=mesh_device,
            model_path=Path(model_path),
            cache_dir=Path(cache_dir),
            tokenizer=tokenizer,
        )

    @property
    def cache_path(self):
        logger.info("Initializing DeepseekV3ForCausalLM cache_path called")
        return self.cache_dir

    def prefill_forward(self, *args, **kwargs):
        logger.info(f"Initializing DeepseekV3ForCausalLM prefill_forward called with args: {args} and kwargs: {kwargs}")
        self._prepare_run_configs("prefill")

        logger.info(f"kwargs['tokens']: {kwargs['tokens'].shape}")
        tokens = kwargs["tokens"]
        lengths = kwargs["prompt_lens"]
        num_of_users = tokens.shape[0]
        last_logits = []
        for user_id in range(num_of_users):
            if lengths[user_id] == 0:
                logger.info(f"Skipping prefill for user {user_id} as prompt length is 0")
                last_logits.append(torch.zeros(self.hf_config.vocab_size))
                continue
            logger.info(f"Running prefill for {user_id}")
            logger.info(
                f"Input to the prefill: {self.tokenizer.decode(tokens[user_id].tolist(), skip_special_tokens=True)}"
            )
            user_out = self._prefill(tokens[user_id], user_id)
            last_logits.append(user_out)
        last_logits = torch.stack(last_logits)

        self._cleanup_run_configs("prefill")
        # breakpoint()
        logger.info(f"prefill_forward last logits: {last_logits.shape}")
        return last_logits

    def decode_forward(self, *args, **kwargs):
        logger.info(f"Initializing DeepseekV3ForCausalLM decode_forward called with args: {args} and kwargs: {kwargs}")

        self._prepare_run_configs("decode")
        # breakpoint()
        logger.info(f"kwargs['tokens']: {kwargs['tokens'].shape}")
        logger.info(f"kwargs['start_pos']: {kwargs['start_pos'].shape}")
        kwargs["tokens"] = kwargs["tokens"].squeeze(1)
        return_value = self._decode_step(kwargs["tokens"], kwargs["start_pos"])
        logger.info(f"decode_forward return_value: {return_value.shape}")
        return return_value

    def allocate_kv_cache(self, *args, **kwargs):
        logger.info(
            f"Initializing DeepseekV3ForCausalLM allocate_kv_cache called with args: {args} and kwargs: {kwargs}"
        )
        return [None] * self.hf_config.num_hidden_layers
