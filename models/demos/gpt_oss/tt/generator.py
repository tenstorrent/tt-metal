# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS specific generator utilities.

Why this exists:
- GPT-OSS attention prefill only supports batch_size=1, but uses a global
  paged KV cache with a user slot index (`user_id`) to choose which cache
  entry to fill.
- For high-throughput profiles (e.g. max_batch_size=128 on multi-row meshes),
  `users_row_sharded=True` requires the page table batch dimension to be the
  global max batch size (divisible by mesh rows). vLLM provides per-request
  block tables; the TT model runner maps requests to stable slots and builds a
  global page table.

This module provides a generator implementation that consumes:
- `tokens`: [unpadded_batch, seq_len]
- `page_table`: [max_batch_size, num_blocks] (global, slot-indexed)
- `empty_slots`: list[int] mapping each token row -> global slot id

and calls into the TT model with `user_id=<slot_id>` while keeping the token
batch dimension 1 for prefill.
"""

from __future__ import annotations

import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import get_block_size, get_padded_prefill_len, num_blocks_in_seq
from models.tt_transformers.tt.generator import Generator


class GPTOSSRowShardedGenerator(Generator):
    """Generator variant for GPT-OSS high-throughput user-slot prefill.

    This overrides only prefill behavior. Decode is handled by the base
    Generator implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prefill_forward_text(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        model_id_warmup=None,
        **kwargs,
    ):
        if page_table is not None:
            assert isinstance(page_table, torch.Tensor), "page_table mush be torch.Tensor"
        else:
            # Only paged attention is supported for prefill
            enable_trace = False

        # self.warmup_prefill_traces(
        #     page_table,
        #     kv_cache,
        #     enable_trace,
        # )

        batch_size, batch_seq_len = tokens.shape
        max_batch_size_per_model = self.model_args[0].max_batch_size
        max_batch_per_mesh_row = max_batch_size_per_model // self.mesh_device.shape[0]

        # Each model expected to run the same model, safe to use 1st vocab size
        output_logits = torch.zeros(batch_size, 1, self.model_args[0].vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch_size)
        if empty_slots is None:
            empty_slots = list(range(batch_size))

        out_list = []
        for idx, user_id in enumerate(empty_slots):
            # if model_id is not None, it means that prefill is called from warmup_prefill_traces
            model_id = user_id // max_batch_size_per_model if model_id_warmup is None else model_id_warmup
            group_user_id = user_id % max_batch_size_per_model if page_table is None else 0
            seq_len = int(prompt_lens[idx])
            last_token_idx = seq_len - 1
            prefill_seq_len = get_padded_prefill_len(seq_len)
            local_kwargs = kwargs.copy()  # Avoid modifying original kwargs
            logger.info(f"Prefilling User {user_id + 1} up to {seq_len} tokens")

            # Extracting data for the current user
            # If page_table is not provided, we keep track of the relative/model user_id through group_user_id
            prefill_ids = torch.cat(
                [tokens[idx : idx + 1, :seq_len], torch.zeros(1, prefill_seq_len - seq_len).long()], dim=-1
            )

            enable_trace_current_prompt = enable_trace and self.model_args[model_id].can_enable_trace(prefill_seq_len)

            logger.info(
                f"Prefill seq len: {prefill_seq_len}, max_prefill_chunk_size: {self.model_args[0].max_prefill_chunk_size}, trace: {enable_trace_current_prompt}"
            )

            page_table_user = (
                self.get_prefill_page_table(
                    page_table,
                    kv_cache[model_id],
                    prefill_seq_len,
                    user_id=(user_id // max_batch_per_mesh_row) * max_batch_per_mesh_row,
                )
                if page_table is not None
                else None
            )
            if page_table is not None:
                page_table = page_table[1:, :]

            model_kv_cache = kv_cache[model_id] if kv_cache is not None else None

            if enable_trace_current_prompt:
                logits = self._easy_trace_prefill(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=0,
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    prefill_seq_len=prefill_seq_len,
                    **local_kwargs,
                )
            else:
                logits = self.prefill_forward_single_user_text(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=0,
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    **local_kwargs,
                )
            if enable_trace_current_prompt:
                # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
                # We need to do this here, because we can't do this part in forward() if we have trace enabled
                # The reason we can't do it in trace is because we can't pass the correct get_last_token to trace
                logits = self.model[model_id].process_logits_after_prefill_trace(logits, last_token_idx)

            # if data parallel is greater than 1, we need to add logits to out_list and do the processing after all the prefill are done
            # otherwise, we can process the logits after prefill immediately
            if self.data_parallel > 1:
                out_list.append(logits)
            else:
                output_logits[idx] = self.model[model_id].process_output_prefill(
                    logits, last_token_idx=(last_token_idx % 32)
                )
                del logits

        # Process the logits after all the prefill are done in data parallel mode
        if self.data_parallel > 1:
            for idx, out in enumerate(out_list):
                seq_len = int(prompt_lens[idx])
                last_token_idx = seq_len - 1
                user_id = empty_slots[idx]
                model_id = user_id // max_batch_size_per_model if model_id_warmup is None else model_id_warmup

                # Since we give unpadded_seq_len, only the tile containing the last token is returned
                output_logits[idx] = self.model[model_id].process_output_prefill(
                    out, last_token_idx=(last_token_idx % 32)
                )

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")
        return output_logits

    def get_prefill_page_table(self, page_table, kv_cache, prefill_len, user_id):
        # Ensure page_table is not padded with extra blocks for paged_fill_cache to work properly

        block_size = get_block_size(kv_cache)
        num_blocks = num_blocks_in_seq(prefill_len, block_size)
        page_table = page_table[:, :num_blocks]
        if page_table.shape[1] < num_blocks:
            # If page table is too short, pad it with -1
            padding = torch.ones(page_table.shape[0], num_blocks - page_table.shape[1], dtype=torch.int32) * -1
            page_table = torch.cat([page_table, padding], dim=1)
        # Pad page table to 32 users
        padded_page_table = torch.ones(128, page_table.shape[1], dtype=torch.int32) * -1
        padded_page_table[user_id, :] = page_table[0, :]
        return padded_page_table

    # --- vLLM hooks (TTModelRunner expects these on the returned model object) ---

    @property
    def cache_path(self):
        # Match `GptOssForCausalLM.cache_path` in `generator_vllm.py`
        return self.model_args[0].weight_cache_path(ttnn.bfloat8_b)

    def allocate_kv_cache(self, *args, **kwargs):
        # Lazy import avoids import-time cycles with `generator_vllm.py`.
        from models.tt_transformers.tt.generator_vllm import allocate_vllm_kv_cache

        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)

    def prefill_forward(self, *args, **kwargs):
        return self.prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward_text(*args, **kwargs)
