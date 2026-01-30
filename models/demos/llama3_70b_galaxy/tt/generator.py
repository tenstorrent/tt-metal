# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger
from typing import List
from collections import defaultdict
from dataclasses import fields, replace

from llama_models.llama3.api.datatypes import (
    InterleavedTextMedia,
    StopReason,
)

from llama_models.llama3.reference_impl.generation import (
    ChatPrediction,
    CompletionPrediction,
)
from models.tt_transformers.tt.common import (
    copy_host_to_device,
    num_blocks_in_seq,
    get_block_size,
)
from models.common.sampling.generator import format_sampling_params
from models.tt_transformers.tt.generator import SamplingParams


def get_padded_prefill_len(seq_len: int) -> int:
    """
    Get the padded prefill length for a given sequence length.
    This is used to pad the sequence length to the nearest power of 2.
    """
    if seq_len <= 128:
        return 128
    if seq_len <= 1024:
        return 1024
    else:
        # return next power of 2 greater than seq_len
        return 2 ** (seq_len - 1).bit_length()


class Generator:
    def __init__(self, model, model_args, mesh_device, tokenizer=None, formatter=None):
        """
        Creating a LlamaVision wrapper requires only a mesh_device and model_args.
        With model_args you have the checkpoint location, can specify max batch size
        and max seqlen, and other model specific parameters.

        LlamaVision is general to text and chat.

        For bringup, make this class general to any backend implementation, as long as it takes torch tensors and returns torch tensors.

        """
        self.model = model
        self.model_args = model_args
        self.mesh_device = mesh_device
        self.formatter = formatter
        if isinstance(self.model_args, List):
            self.model_args = self.model_args[0]
        if isinstance(self.model, List):
            self.model = self.model[0]
        self.tokenizer = self.model_args.tokenizer
        self.trace_id_prefill = defaultdict(lambda: None)
        self.trace_inputs_prefill = defaultdict(lambda: None)
        self.trace_output_prefill = defaultdict(lambda: None)
        # Create persistent buffer for accumulated logits (used for on-device sampling)
        self.tt_logits_accumulated = [
            ttnn.from_torch(
                torch.zeros(1, 1, 1, self.model.args.padded_vocab_size // self.model_args.cluster_shape[0]),
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                dtype=ttnn.bfloat8_b,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
            )
            for _ in range(self.model_args.max_batch_size)
        ]
        self.tt_logits_accumulated_batched = []  # Temporary list for batched prefill
        self.prev_page_table = None
        self.prefill_traces_warmup = False
        self.trace_ids_decode = defaultdict(lambda: None)  # {return_logits: {device_id: trace_id}}
        self.trace_inputs_decode = defaultdict(lambda: None)
        self.trace_output_decode = defaultdict(lambda: None)
        # Split sampling: decode trace captures transformer only, sampling runs separately
        self.enable_split_sampling = True  # Decode trace returns logits, sampling is separate
        self.model.enable_internal_trace = self.enable_split_sampling  # NEVER trace sampling - causes buffer corruption
        self._disable_prefill_tracing = False  # Whether to disable prefill traces
        self._disable_decode_tracing = False  # Whether to disable decode traces
        self._trace_debug_seq = 0  # Monotonic seq for hang-debug logs (trace capture/replay ordering)

    def warmup_prefill_traces(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        enable_trace=True,
        sampling_params=None,
        empty_slots=None,
        tt_out_logits_all_users=None,
    ):
        # Avoids an infinite loop
        self.prefill_traces_warmup = True

        self.model.switch_mode("prefill")
        logger.info("Warming up prefill traces for all supported sequence lengths")
        supported_seqlens = (
            self.model.tt_ccl.support_seqlens
        )  # caching because running prefill can switch mode to decode
        for supported_length in supported_seqlens:
            logger.info(f"Creating warmup tensor for sequence length: {supported_length}")
            # Capture trace for both
            for batch in (1, 32):  # TODO add proper support for batched prefill == b-32
                logger.info(f"Running warmup prefill for sequence length: {supported_length}, batch: {batch}")
                # For batched prefill this needs to be *32
                if batch == 32 and supported_length == 4096:
                    # For batched prefill max batch sequence length is 2048 or lower (128k limit)
                    logger.info(f"Skipping warm up step on batched prefill for sequence length {supported_length}")
                    continue
                if batch == 32:
                    # For warmup, ALL rows need valid block indices (not -1)
                    # because use_batched_prefill=False processes each user individually
                    # and -1 page table entries cause device hangs
                    warmup_page_table = torch.zeros(batch, page_table.shape[1], dtype=torch.int32)
                else:
                    warmup_page_table = page_table
                warmup_tokens = torch.zeros(batch, supported_length, dtype=torch.long)
                warmup_prompt_lens = torch.tensor([supported_length] * batch, dtype=torch.long)
                warmup_empty_slots = list(range(batch))
                self.prefill_forward_text(
                    warmup_tokens,
                    warmup_page_table,
                    kv_cache,
                    warmup_prompt_lens,
                    enable_trace,
                    sampling_params,
                    warmup_empty_slots,
                    tt_out_logits_all_users,
                )
        logger.info("Prefill traces warmup completed")

    def prefill_forward_text(
        self,
        tokens: torch.Tensor,  # All tokens, including the cached ones
        page_table=None,
        kv_cache=None,
        prompt_lens=None,  # Full prompt lengths, including the cached ones
        enable_trace=True,
        sampling_params=None,
        empty_slots=None,
        tt_out_logits_all_users=None,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
        start_pos: list[int] | None = None,  # Cached prefixes lengths
    ):
        if getattr(self, "_disable_prefill_tracing", False):
            enable_trace = False

        if self.prefill_traces_warmup is False:
            self.warmup_prefill_traces(
                tokens,
                page_table,
                kv_cache,
                prompt_lens,
                enable_trace,
                None,
                empty_slots,
                tt_out_logits_all_users,
            )

        return_logits = sampling_params is None

        if self.model.is_prefill_setup is False:
            self.model.switch_mode("prefill")

        kv_cache = kv_cache[0]
        batch, batch_seq_len = tokens.shape
        output_toks = torch.zeros(batch, 1, 1)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch)
        if not isinstance(prompt_lens, list):
            prompt_lens = prompt_lens.tolist()

        # Extract num_cached_tokens from start_pos
        num_cached_tokens_list = [int(start_pos[idx]) if start_pos is not None else 0 for idx in range(batch)]

        # Calculate and pad prefill_seq_lens excluding cached tokens
        prefill_seq_lens = [
            get_padded_prefill_len(seq_len - num_cached_tokens_list[idx]) for idx, seq_len in enumerate(prompt_lens)
        ]
        if page_table is not None:
            assert isinstance(
                page_table, torch.Tensor
            ), "page_table must be a torch.Tensor when passing into prefill_forward"
            block_size = get_block_size(kv_cache)
            if start_pos is not None:
                assert all(
                    start_pos[idx] % block_size == 0 for idx in range(batch)
                ), "start_pos must be aligned to block_size"

        if empty_slots is None:
            empty_slots = list(range(batch))

        # If batch is 32 and prompt_lens are all the same and batch_seq_len * batch is less than 128*1024, use batched prefill
        use_batched_prefill = False
        if batch >= 16 and len(set(prefill_seq_lens)) == 1 and prefill_seq_lens[0] == 128 and (start_pos is None or all(x == 0 for x in start_pos)):
            use_batched_prefill = True

        if return_logits:
            tt_out_logits_all_users = torch.zeros(batch, 1, self.model.args.padded_vocab_size)

        # Prefill has two main modes:
        # - return_logits=True: return logits to host (no on-device sampling)
        # - return_logits=False: produce next-token ids; we only run on-device sampling when logits are not requested
        save_logits_to_host = tt_out_logits_all_users is not None
        do_device_sampling = (not return_logits) and (not save_logits_to_host)

        # Accumulate sharded logits (same format as decode, before all-gather) for on-device sampling.

        all_users = [0] if use_batched_prefill else empty_slots

        for id, user_id in enumerate(all_users):
            logger.info(f"Prefilling User {user_id + 1}, use_batched_prefill: {use_batched_prefill}")
            if use_batched_prefill:
                user_id = empty_slots
                last_token_idx = [(seq_len - 1) for seq_len in prompt_lens]
                prefill_seq_len = prefill_seq_lens[0]
                num_cached_tokens = 0
                seq_len = prompt_lens
            else:
                seq_len = int(prompt_lens[id])
                num_cached_tokens = num_cached_tokens_list[id]
                last_token_idx = seq_len - 1  # Absolute index including cached tokens
                prefill_seq_len = prefill_seq_lens[id]

                if prefill_seq_len not in self.model.tt_ccl.support_seqlens:
                    enable_trace = False

            padded_batch = 32
            if use_batched_prefill:
                # Place each request at its corresponding slot and pad to 32 users
                prefill_ids = torch.zeros(padded_batch, prefill_seq_len, dtype=torch.long, device=tokens.device)
                padded_last_token_idx = [1] * padded_batch  # dummy idx for padded slots
                for local_idx, slot in enumerate(empty_slots):
                    seq_len_local = int(seq_len[local_idx])
                    padded_tokens = torch.cat(
                        [
                            tokens[local_idx : local_idx + 1, :seq_len_local],
                            torch.zeros(1, prefill_seq_len - seq_len_local, dtype=torch.long, device=tokens.device),
                        ],
                        dim=-1,
                    )
                    prefill_ids[slot : slot + 1] = padded_tokens
                    padded_last_token_idx[slot] = last_token_idx[local_idx]
                last_token_idx = padded_last_token_idx
            else:
                seq_len = int(prompt_lens[id])
                last_token_idx = seq_len - 1  # Absolute index including cached tokens
                prefill_seq_len = prefill_seq_lens[id]

                if prefill_seq_len not in self.model.tt_ccl.support_seqlens:
                    enable_trace = False
                # Extract tokens skipping cached ones
                num_cached_tokens = num_cached_tokens_list[id]
                new_tokens_len = seq_len - num_cached_tokens
                prefill_ids = torch.cat(
                    [
                        tokens[id : id + 1, num_cached_tokens:seq_len],  # Skip cached tokens
                        torch.zeros(1, prefill_seq_len - new_tokens_len).long(),
                    ],
                    dim=-1,
                )

            if page_table is not None:
                # For prefix caching, page_table includes both cached and new blocks
                page_table_user = self._get_prefill_user_page_table(
                    page_table,
                    kv_cache,
                    num_cached_tokens + prefill_seq_len,
                    user_id,
                    use_batched_prefill,  # Use full seq_len including cached
                )
                # remove the first user from the page table
                page_table = page_table[1:, :]

            prefill_kwargs = {
                "tokens": prefill_ids,
                "page_table": page_table_user if page_table is not None else None,
                "kv_cache": kv_cache,
                "user_id": 0 if use_batched_prefill else user_id,
                "last_token_idx": last_token_idx,
                "batch_size": padded_batch if use_batched_prefill else 1,
                "num_cached_tokens": num_cached_tokens_list[id] if not use_batched_prefill else 0,
            }

            # Add num_cached_tokens for prefix caching support
            if not use_batched_prefill:
                prefill_kwargs["num_cached_tokens"] = num_cached_tokens_list[id]

            # Save output logits (PCC check / return_logits path)
            tt_out_logits_saved = None
            if save_logits_to_host:
                tt_out_logits_saved = torch.zeros(1, self.model.args.padded_vocab_size)
                prefill_kwargs["tt_out_logits_saved"] = tt_out_logits_saved

            # Disable tracing when prefix caching is active (cached tokens)
            if use_batched_prefill:
                enable_trace_current = enable_trace
            else:
                enable_trace_current = enable_trace and (num_cached_tokens_list[id] == 0)

            if enable_trace_current:
                # For batched prefill, reset to empty list since we use extend()
                if use_batched_prefill and do_device_sampling:
                    self.tt_logits_accumulated_batched = []
                tt_tok = self._easy_trace_prefill(**prefill_kwargs, prefill_seq_len=prefill_seq_len)
            else:
                tt_tok = self.prefill_forward_single_user_text(**prefill_kwargs)

            if not do_device_sampling:
                tt_tok = self.model.process_output_prefill(
                    tt_tok, last_token_idx=last_token_idx, tt_out_logits_saved=tt_out_logits_saved
                )
                if use_batched_prefill:
                    # reverse the reordering of the tokens when empty_slots are not sequential (from vllm)
                    tt_tok_tensor = torch.stack(tt_tok, dim=0)
                    output_toks = tt_tok_tensor[empty_slots].reshape(batch, 1, 1)
                else:
                    output_toks[id] = tt_tok

                if tt_out_logits_all_users is not None and tt_out_logits_saved is not None:
                    tt_out_logits_all_users[id] = tt_out_logits_saved
            else:
                # Process prefill output to get logits (before all-gather) for on-device sampling
                # Returns list of logits in sharded format (same as decode)
                tt_logits_list = self.model.process_output_prefill_logits(tt_tok, last_token_idx=last_token_idx)
                if use_batched_prefill:
                    # Batched prefill: logits list has 32 entries ordered by slot position
                    self.tt_logits_accumulated_batched.extend(tt_logits_list)
                else:
                    # Single user: logits list has 1 entry, copy into persistent buffer
                    ttnn.copy(input_a=tt_logits_list[0], input_b=self.tt_logits_accumulated[user_id])
        # On-device sampling for prefill
        if do_device_sampling:
            padded_batch = 32

            # Use batched list for batched prefill, persistent buffer for non-batched
            logits_source = self.tt_logits_accumulated_batched if use_batched_prefill else self.tt_logits_accumulated

            # Concatenate along slot dimension -> [1, 1, 1[32], vocab_shard]
            tt_logits_batch = ttnn.concat(logits_source, dim=2)
            # Sample using the sampling module
            # Logits are in sharded format (before all-gather), same as decode
            # sampling_params are already padded to 32 by format_sampling_params
            self.model.switch_mode("decode")

            # Setting sampling module up after switch to decode mode
            sampling_params = format_sampling_params(sampling_params, self.model_args.max_batch_size)

            # Reorder sampling params so values sit in their slot positions (except seed).
            def _scatter_params_to_slots(params, slots):
                max_batch = self.model_args.max_batch_size

                def _scatter_list(values):
                    if not isinstance(values, list):
                        return values
                    values = list(values)
                    # Broadcast single-entry lists to match user count
                    if len(values) == 1 and len(slots) > 1:
                        values = values * len(slots)
                    user_vals = values[: len(slots)]
                    filler = values[len(slots)] if len(values) > len(slots) else values[-1]
                    scattered = [filler for _ in range(max_batch)]
                    for val, slot_idx in zip(user_vals, slots):
                        scattered[slot_idx] = val
                    return scattered

                updates = {}
                for f in fields(SamplingParams):
                    if f.name == "seed":
                        # Seeds stay in original order; no reordering to slot indices.
                        updates[f.name] = getattr(params, f.name)
                        continue
                    updates[f.name] = _scatter_list(getattr(params, f.name))
                return replace(params, **updates)

            sampling_params = _scatter_params_to_slots(sampling_params, empty_slots)
            # print("sampling_params_scattered", sampling_params, "empty_slots", empty_slots)
            sampling_module = self.model.sampling

            sampling_module.reset_sampling_params(sampling_params)
            # if prompt_tokens is not None:  # Guard for warmup
            sampling_module.reset_prompt_tokens(prefill_ids)
            sampling_module.reset_output_state()
            sampling_module.seed_manager.reset_seed(sampling_params.seed, empty_slots)
            sampling_module.seed_manager.get_new_values(empty_slots)
            tt_sampled, tt_log_probs = sampling_module.sample(
                tt_logits_batch,
                tt_out_tok=None,
                enable_trace=False,  # Don't trace prefill sampling
            )
            if isinstance(tt_sampled, tuple):
                tt_sampled = tt_sampled[0]
            if isinstance(tt_sampled, list):
                tt_sampled = tt_sampled[0]

            sampled_tokens = ttnn.to_torch(ttnn.get_device_tensors(tt_sampled)[0])

            # sampled_tokens has 32 entries ordered by slot.
            sampled_tensor = sampled_tokens[0, 0, 0, :]  # Shape: [32]
            output_toks = sampled_tensor[empty_slots].reshape(batch, 1, 1)

        if return_logits:
            # TODO: the current solution runs the argmax even if we are returning logits
            # This is inefficient and should be fixed
            # Return logits instead of tokens
            # batch x seq x padded_vocab_size -> batch x seq x vocab_size
            tt_out_logits_all_users = tt_out_logits_all_users[:, :, : self.model.vocab_size]
            return tt_out_logits_all_users

        logger.info(f"[PREFILL_MAIN] Finished prefill for all users up to {batch_seq_len} tokens")
        return output_toks

    def prefill_forward_single_user_text(
        self,
        tokens,  # New tokens to prefill (without the cached tokens), padded by get_padded_prefill_len()
        page_table,  # (32, num_blocks), cached and new pages. All users or just the single user at the given user_id index
        user_id,
        last_token_idx,  # Last token index of the full prompt, including the cached tokens
        kv_cache=None,
        tt_out_logits_saved=None,
        batch_size=1,
        num_cached_tokens=0,  # Number of tokens already cached (for prefix caching)
    ):
        assert num_cached_tokens == 0 or kv_cache is not None, (
            f"kv_cache must be provided for prefix caching when num_cached_tokens > 0: "
            f"num_cached_tokens={num_cached_tokens}"
        )
        seq_len = tokens.shape[-1]
        use_prefix_caching = num_cached_tokens > 0

        # If batch_size is 1, extract the single user's page table row.
        #    Page_table comes from _get_prefill_user_page_table which places user data at row user_id
        #    We extract to (1, num_blocks) so prepare_prefill_inputs_host can use page_table[0, :]
        # If batch size is 32, use the entire page_table.
        if page_table is not None:
            if batch_size == 1:
                page_table_user = page_table[user_id : user_id + 1, :]
            else:
                page_table_user = page_table
        else:
            page_table_user = None

        if use_prefix_caching:
            """
            Prefix caching requires paged attention.
            - page_table must match batch size of inputs (1 for single user)
            - page_table includes both cached and new blocks
            """
            assert page_table_user is not None, "page_table must be provided for prefix caching"
            assert kv_cache is not None, "kv_cache must be provided for prefix caching"
            assert last_token_idx is not None and last_token_idx < seq_len + num_cached_tokens, (
                f"last_token_idx must be provided and less than seq_len + num_cached_tokens: "
                f"last_token_idx={last_token_idx}, seq_len={seq_len}, num_cached_tokens={num_cached_tokens}"
            )

            block_size = get_block_size(kv_cache)
            # Assert that num_cached_tokens is aligned to block_size to ensure correct chunk_page_table calculation
            assert (
                num_cached_tokens % block_size == 0
            ), f"num_cached_tokens ({num_cached_tokens}) must be aligned to block_size ({block_size})."
            # Assert that the end position (num_cached_tokens + seq_len) is also aligned to block_size
            # to ensure the end block calculation using floor division is correct
            assert (num_cached_tokens + seq_len) % block_size == 0, (
                f"End position (num_cached_tokens + seq_len = {num_cached_tokens + seq_len}) must be aligned to "
                f"block_size ({block_size})."
            )
            # Pad page table to match number of blocks in full sequence (including cached)
            num_padding_blocks = num_blocks_in_seq(seq_len + num_cached_tokens, block_size) - page_table_user.shape[1]
            if num_padding_blocks > 0:
                padding = torch.zeros(1, num_padding_blocks, dtype=torch.int32)
                page_table_user_padded = torch.cat([page_table_user, padding], dim=-1)
            else:
                page_table_user_padded = page_table_user

            # For prefix caching, chunk_start_idx is the absolute position where new tokens start
            chunk_start_idx = num_cached_tokens
            chunk_start_block = num_cached_tokens // block_size
            chunk_end_block = (num_cached_tokens + seq_len) // block_size
            chunk_page_table = page_table_user_padded[:, chunk_start_block:chunk_end_block]

            # For prefix caching, get_last_token must be the RELATIVE position within the chunk,
            # not the absolute position in the full sequence. The tensor only contains
            # positions [num_cached_tokens, seq_len), so we need to adjust the index.
            last_token_idx_relative = last_token_idx - num_cached_tokens

            (
                tt_prefill_input,
                tt_user_id,
                tt_page_table,
                tt_chunk_page_table,
                tt_chunk_start_idx,
            ) = self.model.prepare_inputs_prefill(
                tokens,
                user_id=user_id,
                page_table=page_table_user_padded,
                chunk_page_table=chunk_page_table,  # Pass chunk_page_table for conversion to ttnn tensor
                chunk_start_idx=chunk_start_idx,
                batch_size=batch_size,
            )
            full_rot_mats = self.model.get_or_create_prefill_rot_mats()

            tt_toks = self.model.ttnn_prefill_forward(
                x=tt_prefill_input,
                user_id=tt_user_id,
                page_table=tt_page_table,
                chunk_page_table=tt_chunk_page_table,  # Use converted ttnn tensor
                chunk_start_idx=tt_chunk_start_idx,
                start_pos=chunk_start_idx,  # Python int for attention (SDPA path, program config)
                get_last_token=last_token_idx_relative,  # Use RELATIVE index for slicing within chunk
                kv_cache=kv_cache,
                rot_mats=full_rot_mats,
                batch_size=batch_size,
            )
            tt_toks = self.model.process_output_prefill(
                tt_toks,
                last_token_idx=(last_token_idx_relative),
                tt_out_logits_saved=tt_out_logits_saved,
                user_id=user_id,
            )

            return tt_toks
        else:
            # Non-prefix-cached path
            (
                tt_prefill_input,
                tt_user_id,
                tt_page_table,
                tt_chunk_page_table,
                tt_chunk_start_idx,
            ) = self.model.prepare_inputs_prefill(
                tokens,
                user_id=user_id,
                page_table=page_table_user,
                chunk_page_table=None,
                chunk_start_idx=0,
                batch_size=batch_size,
            )
            full_rot_mats = self.model.get_or_create_prefill_rot_mats()
            tt_toks = self.model.ttnn_prefill_forward(
                x=tt_prefill_input,
                user_id=tt_user_id,
                page_table=tt_page_table,
                chunk_page_table=tt_chunk_page_table,
                chunk_start_idx=tt_chunk_start_idx,
                start_pos=0,
                get_last_token=last_token_idx,
                kv_cache=kv_cache,
                rot_mats=full_rot_mats,
                batch_size=batch_size,
            )
            tt_toks = self.model.process_output_prefill(
                tt_toks, last_token_idx=(last_token_idx), tt_out_logits_saved=tt_out_logits_saved, user_id=user_id
            )
            return tt_toks

    def _easy_trace_prefill(
        self,
        tokens,  # New tokens to prefill (without the cached tokens), padded by get_padded_prefill_len()
        last_token_idx,
        prefill_seq_len,
        page_table=None,
        kv_cache=None,
        user_id=0,
        batch_size=1,
        tt_out_logits_saved=None,
        num_cached_tokens=0,  # For prefix caching support
    ):
        """
        Tracing with prefix caching support.
        Trace reuse is valid only for the same
        (prefill_seq_len, num_cached_blocks) which determines start_pos.
        """
        # Extract single user's page table row for batch_size=1
        # page_table comes from _get_prefill_user_page_table which places user data at row user_id
        # We extract to (1, num_blocks) so prepare_prefill_inputs_host can use page_table[0, :]
        if page_table is not None and batch_size == 1:
            page_table = page_table[user_id : user_id + 1, :]

        # Compute prefix caching values
        use_prefix_caching = num_cached_tokens > 0
        chunk_start_idx = num_cached_tokens  # 0 when not prefix caching
        chunk_page_table = None
        block_size = None

        if use_prefix_caching:
            block_size = get_block_size(kv_cache)
            chunk_start_block = num_cached_tokens // block_size
            chunk_end_block = (num_cached_tokens + prefill_seq_len) // block_size
            chunk_page_table = page_table[:, chunk_start_block:chunk_end_block]

            # Pad page_table if needed
            num_padding_blocks = (
                num_blocks_in_seq(prefill_seq_len + num_cached_tokens, block_size) - page_table.shape[1]
            )
            if num_padding_blocks > 0:
                padding = torch.zeros(1, num_padding_blocks, dtype=torch.int32)
                page_table = torch.cat([page_table, padding], dim=-1)

        # Trace key includes num_cached_blocks because page_table shape depends on total blocks
        # (page_table is padded to num_blocks_in_seq(prefill_seq_len + num_cached_tokens, block_size))
        # num_cached_tokens is block-aligned, so we use cached blocks count for the key
        if use_prefix_caching:
            num_cached_blocks = num_cached_tokens // block_size
        else:
            num_cached_blocks = 0
        trace_key = f"{prefill_seq_len}_{batch_size}_{num_cached_blocks}"

        # For prefix caching, the model output has only prefill_seq_len positions (the chunk).
        # get_last_token must be the relative index within the chunk (0..prefill_seq_len-1).
        last_token_idx_for_trace = last_token_idx - num_cached_tokens

        if self.trace_id_prefill[trace_key] is None:
            trace_id, tt_out_trace, *device_inputs = self._capture_trace_prefill(
                tokens,
                last_token_idx_for_trace,  # Relative index for get_last_token (chunk has prefill_seq_len positions)
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                kv_cache=kv_cache,
                user_id=user_id,
                batch_size=batch_size,
                start_pos=chunk_start_idx,  # RoPE start_pos for this trace key
            )
            self.trace_id_prefill[trace_key] = trace_id
            self.trace_inputs_prefill[trace_key] = device_inputs
            self.trace_output_prefill[trace_key] = tt_out_trace
        tt_out_trace = self._prefill_forward_trace_text(
            self.trace_id_prefill[trace_key],
            self.trace_inputs_prefill[trace_key],
            self.trace_output_prefill[trace_key],
            tokens,
            user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            batch_size=batch_size,
            start_pos=chunk_start_idx,  # For position_ids generation
        )
        return tt_out_trace

    def _capture_trace_prefill(
        self,
        tokens,
        last_token_idx,
        user_id,
        page_table=None,
        chunk_page_table=None,  # For prefix caching
        kv_cache=None,
        batch_size=1,
        start_pos=0,  # Absolute start position
    ):
        """
        Captures a trace for the prefill_forward method with prefix caching support.
        Uses full rot mats + chunk_start_idx device tensor; slice is inside the trace.
        """
        # Get host tensors (tokens, user_id, page_table, chunk_page_table, chunk_start_idx)
        host_inputs = self.model.prepare_prefill_inputs_host(
            tokens,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=start_pos,
            batch_size=batch_size,
        )
        tokens_host, user_id_host, tt_page_table_host, tt_chunk_page_table_host, tt_chunk_start_idx_host = host_inputs

        # Copy host tensors to device
        device_inputs = copy_host_to_device(
            (tokens_host, user_id_host, tt_page_table_host, tt_chunk_page_table_host, tt_chunk_start_idx_host),
            mesh_device=self.mesh_device,
        )

        # Transform inputs (no rot_mats computed here)
        transformed_inputs = self.model.transform_prefill_inputs_device(*device_inputs)
        tt_tokens, tt_user_id, tt_page_table, tt_chunk_page_table, tt_chunk_start_idx = transformed_inputs
        full_rot_mats = self.model.get_or_create_prefill_rot_mats()

        # Compile run
        tt_out_trace = self.model.ttnn_prefill_forward(
            x=tt_tokens,
            user_id=tt_user_id,
            page_table=tt_page_table,
            chunk_page_table=tt_chunk_page_table,
            chunk_start_idx=tt_chunk_start_idx,
            start_pos=start_pos,
            kv_cache=kv_cache,
            get_last_token=last_token_idx,
            rot_mats=full_rot_mats,
            batch_size=batch_size,
        )
        ttnn.synchronize_device(self.mesh_device)
        logger.info("Done Compiling Model")

        # Trace capture run
        device_inputs = copy_host_to_device(
            (tokens_host, user_id_host, tt_page_table_host, tt_chunk_page_table_host, tt_chunk_start_idx_host),
            mesh_device=self.mesh_device,
        )
        # Reset CCL indices so the captured trace sees the same state as replay (indices at 0).
        # Otherwise capture runs after compile (indices advanced); replay runs after another
        # trace + process_output_prefill (different indices) and can deadlock on 2048_1_0.
        self.model.tt_ccl.reset_gather_and_buffer_idx()
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        transformed_inputs = self.model.transform_prefill_inputs_device(*device_inputs)
        tt_tokens, tt_user_id, tt_page_table, tt_chunk_page_table, tt_chunk_start_idx = transformed_inputs
        tt_out_trace = self.model.ttnn_prefill_forward(
            x=tt_tokens,
            user_id=tt_user_id,
            page_table=tt_page_table,
            chunk_page_table=tt_chunk_page_table,
            chunk_start_idx=tt_chunk_start_idx,
            start_pos=start_pos,
            kv_cache=kv_cache,
            get_last_token=last_token_idx,
            rot_mats=full_rot_mats,
            batch_size=batch_size,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        logger.info("Done Capturing Prefill Trace")

        return trace_id, tt_out_trace, *device_inputs

    def _prefill_forward_trace_text(
        self,
        trace_id,
        device_inputs,
        tt_out_trace,
        tokens,
        user_id,
        page_table=None,
        chunk_page_table=None,  # For prefix caching
        batch_size=1,
        start_pos=0,  # starting position for this trace key
    ):
        """
        Executes the trace for the prefill_forward method with prefix caching support.
        """
        # Ensure all prior device work is done before running this trace. When warming up
        # multiple keys in sequence (e.g. 4096_1_0 then 2048_1_0), the previous key's
        # process_output_prefill (line_all_gather, etc.) may still be in flight; without
        # this sync, the replayed trace can wait forever at synchronize_device.
        ttnn.synchronize_device(self.mesh_device)
        # Reset CCL indices so every replay starts from the same state as capture (indices at 0).
        # Required when running multiple traces in sequence (e.g. 4096_1_0 then 2048_1_0):
        # process_output_prefill after the first trace advances indices, so the second trace
        # would otherwise see mismatched semaphore/buffer state and can deadlock.
        self.model.tt_ccl.reset_gather_and_buffer_idx()

        # Get host tensors (tokens, user_id, page_table, chunk_page_table, chunk_start_idx)
        host_inputs = self.model.prepare_prefill_inputs_host(
            tokens,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=start_pos,
            batch_size=batch_size,
        )
        tokens_host, user_id_host, page_table_host, chunk_page_table_host, chunk_start_idx_host = host_inputs

        # Copy host tensors into the stored device buffers (same buffers the trace was captured with).
        # Overwritten: tokens, user_id, page_table, chunk_page_table, chunk_start_idx (all five device_inputs).
        # Slice of full rot mats is inside the trace; no ttnn_prefill_forward on replay.
        device_inputs = copy_host_to_device(
            host_tensors=(tokens_host, user_id_host, page_table_host, chunk_page_table_host, chunk_start_idx_host),
            device_tensors=device_inputs,
        )

        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
        # Synchronize before process_output_prefill to prevent CCL state conflicts
        # between trace's CCL operations and process_output_prefill's line_all_gather
        logger.info("Executed trace")
        ttnn.synchronize_device(self.mesh_device)
        return tt_out_trace

    def decode_forward_text(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        async_read=False,
        sampling_params: SamplingParams = None,  # None means returning logits and host sampling.
        reset_inputs=False,  # If false, skip loading inputs, because it's next step of the batch we last had and sampled on device
        tt_out_logits_saved=None,
        is_cur_pos_sharded=False,
        is_page_table_sharded=False,
        reset_batch=False,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
    ):
        if getattr(self, "_disable_decode_tracing", False):
            enable_trace = False

        if sampling_params is None:
            return_logits = True
            reset_inputs = True  # We didn't sample on device, so we need to load inputs.
        else:
            return_logits = False

        if self.prev_page_table is None:
            self.prev_page_table = (
                page_table.clone()
            )  # Make sure we reference a fresh page table, in case it has changed
        if torch.any(self.prev_page_table != page_table).item():
            reset_inputs = True  # doesn't this do what reset_batch does?
            self.prev_page_table = (
                page_table.clone()
            )  # Make sure we reference a fresh page table, in case it has changed

        if self.model.is_decode_setup is False:
            self.model.switch_mode("decode")
            reset_inputs = True  # Last step wasn't decode, so we definitely need to load inputs.

        kv_cache = kv_cache[0]
        decode_kwargs = {
            "current_pos": start_pos,
            "tokens": tokens,
            "page_table": page_table,
            "kv_cache": kv_cache,
            "is_cur_pos_sharded": is_cur_pos_sharded,
            "is_page_table_sharded": is_page_table_sharded,
        }
        self.model.sampling.seed_manager.get_new_values()
        if reset_inputs and sampling_params is not None:
            # If we have new inputs, we need to set up the sampling module again
            sampling_params = format_sampling_params(sampling_params, self.model_args.max_batch_size)

            sampling_module = self.model.sampling
            sampling_module.reset_sampling_params(sampling_params)
            if reset_batch:
                sampling_module.reset_prompt_tokens(prompt_tokens)
                sampling_module.reset_output_state(output_tokens)

        if tt_out_logits_saved is not None:
            decode_kwargs["tt_out_logits_saved"] = tt_out_logits_saved

        if enable_trace:
            tt_tok, tt_log_probs = self._decode_easy_trace_text(
                **decode_kwargs,
                reset_inputs=reset_inputs,
                return_logits=return_logits,
            )
        else:
            tt_tok = self._decode_forward_no_trace_text(
                **decode_kwargs,
                return_logits=return_logits,
            )
            tt_log_probs = None

        if read_from_device:
            # IMPORTANT: If split sampling is enabled, `tt_log_probs` is produced by the sampling
            # module (potentially via its own trace). We must pass it through to the readback path;
            # otherwise `process_output_decode()` will return log_probs=None and host code will fill
            # log_probs with torch.ones(), masking the real values.
            tt_out_for_read = (tt_tok, tt_log_probs) if tt_log_probs is not None else tt_tok
            tt_out = self.read_decode_output(tt_out_for_read, async_read=async_read)
            if async_read:
                return tt_out
            else:
                return self.process_decode_output_host(tt_out, is_tokens=(not return_logits))

        return tt_tok, tt_log_probs

    def _decode_forward_no_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        tt_out_logits_saved=None,
        is_cur_pos_sharded=False,
        is_page_table_sharded=False,
        return_logits=False,
    ):
        """
        Performs text decode step.
        Returns tt_logits on device
        """
        tt_tokens, tt_current_pos, rot_mat_idxs, tt_page_table = self.model.prepare_inputs_decode(
            tokens, current_pos, page_table, is_cur_pos_sharded, is_page_table_sharded
        )
        tt_tok = self.model.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mat_idxs=rot_mat_idxs,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            tt_out_logits_saved=tt_out_logits_saved,
            is_cur_pos_sharded=is_cur_pos_sharded,
            return_logits=return_logits,
            capture_sampling_trace=self.enable_split_sampling,
        )
        # TODO this actually never calls sampling, because we're telling the model we'll do it ourselves.
        # We also never set the sampling module up with the right parameters.
        return tt_tok

    def _capture_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        is_cur_pos_sharded=False,
        is_page_table_sharded=False,
        return_logits=False,
    ):
        """
        Captures a trace for the decode_forward method.
        """
        # Compile run
        self._decode_forward_no_trace_text(
            tokens,
            current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            is_cur_pos_sharded=is_cur_pos_sharded,
            is_page_table_sharded=is_page_table_sharded,
            return_logits=return_logits,
        )
        logger.info("Done Compiling Model")

        # Get inputs ready for trace run
        tokens_tt, current_pos_tt, rope_idxs_tt, page_table_tt = self.model.prepare_inputs_decode(
            tokens, current_pos, page_table, is_cur_pos_sharded, is_page_table_sharded
        )

        # Save the buffer addresses for preallocated tensors
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        tt_out_tok = self.model.ttnn_decode_forward(
            tokens_tt,
            current_pos_tt,
            rope_idxs_tt,
            page_table_tt,
            kv_cache=kv_cache,
            is_cur_pos_sharded=is_cur_pos_sharded,
            return_logits=return_logits,
            capture_sampling_trace=self.enable_split_sampling,
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        return trace_id, tt_out_tok, tokens_tt, current_pos_tt, rope_idxs_tt, page_table_tt

    def _decode_forward_trace_text(
        self,
        trace_id,
        device_inputs,
        tt_out_trace,
        tokens,
        current_pos,
        page_table=None,
    ):
        """
        Executes the trace for the decode_forward method but does not read back outputs.
        """
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
        return tt_out_trace

    def _decode_easy_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        reset_inputs=False,
        is_cur_pos_sharded=False,
        is_page_table_sharded=False,
        return_logits=False,
    ):
        """
        Run decode forward text with tracing
        """
        tokens = tokens.view(-1, 1)
        # The trace is different depending on whether we are returning logits or sampling on device
        if not self.trace_ids_decode[return_logits]:
            trace_id, tt_out_tok, *device_inputs = self._capture_trace_text(
                tokens,
                current_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                is_cur_pos_sharded=is_cur_pos_sharded,
                is_page_table_sharded=is_page_table_sharded,
                return_logits=return_logits,
            )
            self.trace_ids_decode[return_logits] = trace_id
            self.trace_inputs_decode[return_logits] = device_inputs
            self.trace_output_decode[return_logits] = tt_out_tok
        if reset_inputs:
            host_inputs = self.model.prepare_decode_inputs_host(
                tokens, current_pos, page_table, is_cur_pos_sharded, is_page_table_sharded
            )
            shard_specs = self.model.prepare_decode_shard_configs(is_cur_pos_sharded, is_page_table_sharded)
            device_inputs = copy_host_to_device(
                host_tensors=host_inputs,
                device_tensors=self.trace_inputs_decode[return_logits],
                shard_specs=shard_specs,
            )

        trace_tok_rm = self._decode_forward_trace_text(
            self.trace_ids_decode[return_logits],
            self.trace_inputs_decode[return_logits],
            self.trace_output_decode[return_logits],
            tokens,
            current_pos,
            page_table=page_table,
        )

        if self.enable_split_sampling and not return_logits:
            return self.model.sampling.sample(
                logits=trace_tok_rm[0],
                tt_out_tok=self.trace_inputs_decode[return_logits][0],
            )

        return trace_tok_rm

    def read_decode_output(self, tt_out, async_read=True):
        if not async_read:
            tt_log_probs_cpu = None
            if isinstance(tt_out, tuple):
                tt_log_probs = tt_out[1]
                tt_out = tt_out[0]
                if tt_log_probs is not None:
                    tt_log_probs_cpu = tt_log_probs.cpu()

            return tt_out.cpu(), tt_log_probs_cpu

        logits, log_probs, read_event = self.model.process_output_decode(tt_out)
        return (logits, log_probs), [read_event]

    def process_decode_output_host(self, tt_out, is_tokens=True):
        if isinstance(tt_out, tuple):
            tt_log_probs = tt_out[1]
            tt_out = tt_out[0]
            tt_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
            if tt_log_probs is not None:
                tt_log_probs = ttnn.to_torch(ttnn.get_device_tensors(tt_log_probs)[0])
            else:
                tt_log_probs = torch.ones(tt_out.shape)
        else:
            tt_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
            tt_log_probs = torch.ones(tt_out.shape)
        # Check if tensor is distributed across mesh devices (vocab_size // 8 indicates sharding)
        # If so, convert from distributed TT tensor to consolidated torch tensor
        if tt_out.shape[-1] >= self.model.vocab_size // 8:
            ttnn.synchronize_device(self.mesh_device)
            return tt_out[0, 0, :, : self.model.vocab_size].unsqueeze(1), tt_log_probs[0, 0, :, :]

        # If not sharded (it is a sampled token), convert directly from device tensor to torch tensor
        return tt_out[0, 0, 0, :], tt_log_probs[0, 0, 0, :]

    def chat_completion(
        self,
        messages,
        temperature=0.6,
        top_p: float = 0.9,
        max_gen_len=None,
    ):
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model.configuration.max_seq_len:
            max_gen_len = self.model.configuration.max_seq_len - 1

        tokens = []

        stop_reason = None
        for result in self.generate(
            model_input=self.formatter.encode_dialog_prompt(messages, tool_prompt_format=False),
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        ):
            tokens.append(result.token)
            if result.text == "<|eot_id|>":
                stop_reason = StopReason.end_of_turn
            elif result.text == "<|eom_id|>":
                stop_reason = StopReason.end_of_message

        if stop_reason is None:
            stop_reason = StopReason.out_of_tokens

        message = self.formatter.decode_assistant_message(tokens, stop_reason)

        return ChatPrediction(generation=message)

    def text_completion(
        self,
        content: InterleavedTextMedia,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len=None,
    ):
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model.configuration.max_seq_len:
            max_gen_len = self.model.configuration.max_seq_len - 1

        model_input = self.formatter.encode_content(content)

        tokens = []

        for result in self.generate(
            model_input=model_input,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        ):
            tokens.append(result.token)

        generation = self.tokenizer.decode(tokens)

        return CompletionPrediction(generation=generation)

    def _get_prefill_user_page_table(self, page_table, kv_cache, prefill_len, user_id, use_batched_prefill=False):
        # Output shape: (32, num_blocks)
        # Either all 32 users or just the single user at the given user_id index
        # Ensure page_table is not padded with extra blocks for paged_fill_cache to work properly
        block_size = get_block_size(kv_cache)
        num_blocks = num_blocks_in_seq(prefill_len, block_size)
        page_table = page_table[:, :num_blocks]
        if page_table.shape[1] < num_blocks:
            # If page table is too short, pad it with -1
            padding = torch.ones(page_table.shape[0], num_blocks - page_table.shape[1], dtype=torch.int32) * -1
            page_table = torch.cat([page_table, padding], dim=1)
        # Pad page table to 32 users
        padded_page_table = torch.ones(32, page_table.shape[1], dtype=torch.int32) * -1
        if use_batched_prefill:
            for i, user in enumerate(user_id):
                padded_page_table[user, :] = page_table[i, :]
        else:
            padded_page_table[user_id, :] = page_table[0, :]
        return padded_page_table

    def warmup_model_prefill(self, kv_cache, enable_trace, sampling_params) -> None:
        # page_table gets padded properly in prefill_forward_text
        # be sure to pad correctly for non traced sequences in future warmup calls
        page_table = torch.zeros(1, 1, dtype=torch.int32)
        self.warmup_prefill_traces(
            tokens=None,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=None,
            enable_trace=enable_trace,
            sampling_params=None,
            empty_slots=None,
            tt_out_logits_all_users=None,
        )

    ## Destructor (used to delete ttnn trace if exists)

    def __del__(self):
        # Release prefill traces (this class uses trace_id_prefill dict, not trace_id)
        if hasattr(self, "trace_id_prefill"):
            for trace_id in self.trace_id_prefill.values():
                if trace_id is not None:
                    ttnn.release_trace(self.mesh_device, trace_id)
        # Release decode traces (this class uses trace_ids_decode dict, not trace_id_text)
        if hasattr(self, "trace_ids_decode"):
            for trace_id in self.trace_ids_decode.values():
                if trace_id is not None:
                    ttnn.release_trace(self.mesh_device, trace_id)

        self.model.tt_ccl.close()

        if hasattr(super(Generator, self), "__del__"):
            super().__del__()
