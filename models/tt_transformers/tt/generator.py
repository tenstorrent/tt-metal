# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from dataclasses import dataclass, fields
from typing import List

import torch
from loguru import logger

import ttnn
from models.common.llama_models import (
    CompletionMessage,
    StopReason,
    TokenResult,
    create_vision_mask,
    encode_content,
    extract_images_from_messages,
    sample_top_p,
)
from models.common.sampling.generator import format_sampling_params
from models.tt_transformers.tt.common import (
    copy_host_to_device,
    get_block_size,
    get_max_prefill_chunk_size,
    get_padded_prefill_len,
    num_blocks_in_seq,
)


@dataclass(frozen=True)
class SamplingParams:
    """
    Used in Generator decode forward functions for greedy decoding / sampling on device.
    The same data class exists in vLLM at vllm/worker/tt_model_runner.py.
    """

    temperature: float | list[float]
    top_k: int | list[int]
    top_p: float | list[float]
    presence_penalty: float | list[float] = 0.0
    frequency_penalty: float | list[float] = 0.0
    repetition_penalty: float | list[float] = 1.0
    seed: int | list[int] = 0
    enable_log_probs: bool | list[bool] = False


SAMPLING_PARAM_FIELDS = tuple(f.name for f in fields(SamplingParams))


# Split lists into chunks
def split_list(lst, n):
    """Split list into n equal parts"""
    chunk_size = len(lst) // n
    chunks = []
    start = 0
    for i in range(n):
        chunks.append(list(lst[start : start + chunk_size]))  # Convert to list explicitly
        start += chunk_size
    return chunks


def max_prefill_chunk_size_cutoff(sequence_length, max_prefill_chunk_size):
    return sequence_length > max_prefill_chunk_size


class Generator:
    def __init__(self, model, model_args, mesh_device, processor=None, tokenizer=None):
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
        self.processor = processor
        self.tokenizer = tokenizer
        self.data_parallel = len(self.model)
        self.prev_page_table = None
        self.trace_id_prefill = defaultdict(lambda: None)
        self.trace_inputs_prefill = defaultdict(lambda: None)
        self.trace_output_prefill = defaultdict(lambda: None)
        self.trace_ids_decode = defaultdict(lambda: None)  # {device_sampling_bool: {device_id: trace_id}}
        self.trace_inputs_decode = defaultdict(lambda: None)
        self.trace_output_decode = defaultdict(lambda: None)
        self.prefill_traces_warmup = False
        self.already_warmed_up_prefill = False
        # By default, enable split sampling (break the decode trace into two parts: upto logits, then sampling step)
        self.enable_split_sampling = True
        self.mode = None

    # Class-level capabilities (VLLM specific, to be overridden by subclasses)
    model_capabilities = {
        "supports_prefix_caching": True,
    }

    def _chunk_sampling_param(self, values):
        if isinstance(values, List):
            return split_list(values, self.data_parallel)
        return [values] * self.data_parallel

    def _set_sampling_trace_mode(self, enabled: bool):
        for model_instance in self.model:
            sampling_module = getattr(model_instance, "sampling", None)
            if sampling_module is not None:
                sampling_module.enable_internal_trace = enabled

    def warmup_model_prefill(
        self,
        kv_cache,
        enable_trace,
        sampling_params=None,
    ):
        if self.already_warmed_up_prefill:
            return
        self.already_warmed_up_prefill = True

        sequence_lengths_to_warmup = self.model_args[0].get_warmup_prefill_supported_seq_lens()

        for model_id in range(self.data_parallel):
            for supported_length in sequence_lengths_to_warmup:
                # When model_id = 0, we compile all operators for the first time
                # Since operators are compiled, we only need to run sequence lengths that can be traced (each mesh has its own captured traces)
                if model_id != 0 and (
                    supported_length not in self.model_args[0].trace_prefill_supported_seq_lens or not enable_trace
                ):
                    continue

                warmup_tokens = torch.zeros(1, supported_length, dtype=torch.long)
                warmup_prompt_lens = torch.tensor([supported_length], dtype=torch.long)
                warmup_empty_slots = list(range(1))

                logger.info(f"Warming up prefill for sequence length: {supported_length}")

                page_table_warmup = None
                # second check is some tests set the kv_cache to [None] instead of None
                if kv_cache is not None and kv_cache[model_id] is not None:
                    block_size = get_block_size(kv_cache[model_id])
                    num_blocks = num_blocks_in_seq(supported_length, block_size)
                    page_table_warmup = torch.zeros(1, num_blocks, dtype=torch.int32)

                # chunked prefill not supported without paged attention
                if page_table_warmup is None and max_prefill_chunk_size_cutoff(
                    supported_length, self.model_args[0].max_prefill_chunk_size
                ):
                    logger.warning(
                        "Skipping warmup for sequence lengths after: {supported_length} because they are greater than the max prefill chunk size and paged attention is disabled"
                    )
                    break

                self.prefill_forward_text(
                    warmup_tokens,
                    page_table_warmup,
                    kv_cache,
                    warmup_prompt_lens,
                    warmup_empty_slots,
                    enable_trace,
                    model_id,
                )

    def _capture_trace_prefill(
        self,
        prefill_ids,
        page_table=None,
        kv_cache=None,
        model_id=-1,
    ):
        host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, page_table=page_table)
        # These matrices will actually be pointing to the whole cos_matrix and sin_matrix that was allocated on device in the RotarySetup class
        tt_rot_mats_prefill_global = host_inputs[1]
        tt_rot_mats_prefill_local = host_inputs[2]
        host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
        transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
        tt_out_trace = self.model[model_id].ttnn_prefill_forward(
            x=transformed_inputs[0],
            rot_mats_global=tt_rot_mats_prefill_global,
            rot_mats_local=tt_rot_mats_prefill_local,
            page_table=transformed_inputs[1],
            chunk_page_table=transformed_inputs[2],
            kv_cache=kv_cache,
        )
        logger.info("Done Compiling Model")

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.model_args[model_id].mesh_device)
        trace_id = ttnn.begin_trace_capture(self.model_args[model_id].mesh_device, cq_id=0)
        transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
        tt_out_trace = self.model[model_id].ttnn_prefill_forward(
            x=transformed_inputs[0],
            rot_mats_global=tt_rot_mats_prefill_global,
            rot_mats_local=tt_rot_mats_prefill_local,
            page_table=transformed_inputs[1],
            chunk_page_table=transformed_inputs[2],
            kv_cache=kv_cache,
        )
        ttnn.end_trace_capture(self.model_args[model_id].mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Prefill Trace")
        return trace_id, tt_out_trace, *device_inputs

    def _easy_trace_prefill(
        self,
        prefill_ids,
        page_table=None,
        user_id=0,
        last_token_idx=None,
        kv_cache=None,
        model_id=-1,
        prefill_seq_len=None,
        **kwargs,
    ):
        # We are not appending host/device here because we never do device sampling in prefill with TTT
        trace_key = f"{prefill_seq_len}_{model_id}"
        if self.trace_id_prefill[trace_key] is None:
            trace_id, tt_out_trace, *device_inputs = self._capture_trace_prefill(
                prefill_ids,
                page_table=page_table,
                kv_cache=kv_cache,
                model_id=model_id,
            )
            self.trace_id_prefill[trace_key] = trace_id
            self.trace_inputs_prefill[trace_key] = device_inputs
            self.trace_output_prefill[trace_key] = tt_out_trace

        tt_out_trace = self._prefill_forward_trace(
            self.trace_id_prefill[trace_key],
            self.trace_inputs_prefill[trace_key],
            self.trace_output_prefill[trace_key],
            prefill_ids,
            page_table=page_table,
            model_id=model_id,
        )

        return tt_out_trace

    def _prefill_forward_trace(
        self,
        trace_id,
        device_inputs,
        tt_out_trace,
        prefill_ids,
        user_id=0,
        page_table=None,
        model_id=-1,
    ):
        host_inputs = self.model[model_id].prepare_prefill_inputs_trace(prefill_ids, page_table=page_table)
        host_inputs = (host_inputs[0], host_inputs[3], host_inputs[4])

        device_inputs = copy_host_to_device(
            host_inputs, device_tensors=device_inputs, mesh_device=self.model_args[model_id].mesh_device
        )

        ttnn.execute_trace(self.model_args[model_id].mesh_device, trace_id, cq_id=0, blocking=False)

        return tt_out_trace

    # Note: This function is called by vLLM
    def prefill_forward_text(
        self,
        tokens: torch.Tensor,  # All tokens, including the cached ones
        page_table=None,
        kv_cache=None,
        prompt_lens=None,  # Full prompt lengths, including the cached ones
        empty_slots=None,
        enable_trace=True,
        model_id_warmup=None,
        start_pos: list[int] = None,  # Cached prefixes lengths
        **kwargs,
    ):
        self.mode = "prefill"
        if page_table is not None:
            assert isinstance(page_table, torch.Tensor), "page_table mush be torch.Tensor"
        else:
            # Only paged attention is supported for prefill
            enable_trace = False

        # we need this here becuase of tt-metal tests
        self.warmup_model_prefill(kv_cache, enable_trace)

        batch_size, batch_seq_len = tokens.shape
        max_batch_size_per_model = self.model_args[0].max_batch_size

        # Each model expected to run the same model, safe to use 1st vocab size
        output_logits = torch.zeros(batch_size, 1, self.model_args[0].vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch_size)

        if empty_slots is None:
            empty_slots = list(range(batch_size))

        out_list = []
        for idx, user_id in enumerate(empty_slots):
            # if model_id is not None, it means that prefill is called from warmup_prefill
            model_id = user_id // max_batch_size_per_model if model_id_warmup is None else model_id_warmup
            group_user_id = user_id % max_batch_size_per_model if page_table is None else 0
            seq_len = int(prompt_lens[idx])  # Full length of the current prompt
            num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
            last_token_idx = seq_len - 1  # Last token index of the current full prompt, including the cached tokens
            prefill_seq_len = get_padded_prefill_len(
                seq_len - num_cached_tokens
            )  # Without the cached tokens, then padded
            local_kwargs = kwargs.copy()  # Avoid modifying original kwargs

            logger.info(f"Prefilling User {user_id + 1} up to {seq_len} tokens")

            # Extracting data for the current user
            # If page_table is not provided, we keep track of the relative/model user_id through group_user_id
            prefill_ids = torch.cat(
                [
                    tokens[idx : idx + 1, num_cached_tokens:seq_len],  # Select this user, skip the cached tokens
                    torch.zeros(1, prefill_seq_len - (seq_len - num_cached_tokens)).long(),  # Pad
                ],
                dim=-1,
            )

            enable_trace_current_prompt = enable_trace and self.model_args[model_id].can_enable_trace(
                prefill_seq_len, num_cached_tokens
            )

            logger.info(
                f"Prefill seq len: {prefill_seq_len}, max_prefill_chunk_size: {self.model_args[0].max_prefill_chunk_size}, trace: {enable_trace_current_prompt}"
            )

            page_table_user = (
                self._get_prefill_user_page_table(
                    page_table=page_table[idx : idx + 1],  # Slice page table for the current user
                    kv_cache=kv_cache[model_id],
                    prefill_len=seq_len,  # Full length of the current prompt
                    trace_enabled=enable_trace_current_prompt,
                    prefill_seq_len=prefill_seq_len,
                )
                if page_table is not None
                else None
            )
            model_kv_cache = kv_cache[model_id] if kv_cache is not None else None

            # Check if 'pixel_values' exists and index it safely
            if local_kwargs.get("pixel_values", None) is not None:
                local_kwargs["pixel_values"] = local_kwargs["pixel_values"][idx]
                if "image_grid_thw" in local_kwargs:
                    local_kwargs["image_grid_thw"] = local_kwargs["image_grid_thw"][idx]

            if enable_trace_current_prompt:
                logits = self._easy_trace_prefill(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=group_user_id,
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
                    user_id=group_user_id,
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    num_cached_tokens=num_cached_tokens,
                    **local_kwargs,
                )
            if enable_trace_current_prompt:
                # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
                # We need to do this here, because we can't do this part in forward() if we have trace enabled
                # The reason we can't do it in trace is because we can't pass the correct get_last_token to trace
                logits = self.model[model_id].process_logits_after_prefill_trace(logits, last_token_idx)

            # We have to dispatch copy to host to avoid corruption by the next user's prefill
            out_list.append(logits.cpu(blocking=False))

        # Process the logits after all the prefill are done in data parallel mode
        for idx, out in enumerate(out_list):
            seq_len = int(prompt_lens[idx])
            last_token_idx = seq_len - 1
            num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
            last_token_idx_relative = last_token_idx - num_cached_tokens
            user_id = empty_slots[idx]
            model_id = user_id // max_batch_size_per_model if model_id_warmup is None else model_id_warmup

            # Ensure all copying is done
            ttnn.synchronize_device(self.model[model_id].mesh_device)

            # Since we give unpadded_seq_len, only the tile containing the last token is returned
            output_logits[idx] = self.model[model_id].process_output_prefill(
                out, last_token_idx=((last_token_idx_relative) % 32)
            )

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")
        return output_logits

    def prefill_forward_single_user_text(
        self,
        tokens,  # New tokens to prefill (without the cached tokens), padded by get_padded_prefill_len()
        page_table,  # Cached and new pages
        user_id,
        last_token_idx,  # Last token index of the full prompt, including the cached tokens
        kv_cache=None,
        model_id=-1,
        num_cached_tokens: int = 0,
        **kwargs,
    ):
        seq_len = tokens.shape[-1]
        use_chunked_prefill = seq_len > self.model_args[model_id].max_prefill_chunk_size
        use_prefix_caching = num_cached_tokens > 0
        if use_chunked_prefill or use_prefix_caching:
            """
            Chunked prefill requires paged attention. There are some strange constraints which we must meet:
             - page_table, which is used in SDPA, must match batch size of inputs, which is 1. This is because SDPA
             checks that page table batch dim matches input batch dim. Therefore we must slice the page table for the current user.
             - page_table must also have enough entries in each chunk, so it will be padded with zeros if necessary.
             - chunked_page_table is the slice of the page table for the current chunk. This is used by paged_fill_cache
             to keep it otherwise unaware that it is operating on a chunk.
             - due to the above point, we must always set user_id to 0 for chunked prefill.
            """
            assert page_table is not None, "page_table must be provided for chunked prefill"
            assert kv_cache is not None, "kv_cache must be provided for chunked prefill"
            assert last_token_idx is not None and last_token_idx < seq_len + num_cached_tokens, (
                f"last_token_idx must be provided and less than seq_len + num_cached_tokens: "
                f"last_token_idx={last_token_idx}, seq_len={seq_len}, num_cached_tokens={num_cached_tokens}"
            )

            if use_chunked_prefill:
                # If chunked prefill (more than one chunk is needed), we want to use the maximum chunk size.
                chunk_size = get_max_prefill_chunk_size(seq_len, self.model_args[model_id].max_prefill_chunk_size)
            else:
                # Otherwise we only have one chunk.
                chunk_size = seq_len

            last_token_idx_in_seq = last_token_idx - num_cached_tokens  # Excluding the cached tokens
            block_size = get_block_size(kv_cache)
            last_token_idx_in_chunk = last_token_idx_in_seq % chunk_size
            # Calculate which chunk contains the last_token_idx
            last_chunk_start = (last_token_idx_in_seq // chunk_size) * chunk_size
            page_table_user = page_table[user_id : user_id + 1, :]
            # Pad page table to match number of blocks in seq_len
            num_padding_blocks = num_blocks_in_seq(seq_len + num_cached_tokens, block_size) - page_table_user.shape[1]
            page_table_user_padded = torch.cat(
                [page_table_user, torch.zeros(1, num_padding_blocks, dtype=torch.int32)], dim=-1
            )
            CHUNK_USER_ID = 0

            for chunk_start in range(num_cached_tokens, num_cached_tokens + seq_len, chunk_size):
                # These are absolute, i.e. including the cached tokens
                chunk_end = chunk_start + chunk_size
                # These are relative, i.e. excluding the cached tokens
                chunk_start_relative = chunk_start - num_cached_tokens
                chunk_end_relative = chunk_end - num_cached_tokens
                assert chunk_end <= num_cached_tokens + seq_len, (
                    f"chunk_end should be less or equal to "
                    f"num_cached_tokens + seq_len. "
                    f"Got: chunk_end={chunk_end}, "
                    f"num_cached_tokens={num_cached_tokens}, seq_len={seq_len}"
                )

                # Select tokens for the current chunk.
                # Cached tokens were allready excluded (not part of the input),
                # so using relative indexes.
                chunk_tokens = tokens[:, chunk_start_relative:chunk_end_relative]

                # Select pages for the current chunk.
                # Cached pages must be skipped as well,
                # so using absolute indexes.
                chunk_page_table = page_table_user_padded[:, chunk_start // block_size : chunk_end // block_size]

                (
                    chunk_prefill_input,
                    chunk_rot_mats_global_prefill,
                    chunk_rot_mats_local_prefill,
                    page_table_tt,
                    chunk_page_table_tt,
                ) = self.model[model_id].prepare_inputs_prefill(
                    chunk_tokens,
                    start_pos=chunk_start,
                    page_table=page_table_user_padded,
                    chunk_page_table=chunk_page_table,
                    last_token_idx=last_token_idx,
                    **kwargs,
                )
                tt_logits = self.model[model_id].ttnn_prefill_forward(
                    chunk_prefill_input,
                    rot_mats_global=chunk_rot_mats_global_prefill,
                    rot_mats_local=chunk_rot_mats_local_prefill,
                    user_id=CHUNK_USER_ID,
                    page_table=page_table_tt,
                    chunk_page_table=chunk_page_table_tt,
                    chunk_start_idx=chunk_start,
                    get_last_token=(last_token_idx_in_chunk // 32) * 32,
                    kv_cache=kv_cache,
                )

                if chunk_start_relative == last_chunk_start:
                    return tt_logits
                else:
                    del tt_logits
        else:
            (
                prefill_input,
                rot_mats_global_prefill,
                rot_mats_local_prefill,
                page_table_tt,
                _,
            ) = self.model[model_id].prepare_inputs_prefill(
                tokens,
                page_table=page_table,
                last_token_idx=last_token_idx,
                **kwargs,
            )

            tt_logits = self.model[model_id].ttnn_prefill_forward(
                prefill_input,
                rot_mats_global=rot_mats_global_prefill,
                rot_mats_local=rot_mats_local_prefill,
                user_id=user_id,
                page_table=page_table_tt,
                get_last_token=(last_token_idx // 32) * 32,
                kv_cache=kv_cache,
            )
            return tt_logits

    # Note: This function is called by vLLM
    def decode_forward_text(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params: SamplingParams = None,  # Should be None if not greedy decoding / sampling on device.
        reset_batch=False,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
    ):
        mode_switched = False
        if self.mode != "decode":
            self.mode = "decode"
            mode_switched = True
        sampling_on_device = sampling_params is not None
        split_sampling_enabled = bool(self.enable_split_sampling and sampling_on_device)
        self._set_sampling_trace_mode(split_sampling_enabled)

        B = tokens.shape[0]

        tokens = torch.chunk(tokens, self.data_parallel, 0)
        start_pos = torch.chunk(start_pos, self.data_parallel, 0)
        page_table = torch.chunk(page_table, self.data_parallel, 0) if page_table is not None else None
        sampling_params_list = None
        if sampling_params is not None:
            # Fall back to dataclass defaults when optional fields are omitted
            chunked_fields = {}
            for field in SAMPLING_PARAM_FIELDS:
                try:
                    val = getattr(sampling_params, field)
                except AttributeError:
                    if hasattr(SamplingParams, field):
                        val = getattr(SamplingParams, field)
                    else:
                        raise
                chunked_fields[field] = self._chunk_sampling_param(val)
            prompt_chunks = (
                torch.chunk(prompt_tokens, self.data_parallel, 0)
                if prompt_tokens is not None
                else [None] * self.data_parallel
            )
            output_chunks = (
                torch.chunk(output_tokens, self.data_parallel, 0)
                if output_tokens is not None
                else [None] * self.data_parallel
            )
            sampling_params_list = [
                SamplingParams(**{field: chunked_fields[field][i] for field in SAMPLING_PARAM_FIELDS})
                for i in range(self.data_parallel)
            ]
            for i in range(self.data_parallel):
                formatted_params = format_sampling_params(
                    sampling_params_list[i], 32
                )  # Sampling needs params padded to 32 regardless of batch_size
                sampling_module = getattr(self.model[i], "sampling", None)
                assert sampling_module is not None, "Sampling module not found in model for sampling on device."
                sampling_module.reset_sampling_params(formatted_params)
                if reset_batch:
                    sampling_module.reset_seed(formatted_params.seed)
                    sampling_module.reset_prompt_tokens(prompt_chunks[i])
                    sampling_module.reset_output_state(output_chunks[i])

        decode_kwargs = {
            "current_pos": start_pos,
            "tokens": tokens,
            "page_table": page_table,
            "kv_cache": kv_cache,
            "sampling_on_device": sampling_on_device,
        }
        if enable_trace:
            tt_decode_output = self._decode_forward_trace_text(**decode_kwargs, reset_batch=mode_switched)
        else:
            tt_decode_output = self._decode_forward_no_trace_text(**decode_kwargs)

        if read_from_device:
            to_host = self.read_decode_output(tt_decode_output)
            return self.process_decode_output_host(to_host, is_tokens=(sampling_params is not None))
        return tt_decode_output

    def _decode_forward_no_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        sampling_on_device=False,
    ):
        """
        Performs text decode step.
        Returns tt_logits on device
        """
        tt_output = []
        tt_tokens = []
        tt_current_pos = []
        tt_rot_mat_idxs = []
        tt_page_table = []
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None
            model_i = self.model[i]
            (
                tt_tokens_i,
                tt_current_pos_i,
                tt_rot_mat_idxs_i,
                tt_page_table_i,
            ) = model_i.prepare_inputs_decode(tokens[i], current_pos[i], user_page_table)
            tt_tokens.append(tt_tokens_i)
            tt_current_pos.append(tt_current_pos_i)
            tt_rot_mat_idxs.append(tt_rot_mat_idxs_i)
            tt_page_table.append(tt_page_table_i)

        for i in range(self.data_parallel):
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            tt_logits_i, tt_log_probs_i = self.model[i].ttnn_decode_forward(
                tt_tokens[i],
                tt_current_pos[i],
                rot_mat_idxs=tt_rot_mat_idxs[i],
                page_table=tt_page_table[i],
                kv_cache=user_kv_cache,
                sampling_on_device=sampling_on_device,
            )
            tt_output.append((tt_logits_i, tt_log_probs_i))

        return tt_output

    def _capture_decode_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        sampling_on_device=False,
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
            sampling_on_device=sampling_on_device,
        )
        logger.info("Done Compiling Model")

        # Get inputs ready for trace run
        device_inputs = []
        tt_out_trace = []
        trace_ids = {}
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None

            host_inputs = self.model[i].prepare_decode_inputs_host(
                tokens[i], current_pos[i], page_table=user_page_table
            )

            device_inputs_i = copy_host_to_device(host_inputs, mesh_device=self.model_args[i].mesh_device)
            device_inputs.append(device_inputs_i)

        for i in range(self.data_parallel):
            sampling_module = getattr(self.model[i], "sampling", None)
            split_enabled = (
                sampling_on_device
                and sampling_module is not None
                and getattr(sampling_module, "enable_internal_trace", False)
            )
            trace_id = ttnn.begin_trace_capture(self.model_args[i].mesh_device, cq_id=0)
            trace_ids[i] = trace_id
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            tt_out_trace.append(
                self.model[i].ttnn_decode_forward(
                    *device_inputs[i],
                    kv_cache=user_kv_cache,
                    sampling_on_device=sampling_on_device,
                    capture_sampling_trace=split_enabled,
                )
            )
            ttnn.end_trace_capture(self.model_args[i].mesh_device, trace_id, cq_id=0)

            if split_enabled:
                sampling_module.capture_trace(logits=tt_out_trace[i], tt_out_tok=device_inputs[i][0])
        logger.info("Done Capturing Decode Trace")

        return trace_ids, tt_out_trace, *device_inputs

    def _decode_forward_trace_text(
        self, tokens, current_pos, page_table=None, kv_cache=None, sampling_on_device=False, reset_batch=False
    ):
        """
        Run decode forward text with tracing
        """
        # The trace is different depending on whether we are doing device sampling or not
        if not self.trace_ids_decode[sampling_on_device]:
            trace_ids, tt_out_trace, *device_inputs = self._capture_decode_trace_text(
                tokens, current_pos, page_table=page_table, kv_cache=kv_cache, sampling_on_device=sampling_on_device
            )
            self.trace_ids_decode[sampling_on_device] = trace_ids
            self.trace_inputs_decode[sampling_on_device] = device_inputs
            self.trace_output_decode[sampling_on_device] = tt_out_trace

        # reset inputs when mode switches from prefill to decode
        reset_inputs = reset_batch or not sampling_on_device
        if self.prev_page_table is None or any(
            not torch.equal(prev, curr) for prev, curr in zip(self.prev_page_table, page_table)
        ):
            # If the page table has changed, it means additional pages have been added or inputs are shuffled
            reset_inputs = True
            if page_table is not None:
                self.prev_page_table = tuple(pt.clone() for pt in page_table)

        if reset_inputs:
            for i in range(self.data_parallel):
                user_page_table = page_table[i] if page_table is not None else None
                host_inputs_i = self.model[i].prepare_decode_inputs_host(tokens[i], current_pos[i], user_page_table)

                copy_host_to_device(
                    host_tensors=host_inputs_i,
                    device_tensors=self.trace_inputs_decode[sampling_on_device][i],
                )
        for i, trace_id in self.trace_ids_decode[sampling_on_device].items():
            ttnn.execute_trace(self.model_args[i].mesh_device, trace_id, cq_id=0, blocking=False)
        outputs = self.trace_output_decode[sampling_on_device]
        if sampling_on_device:
            new_outputs = []
            for i in range(self.data_parallel):
                sampling_module = getattr(self.model[i], "sampling", None)
                if sampling_module is None or not getattr(sampling_module, "enable_internal_trace", False):
                    new_outputs.append(outputs[i])
                    continue

                new_outputs.append(
                    sampling_module.sample(
                        logits=outputs[i],
                        tt_out_tok=self.trace_inputs_decode[sampling_on_device][i][0],
                    )
                )
            return new_outputs
        return outputs

    def _prefill_forward_single_user(
        self,
        vision_images,
        vision_mask,
        tokens,
        xattn_caches,
        user_id,
        total_len,
        prefill_len,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        model_id=-1,
    ):
        """
        Performs vision encode step then text prefill.
        Returns (xattn_caches, cross_attention_masks, full_text_row_masked_out_mask, logits)
        """
        B = tokens.shape[0]
        last_token_idx = prefill_len - 1

        text_only_inference = vision_images is None
        if not text_only_inference:
            (
                vision_tokens,
                prefill_cross_attention_masks,
                prefill_full_text_row_masked_out_mask,
                decode_cross_attention_masks,
                decode_full_text_row_masked_out_mask,
            ) = self.model[model_id].compute_vision_tokens_masks(
                batch_images=[vision_images],
                batch_masks=[vision_mask],
                total_len=total_len,
                prefill_len=prefill_len,
            )

            if cross_page_table is not None:
                num_vision_tokens = vision_tokens.shape[2]
                cross_page_table = self._get_prefill_user_page_table(cross_page_table, kv_cache, num_vision_tokens)
        else:
            (
                vision_tokens,
                prefill_cross_attention_masks,
                prefill_full_text_row_masked_out_mask,
                decode_cross_attention_masks,
                decode_full_text_row_masked_out_mask,
            ) = (None, None, None, None, None)

        if page_table is not None:
            page_table = self._get_prefill_user_page_table(page_table, kv_cache, prefill_len)

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            rot_mats,
            tt_page_table,
            tt_cross_page_table,
        ) = self.model[model_id].prepare_inputs_prefill(
            tokens,
            prefill_cross_attention_masks,
            prefill_full_text_row_masked_out_mask,
            prefill_len=prefill_len,
            page_table=page_table,
            cross_page_table=cross_page_table,
            text_only_inference=text_only_inference,
        )

        tt_logits = self.model[model_id].ttnn_prefill_forward(
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            xattn_caches,
            rot_mats,
            user_id,
            vision_tokens,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            get_last_token=(last_token_idx // 32) * 32,
            cross_page_table=tt_cross_page_table,
            text_only_inference=text_only_inference,
        )

        del tt_page_table
        del tt_cross_page_table

        return (
            xattn_caches,
            prefill_cross_attention_masks,
            prefill_full_text_row_masked_out_mask,
            decode_cross_attention_masks,
            decode_full_text_row_masked_out_mask,
            tt_logits,
        )

    # Note: This function is called by vLLM
    def prefill_forward(
        self,
        vision_images,
        vision_masks,
        tokens,
        xattn_caches,
        total_lens,
        prompt_lens,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        empty_slots=None,
        **kwargs,
    ):
        if not self.model_args[0].is_llama_vision():
            logits = self.prefill_forward_text(
                tokens,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                pixel_values=vision_images,
                **kwargs,
            )

            return logits, None, None, None, None

        else:
            (
                output_logits,
                prefill_output_xattn_masks,
                prefill_output_full_text_row_masked_out_masks,
                decode_output_xattn_masks,
                decode_output_full_text_row_masked_out_masks,
            ) = self.prefill_forward_llama_vision(
                vision_images,
                vision_masks,
                tokens,
                xattn_caches,
                total_lens,
                prompt_lens,
                page_table=page_table,
                kv_cache=kv_cache,
                cross_page_table=cross_page_table,
                empty_slots=empty_slots,
            )

            return (
                output_logits,
                prefill_output_xattn_masks,
                prefill_output_full_text_row_masked_out_masks,
                decode_output_xattn_masks,
                decode_output_full_text_row_masked_out_masks,
            )

    # Note: This function is called by vLLM
    def prefill_forward_llama_vision(
        self,
        vision_images,
        vision_masks,
        tokens: torch.Tensor,
        xattn_caches,
        total_lens,
        prompt_lens,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        empty_slots=None,
    ):
        """
        Batched version of _prefill_forward_single_user for vision model.
        """
        if page_table is not None:
            assert isinstance(page_table, torch.Tensor), "page_table mush be torch.Tensor"
        if cross_page_table is not None:
            assert isinstance(cross_page_table, torch.Tensor), "cross_page_table mush be torch.Tensor"

        batch_size, batch_seq_len = tokens.shape
        max_batch_size_per_model = self.model_args[0].max_batch_size

        output_logits = torch.zeros(batch_size, 1, self.model_args[0].vocab_size)

        out_list = []
        prefill_output_xattn_masks = []
        prefill_output_full_text_row_masked_out_masks = []
        decode_output_xattn_masks = []
        decode_output_full_text_row_masked_out_masks = []

        if empty_slots is None:
            empty_slots = list(range(batch_size))

        for idx, user_id in enumerate(empty_slots):
            model_id = user_id // max_batch_size_per_model
            group_user_id = user_id % max_batch_size_per_model if page_table is None else 0
            seq_len = int(prompt_lens[idx])

            logger.info(f"Prefilling User {user_id + 1} up to {seq_len} tokens")

            user_page_table = page_table[idx : idx + 1] if page_table is not None else None
            user_cross_page_table = cross_page_table[idx : idx + 1] if kv_cache is not None else None
            model_kv_cache = kv_cache[model_id] if kv_cache is not None else None
            model_xattn_cache = xattn_caches[model_id] if xattn_caches is not None else None

            (
                model_xattn_cache,
                prefill_cross_attention_masks,
                prefill_full_text_row_masked_out_mask,
                decode_cross_attention_masks,
                decode_full_text_row_masked_out_mask,
                logits,
            ) = self._prefill_forward_single_user(
                vision_images=vision_images[idx],
                vision_mask=vision_masks[idx],
                tokens=tokens[idx : idx + 1, :seq_len],  # Keep batch dimension
                xattn_caches=model_xattn_cache,
                user_id=group_user_id,
                total_len=total_lens[idx],
                prefill_len=seq_len,
                page_table=user_page_table,
                kv_cache=model_kv_cache,
                cross_page_table=user_cross_page_table,
                model_id=model_id,
            )

            if xattn_caches is not None:
                xattn_caches[model_id] = model_xattn_cache

            out_list.append(logits)
            prefill_output_xattn_masks.append(prefill_cross_attention_masks)
            prefill_output_full_text_row_masked_out_masks.append(prefill_full_text_row_masked_out_mask)
            decode_output_xattn_masks.append(decode_cross_attention_masks)
            decode_output_full_text_row_masked_out_masks.append(decode_full_text_row_masked_out_mask)

        # We gather prefill output at the end of prefill to reduce unnecessary device sync
        for idx, user_id in enumerate(empty_slots):
            model_id = user_id // max_batch_size_per_model

            last_token_idx = prompt_lens[idx] - 1
            output_logits[idx] = self.model[model_id].process_output_prefill(
                out_list[idx].cpu(), 1, last_token_idx=(last_token_idx % 32)
            )

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")

        return (
            output_logits,
            prefill_output_xattn_masks,
            prefill_output_full_text_row_masked_out_masks,
            decode_output_xattn_masks,
            decode_output_full_text_row_masked_out_masks,
        )

    # Note: This function is called by vLLM
    def decode_forward_llama_vision(
        self,
        start_pos,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        xattn_caches=None,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        enable_trace=True,
        read_from_device=True,
    ):
        B = tokens.shape[0]
        data_parallel = min(B, self.data_parallel)
        batch_per_device = B // data_parallel
        tokens = torch.chunk(tokens, self.data_parallel, 0)
        start_pos = torch.chunk(start_pos, self.data_parallel, 0)
        prefill_cross_attention_masks = [
            prefill_cross_attention_masks[i * batch_per_device : (i + 1) * batch_per_device]
            for i in range(data_parallel)
        ]
        prefill_full_text_row_masked_out_mask = [
            prefill_full_text_row_masked_out_mask[i * batch_per_device : (i + 1) * batch_per_device]
            for i in range(data_parallel)
        ]
        decode_cross_attention_masks = [
            decode_cross_attention_masks[i * batch_per_device : (i + 1) * batch_per_device]
            for i in range(data_parallel)
        ]
        decode_full_text_row_masked_out_mask = [
            decode_full_text_row_masked_out_mask[i * batch_per_device : (i + 1) * batch_per_device]
            for i in range(data_parallel)
        ]
        page_table = torch.chunk(page_table, self.data_parallel, 0) if page_table is not None else None
        cross_page_table = (
            torch.chunk(cross_page_table, self.data_parallel, 0) if cross_page_table is not None else None
        )

        decode_kwargs = {
            "position_id": start_pos,
            "tokens": tokens,
            "prefill_cross_attention_masks": prefill_cross_attention_masks,
            "prefill_full_text_row_masked_out_mask": prefill_full_text_row_masked_out_mask,
            "decode_cross_attention_masks": decode_cross_attention_masks,
            "decode_full_text_row_masked_out_mask": decode_full_text_row_masked_out_mask,
            "xattn_caches": xattn_caches,
            "page_table": page_table,
            "kv_cache": kv_cache,
            "cross_page_table": cross_page_table,
        }
        if enable_trace:
            tt_logits = self._easy_trace(**decode_kwargs)
        else:
            tt_logits = self._decode_forward_no_trace(**decode_kwargs)

        if read_from_device:
            to_host = self.read_decode_output(tt_logits)
            return self.process_decode_output_host(to_host)
        else:
            return tt_logits

    def decode_forward(
        self,
        start_pos,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        xattn_caches=None,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        enable_trace=True,
        read_from_device=True,
    ):
        if not self.model_args[0].is_llama_vision():
            output = self.decode_forward_text(
                tokens,
                start_pos,
                enable_trace=enable_trace,
                page_table=page_table,
                kv_cache=kv_cache,
            )
        else:
            output = self.decode_forward_llama_vision(
                start_pos,
                tokens,
                prefill_cross_attention_masks,
                prefill_full_text_row_masked_out_mask,
                decode_cross_attention_masks,
                decode_full_text_row_masked_out_mask,
                xattn_caches,
                page_table,
                kv_cache,
                cross_page_table,
                enable_trace,
                read_from_device,
            )
        # skip returning log-probs
        if isinstance(output, tuple):
            return output[0]
        else:
            return output

    # Note: This function is called by vLLM
    def read_decode_output(self, tt_out, async_read=False):
        """
        Input tt_out is list of tuples of (tt_out_tok, tt_log_probs)

        """
        if not async_read:
            # output is a tuple of (tt_out_tok, tt_log_probs)
            if isinstance(tt_out[0], tuple):
                return [
                    (out[0].cpu(), out[1].cpu() if out[1] is not None else None)  # logits should never be None
                    for out in tt_out
                ]
            elif isinstance(tt_out[0], ttnn.Tensor):
                return [out.cpu() for out in tt_out]

        host_outputs = []
        read_events = []
        for i in range(self.data_parallel):
            if isinstance(tt_out[i], tuple):
                outputs = (tt_out[i][0].cpu(blocking=False), tt_out[i][1].cpu(blocking=False))  # logits  # log-probs
                host_outputs.append(outputs)
            elif isinstance(tt_out[i], ttnn.Tensor):
                outputs = tt_out[i].cpu(blocking=False)
                host_outputs.append(outputs)

            read_events.append(ttnn.record_event(self.model[i].mesh_device, 0))

        return host_outputs, read_events

    # Note: This function is called by vLLM
    def process_decode_output_host(self, tt_out, is_tokens=False):
        """
        Converts the input ttnn host tensors to a torch tensor.
        The input can be logits (if is_tokens=False) or tokens (if is_tokens=True).
        """
        max_batch_size_per_model = self.model_args[0].max_batch_size

        logits = []
        log_probs = []
        for i in range(self.data_parallel):
            if isinstance(tt_out[i], tuple):
                logits_i = self.model[i].process_output_decode(
                    tt_out[i][0], max_batch_size_per_model, S=1, is_tokens=is_tokens
                )
                log_probs_i = (
                    self.model[i].process_output_decode(
                        tt_out[i][1], max_batch_size_per_model, S=1, is_tokens=is_tokens, is_log_probs=True
                    )
                    if tt_out[i][1] is not None
                    else torch.ones(logits_i.shape)
                )
                logits.append(logits_i)
                log_probs.append(log_probs_i)
            elif isinstance(tt_out[i], ttnn.Tensor):
                logits_i = self.model[i].process_output_decode(
                    tt_out[i], max_batch_size_per_model, S=1, is_tokens=is_tokens
                )
                logits.append(logits_i)
                # add dummy tensor for log_probs
                log_probs.append(torch.ones(logits_i.shape))
            else:
                raise ValueError(f"Invalid type of tt_out: {type(tt_out[i])}")
        return (torch.cat(logits, 0), torch.cat(log_probs, 0))

    def _decode_forward_no_trace(
        self,
        position_id,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        xattn_caches=None,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
    ):
        """
        Performs text decode step.
        Returns tt_logits on device
        """

        # forward_decode should be traced callable
        # decorator does compilation, capture, execute
        tt_h = []
        tt_xattn_mask = []
        tt_full_text_mask_expand_1NSH = []
        tt_full_text_mask_expand_11SD = []
        tt_position_id = []
        tt_rot_mats = []
        tt_page_table = []
        tt_cross_page_table = []

        for i in range(self.data_parallel):
            B, S = tokens[i].shape
            assert S == 1

            user_page_table = page_table[i] if page_table is not None else None
            user_cross_page_table = cross_page_table[i] if cross_page_table is not None else None
            (
                tt_h_i,
                tt_xattn_mask_i,
                tt_full_text_mask_expand_1NSH_i,
                tt_full_text_mask_expand_11SD_i,
                tt_position_id_i,
                tt_rot_mats_i,
                tt_page_table_i,
                tt_cross_page_table_i,
            ) = self.model[i].prepare_inputs_decode(
                tokens[i],
                prefill_cross_attention_masks[i],
                prefill_full_text_row_masked_out_mask[i],
                decode_cross_attention_masks[i],
                decode_full_text_row_masked_out_mask[i],
                position_id=position_id[i],
                page_table=user_page_table,
                cross_page_table=user_cross_page_table,
            )

            tt_h.append(tt_h_i)
            tt_xattn_mask.append(tt_xattn_mask_i)
            tt_full_text_mask_expand_1NSH.append(tt_full_text_mask_expand_1NSH_i)
            tt_full_text_mask_expand_11SD.append(tt_full_text_mask_expand_11SD_i)
            tt_position_id.append(tt_position_id_i)
            tt_rot_mats.append(tt_rot_mats_i)
            tt_page_table.append(tt_page_table_i)
            tt_cross_page_table.append(tt_cross_page_table_i)

        tt_logits = []
        tt_log_probs = []
        for i in range(self.data_parallel):
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            xattn_cache = xattn_caches[i] if xattn_caches is not None else None
            tt_logits_i, tt_log_probs_i = self.model[i].ttnn_decode_forward(
                tt_h[i],
                tt_xattn_mask[i],
                tt_full_text_mask_expand_1NSH[i],
                tt_full_text_mask_expand_11SD[i],
                xattn_cache,
                tt_position_id[i],
                tt_rot_mats[i],
                page_table=tt_page_table[i],
                kv_cache=user_kv_cache,
                cross_page_table=tt_cross_page_table[i],
            )
            tt_logits.append(tt_logits_i)
            tt_log_probs.append(tt_log_probs_i)

        return tt_logits, tt_log_probs

    def _capture_trace(
        self,
        position_id,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        xattn_caches,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
    ):
        """
        Captures a trace for the decode_forward method.
        """
        tt_h = []
        tt_xattn_mask = []
        tt_full_text_mask_expand_1NSH = []
        tt_full_text_mask_expand_11SD = []
        tt_position_id = []
        tt_rot_mats = []
        tt_page_table = []
        tt_cross_page_table = []
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None
            user_cross_page_table = cross_page_table[i] if cross_page_table is not None else None
            (
                tt_h_i,
                tt_xattn_mask_i,
                tt_full_text_mask_expand_1NSH_i,
                tt_full_text_mask_expand_11SD_i,
                tt_position_id_i,
                tt_rot_mats_i,
                tt_page_table_i,
                tt_cross_page_table_i,
            ) = self.model[i].prepare_inputs_decode(
                tokens[i],
                prefill_cross_attention_masks[i],
                prefill_full_text_row_masked_out_mask[i],
                decode_cross_attention_masks[i],
                decode_full_text_row_masked_out_mask[i],
                position_id=position_id[i],
                page_table=user_page_table,
                cross_page_table=user_cross_page_table,
            )

            tt_h.append(tt_h_i)
            tt_xattn_mask.append(tt_xattn_mask_i)
            tt_full_text_mask_expand_1NSH.append(tt_full_text_mask_expand_1NSH_i)
            tt_full_text_mask_expand_11SD.append(tt_full_text_mask_expand_11SD_i)
            tt_position_id.append(tt_position_id_i)
            tt_rot_mats.append(tt_rot_mats_i)
            tt_page_table.append(tt_page_table_i)
            tt_cross_page_table.append(tt_cross_page_table_i)

        # Compile run
        for i in range(self.data_parallel):
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            xattn_cache = xattn_caches[i] if xattn_caches is not None else None
            # tt_logits_rm and tt_log_probs_rm unused later, no need to make a list
            tt_logits_rm, tt_log_probs_rm = self.model[i].ttnn_decode_forward(
                tt_h[i],
                tt_xattn_mask[i],
                tt_full_text_mask_expand_1NSH[i],
                tt_full_text_mask_expand_11SD[i],
                xattn_cache,
                tt_position_id[i],
                tt_rot_mats[i],
                page_table=tt_page_table[i],
                kv_cache=user_kv_cache,
                cross_page_table=tt_cross_page_table[i],
            )
        logger.info("Done Compiling Model")

        # Get inputs ready for trace run
        tt_h = []
        tt_xattn_mask = []
        tt_full_text_mask_expand_1NSH = []
        tt_full_text_mask_expand_11SD = []
        tt_position_id = []
        tt_rope_id = []
        tt_page_table = []
        tt_cross_page_table = []
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None
            user_cross_page_table = cross_page_table[i] if cross_page_table is not None else None
            (
                tt_h_i,
                tt_xattn_mask_i,
                tt_full_text_mask_expand_1NSH_i,
                tt_full_text_mask_expand_11SD_i,
                tt_position_id_i,
                tt_rope_id_i,
                tt_page_table_i,
                tt_cross_page_table_i,
            ) = self.model[i].prepare_decode_inputs_host(
                tokens[i],
                prefill_cross_attention_masks[i],
                prefill_full_text_row_masked_out_mask[i],
                decode_cross_attention_masks[i],
                decode_full_text_row_masked_out_mask[i],
                position_id[i],
                page_table=user_page_table,
                cross_page_table=user_cross_page_table,
            )

            (
                tt_h_i,
                tt_xattn_mask_i,
                tt_full_text_mask_expand_1NSH_i,
                tt_full_text_mask_expand_11SD_i,
                tt_position_id_i,
                tt_rope_id_i,
                tt_page_table_i,
                tt_cross_page_table_i,
            ) = copy_host_to_device(
                (
                    tt_h_i,
                    tt_xattn_mask_i,
                    tt_full_text_mask_expand_1NSH_i,
                    tt_full_text_mask_expand_11SD_i,
                    tt_position_id_i,
                    tt_rope_id_i,
                    tt_page_table_i,
                    tt_cross_page_table_i,
                ),
                mesh_device=self.model_args[i].mesh_device,
            )

            tt_h.append(tt_h_i)
            tt_xattn_mask.append(tt_xattn_mask_i)
            tt_full_text_mask_expand_1NSH.append(tt_full_text_mask_expand_1NSH_i)
            tt_full_text_mask_expand_11SD.append(tt_full_text_mask_expand_11SD_i)
            tt_position_id.append(tt_position_id_i)
            tt_rope_id.append(tt_rope_id_i)
            tt_page_table.append(tt_page_table_i)
            tt_cross_page_table.append(tt_cross_page_table_i)

        tt_h_trace_input = tt_h

        tt_logits_rm = []
        tt_log_probs_rm = []
        trace_ids = {}
        # Do on-device transformations of inputs before forward
        for i in range(self.data_parallel):
            trace_id = ttnn.begin_trace_capture(self.model_args[i].mesh_device, cq_id=0)
            trace_ids[i] = trace_id
            B = tokens[i].shape[0]
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            xattn_cache = xattn_caches[i] if xattn_caches is not None else None
            (
                tt_h_transform,
                tt_rot_mats,
                tt_xattn_mask_transform,
                tt_full_text_mask_expand_1NSH_transform,
                tt_full_text_mask_expand_11SD_transform,
            ) = self.model[i].transform_decode_inputs_device(
                tt_h[i],
                tt_rope_id[i],
                tt_xattn_mask[i],
                tt_full_text_mask_expand_1NSH[i],
                tt_full_text_mask_expand_11SD[i],
                B=B,
            )

            tt_logits_rm_i, tt_log_probs_rm_i = self.model[i].ttnn_decode_forward(
                tt_h_transform,
                tt_xattn_mask_transform,
                tt_full_text_mask_expand_1NSH_transform,
                tt_full_text_mask_expand_11SD_transform,
                xattn_cache,
                tt_position_id[i],
                tt_rot_mats,
                page_table=tt_page_table[i],
                kv_cache=user_kv_cache,
                cross_page_table=tt_cross_page_table[i],
            )
            tt_logits_rm.append(tt_logits_rm_i)
            tt_log_probs_rm.append(tt_log_probs_rm_i)
            ttnn.end_trace_capture(self.model_args[i].mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Decode Trace")

        return (
            trace_ids,
            tt_logits_rm,
            tt_log_probs_rm,
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            tt_position_id,
            tt_rope_id,
            tt_page_table,
            tt_cross_page_table,
        )

    def _decode_forward_trace(
        self,
        position_id,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        page_table,
        cross_page_table,
        trace_ids,
        trace_logits_rm,
        trace_h,
        trace_xattn_mask,
        trace_full_text_mask_expand_1NSH,
        trace_full_text_mask_expand_11SD,
        trace_position_id,
        trace_rope_id,
        trace_page_table,
        trace_cross_page_table,
    ):
        """
        Executes the trace for the decode_forward method but does not read back outputs.
        """
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None
            user_cross_page_table = cross_page_table[i] if cross_page_table is not None else None
            (
                tt_h,
                tt_xattn_mask,
                tt_full_text_mask_expand_1NSH,
                tt_full_text_mask_expand_11SD,
                tt_position_id,
                tt_rope_id,
                tt_page_table,
                tt_cross_page_table,
            ) = self.model[i].prepare_decode_inputs_host(
                tokens[i],
                prefill_cross_attention_masks[i],
                prefill_full_text_row_masked_out_mask[i],
                decode_cross_attention_masks[i],
                decode_full_text_row_masked_out_mask[i],
                position_id=position_id[i],
                page_table=user_page_table,
                cross_page_table=user_cross_page_table,
            )

            copy_host_to_device(
                host_tensors=(
                    tt_h,
                    tt_xattn_mask,
                    tt_full_text_mask_expand_1NSH,
                    tt_full_text_mask_expand_11SD,
                    tt_position_id,
                    tt_rope_id,
                    tt_page_table,
                    tt_cross_page_table,
                ),
                device_tensors=(
                    trace_h[i],
                    trace_xattn_mask[i],
                    trace_full_text_mask_expand_1NSH[i],
                    trace_full_text_mask_expand_11SD[i],
                    trace_position_id[i],
                    trace_rope_id[i],
                    trace_page_table[i],
                    trace_cross_page_table[i],
                ),
            )
        for i, trace_id in trace_ids.items():
            ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        return trace_logits_rm

    def _easy_trace(
        self,
        position_id,
        tokens,
        prefill_cross_attention_masks,
        prefill_full_text_row_masked_out_mask,
        decode_cross_attention_masks,
        decode_full_text_row_masked_out_mask,
        xattn_caches=None,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
    ):
        """
        Tracing is easy! Just call this method and we'll handle tracing for you.
        """
        if not hasattr(self, "trace_ids"):
            (
                trace_ids,
                tt_logits_rm,
                tt_log_probs_rm,
                tt_h,
                tt_xattn_mask,
                tt_full_text_mask_expand_1NSH,
                tt_full_text_mask_expand_11SD,
                tt_position_id,
                tt_rope_id,
                tt_page_table,
                tt_cross_page_table,
            ) = self._capture_trace(
                position_id,
                tokens,
                prefill_cross_attention_masks,
                prefill_full_text_row_masked_out_mask,
                decode_cross_attention_masks,
                decode_full_text_row_masked_out_mask,
                xattn_caches,
                page_table=page_table,
                kv_cache=kv_cache,
                cross_page_table=cross_page_table,
            )
            self.trace_ids = trace_ids
            self.trace_inputs = {
                "tt_h": tt_h,
                "tt_xattn_mask": tt_xattn_mask,
                "tt_full_text_mask_expand_1NSH": tt_full_text_mask_expand_1NSH,
                "tt_full_text_mask_expand_11SD": tt_full_text_mask_expand_11SD,
                "tt_position_id": tt_position_id,
                "tt_rope_id": tt_rope_id,
                "tt_page_table": tt_page_table,
                "tt_cross_page_table": tt_cross_page_table,
            }
            self.trace_outputs = {
                "tt_logits_rm": tt_logits_rm,
            }

        trace_logits_rm = self._decode_forward_trace(
            position_id,
            tokens,
            prefill_cross_attention_masks,
            prefill_full_text_row_masked_out_mask,
            decode_cross_attention_masks,
            decode_full_text_row_masked_out_mask,
            page_table,
            cross_page_table,
            self.trace_ids,
            self.trace_outputs["tt_logits_rm"],
            self.trace_inputs["tt_h"],
            self.trace_inputs["tt_xattn_mask"],
            self.trace_inputs["tt_full_text_mask_expand_1NSH"],
            self.trace_inputs["tt_full_text_mask_expand_11SD"],
            self.trace_inputs["tt_position_id"],
            self.trace_inputs["tt_rope_id"],
            self.trace_inputs["tt_page_table"],
            self.trace_inputs["tt_cross_page_table"],
        )

        return trace_logits_rm

    def generate(
        self,
        vision_images,
        vision_mask,
        prompt_tokens,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ):
        # Do initial prefill
        prefill_len = len(prompt_tokens)
        total_len = prefill_len + max_gen_len  # Prepares mask for full length of output

        prompt_tokens_tensor = torch.tensor(prompt_tokens, dtype=torch.long).reshape(1, -1)  # B, S
        # Suboptimal to allocate caches every time
        model_id = 0
        xattn_caches = self.model[model_id].setup_cache(self.model_args[model_id].max_batch_size)
        (
            xattn_caches,
            prefill_cross_attention_masks,
            prefill_full_text_row_masked_out_mask,
            decode_cross_attention_masks,
            decode_full_text_row_masked_out_mask,
            logits,
        ) = self._prefill_forward_single_user(
            vision_images,
            vision_mask,
            prompt_tokens_tensor,
            xattn_caches,
            user_id=0,
            total_len=total_len,
            prefill_len=prefill_len,
            model_id=model_id,
        )

        last_token_idx = prefill_len - 1
        logits = self.model[model_id].process_output_prefill(logits.cpu(), 1, last_token_idx=(last_token_idx % 32))
        logits = logits.view(1, 1, self.model_args[model_id].vocab_size)

        prefill_output_xattn_masks = [[] for _ in range(self.data_parallel)]
        prefill_output_full_text_row_masked_out_masks = [[] for _ in range(self.data_parallel)]
        decode_output_xattn_masks = [[] for _ in range(self.data_parallel)]
        decode_output_full_text_row_masked_out_masks = [[] for _ in range(self.data_parallel)]

        prefill_output_xattn_masks[model_id].append(prefill_cross_attention_masks)
        prefill_output_full_text_row_masked_out_masks[model_id].append(prefill_full_text_row_masked_out_mask)
        decode_output_xattn_masks[model_id].append(decode_cross_attention_masks)
        decode_output_full_text_row_masked_out_masks[model_id].append(decode_full_text_row_masked_out_mask)

        def sample(logits):
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            decoder = self.tokenizer or self.processor
            return next_token, decoder.decode(next_token.tolist())

        next_token, text = sample(logits)

        yield TokenResult(
            token=next_token[0].item(),
            text=text,
        )

        for gen_idx in range(max_gen_len - 1):
            position_id = torch.tensor([prefill_len + gen_idx])
            next_token_tensor = next_token.reshape(1, 1)  # B, S

            logits = self.decode_forward(
                position_id,
                next_token_tensor,
                prefill_output_xattn_masks,
                prefill_output_full_text_row_masked_out_masks,
                decode_output_xattn_masks,
                decode_output_full_text_row_masked_out_masks,
                [xattn_caches],
                enable_trace=False,
            )
            next_token, text = sample(logits)
            yield TokenResult(
                token=next_token[0].item(),
                text=text,
            )

    def chat_completion(
        self,
        messages,
        temperature=0.6,
        top_p: float = 0.9,
        max_gen_len=None,
    ):
        model_id = 0
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model[model_id].configuration.max_seq_len:
            max_gen_len = self.model[model_id].configuration.max_seq_len - 1

        encoder = self.processor or self.tokenizer
        model_input = encoder.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        vision_images = extract_images_from_messages(messages) or None
        vision_mask = None
        if vision_images is not None:
            vision_mask = create_vision_mask(model_input["input_ids"][0], encoder.image_token_id) or None

        tokens = []

        stop_reason = None
        for result in self.generate(
            vision_images=vision_images,
            vision_mask=vision_mask,
            prompt_tokens=model_input["input_ids"][0],
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

        decoder = self.tokenizer or self.processor
        message = decoder.decode(tokens, skip_special_tokens=True)

        return CompletionMessage(message)

    def text_completion(
        self,
        content,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len=None,
    ):
        """Supports only vision models at the moment"""
        model_id = 0
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model[model_id].configuration.max_seq_len:
            max_gen_len = self.model[model_id].configuration.max_seq_len - 1

        vision_images = []
        image_token = getattr(self.processor, "image_token", None) or getattr(self.tokenizer, "image_token", None)
        text = encode_content(content, vision_images, image_token)
        vision_images = vision_images or None
        model_input = self.processor(text=text, images=vision_images, add_special_tokens=False)
        vision_mask = None
        if vision_images is not None:
            vision_mask = create_vision_mask(model_input["input_ids"][0], self.processor.image_token_id) or None

        tokens = []

        for result in self.generate(
            vision_images=vision_images,
            vision_mask=vision_mask,
            prompt_tokens=model_input["input_ids"],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        ):
            tokens.append(result.token)

        decoder = self.tokenizer or self.processor
        generation = decoder.decode(tokens, skip_special_tokens=True)

        return generation

    def _get_prefill_user_page_table(
        self, page_table, kv_cache, prefill_len, trace_enabled=False, prefill_seq_len=None
    ):
        # Ensure page_table is not padded with extra blocks for paged_fill_cache to work properly
        block_size = get_block_size(kv_cache)
        num_blocks = 0
        if trace_enabled:
            num_blocks = num_blocks_in_seq(prefill_seq_len, block_size)
        else:
            num_blocks = num_blocks_in_seq(prefill_len, block_size)
        if trace_enabled:
            if page_table.shape[1] < num_blocks:
                # If page table is too short, pad it with -1
                padding = torch.ones(1, num_blocks - page_table.shape[1], dtype=torch.int32) * -1
                page_table = torch.cat([page_table, padding], dim=1)
        return page_table[:, :num_blocks]

    ## Destructor

    def __del__(self):
        # Workaround for issue #19052
        if self.data_parallel > 1:
            for m in self.model:
                ttnn.close_mesh_device(m.mesh_device)

        if hasattr(super(Generator, self), "__del__"):
            super().__del__()


def create_submeshes(mesh_device, data_parallel):
    if not isinstance(mesh_device, ttnn.MeshDevice) or data_parallel == 1:
        return [mesh_device]

    num_rows, num_cols = mesh_device.shape
    num_devices = num_rows * num_cols
    assert num_devices % data_parallel == 0, f"Unsupported device split: {num_devices} devices, {data_parallel} groups"

    if num_rows == 8 and num_cols == 4 and num_cols % data_parallel == 0:
        submeshes = mesh_device.create_submeshes(ttnn.MeshShape(num_rows, num_cols // data_parallel))
        for submesh in submeshes:
            submesh.reshape(ttnn.MeshShape(1, num_devices // data_parallel))
        return submeshes

    return mesh_device.create_submeshes(ttnn.MeshShape(1, num_devices // data_parallel))
