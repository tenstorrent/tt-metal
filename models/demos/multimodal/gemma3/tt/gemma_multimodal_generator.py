# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Gemma3 multimodal wrapper around tt_transformers ``Generator``.

Vision inputs are normalized here before delegating to the shared text-prefill
core, so Gemma3 exposes a multimodal-specific public entrypoint
"""

from __future__ import annotations

import os

import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingParams, broadcast_sampling_params, format_sampling_params
from models.common.sampling.tt_log_probs import LogProbsResult, reformat_logprobs
from models.tt_transformers.tt.common import Mode, get_padded_prefill_len
from models.tt_transformers.tt.generator import (
    MAX_BATCHED_PREFILL_SEQ_LEN,
    SUPPORTED_PREFILL_BATCH_SIZES,
    Generator,
    max_prefill_chunk_size_cutoff,
)


def _deepseek_kvdbg_enabled() -> bool:
    return os.getenv("DEEPSEEK_KVDBG", "").lower() in ("1", "true", "yes", "y")


class GemmaMultimodalGenerator(Generator):
    def __init__(self, model, model_args, mesh_device, processor=None, tokenizer=None):
        super().__init__(model, model_args, mesh_device, processor, tokenizer)

    def encode_vision_for_prefill(self, pixel_values: list):
        if not hasattr(self.model[0], "encode_vision_embeddings_from_pixels"):
            raise TypeError(
                "GemmaMultimodalGenerator requires TtGemmaModel (multimodal). "
                "text_demo uses tt_transformers.Generator with a plain Transformer."
            )
        return [
            self.model[0].encode_vision_embeddings_from_pixels(pv) if pv is not None else None for pv in pixel_values
        ]

    def _prepare_multimodal_prefill_kwargs(self, **kwargs):
        if kwargs.get("vision_embeddings") is None and kwargs.get("pixel_values") is not None:
            kwargs = dict(kwargs)
            kwargs["vision_embeddings"] = self.encode_vision_for_prefill(kwargs["pixel_values"])
            kwargs.pop("pixel_values", None)
        return kwargs

    def prefill_forward_multimodal(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace: bool = True,
        model_id_warmup=None,
        sampling_params: SamplingParams | None = None,
        start_pos: list[int] | None = None,
        return_hidden_states: bool = False,
        warmup_prefill: bool = True,
        **kwargs,
    ):
        kwargs = self._prepare_multimodal_prefill_kwargs(**kwargs)
        # Use GemmaMultimodalGenerator.prefill_forward_text (local copy), not
        # Generator.prefill_forward_text in tt_transformers, so vision kwargs stay
        # consistent with this file (e.g. vision_embeddings list handling).
        return self.prefill_forward_text(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            enable_trace=enable_trace,
            model_id_warmup=model_id_warmup,
            sampling_params=sampling_params,
            start_pos=start_pos,
            return_hidden_states=return_hidden_states,
            warmup_prefill=warmup_prefill,
            **kwargs,
        )

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
        del vision_masks, xattn_caches, total_lens, cross_page_table
        return self.prefill_forward_multimodal(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            pixel_values=vision_images,
            **kwargs,
        )

    def prefill_forward_text(
        self,
        tokens: torch.Tensor,  # All tokens, including the cached ones
        page_table=None,
        kv_cache=None,
        prompt_lens=None,  # Full prompt lengths, including the cached ones
        empty_slots=None,
        enable_trace=True,
        model_id_warmup=None,
        sampling_params: SamplingParams | None = None,
        start_pos: list[int] = None,  # Cached prefixes lengths
        return_hidden_states=False,
        warmup_prefill=True,
        **kwargs,
    ):
        kwargs = self._prepare_multimodal_prefill_kwargs(**kwargs)
        self.mode = Mode.PREFILL
        if page_table is not None:
            assert isinstance(page_table, torch.Tensor), "page_table mush be torch.Tensor"
        else:
            # Only paged attention is supported for prefill
            enable_trace = False

        on_device_sampling_requested = sampling_params is not None

        # we need this here because of tt-metal tests
        if warmup_prefill:
            on_device_sampling_enabled = (
                getattr(self.model[0], "_supports_on_device_sampling", False)
                and getattr(self.model[0], "sampling", None) is not None
            )

            self.warmup_model_prefill(
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                can_sample_on_device=on_device_sampling_enabled,
            )

        batch_size, batch_seq_len = tokens.shape
        max_batch_size_per_model = self.model_args[0].max_batch_size

        # Output shape depends on whether we're returning logits or hidden states
        if return_hidden_states:
            # For hidden states, output shape is [batch_size, hidden_size]
            # Note: dim is the hidden dimension size
            hidden_size = self.model_args[0].dim
            output_tensor = torch.zeros(batch_size, hidden_size)
        else:
            # Each model expected to run the same model, safe to use 1st vocab size
            output_tensor = torch.zeros(batch_size, 1, self.model_args[0].vocab_size)
            output_tokens = torch.zeros(batch_size, 1, dtype=torch.int64)
            output_log_probs = [None] * batch_size
        sampling_executed = False
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch_size)

        if empty_slots is None:
            empty_slots = list(range(batch_size))

        # For row-sharded users, use max_local_batch_size (users per row) for group_user_id
        local_batch_size = getattr(self.model_args[0], "max_local_batch_size", max_batch_size_per_model)

        if not isinstance(prompt_lens, list):
            prompt_lens = prompt_lens.tolist()

        prefill_seq_lens = [get_padded_prefill_len(seq_len) for seq_len in prompt_lens]
        # Row-sharded batched prefill: process 1 user per row per iteration.
        # Only used when device sampling is active (sampling_params is not None)
        # and the prompt uses the harmony chat template (first token is <|start|>=200006).
        # Host sampling (sampling_params=None) needs the single-user prefill path
        # that returns full logits per user.
        model_0 = self.model[0]
        is_harmony = tokens.shape[1] > 0 and int(tokens[0, 0]) == 200006
        if (
            getattr(model_0, "users_row_sharded", False)
            and batch_size > 1
            and sampling_params is not None
            and is_harmony
        ):
            return self._row_sharded_batched_prefill(
                tokens,
                page_table,
                kv_cache,
                prompt_lens,
                prefill_seq_lens=prefill_seq_lens,
                enable_trace=enable_trace,
                sampling_params=sampling_params,
            )

        # Batched prefill: all prompts share the same padded length so they can
        # be processed in a single forward pass. padded_batch is rounded up to
        # the nearest SUPPORTED_PREFILL_BATCH_SIZES entry (not max_batch_size)
        # to keep all_gather buffers within DRAM limits.
        use_batched_prefill = (
            batch_size > 1
            and len(set(prefill_seq_lens)) == 1
            and self.data_parallel == 1
            and not getattr(self.model_args[0], "disable_batched_prefill", False)
        )

        if use_batched_prefill and on_device_sampling_requested:
            sampling_module, sampling_dp, _, _ = self._get_sampling_contract(0)
            if sampling_module is not None and sampling_dp > 1:
                # NOTE: Batched prefill disabled: on-device sampling
                # must fall back to sequential prefill until a row-sharded
                # batched-prefill sampling contract is implemented.
                use_batched_prefill = False

        if use_batched_prefill:
            padded_batch = next(
                (b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= batch_size),
                self.model_args[0].max_batch_size,
            )
            if padded_batch > self.model_args[0].max_batch_size:
                logger.info(
                    f"Batched prefill disabled: padded_batch {padded_batch} exceeds "
                    f"max_batch_size {self.model_args[0].max_batch_size}"
                )
                use_batched_prefill = False
            elif padded_batch * prefill_seq_lens[0] >= MAX_BATCHED_PREFILL_SEQ_LEN:
                logger.info(
                    f"Batched prefill disabled: {padded_batch} x {prefill_seq_lens[0]} = "
                    f"{padded_batch * prefill_seq_lens[0]} tokens exceeds limit {MAX_BATCHED_PREFILL_SEQ_LEN}"
                )
                use_batched_prefill = False
        if not use_batched_prefill:
            padded_batch = self.model_args[0].max_batch_size

        all_users = [0] if use_batched_prefill else empty_slots

        sampling_params_per_out: list[SamplingParams | None] = [None] * len(empty_slots)
        prompt_tokens_per_out: list[torch.Tensor | None] = [None] * len(empty_slots)
        prefill_results: list[dict] = []

        for idx, user_id in enumerate(all_users):
            model_id = user_id // max_batch_size_per_model if model_id_warmup is None else model_id_warmup
            group_user_id = user_id % local_batch_size if page_table is None else 0

            if use_batched_prefill:
                batch_user_ids = empty_slots
                last_token_idx = [(seq_len - 1) for seq_len in prompt_lens]
                prefill_seq_len = prefill_seq_lens[0]
                seq_len = prompt_lens
            else:
                batch_user_ids = None
                seq_len = int(prompt_lens[idx])
                num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
                last_token_idx = seq_len - 1
                prefill_seq_len = prefill_seq_lens[idx]
                logger.info(f"Prefilling User {user_id + 1} up to {seq_len} tokens")
            local_kwargs = kwargs.copy()  # Avoid modifying original kwargs
            if getattr(self.model[model_id], "users_row_sharded", False):
                local_kwargs["global_user_id"] = batch_user_ids if use_batched_prefill else user_id
            sampling_enabled = (
                on_device_sampling_requested
                and getattr(self.model[model_id], "_supports_on_device_sampling", False)
                and getattr(self.model[model_id], "sampling", None) is not None
            )

            if use_batched_prefill:
                # Galaxy 70B approach: slot-based placement with shape [padded_batch, prefill_seq_len]
                # Each request is placed at its corresponding slot index
                prefill_ids = torch.zeros(padded_batch, prefill_seq_len, dtype=torch.long, device=tokens.device)
                padded_last_token_idx = [0] * padded_batch  # dummy idx for padded slots
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
                num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
                prefill_ids = torch.cat(
                    [
                        tokens[idx : idx + 1, num_cached_tokens:seq_len],
                        torch.zeros(1, prefill_seq_len - (seq_len - num_cached_tokens)).long(),
                    ],
                    dim=-1,
                )

            enable_trace_current_prompt = enable_trace and self.model_args[model_id].can_enable_trace(
                prefill_seq_len, num_cached_tokens if not use_batched_prefill else 0
            )

            logger.info(
                f"Prefill seq len: {prefill_seq_len}, max_prefill_chunk_size: {self.model_args[0].max_prefill_chunk_size}, trace: {enable_trace_current_prompt}"
            )

            if page_table is not None:
                # For batched prefill: pass full page_table (function handles slot placement)
                # For non-batched prefill: pass sliced page_table for current user (like original code)
                page_table_for_user = page_table if use_batched_prefill else page_table[idx : idx + 1]
                page_table_user = self._get_prefill_user_page_table(
                    page_table_for_user,
                    kv_cache[model_id],
                    seq_len,
                    trace_enabled=enable_trace_current_prompt,
                    prefill_seq_len=prefill_seq_len,
                    use_batched_prefill=use_batched_prefill,
                    user_id=batch_user_ids if use_batched_prefill else user_id,
                    padded_batch_size=padded_batch if use_batched_prefill else None,
                )
            else:
                page_table_user = None
            if page_table_user is not None and _deepseek_kvdbg_enabled():
                sample = []
                if page_table_user.numel():
                    flat = page_table_user.reshape(-1)
                    sample = flat[: min(16, flat.numel())].tolist()
                logger.debug(
                    "KVDBG deepseek prefill user global={} local={} seq_len={} cached={} page_table_shape={} sample={}",
                    user_id,
                    group_user_id,
                    seq_len,
                    num_cached_tokens,
                    list(page_table_user.shape),
                    sample,
                )
            model_kv_cache = kv_cache[model_id] if kv_cache is not None else None

            # Per-user multimodal kwargs (Gemma3 uses vision_embeddings via GemmaMultimodalGenerator;
            # other models typically omit these keys.)
            if "vision_embeddings" in local_kwargs and local_kwargs["vision_embeddings"] is not None:
                local_kwargs["vision_embeddings"] = local_kwargs["vision_embeddings"][idx]
            if local_kwargs.get("pixel_values", None) is not None:
                local_kwargs["pixel_values"] = local_kwargs["pixel_values"][idx]
                if "image_grid_thw" in local_kwargs:
                    local_kwargs["image_grid_thw"] = local_kwargs["image_grid_thw"][idx]
                if "image_sizes" in local_kwargs and local_kwargs["image_sizes"] is not None:
                    local_kwargs["image_sizes"] = local_kwargs["image_sizes"][idx]

            if sampling_enabled and not use_batched_prefill:
                sampling_executed = True
                sampling_dp = getattr(self.model[model_id], "sampling_dp", 1)
                total_batch = self.model[model_id].sampling.tt_sampling.max_batch_size * sampling_dp
                per_request_params = format_sampling_params(
                    broadcast_sampling_params(sampling_params, idx, slot_len=total_batch), total_batch
                )
                assert per_request_params is not None, "Sampling was executed but missing per-request sampling params"
                # empty_slots uses max_batch_size_per_model (not total_batch) because
                # the seed manager operates on per-row slots (0..31).  When sampling_dp > 1
                # the params are already broadcast across all rows by broadcast_sampling_params.
                self.model[model_id].sampling.apply_prefill_state(
                    sampling_params=per_request_params,
                    prompt_tokens=prefill_ids[:, :seq_len].repeat(total_batch, 1),
                    empty_slots=[user_id % max_batch_size_per_model],
                )

            if enable_trace_current_prompt:
                logits = self._easy_trace_prefill(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=batch_user_ids if use_batched_prefill else group_user_id,
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    prefill_seq_len=prefill_seq_len,
                    batch_size=padded_batch if use_batched_prefill else 1,
                    **local_kwargs,
                )
            else:
                logits = self.prefill_forward_single_user_text(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=batch_user_ids if use_batched_prefill else group_user_id,
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    num_cached_tokens=0 if use_batched_prefill else num_cached_tokens,
                    batch_size=padded_batch if use_batched_prefill else 1,
                    **local_kwargs,
                )
            if use_batched_prefill:
                hidden_dim = logits.shape[-1]
                logits = ttnn.reshape(logits, [padded_batch, 1, prefill_seq_len, hidden_dim])

                if sampling_enabled:
                    sampling_executed = True

                    sampling_module, sampling_dp, sampling_batch, _ = self._get_sampling_contract(model_id)
                    assert sampling_module is not None
                    assert sampling_batch is not None
                    combined_params = format_sampling_params(sampling_params, sampling_batch)
                    max_prompt_len = max(int(prompt_lens[i]) for i in range(len(empty_slots)))
                    combined_prompt_tokens = torch.zeros(sampling_batch, max_prompt_len, dtype=torch.long)
                    for local_idx, slot in enumerate(empty_slots):
                        plen = int(prompt_lens[local_idx])
                        combined_prompt_tokens[slot, :plen] = prefill_ids[slot, :plen]

                    sampling_module.apply_prefill_state(
                        sampling_params=combined_params,
                        prompt_tokens=combined_prompt_tokens,
                        empty_slots=empty_slots,
                        replicate_seeds=False,
                    )

                    user_hidden = self.model[model_id].extract_last_tokens_batched_prefill(
                        logits,
                        last_token_idx,
                        padded_batch,
                        prefill_seq_len,
                        target_batch=sampling_batch,
                    )

                    sampling_trace_key = f"sampling_{prefill_seq_len}_{model_id}_{sampling_batch}_{sampling_dp}"
                    if enable_trace_current_prompt:
                        if self.trace_id_prefill_sampling[sampling_trace_key] is None:
                            (
                                s_trace_id,
                                s_trace_output,
                                s_trace_input,
                            ) = self._capture_trace_prefill_sampling(model_id, sampling_batch)
                            self.trace_id_prefill_sampling[sampling_trace_key] = s_trace_id
                            self.trace_output_prefill_sampling[sampling_trace_key] = s_trace_output
                            self.trace_input_prefill_sampling[sampling_trace_key] = s_trace_input

                        s_trace_input = self.trace_input_prefill_sampling[sampling_trace_key]
                        user_hidden_host = user_hidden.cpu()
                        ttnn.copy_host_to_device_tensor(user_hidden_host, s_trace_input)
                        ttnn.execute_trace(
                            self.model_args[model_id].mesh_device,
                            self.trace_id_prefill_sampling[sampling_trace_key],
                            cq_id=0,
                            blocking=False,
                        )
                        tt_tokens, tt_log_probs = self.trace_output_prefill_sampling[sampling_trace_key]
                    else:
                        batched_logits = self.model[model_id]._apply_norm_and_lm_head(user_hidden)
                        tt_tokens, tt_log_probs = self.model[model_id].sampling.sample(
                            batched_logits,
                            enable_trace=False,
                        )

                    ttnn.synchronize_device(self.model[model_id].mesh_device)

                    tokens_host = ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0]).reshape(-1)
                    log_probs_host = (
                        ttnn.to_torch(ttnn.get_device_tensors(tt_log_probs)[0]).reshape(-1)
                        if tt_log_probs is not None
                        else None
                    )
                    for local_idx, slot in enumerate(empty_slots):
                        output_tokens[slot] = tokens_host[slot]
                        if log_probs_host is not None:
                            output_log_probs[slot] = log_probs_host[slot]
                else:
                    for local_idx, slot in enumerate(empty_slots):
                        user_logits = logits[slot : slot + 1, :, :, :]
                        _logits = self.model[model_id].process_logits_after_prefill_trace(
                            user_logits, last_token_idx[slot]
                        )
                        _logits = ttnn.to_layout(_logits, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        output_tensor[slot] = self.model[model_id].process_output_prefill(
                            _logits.cpu(), last_token_idx=(last_token_idx[slot] % 32)
                        )
                break

            # Non-batched prefill path
            if enable_trace_current_prompt:
                if return_hidden_states:
                    hidden_states = self.model[model_id].process_hidden_states_after_prefill_trace(
                        logits, last_token_idx
                    )
                    prefill_results.append(
                        {
                            "idx": idx,
                            "model_id": model_id,
                            "last_token_idx": last_token_idx,
                            "hidden_states": hidden_states.cpu(blocking=False),
                        }
                    )
                    continue
                else:
                    logits = self.model[model_id].process_logits_after_prefill_trace(logits, last_token_idx)
            else:
                if return_hidden_states:
                    raise NotImplementedError("return_hidden_states=True requires enable_trace=True")

            if sampling_enabled:
                tt_tokens, tt_log_probs = self.model[model_id].sampling.sample(
                    logits,
                    enable_trace=False,
                )
                prefill_results.append(
                    {
                        "idx": idx,
                        "model_id": model_id,
                        "last_token_idx": last_token_idx,
                        "logits": [
                            tt_tokens.cpu(blocking=False),
                            tt_log_probs.cpu(blocking=False) if tt_log_probs is not None else None,
                        ],
                        "sampling": sampling_enabled,
                    }
                )
            else:
                logits = ttnn.untilize(logits, use_multicore=True)
                prefill_results.append(
                    {
                        "idx": idx,
                        "model_id": model_id,
                        "last_token_idx": last_token_idx,
                        "logits": logits.cpu(blocking=False),
                        "sampling": sampling_enabled,
                    }
                )

        if len(prefill_results) > 0:
            for elem_idx, res in enumerate(prefill_results):
                idx = res["idx"]
                last_token_idx = res["last_token_idx"]
                model_id = res["model_id"]
                num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
                last_token_idx_relative = last_token_idx - num_cached_tokens
                ttnn.synchronize_device(self.model[model_id].mesh_device)

                if "hidden_states" in res:
                    output_tensor[idx] = self.model[model_id].process_output_prefill_hidden_states(
                        res["hidden_states"], last_token_idx=(last_token_idx_relative % 32)
                    )
                elif res["sampling"]:
                    tt_tokens = res["logits"][0]
                    tt_log_probs = res["logits"][1]
                    tokens_host = ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0]).reshape(-1)[
                        (
                            last_token_idx % 32
                        )  # TODO: Check if here should be used last_token_idx_relative instead of last_token_idx
                    ]
                    if isinstance(tt_log_probs, LogProbsResult):
                        log_probs_host = tt_log_probs.extract_user(last_token_idx % 32)
                    elif tt_log_probs is not None:
                        log_probs_host = ttnn.to_torch(ttnn.get_device_tensors(tt_log_probs)[0]).reshape(-1)[
                            (last_token_idx % 32)
                        ]  # TODO: Check if here should be used last_token_idx_relative instead of last_token_idx
                    else:
                        log_probs_host = None
                    output_tokens[idx] = tokens_host
                    if log_probs_host is not None:
                        output_log_probs[idx] = log_probs_host
                else:
                    output_tensor[idx] = self.model[model_id].process_output_prefill(
                        res["logits"], last_token_idx=(last_token_idx_relative % 32)
                    )

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")

        if sampling_executed:
            return output_tokens, reformat_logprobs(output_log_probs, batch_size)
        else:
            return output_tensor

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, greedy_only: bool = False):
        if self.already_warmed_up_prefill:
            return
        self.already_warmed_up_prefill = True

        sequence_lengths_to_warmup = self.model_args[0].get_warmup_prefill_supported_seq_lens()
        warmup_batch_sizes = (1,)

        skip_sequence_lengths = False

        # Sweep all sampling parameters for prefill warmup just once since it is sequence length agnostic
        sampling_parameters_sweeped = False

        if enable_trace:
            logger.info("Using batch-1-only traced prefill warmup; runtime batched prefill remains enabled")

        for model_id in range(self.data_parallel):
            for supported_length in sequence_lengths_to_warmup:
                if model_id != 0 and (
                    supported_length not in self.model_args[0].trace_prefill_supported_seq_lens or not enable_trace
                ):
                    continue

                # Token-limit guard below skips combinations that would
                # exceed MAX_BATCHED_PREFILL_SEQ_LEN.
                for batch_size in warmup_batch_sizes:
                    if batch_size > 1 and batch_size * supported_length >= MAX_BATCHED_PREFILL_SEQ_LEN:
                        logger.info(
                            f"Skipping batched prefill warmup for batch_size={batch_size}, "
                            f"seq_len={supported_length}: exceeds token limit"
                        )
                        continue

                    warmup_args = self._mock_tokens(batch_size, supported_length, kv_cache, model_id)

                    # chunked prefill not supported without paged attention
                    if warmup_args["page_table"] is None and max_prefill_chunk_size_cutoff(
                        supported_length, self.model_args[0].max_prefill_chunk_size
                    ):
                        logger.warning(
                            f"Skipping warmup for sequence lengths after: {supported_length} because they are greater than the max prefill chunk size and paged attention is disabled"
                        )
                        skip_sequence_lengths = True
                        break

                    if not sampling_parameters_sweeped:
                        sampling_params = self._create_sampling_params(
                            can_sample_on_device=can_sample_on_device,
                            batch_size=batch_size,
                            greedy_only=greedy_only,
                        )
                    else:
                        sampling_params = [None]

                    for param in sampling_params:
                        logger.info(
                            f"Warming up prefill for sequence length: {supported_length} for batch size: {batch_size} with sampling params: {param}"
                        )
                        self.prefill_forward_text(
                            **warmup_args,
                            kv_cache=kv_cache,
                            enable_trace=enable_trace,
                            model_id_warmup=model_id,
                            sampling_params=param,
                        )

                    sampling_parameters_sweeped = True

                if skip_sequence_lengths:
                    break

        # Vision compile for multimodal models
        if getattr(self.model_args[0], "is_multimodal", False) and hasattr(
            self.model[0], "encode_vision_embeddings_from_pixels"
        ):
            vision_chunk_size = getattr(self.model_args[0], "vision_chunk_size", 896)
            vision_channels = getattr(self.model_args[0], "vision_in_channels", 3)
            model_id = 0

            # Create synthetic image for vision warmup
            # pixel_values is a list (one per user), each element is (num_images, C, H, W)
            warmup_pixel_values = [torch.zeros((1, vision_channels, vision_chunk_size, vision_chunk_size))]

            # Minimal text tokens for vision warmup pass, prefill expects non-empty tokens
            batch_size = 1  # VLMs support only batch=1 for now
            prefill_forward_args = self._mock_tokens(batch_size, 128, kv_cache, model_id)

            logger.info(f"Warming up vision encoder with image size {vision_chunk_size}x{vision_chunk_size}")

            multimodal_prefill = getattr(self, "prefill_forward_multimodal", None)
            if callable(multimodal_prefill):
                multimodal_prefill(
                    prefill_forward_args["tokens"],
                    page_table=prefill_forward_args["page_table"],
                    kv_cache=kv_cache,
                    prompt_lens=prefill_forward_args["prompt_lens"],
                    empty_slots=prefill_forward_args["empty_slots"],
                    enable_trace=False,  # Vision encoder warmup doesn't support trace
                    model_id_warmup=model_id,
                    sampling_params=None,
                    pixel_values=warmup_pixel_values,
                    image_sizes=[(vision_chunk_size, vision_chunk_size)],
                )
            else:
                self.prefill_forward_text(
                    **prefill_forward_args,
                    kv_cache=kv_cache,
                    enable_trace=False,  # Vision encoder warmup doesn't support trace
                    model_id_warmup=model_id,
                    sampling_params=None,
                    pixel_values=warmup_pixel_values,
                    image_sizes=[(vision_chunk_size, vision_chunk_size)],
                )
            logger.info("Vision encoder warmup completed")
