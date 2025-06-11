# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from llama_models.llama3.api.datatypes import InterleavedTextMedia, StopReason
from llama_models.llama3.reference_impl.generation import (
    ChatPrediction,
    CompletionPrediction,
    TokenResult,
    sample_top_p,
)
from loguru import logger

import ttnn
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

    temperature: float
    top_k: int
    top_p: float


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
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.data_parallel = len(self.model)

    # Note: This function is called by vLLM
    def prefill_forward_text(self, tokens: torch.Tensor, page_table=None, kv_cache=None, prompt_lens=None):
        batch, batch_seq_len = tokens.shape

        # Each model expected to run the same model, safe to use 1st vocab size
        output_logits = torch.zeros(batch, 1, self.model_args[0].vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch)

        data_parallel = min(batch, self.data_parallel)
        batch_per_device = batch // data_parallel

        if page_table is not None:
            assert isinstance(page_table, torch.Tensor), "page_table mush be torch.Tensor"
            page_table = torch.chunk(page_table, self.data_parallel, 0)

        out_list = []
        for group_user_id in range(batch_per_device):
            for model_id in range(data_parallel):
                user_id = group_user_id + model_id * batch_per_device

                logger.info(f"Prefilling User {user_id + 1}")
                seq_len = int(prompt_lens[user_id])
                last_token_idx = seq_len - 1

                prefill_seq_len = get_padded_prefill_len(seq_len)
                prefill_ids = torch.cat(
                    [tokens[user_id : user_id + 1, :seq_len], torch.zeros(1, prefill_seq_len - seq_len).long()], dim=-1
                )
                if page_table is not None:
                    page_table_user = self._get_prefill_user_page_table(
                        page_table[model_id], kv_cache[model_id], seq_len
                    )

                logits = self.prefill_forward_single_user_text(
                    prefill_ids,
                    page_table=page_table_user if page_table is not None else None,
                    user_id=group_user_id,
                    last_token_idx=last_token_idx,
                    kv_cache=kv_cache[model_id] if kv_cache is not None else None,
                    model_id=model_id,
                )
                out_list.append(logits)

        # We gather data back to how at the end of prefill
        for idx, out in enumerate(out_list):
            model_id = idx % self.data_parallel
            group_user_id = idx // self.data_parallel
            user_id = group_user_id + model_id * batch_per_device

            seq_len = int(prompt_lens[user_id])
            last_token_idx = seq_len - 1

            # Since we give unpadded_seq_len, only the tile containing the last token is returned
            output_logits[user_id] = self.model[model_id].process_output_prefill(
                out, last_token_idx=(last_token_idx % 32)
            )

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")
        return output_logits

    def prefill_forward_single_user_text(self, tokens, page_table, user_id, last_token_idx, kv_cache=None, model_id=-1):
        seq_len = tokens.shape[-1]
        use_chunked_prefill = seq_len > self.model_args[model_id].max_prefill_chunk_size
        if use_chunked_prefill:
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
            assert (
                last_token_idx is not None and last_token_idx < seq_len
            ), "last_token_idx must be provided and less than seq_len"
            chunk_size = get_max_prefill_chunk_size(seq_len, self.model_args[model_id].max_prefill_chunk_size)
            block_size = get_block_size(kv_cache)
            last_token_idx_in_chunk = last_token_idx % chunk_size
            # Calculate which chunk contains the last_token_idx
            last_chunk_start = (last_token_idx // chunk_size) * chunk_size
            page_table_user = page_table[user_id : user_id + 1, :]
            # Pad page table to match number of blocks in seq_len
            num_padding_blocks = num_blocks_in_seq(seq_len, block_size) - page_table_user.shape[1]
            page_table_user_padded = torch.cat(
                [page_table_user, torch.zeros(1, num_padding_blocks, dtype=torch.int32)], dim=-1
            )
            CHUNK_USER_ID = 0

            for chunk_start in range(0, seq_len, chunk_size):
                chunk_end = chunk_start + chunk_size
                assert (
                    chunk_end <= seq_len
                ), f"Chunk end should be less than seq_len, got chunk_end={chunk_end} and seq_len={seq_len}"
                chunk_tokens = tokens[:, chunk_start:chunk_end]
                chunk_page_table = page_table_user[:, chunk_start // block_size : chunk_end // block_size]

                (
                    chunk_prefill_input,
                    chunk_rot_mats_prefill,
                    page_table_tt,
                    chunk_page_table_tt,
                ) = self.model[model_id].prepare_inputs_prefill(
                    chunk_tokens,
                    start_pos=chunk_start,
                    page_table=page_table_user_padded,
                    chunk_page_table=chunk_page_table,
                )
                tt_logits = self.model[model_id].ttnn_prefill_forward(
                    chunk_prefill_input,
                    rot_mats=chunk_rot_mats_prefill,
                    user_id=CHUNK_USER_ID,
                    page_table=page_table_tt,
                    chunk_page_table=chunk_page_table_tt,
                    chunk_start_idx=chunk_start,
                    get_last_token=(last_token_idx_in_chunk // 32) * 32,
                    kv_cache=kv_cache,
                )

                if chunk_start == last_chunk_start:
                    return tt_logits
                else:
                    del tt_logits
        else:
            prefill_input, rot_mats_prefill, page_table_tt, _ = self.model[model_id].prepare_inputs_prefill(
                tokens,
                page_table=page_table,
            )

            tt_logits = self.model[model_id].ttnn_prefill_forward(
                prefill_input,
                rot_mats=rot_mats_prefill,
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
    ):
        assert (
            sampling_params is None or sampling_params.temperature == 0
        ), "Currently only supporting greedy decoding (temperature=0) on device"
        argmax_on_device = sampling_params is not None and sampling_params.temperature == 0

        B = tokens.shape[0]
        tokens = torch.chunk(tokens, self.data_parallel, 0)
        start_pos = torch.chunk(start_pos, self.data_parallel, 0)
        page_table = torch.chunk(page_table, self.data_parallel, 0) if page_table is not None else None

        decode_kwargs = {
            "current_pos": start_pos,
            "tokens": tokens,
            "page_table": page_table,
            "kv_cache": kv_cache,
            "argmax_on_device": argmax_on_device,
        }
        if enable_trace:
            tt_logits = self._easy_trace_text(**decode_kwargs)
        else:
            tt_logits = self._decode_forward_no_trace_text(**decode_kwargs)

        if read_from_device:
            to_host = self.read_decode_output(tt_logits, B, is_tokens=(sampling_params is not None))
            return to_host
        else:
            return tt_logits

    def _decode_forward_no_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        argmax_on_device=False,
    ):
        """
        Performs text decode step.
        Returns tt_logits on device
        """
        tt_logits = []

        tt_tokens = []
        tt_current_pos = []
        tt_rot_mats = []
        tt_page_table = []

        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None
            tt_tokens_i, tt_current_pos_i, tt_rot_mats_i, tt_page_table_i = self.model[i].prepare_inputs_decode(
                tokens[i], current_pos[i], user_page_table
            )
            tt_tokens.append(tt_tokens_i)
            tt_current_pos.append(tt_current_pos_i)
            tt_rot_mats.append(tt_rot_mats_i)
            tt_page_table.append(tt_page_table_i)

        for i in range(self.data_parallel):
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            tt_logits_i = self.model[i].ttnn_decode_forward(
                tt_tokens[i],
                tt_current_pos[i],
                rot_mats=tt_rot_mats[i],
                page_table=tt_page_table[i],
                kv_cache=user_kv_cache,
                argmax_on_device=argmax_on_device,
            )
            tt_logits.append(tt_logits_i)

        return tt_logits

    def _capture_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        argmax_on_device=False,
    ):
        """
        Captures a trace for the decode_forward method.
        """

        # Compile run
        self._decode_forward_no_trace_text(
            tokens, current_pos, page_table=page_table, kv_cache=kv_cache, argmax_on_device=argmax_on_device
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
            trace_id = ttnn.begin_trace_capture(self.model_args[i].mesh_device, cq_id=0)
            trace_ids[i] = trace_id
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            transformed_inputs = self.model[i].transform_decode_inputs_device(*(device_inputs[i]))
            tt_out_trace.append(
                self.model[i].ttnn_decode_forward(
                    *transformed_inputs, kv_cache=user_kv_cache, argmax_on_device=argmax_on_device
                )
            )
            ttnn.end_trace_capture(self.model_args[i].mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Decode Trace")
        return trace_ids, tt_out_trace, *device_inputs

    def _decode_forward_trace_text(
        self,
        trace_ids,
        device_inputs,
        tt_out_trace,
        tokens,
        current_pos,
        page_table=None,
    ):
        """
        Executes the trace for the decode_forward method but does not read back outputs.
        """
        host_inputs = []
        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None
            host_inputs_i = self.model[i].prepare_decode_inputs_host(tokens[i], current_pos[i], user_page_table)
            host_inputs.append(host_inputs_i)

        to_device = []
        for i in range(self.data_parallel):
            to_device.append(
                copy_host_to_device(
                    host_tensors=host_inputs[i],
                    device_tensors=device_inputs[i],
                )
            )
        device_inputs = to_device
        for i, trace_id in trace_ids.items():
            ttnn.execute_trace(self.model_args[i].mesh_device, trace_id, cq_id=0, blocking=False)

        return tt_out_trace

    def _easy_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        argmax_on_device=False,
    ):
        """
        Tracing is easy! Just call this method and we'll handle tracing for you.
        """
        if not hasattr(self, "trace_ids_text"):
            trace_ids, tt_out_trace, *device_inputs = self._capture_trace_text(
                tokens, current_pos, page_table=page_table, kv_cache=kv_cache, argmax_on_device=argmax_on_device
            )
            self.trace_ids_text = trace_ids
            self.trace_inputs_text = device_inputs
            self.trace_output_text = tt_out_trace

        trace_logits_rm = self._decode_forward_trace_text(
            self.trace_ids_text,
            self.trace_inputs_text,
            self.trace_output_text,
            tokens,
            current_pos,
            page_table=page_table,
        )

        return trace_logits_rm

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
                cross_attention_masks,
                full_text_row_masked_out_mask,
            ) = self.model[model_id].compute_vision_tokens_masks(
                batch_images=[vision_images],
                batch_masks=[vision_mask],
                total_len=total_len,
            )

            if cross_page_table is not None:
                num_vision_tokens = vision_tokens.shape[2]
                cross_page_table = self._get_prefill_user_page_table(cross_page_table, kv_cache, num_vision_tokens)
        else:
            vision_tokens, cross_attention_masks, full_text_row_masked_out_mask = None, None, None

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
            cross_attention_masks,
            full_text_row_masked_out_mask,
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

        return xattn_caches, cross_attention_masks, full_text_row_masked_out_mask, tt_logits

    # Note: This function is called by vLLM
    def prefill_forward(
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
    ):
        """
        Batched version of _prefill_forward_single_user for vision model.
        """
        batch, batch_seq_len = tokens.shape
        output_logits = torch.zeros(batch, 1, self.model_args[0].vocab_size)

        data_parallel = min(batch, self.data_parallel)
        batch_per_device = batch // data_parallel

        out_list = [[] for _ in range(data_parallel)]
        output_xattn_masks = [None for _ in range(batch)]
        output_full_text_row_masked_out_masks = [None for _ in range(batch)]

        if page_table is not None:
            assert isinstance(page_table, torch.Tensor), "page_table mush be torch.Tensor"
            page_table = torch.chunk(page_table, self.data_parallel, 0)  # cross_page_table
        if cross_page_table is not None:
            assert isinstance(cross_page_table, torch.Tensor), "cross_page_table mush be torch.Tensor"
            cross_page_table = torch.chunk(cross_page_table, self.data_parallel, 0)

        for group_user_id in range(batch_per_device):
            for model_id in range(data_parallel):
                user_id = group_user_id + model_id * batch_per_device

                logger.info(f"Prefilling User {user_id + 1}")
                seq_len = int(prompt_lens[user_id])
                user_page_table = page_table[model_id] if page_table is not None else None
                user_kv_cache = kv_cache[model_id] if kv_cache is not None else None
                user_cross_page_table = cross_page_table[model_id] if kv_cache is not None else None
                xattn_cache = xattn_caches[model_id] if xattn_caches is not None else None
                (
                    xattn_cache,
                    cross_attention_masks,
                    full_text_row_masked_out_mask,
                    logits,
                ) = self._prefill_forward_single_user(
                    vision_images=vision_images[user_id],
                    vision_mask=vision_masks[user_id],
                    tokens=tokens[user_id : user_id + 1, :seq_len],  # Keep batch dimension
                    xattn_caches=xattn_cache,
                    user_id=group_user_id,
                    total_len=total_lens[user_id],
                    prefill_len=seq_len,
                    page_table=user_page_table,
                    kv_cache=user_kv_cache,
                    cross_page_table=user_cross_page_table,
                    model_id=model_id,
                )
                if xattn_caches is not None:
                    xattn_caches[model_id] = xattn_cache
                out_list[model_id].append(logits)
                output_xattn_masks[user_id] = cross_attention_masks
                output_full_text_row_masked_out_masks[user_id] = full_text_row_masked_out_mask

        # We gather prefill output at the end of prefill to reduce unnecessary device sync
        for group_user_id in range(batch_per_device):
            for model_id in range(data_parallel):
                user_id = group_user_id + model_id * batch_per_device
                last_token_idx = prompt_lens[user_id] - 1
                output_logits[user_id] = self.model[model_id].process_output_prefill(
                    out_list[model_id][group_user_id], 1, last_token_idx=(last_token_idx % 32)
                )

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")

        return output_logits, output_xattn_masks, output_full_text_row_masked_out_masks

    # Note: This function is called by vLLM
    def decode_forward(
        self,
        start_pos,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
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
        cross_attention_masks = [
            cross_attention_masks[i * batch_per_device : (i + 1) * batch_per_device] for i in range(data_parallel)
        ]
        full_text_row_masked_out_mask = [
            full_text_row_masked_out_mask[i * batch_per_device : (i + 1) * batch_per_device]
            for i in range(data_parallel)
        ]
        page_table = torch.chunk(page_table, self.data_parallel, 0) if page_table is not None else None
        cross_page_table = (
            torch.chunk(cross_page_table, self.data_parallel, 0) if cross_page_table is not None else None
        )

        decode_kwargs = {
            "position_id": start_pos,
            "tokens": tokens,
            "cross_attention_masks": cross_attention_masks,
            "full_text_row_masked_out_mask": full_text_row_masked_out_mask,
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
            to_host = self.read_decode_output(tt_logits, B)
            return to_host
        else:
            return tt_logits

    # Note: This function is called by vLLM
    def read_decode_output(self, tt_out, unpadded_batch, is_tokens=False):
        """
        Input is ttnn device tensor of logits if is_tokens=False, otherwise tokens. Output is the corresponding torch tensor.
        """
        logits = []
        for i in range(self.data_parallel):
            logits_i = self.model[i].process_output_decode(
                tt_out[i], B=self.model_args[i].max_batch_size, S=1, is_tokens=is_tokens
            )
            logits.append(logits_i)
        logits = torch.cat(logits, 0)
        return logits[:unpadded_batch]

    def _decode_forward_no_trace(
        self,
        position_id,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
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
                cross_attention_masks[i],
                full_text_row_masked_out_mask[i],
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
        for i in range(self.data_parallel):
            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            xattn_cache = xattn_caches[i] if xattn_caches is not None else None
            tt_logits_i = self.model[i].ttnn_decode_forward(
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

        return tt_logits

    def _capture_trace(
        self,
        position_id,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
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
                cross_attention_masks[i],
                full_text_row_masked_out_mask[i],
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
            # tt_logits_rm unused later, no need to make a list
            tt_logits_rm = self.model[i].ttnn_decode_forward(
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
                cross_attention_masks[i],
                full_text_row_masked_out_mask[i],
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

            tt_logits_rm_i = self.model[i].ttnn_decode_forward(
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
            ttnn.end_trace_capture(self.model_args[i].mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Decode Trace")

        return (
            trace_ids,
            tt_logits_rm,
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
        cross_attention_masks,
        full_text_row_masked_out_mask,
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
                cross_attention_masks[i],
                full_text_row_masked_out_mask[i],
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
        cross_attention_masks,
        full_text_row_masked_out_mask,
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
                cross_attention_masks,
                full_text_row_masked_out_mask,
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
            cross_attention_masks,
            full_text_row_masked_out_mask,
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
        model_input,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ):
        # Do initial prefill
        vision_images = model_input.vision.images
        vision_mask = model_input.vision.mask
        prompt_tokens = model_input.tokens
        prefill_len = len(prompt_tokens)
        total_len = prefill_len + max_gen_len  # Prepares mask for full length of output

        prompt_tokens_tensor = torch.tensor(prompt_tokens, dtype=torch.long).reshape(1, -1)  # B, S
        # Suboptimal to allocate caches every time
        model_id = 0
        xattn_caches = self.model[model_id].setup_cache(self.model_args[model_id].max_batch_size)
        (
            xattn_caches,
            cross_attention_masks,
            full_text_row_masked_out_mask,
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
        logits = self.model[model_id].process_output_prefill(logits, 1, last_token_idx=(last_token_idx % 32))
        logits = logits.view(1, 1, self.model_args[model_id].vocab_size)

        output_xattn_masks = [[] for _ in range(self.data_parallel)]
        output_full_text_row_masked_out_masks = [[] for _ in range(self.data_parallel)]
        output_xattn_masks[model_id].append(cross_attention_masks)
        output_full_text_row_masked_out_masks[model_id].append(full_text_row_masked_out_mask)

        def sample(logits):
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            return next_token, self.tokenizer.decode(next_token.tolist())

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
                output_xattn_masks,
                output_full_text_row_masked_out_masks,
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
        model_id = 0
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model[model_id].configuration.max_seq_len:
            max_gen_len = self.model[model_id].configuration.max_seq_len - 1

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

    def _get_prefill_user_page_table(self, page_table, kv_cache, prefill_len):
        # Ensure page_table is not padded with extra blocks for paged_fill_cache to work properly
        block_size = get_block_size(kv_cache)
        num_blocks = num_blocks_in_seq(prefill_len, block_size)
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

    # Check if the mesh is 8x4 (expected shape for TG) and perfer row split
    # Submeshes with 8 devices are expected to be in ring topology hence the row split
    if num_rows == 8 and num_cols == 4 and num_rows % data_parallel == 0:
        submeshes = mesh_device.create_submeshes(ttnn.MeshShape(num_rows // data_parallel, num_cols))
        for submesh in submeshes:
            submesh.reshape(ttnn.MeshShape(1, num_devices // data_parallel))
        return submeshes

    return mesh_device.create_submeshes(ttnn.MeshShape(1, num_devices // data_parallel))
