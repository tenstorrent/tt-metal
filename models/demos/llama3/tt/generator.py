# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger

from llama_models.llama3.api.datatypes import (
    InterleavedTextMedia,
    StopReason,
)

from llama_models.llama3.reference_impl.generation import (
    ChatPrediction,
    CompletionPrediction,
    TokenResult,
    sample_top_p,
)
from models.demos.llama3.tt.llama_common import (
    copy_host_to_device,
    get_padded_prefill_len,
    num_blocks_in_seq,
    get_block_size,
    get_max_prefill_chunk_size,
)


class LlamaGenerator:
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

    def prefill_forward_text(self, tokens: torch.Tensor, page_table=None, kv_cache=None, prompt_lens=None):
        batch, batch_seq_len = tokens.shape
        output_logits = torch.zeros(batch, 1, self.model_args.vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch)

        if page_table is not None:
            assert isinstance(
                page_table, torch.Tensor
            ), "page_table must be a torch.Tensor when passing into prefill_forward"

        for user_id in range(batch):
            logger.info(f"Prefilling User {user_id + 1}")
            seq_len = prompt_lens[user_id]
            last_token_idx = seq_len - 1

            prefill_seq_len = get_padded_prefill_len(seq_len)
            prefill_ids = torch.cat(
                [tokens[user_id : user_id + 1, :seq_len], torch.zeros(1, prefill_seq_len - seq_len).long()], dim=-1
            )
            if page_table is not None:
                page_table_user = self._get_prefill_user_page_table(page_table, kv_cache, seq_len)

            logits = self.prefill_forward_single_user_text(
                prefill_ids,
                page_table=page_table_user if page_table is not None else None,
                user_id=user_id,
                last_token_idx=last_token_idx,
                kv_cache=kv_cache,
            )

            # Since we give unpadded_seq_len, only the tile containing the last token is returned
            output_logits[user_id] = logits

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")

        return output_logits

    def prefill_forward_single_user_text(self, tokens, page_table, user_id, last_token_idx, kv_cache=None):
        seq_len = tokens.shape[-1]
        use_chunked_prefill = seq_len > self.model_args.max_prefill_chunk_size
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
            chunk_size = get_max_prefill_chunk_size(seq_len, self.model_args.max_prefill_chunk_size)
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
                ) = self.model.prepare_inputs_prefill(
                    chunk_tokens,
                    start_pos=chunk_start,
                    page_table=page_table_user_padded,
                    chunk_page_table=chunk_page_table,
                )
                tt_logits = self.model.ttnn_prefill_forward(
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
                    logits = self.model.process_output_prefill(tt_logits, last_token_idx=(last_token_idx_in_chunk % 32))
                    return logits
        else:
            prefill_input, rot_mats_prefill, page_table_tt, _ = self.model.prepare_inputs_prefill(
                tokens,
                page_table=page_table,
            )

            tt_logits = self.model.ttnn_prefill_forward(
                prefill_input,
                rot_mats=rot_mats_prefill,
                user_id=user_id,
                page_table=page_table_tt,
                get_last_token=(last_token_idx // 32) * 32,
                kv_cache=kv_cache,
            )

            logits = self.model.process_output_prefill(tt_logits, last_token_idx=(last_token_idx % 32))

            return logits

    def decode_forward_text(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
    ):
        decode_kwargs = {
            "current_pos": start_pos,
            "tokens": tokens,
            "page_table": page_table,
            "kv_cache": kv_cache,
        }
        if enable_trace:
            tt_logits = self._easy_trace_text(**decode_kwargs)
        else:
            tt_logits = self._decode_forward_no_trace_text(**decode_kwargs)

        if read_from_device:
            return self.read_decode_output(tt_logits, tokens.shape[0])
        else:
            return tt_logits

    def _decode_forward_no_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
    ):
        """
        Performs text decode step.
        Returns tt_logits on device
        """
        tt_tokens, tt_current_pos, tt_rot_mats, tt_page_table = self.model.prepare_inputs_decode(
            tokens, current_pos, page_table
        )

        tt_logits = self.model.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mats=tt_rot_mats,
            page_table=tt_page_table,
            kv_cache=kv_cache,
        )

        return tt_logits

    def _capture_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
    ):
        """
        Captures a trace for the decode_forward method.
        """

        # Compile run
        self._decode_forward_no_trace_text(tokens, current_pos, page_table=page_table, kv_cache=kv_cache)
        logger.info("Done Compiling Model")

        # Get inputs ready for trace run
        host_inputs = self.model.prepare_decode_inputs_host(tokens, current_pos, page_table=page_table)

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        transformed_inputs = self.model.transform_decode_inputs_device(*device_inputs)
        tt_out_trace = self.model.ttnn_decode_forward(*transformed_inputs, kv_cache=kv_cache)

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Decode Trace")

        return trace_id, tt_out_trace, *device_inputs

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
        host_inputs = self.model.prepare_decode_inputs_host(tokens, current_pos, page_table)

        device_inputs = copy_host_to_device(
            host_tensors=host_inputs,
            device_tensors=device_inputs,
        )

        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        return tt_out_trace

    def _easy_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
    ):
        """
        Tracing is easy! Just call this method and we'll handle tracing for you.
        """
        if not hasattr(self, "trace_id_text"):
            trace_id, tt_out_trace, *device_inputs = self._capture_trace_text(
                tokens, current_pos, page_table=page_table, kv_cache=kv_cache
            )
            self.trace_id_text = trace_id
            self.trace_inputs_text = device_inputs
            self.trace_output_text = tt_out_trace

        trace_logits_rm = self._decode_forward_trace_text(
            self.trace_id_text,
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
    ):
        """
        Performs vision encode step then text prefill.
        Returns (xattn_caches, cross_attention_masks, full_text_row_masked_out_mask, logits)
        """
        B = tokens.shape[0]
        last_token_idx = prefill_len - 1
        vision_tokens, cross_attention_masks, full_text_row_masked_out_mask = self.model.compute_vision_tokens_masks(
            batch_images=[vision_images],
            batch_masks=[vision_mask],
            total_len=total_len,
        )

        if page_table is not None:
            page_table = self._get_prefill_user_page_table(page_table, kv_cache, prefill_len)

        if cross_page_table is not None:
            num_vision_tokens = vision_tokens.shape[2]
            cross_page_table = self._get_prefill_user_page_table(cross_page_table, kv_cache, num_vision_tokens)

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            rot_mats,
            tt_page_table,
            tt_cross_page_table,
        ) = self.model.prepare_inputs_prefill(
            tokens,
            cross_attention_masks,
            full_text_row_masked_out_mask,
            prefill_len=prefill_len,
            page_table=page_table,
            cross_page_table=cross_page_table,
        )

        tt_logits = self.model.ttnn_prefill_forward(
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
        )

        del tt_page_table
        del tt_cross_page_table

        logits = self.model.process_output_prefill(tt_logits, B, last_token_idx=(last_token_idx % 32))

        return xattn_caches, cross_attention_masks, full_text_row_masked_out_mask, logits

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
        output_logits = torch.zeros(batch, 1, self.model_args.vocab_size)
        output_xattn_masks = []
        output_full_text_row_masked_out_masks = []

        for user_id in range(batch):
            logger.info(f"Prefilling User {user_id + 1}")
            seq_len = prompt_lens[user_id]
            (
                xattn_caches,
                cross_attention_masks,
                full_text_row_masked_out_mask,
                logits,
            ) = self._prefill_forward_single_user(
                vision_images=vision_images[user_id],
                vision_mask=vision_masks[user_id],
                tokens=tokens[user_id : user_id + 1, :seq_len],  # Keep batch dimension
                xattn_caches=xattn_caches,
                user_id=user_id,
                total_len=total_lens[user_id],
                prefill_len=seq_len,
                page_table=page_table,
                kv_cache=kv_cache,
                cross_page_table=cross_page_table,
            )
            output_logits[user_id] = logits
            output_xattn_masks.append(cross_attention_masks)
            output_full_text_row_masked_out_masks.append(full_text_row_masked_out_mask)

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")

        return output_logits, output_xattn_masks, output_full_text_row_masked_out_masks

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
            return self.read_decode_output(tt_logits, tokens.shape[0])
        else:
            return tt_logits

    def read_decode_output(self, tt_logits, unpadded_batch):
        logits = self.model.process_output_decode(tt_logits, B=unpadded_batch, S=1)
        return logits

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
        B, S = tokens.shape
        assert S == 1

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_position_id,
            tt_rot_mats,
            tt_page_table,
            tt_cross_page_table,
        ) = self.model.prepare_inputs_decode(
            tokens,
            cross_attention_masks,
            full_text_row_masked_out_mask,
            position_id=position_id,
            page_table=page_table,
            cross_page_table=cross_page_table,
        )

        tt_logits = self.model.ttnn_decode_forward(
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            xattn_caches,
            tt_position_id,
            tt_rot_mats,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            cross_page_table=tt_cross_page_table,
        )

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
        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_position_id,
            tt_rot_mats,
            tt_page_table,
            tt_cross_page_table,
        ) = self.model.prepare_inputs_decode(
            tokens,
            cross_attention_masks,
            full_text_row_masked_out_mask,
            position_id=position_id,
            page_table=page_table,
            cross_page_table=cross_page_table,
        )

        # Compile run
        tt_logits_rm = self.model.ttnn_decode_forward(
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            xattn_caches,
            tt_position_id,
            tt_rot_mats,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            cross_page_table=tt_cross_page_table,
        )
        logger.info("Done Compiling Model")

        # Get inputs ready for trace run
        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_position_id,
            tt_rope_id,
            tt_page_table,
            tt_cross_page_table,
        ) = self.model.prepare_decode_inputs_host(
            tokens,
            cross_attention_masks,
            full_text_row_masked_out_mask,
            position_id,
            page_table=page_table,
            cross_page_table=cross_page_table,
        )

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_position_id,
            tt_rope_id,
            tt_page_table,
            tt_cross_page_table,
        ) = copy_host_to_device(
            (
                tt_h,
                tt_xattn_mask,
                tt_full_text_mask_expand_1NSH,
                tt_position_id,
                tt_rope_id,
                tt_page_table,
                tt_cross_page_table,
            ),
            mesh_device=self.mesh_device,
        )

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        tt_h_trace_input = tt_h
        B = tokens.shape[0]
        # Do on-device transformations of inputs before forward
        (
            tt_h_transform,
            tt_rot_mats,
            tt_xattn_mask_transform,
            tt_full_text_mask_expand_1NSH_transform,
        ) = self.model.transform_decode_inputs_device(
            tt_h,
            tt_rope_id,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            B=B,
        )

        tt_logits_rm = self.model.ttnn_decode_forward(
            tt_h_transform,
            tt_xattn_mask_transform,
            tt_full_text_mask_expand_1NSH_transform,
            xattn_caches,
            tt_position_id,
            tt_rot_mats,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            cross_page_table=tt_cross_page_table,
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Decode Trace")

        return (
            trace_id,
            tt_logits_rm,
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
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
        trace_id,
        trace_logits_rm,
        trace_h,
        trace_xattn_mask,
        trace_full_text_mask_expand_1NSH,
        trace_position_id,
        trace_rope_id,
        trace_page_table,
        trace_cross_page_table,
    ):
        """
        Executes the trace for the decode_forward method but does not read back outputs.
        """
        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_position_id,
            tt_rope_id,
            tt_page_table,
            tt_cross_page_table,
        ) = self.model.prepare_decode_inputs_host(
            tokens,
            cross_attention_masks,
            full_text_row_masked_out_mask,
            position_id=position_id,
            page_table=page_table,
            cross_page_table=cross_page_table,
        )

        copy_host_to_device(
            host_tensors=(
                tt_h,
                tt_xattn_mask,
                tt_full_text_mask_expand_1NSH,
                tt_position_id,
                tt_rope_id,
                tt_page_table,
                tt_cross_page_table,
            ),
            device_tensors=(
                trace_h,
                trace_xattn_mask,
                trace_full_text_mask_expand_1NSH,
                trace_position_id,
                trace_rope_id,
                trace_page_table,
                trace_cross_page_table,
            ),
        )

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
        if not hasattr(self, "trace_id"):
            (
                trace_id,
                tt_logits_rm,
                tt_h,
                tt_xattn_mask,
                tt_full_text_mask_expand_1NSH,
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
            self.trace_id = trace_id
            self.trace_inputs = {
                "tt_h": tt_h,
                "tt_xattn_mask": tt_xattn_mask,
                "tt_full_text_mask_expand_1NSH": tt_full_text_mask_expand_1NSH,
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
            self.trace_id,
            self.trace_outputs["tt_logits_rm"],
            self.trace_inputs["tt_h"],
            self.trace_inputs["tt_xattn_mask"],
            self.trace_inputs["tt_full_text_mask_expand_1NSH"],
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
        xattn_caches = self.model.setup_cache(self.model_args.max_batch_size)
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
        )

        logits = logits.view(1, 1, self.model_args.vocab_size)

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
                [cross_attention_masks],
                [full_text_row_masked_out_mask],
                xattn_caches,
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

    def _get_prefill_user_page_table(self, page_table, kv_cache, prefill_len):
        # Ensure page_table is not padded with extra blocks for paged_fill_cache to work properly
        block_size = get_block_size(kv_cache)
        num_blocks = num_blocks_in_seq(prefill_len, block_size)
        return page_table[:, :num_blocks]

    ## Destructor (used to delete ttnn trace if exists)

    def __del__(self):
        if hasattr(self, "trace_id"):
            ttnn.release_trace(self.mesh_device, self.trace_id)

        if hasattr(self, "trace_id_text"):
            ttnn.release_trace(self.mesh_device, self.trace_id_text)

        if hasattr(super(LlamaGenerator, self), "__del__"):
            super().__del__()
