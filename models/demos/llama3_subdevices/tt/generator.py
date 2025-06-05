# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger
from typing import List
from collections import defaultdict

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
    get_max_prefill_chunk_size,
)

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
        self.empty_slots = list(range(32))
        self.seq_groups_to_batch_slot = {}

    def prefill_forward_text(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        enable_trace=True,
        sampling_params=SamplingParams(temperature=0.0, top_k=-1, top_p=1.0),
        seq_groups=None,
        finished_requests_ids=[],
    ):
        assert sampling_params.temperature == 0, "Currently only supporting greedy decoding (temperature=0) on device"

        if self.model.is_prefill_setup is False:
            self.model.switch_mode("prefill")
        kv_cache = kv_cache[0]
        batch, batch_seq_len = tokens.shape
        output_logits = torch.zeros(batch, 1, 1)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch)
        if page_table is not None:
            assert isinstance(
                page_table, torch.Tensor
            ), "page_table must be a torch.Tensor when passing into prefill_forward"

        for req in finished_requests_ids:
            empty_batch_slot = self.seq_groups_to_batch_slot[req]
            self.empty_slots.append(empty_batch_slot)
            del self.seq_groups_to_batch_slot[req]

        for id, user_id in enumerate(self.empty_slots[:batch]):
            logger.info(f"Prefilling User {user_id + 1}")
            seq_len = int(prompt_lens[id])
            last_token_idx = seq_len - 1

            prefill_seq_len = get_padded_prefill_len(seq_len)
            prefill_ids = torch.cat(
                [tokens[id : id + 1, :seq_len], torch.zeros(1, prefill_seq_len - seq_len).long()], dim=-1
            )
            if page_table is not None:
                page_table_user = self._get_prefill_user_page_table(page_table, kv_cache, prefill_seq_len, user_id)
                # remove the first user from the page table
                page_table = page_table[1:, :]
            prefill_kwargs = {
                "tokens": prefill_ids,
                "page_table": page_table_user if page_table is not None else None,
                "kv_cache": kv_cache,
                "user_id": user_id,
                "last_token_idx": last_token_idx,
            }
            if enable_trace:
                tt_logits = self._easy_trace_prefill(**prefill_kwargs, prefill_seq_len=prefill_seq_len)
            else:
                tt_logits = self.prefill_forward_single_user_text(**prefill_kwargs)

            output_logits[id] = tt_logits

        if seq_groups is not None:
            # update the batch slot table
            recently_filled_slots = self.empty_slots[:batch]
            self.empty_slots = self.empty_slots[batch:]

            for s in seq_groups:
                self.seq_groups_to_batch_slot[s] = recently_filled_slots.pop(0)

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
                    del tt_logits
        else:
            prefill_input, tt_user_id, page_table_tt, _ = self.model.prepare_inputs_prefill(
                tokens,
                user_id=user_id,
                page_table=page_table,
            )
            tt_logits = self.model.ttnn_prefill_forward(
                prefill_input,
                rot_mats=None,
                user_id=tt_user_id,
                page_table=page_table_tt,
                get_last_token=last_token_idx,  # (last_token_idx // 32) * 32,
                kv_cache=kv_cache,
            )

            logits = self.model.process_output_prefill(tt_logits, last_token_idx=last_token_idx)

            return logits

    def _easy_trace_prefill(self, tokens, last_token_idx, prefill_seq_len, page_table=None, kv_cache=None, user_id=0):
        """
        Tracing is easy! Just call this method and we'll handle tracing for you.
        """
        if self.trace_id_prefill[prefill_seq_len] is None:
            trace_id, tt_out_trace, *device_inputs = self._capture_trace_prefill(
                tokens, last_token_idx, page_table=page_table, kv_cache=kv_cache, user_id=user_id
            )
            self.trace_id_prefill[prefill_seq_len] = trace_id
            self.trace_inputs_prefill[prefill_seq_len] = device_inputs
            self.trace_output_prefill[prefill_seq_len] = tt_out_trace

        logger.info("Executing prefill trace")
        tt_out_trace = self._prefill_forward_trace_text(
            self.trace_id_prefill[prefill_seq_len],
            self.trace_inputs_prefill[prefill_seq_len],
            self.trace_output_prefill[prefill_seq_len],
            tokens,
            user_id,
            page_table=page_table,
        )
        logits = self.model.process_output_prefill(tt_out_trace, last_token_idx=last_token_idx)
        return logits

    def _capture_trace_prefill(
        self,
        tokens,
        last_token_idx,
        user_id,
        page_table=None,
        kv_cache=None,
    ):
        """
        Captures a trace for the decode_forward method.
        """

        # Compile run
        # self.prefill_forward_single_user_text(tokens, page_table, user_id, last_token_idx, kv_cache)

        # Get inputs ready for trace run
        host_inputs = self.model.prepare_prefill_inputs_host(tokens, page_table=page_table)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        transformed_inputs = self.model.transform_prefill_inputs_device(*device_inputs)

        tt_out_trace = self.model.ttnn_prefill_forward(
            *transformed_inputs, kv_cache=kv_cache, get_last_token=last_token_idx
        )
        ttnn.synchronize_device(self.mesh_device)
        logger.info("Done Compiling Model")

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        transformed_inputs = self.model.transform_prefill_inputs_device(*device_inputs)
        tt_out_trace = self.model.ttnn_prefill_forward(
            *transformed_inputs, kv_cache=kv_cache, get_last_token=last_token_idx
        )

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        transformed_inputs = self.model.transform_prefill_inputs_device(*device_inputs)

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        transformed_inputs = self.model.transform_prefill_inputs_device(*device_inputs)
        tt_out_trace = self.model.ttnn_prefill_forward(
            *transformed_inputs, kv_cache=kv_cache, get_last_token=last_token_idx
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
    ):
        """
        Executes the trace for the decode_forward method but does not read back outputs.
        """
        host_inputs = self.model.prepare_prefill_inputs_host(tokens, user_id, page_table)

        device_inputs = copy_host_to_device(
            host_tensors=host_inputs,
            device_tensors=device_inputs,
        )

        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        return tt_out_trace

    def decode_forward_text(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params: SamplingParams = None,  # Should be None if not greedy decoding / sampling on device.
        reset_inputs=True,
        argmax_on_device=False,
        seq_groups=None,
        finished_requests_ids=None,
    ):
        assert (
            sampling_params is None or sampling_params.temperature == 0
        ), "Currently only supporting greedy decoding (temperature=0) on device"
        # argmax_on_device = torch.all(start_pos[1:] == -1).item() and (
        #     sampling_params is not None and sampling_params.temperature == 0
        # )
        self.perm_table_tensor = None
        if seq_groups is not None:
            for req in finished_requests_ids:
                empty_batch_slot = self.seq_groups_to_batch_slot[req]
                self.empty_slots.append(empty_batch_slot)
                del self.seq_groups_to_batch_slot[req]

            # Calculate perm_table_tensor: perm_table_tensor[new_idx] = current_slot_idx that should move to new_idx
            perm_table_tensor = torch.as_tensor(
                [self.seq_groups_to_batch_slot[s] for s in seq_groups] + self.empty_slots,
                dtype=torch.long,
                device=start_pos.device,
            )
            self.perm_table_tensor = perm_table_tensor
            # Calculate inverse_perm_indices: inverse_perm_indices[current_slot_idx] = new_idx where current_slot_idx should go
            inverse_perm_indices = torch.empty_like(perm_table_tensor)
            inverse_perm_indices[perm_table_tensor] = torch.arange(
                perm_table_tensor.size(0),
                dtype=torch.long,
                device=perm_table_tensor.device,
            )

            # permute the start_pos, tokens, and page_table
            start_pos = start_pos[inverse_perm_indices]
            tokens = tokens[inverse_perm_indices, :]
            page_table = page_table[inverse_perm_indices, :]

        if self.model.is_decode_setup is False:
            self.model.switch_mode("decode")
        kv_cache = kv_cache[0]
        decode_kwargs = {
            "current_pos": start_pos,
            "tokens": tokens,
            "page_table": page_table,
            "kv_cache": kv_cache,
            "argmax_on_device": argmax_on_device,
        }
        if enable_trace:
            tt_logits = self._easy_trace_text(**decode_kwargs, reset_inputs=reset_inputs)
        else:
            tt_logits = self._decode_forward_no_trace_text(**decode_kwargs)
        if read_from_device:
            tt_logits = self.read_decode_output(tt_logits, tokens.shape[0], is_tokens=(argmax_on_device))

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
        tt_tokens, tt_current_pos, rot_mat_idxs, tt_page_table = self.model.prepare_inputs_decode(
            tokens, current_pos, page_table
        )
        tt_logits = self.model.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mat_idxs=rot_mat_idxs,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            argmax_on_device=argmax_on_device,
        )

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
        host_inputs = self.model.prepare_decode_inputs_host(tokens, current_pos, page_table=page_table)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        tt_out_trace = self.model.ttnn_decode_forward(
            *device_inputs, kv_cache=kv_cache, argmax_on_device=argmax_on_device
        )

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
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        return tt_out_trace

    def _easy_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
        kv_cache=None,
        argmax_on_device=False,
        reset_inputs=False,
    ):
        """
        Tracing is easy! Just call this method and we'll handle tracing for you.
        """
        tokens = tokens.view(-1, 1)
        if not hasattr(self, "trace_id_text"):
            trace_id, tt_out_trace, *device_inputs = self._capture_trace_text(
                tokens, current_pos, page_table=page_table, kv_cache=kv_cache, argmax_on_device=argmax_on_device
            )
            self.trace_id_text = trace_id
            self.trace_inputs_text = device_inputs
            self.trace_output_text = tt_out_trace
        if reset_inputs:
            host_inputs = self.model.prepare_decode_inputs_host(tokens, current_pos, page_table)
            device_inputs = copy_host_to_device(
                host_tensors=host_inputs,
                device_tensors=self.trace_inputs_text,
            )
        trace_logits_rm = self._decode_forward_trace_text(
            self.trace_id_text,
            self.trace_inputs_text,
            self.trace_output_text,
            tokens,
            current_pos,
            page_table=page_table,
        )

        return trace_logits_rm

    def read_decode_output(self, tt_logits, unpadded_batch, is_tokens=False):
        logits = self.model.process_output_decode(tt_logits, B=unpadded_batch, S=1, is_tokens=is_tokens)
        if self.perm_table_tensor is not None:
            logits = logits[self.perm_table_tensor, :]
        return logits

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

    def _get_prefill_user_page_table(self, page_table, kv_cache, prefill_len, user_id):
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
        padded_page_table[user_id, :] = page_table[0, :]
        return padded_page_table

    ## Destructor (used to delete ttnn trace if exists)

    def __del__(self):
        if hasattr(self, "trace_id"):
            ttnn.release_trace(self.mesh_device, self.trace_id)

        if hasattr(self, "trace_id_text"):
            ttnn.release_trace(self.mesh_device, self.trace_id_text)

        self.model.tt_ccl.close()

        if hasattr(super(Generator, self), "__del__"):
            super().__del__()
