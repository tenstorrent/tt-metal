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
        self.prev_page_table = None
        self.prefill_traces_warmup = False
        self.trace_ids_decode = defaultdict(lambda: None)  # {return_logits: {device_id: trace_id}}
        self.trace_inputs_decode = defaultdict(lambda: None)
        self.trace_output_decode = defaultdict(lambda: None)
        self.enable_split_sampling = True
        self.model.enable_internal_trace = self.enable_split_sampling

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

        logger.info("Warming up prefill traces for all supported sequence lengths")
        for supported_length in self.model.tt_ccl.support_seqlens:
            logger.info(f"Creating warmup tensor for sequence length: {supported_length}")
            # Capture trace for both
            for batch in (1, 32):  # TODO add proper support for batched prefill == b-32
                # For batched prefill this needs to be *32
                if batch == 32 and supported_length == 4096:
                    # For batched prefill max batch sequence length is 2048 or lower (128k limit)
                    logger.info(f"Skipping warm up step on batched prefill for sequence length {supported_length}")
                    continue
                if batch == 32:
                    current_batch = page_table.shape[0]
                    if current_batch < batch:
                        pad_rows = batch - current_batch
                        padding = torch.full((pad_rows, page_table.shape[1]), -1, dtype=torch.int32)
                        warmup_page_table = torch.cat([page_table, padding], dim=0)
                    else:
                        warmup_page_table = page_table
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
        # trace_id_prefill dict check
        logger.info("Prefill traces warmup completed")

    def prefill_forward_text(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        enable_trace=True,
        sampling_params=None,
        empty_slots=None,
        tt_out_logits_all_users=None,
        start_pos: list[int] = None,  # Cached prefixes lengths, ignored for now
    ):
        assert (start_pos is None) or all(
            x == 0 for x in start_pos
        ), f"Prefix caching is not supported for llama3_70b_galaxy, got start_pos: {start_pos}"

        if self.prefill_traces_warmup is False:
            self.warmup_prefill_traces(
                tokens,
                page_table,
                kv_cache,
                prompt_lens,
                enable_trace,
                sampling_params,
                empty_slots,
                tt_out_logits_all_users,
            )

        if sampling_params is None:
            return_logits = True
        else:
            return_logits = False
        if self.model.is_prefill_setup is False:
            self.model.switch_mode("prefill")

        kv_cache = kv_cache[0]
        batch, batch_seq_len = tokens.shape
        output_toks = torch.zeros(batch, 1, 1)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch)
        if not isinstance(prompt_lens, list):
            prompt_lens = prompt_lens.tolist()
        prefill_seq_lens = [get_padded_prefill_len(seq_len) for seq_len in prompt_lens]
        if page_table is not None:
            assert isinstance(
                page_table, torch.Tensor
            ), "page_table must be a torch.Tensor when passing into prefill_forward"

        if empty_slots is None:
            empty_slots = list(range(batch))

        # If batch is 32 and prompt_lens are all the same and batch_seq_len* batch is less than 128*1024, use batched prefill
        use_batched_prefill = False
        if (
            batch == 32
            and len(set(prefill_seq_lens)) == 1
            and prefill_seq_lens[0] * batch < 128 * 1024
            and tt_out_logits_all_users is None
            and not return_logits
        ):
            use_batched_prefill = True

        if return_logits:
            tt_out_logits_all_users = torch.zeros(batch, 1, self.model.args.padded_vocab_size)

        all_users = [0] if use_batched_prefill else empty_slots

        for id, user_id in enumerate(all_users):
            logger.info(f"Prefilling User {user_id + 1}, use_batched_prefill: {use_batched_prefill}")
            if use_batched_prefill:
                user_id = empty_slots
                last_token_idx = [(seq_len - 1) for seq_len in prompt_lens]
                prefill_seq_len = prefill_seq_lens[0]
                seq_len = prompt_lens
            else:
                seq_len = int(prompt_lens[id])
                last_token_idx = seq_len - 1
                prefill_seq_len = prefill_seq_lens[id]

                if prefill_seq_len not in self.model.tt_ccl.support_seqlens:
                    enable_trace = False

            if use_batched_prefill:
                # reordering the tokens when empty_slots are not sequential (from vllm)
                inverse_empty_slots = [empty_slots.index(i) for i in range(batch)]
                prefill_ids = torch.cat(
                    [
                        torch.cat(
                            [tokens[id : id + 1, : seq_len[id]], torch.zeros(1, prefill_seq_len - seq_len[id]).long()],
                            dim=-1,
                        )
                        for id in inverse_empty_slots
                    ],
                    dim=-1,
                )
                last_token_idx = [last_token_idx[id] for id in inverse_empty_slots]
            else:
                prefill_ids = torch.cat(
                    [tokens[id : id + 1, :seq_len], torch.zeros(1, prefill_seq_len - seq_len).long()], dim=-1
                )
            if page_table is not None:
                page_table_user = self._get_prefill_user_page_table(
                    page_table, kv_cache, prefill_seq_len, user_id, use_batched_prefill
                )
                # remove the first user from the page table
                page_table = page_table[1:, :]

            prefill_kwargs = {
                "tokens": prefill_ids,
                "page_table": page_table_user if page_table is not None else None,
                "kv_cache": kv_cache,
                "user_id": 0 if use_batched_prefill else user_id,
                "last_token_idx": last_token_idx,
                "batch_size": batch if use_batched_prefill else 1,
            }

            # If PCC check enabled or return_logits is True (we save output logits)
            if tt_out_logits_all_users is not None or return_logits:
                tt_out_logits_saved = torch.zeros(1, self.model.args.padded_vocab_size)
                prefill_kwargs["tt_out_logits_saved"] = tt_out_logits_saved

            if enable_trace:
                tt_tok = self._easy_trace_prefill(**prefill_kwargs, prefill_seq_len=prefill_seq_len)
            else:
                tt_tok = self.prefill_forward_single_user_text(**prefill_kwargs)
            if use_batched_prefill:
                # reverse the reordering of the tokens when empty_slots are not sequential (from vllm)
                output_toks = torch.cat(tt_tok, dim=0).reshape(batch, 1, 1)[empty_slots]
            else:
                output_toks[id] = tt_tok

            if tt_out_logits_all_users is not None and tt_out_logits_saved is not None:
                tt_out_logits_all_users[id] = tt_out_logits_saved

        if return_logits:
            # TODO: the current solution runs the argmax even if we are returning logits
            # This is inefficient and should be fixed
            # Return logits instead of tokens
            # batch x seq x padded_vocab_size -> batch x seq x vocab_size
            tt_out_logits_all_users = tt_out_logits_all_users[:, :, : self.model.vocab_size]
            return tt_out_logits_all_users

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")
        return output_toks

    def prefill_forward_single_user_text(
        self, tokens, page_table, user_id, last_token_idx, kv_cache=None, tt_out_logits_saved=None, batch_size=1
    ):
        seq_len = tokens.shape[-1]

        prefill_input, tt_user_id, page_table_tt, _ = self.model.prepare_inputs_prefill(
            tokens,
            user_id=user_id,
            page_table=page_table,
            batch_size=batch_size,
        )

        tt_toks = self.model.ttnn_prefill_forward(
            prefill_input,
            rot_mats=None,
            user_id=tt_user_id,
            page_table=page_table_tt,
            get_last_token=last_token_idx,  # (last_token_idx // 32) * 32,
            kv_cache=kv_cache,
            batch_size=batch_size,
        )

        tt_toks = self.model.process_output_prefill(
            tt_toks, last_token_idx=last_token_idx, tt_out_logits_saved=tt_out_logits_saved
        )

        return tt_toks

    def _easy_trace_prefill(
        self,
        tokens,
        last_token_idx,
        prefill_seq_len,
        page_table=None,
        kv_cache=None,
        user_id=0,
        batch_size=1,
        tt_out_logits_saved=None,
    ):
        """
        Tracing is easy! Just call this method and we'll handle tracing for you.
        """
        # Trace is independent of whether we are returning logits or sampling on device
        # The difference happens outside of the trace, in process_output_prefill
        trace_key = f"{prefill_seq_len}_{batch_size}"
        if self.trace_id_prefill[trace_key] is None:
            trace_id, tt_out_trace, *device_inputs = self._capture_trace_prefill(
                tokens, last_token_idx, page_table=page_table, kv_cache=kv_cache, user_id=user_id, batch_size=batch_size
            )
            self.trace_id_prefill[trace_key] = trace_id
            self.trace_inputs_prefill[trace_key] = device_inputs
            self.trace_output_prefill[trace_key] = tt_out_trace

        logger.info("Executing prefill trace")
        tt_out_trace = self._prefill_forward_trace_text(
            self.trace_id_prefill[trace_key],
            self.trace_inputs_prefill[trace_key],
            self.trace_output_prefill[trace_key],
            tokens,
            user_id,
            page_table=page_table,
            batch_size=batch_size,
        )
        toks = self.model.process_output_prefill(
            tt_out_trace, last_token_idx=last_token_idx, tt_out_logits_saved=tt_out_logits_saved
        )
        return toks

    def _capture_trace_prefill(
        self,
        tokens,
        last_token_idx,
        user_id,
        page_table=None,
        kv_cache=None,
        batch_size=1,
    ):
        """
        Captures a trace for the prefill_forward method.
        """

        # Compile run
        # self.prefill_forward_single_user_text(tokens, page_table, user_id, last_token_idx, kv_cache)

        # Get inputs ready for trace run
        host_inputs = self.model.prepare_prefill_inputs_host(tokens, page_table=page_table, batch_size=batch_size)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        transformed_inputs = self.model.transform_prefill_inputs_device(*device_inputs)

        tt_out_trace = self.model.ttnn_prefill_forward(
            *transformed_inputs, kv_cache=kv_cache, get_last_token=last_token_idx, batch_size=batch_size
        )
        ttnn.synchronize_device(self.mesh_device)
        logger.info("Done Compiling Model")

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        transformed_inputs = self.model.transform_prefill_inputs_device(*device_inputs)
        tt_out_trace = self.model.ttnn_prefill_forward(
            *transformed_inputs, kv_cache=kv_cache, get_last_token=last_token_idx, batch_size=batch_size
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
        batch_size=1,
    ):
        """
        Executes the trace for the prefill_forward method but does not read back outputs.
        """
        host_inputs = self.model.prepare_prefill_inputs_host(tokens, user_id, page_table, batch_size=batch_size)

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
        async_read=False,
        sampling_params: SamplingParams = None,  # Should be None if not greedy decoding / sampling on device.
        reset_inputs=False,
        tt_out_logits_saved=None,
        is_cur_pos_sharded=False,
        is_page_table_sharded=False,
        reset_batch=False,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
    ):
        if sampling_params is None:
            return_logits = True
            reset_inputs = True
        else:
            return_logits = False

        if self.prev_page_table is None:
            self.prev_page_table = (
                page_table.clone()
            )  # Make sure we reference a fresh page table, in case it has changed
        if torch.any(self.prev_page_table != page_table).item():
            reset_inputs = True
            self.prev_page_table = (
                page_table.clone()
            )  # Make sure we reference a fresh page table, in case it has changed

        if self.model.is_decode_setup is False:
            self.model.switch_mode("decode")
            reset_inputs = True

        kv_cache = kv_cache[0]
        decode_kwargs = {
            "current_pos": start_pos,
            "tokens": tokens,
            "page_table": page_table,
            "kv_cache": kv_cache,
            "is_cur_pos_sharded": is_cur_pos_sharded,
            "is_page_table_sharded": is_page_table_sharded,
        }
        if reset_inputs and sampling_params is not None:
            sampling_params = format_sampling_params(sampling_params, self.model_args.max_batch_size)

            sampling_module = self.model.sampling
            sampling_module.reset_sampling_params(sampling_params)
            if reset_batch:
                sampling_module.reset_prompt_tokens(prompt_tokens)
                sampling_module.reset_output_state(output_tokens)
                sampling_module.reset_seed(sampling_params.seed)

        if tt_out_logits_saved is not None:
            decode_kwargs["tt_out_logits_saved"] = tt_out_logits_saved
        if enable_trace:
            tt_tok = self._decode_easy_trace_text(
                **decode_kwargs,
                reset_inputs=reset_inputs,
                return_logits=return_logits,
            )
        else:
            tt_tok = self._decode_forward_no_trace_text(
                **decode_kwargs,
                return_logits=return_logits,
            )

        if read_from_device:
            tt_out = self.read_decode_output(tt_tok, async_read=async_read)
            if async_read:
                tt_tok, read_event = tt_out
                return tt_tok, read_event
            else:
                return self.process_decode_output_host(tt_out, is_tokens=(not return_logits))

        return tt_tok

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

        # Try allocating our persistent tensors here and verifying it matches the address that trace captured
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Decode Trace")

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

    def read_decode_output(self, tt_out, async_read=False):
        if not async_read:
            if isinstance(tt_out, tuple):
                # Get logits and skip log-probs
                tt_out = tt_out[0]
            return tt_out.cpu()

        logits, read_event = self.model.process_output_decode(tt_out)
        return logits, [read_event]

    def process_decode_output_host(self, tt_out, is_tokens=True):
        if isinstance(tt_out, tuple):
            tt_out = tt_out[0]
        tt_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
        # Check if tensor is distributed across mesh devices (vocab_size // 8 indicates sharding)
        # If so, convert from distributed TT tensor to consolidated torch tensor
        if tt_out.shape[-1] >= self.model.vocab_size // 8:
            ttnn.synchronize_device(self.mesh_device)
            return tt_out[0, 0, :, : self.model.vocab_size].unsqueeze(1)

        # If not sharded (it is a sampled token), convert directly from device tensor to torch tensor
        return tt_out[0, 0, 0, :]

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
        # in case of multiple sampling parameters, we need to warmup for each one
        for s in sampling_params:
            self.warmup_prefill_traces(
                tokens=None,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=None,
                enable_trace=enable_trace,
                sampling_params=s,
                empty_slots=None,
                tt_out_logits_all_users=None,
            )

    ## Destructor (used to delete ttnn trace if exists)

    def __del__(self):
        if hasattr(self, "trace_id"):
            ttnn.release_trace(self.mesh_device, self.trace_id)

        if hasattr(self, "trace_id_text"):
            ttnn.release_trace(self.mesh_device, self.trace_id_text)

        self.model.tt_ccl.close()

        if hasattr(super(Generator, self), "__del__"):
            super().__del__()
