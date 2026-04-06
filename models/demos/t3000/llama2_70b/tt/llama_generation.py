# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import copy
import math

import torch
from loguru import logger

import ttnn
from models.demos.t3000.llama2_70b.tt.llama_common import BASE_URL
from models.demos.t3000.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized as TtLlamaModel
from models.demos.t3000.llama2_70b.tt.model_config import get_model_config
from ttnn import ConcatMeshToTensor


class TtLlamaModelForGeneration:
    def __init__(self, configuration, state_dict, model_args, tt_args, paged_attention_config=None, vllm=False):
        # Cache Weights setup
        n_layers = model_args.num_layers or 80

        self.params = copy.deepcopy(configuration)

        self.llama_version = model_args.llama_version
        self.max_batch_size = model_args.max_batch_size
        self.max_kv_context_len = model_args.max_kv_context_len

        self.mesh_device = tt_args.mesh_device

        # Initial model_config is set in decode mode
        model_config = get_model_config(
            llama_version=self.llama_version,
            max_batch_size=self.max_batch_size,
            max_context_len=self.max_kv_context_len,
            vllm=vllm,
        )
        self.model_config = model_config

        # TT model -------------------------------------------------------------
        self.tt_model = TtLlamaModel(
            self.mesh_device,
            state_dict,
            BASE_URL,
            n_layers,
            model_config,
            self.params,
            cache_path=tt_args.cache_path,
            read_cache=False,
            paged_attention_config=paged_attention_config,
            vllm=vllm,
        )

        del state_dict

    def forward(self, tokens: torch.Tensor, start_pos: int, page_table=None, kv_cache=None, prompt_lens=None):
        _, seq_len = tokens.shape
        if seq_len == 1:
            return self.decode_forward(tokens, start_pos, page_table=page_table, kv_cache=kv_cache)
        else:
            return self.prefill_forward(
                tokens, start_pos, page_table=page_table, kv_cache=kv_cache, prompt_lens=prompt_lens
            )

    def capture_trace(self, tokens: torch.Tensor, start_pos: int, page_table=None, kv_cache=None):
        # Get inputs on device
        (
            tt_inp_emb,
            start_pos,
            rot_mat,
            cache_idxs_tt,
            tt_page_table,
            tt_inp,
            rot_idxs_tt,
        ) = self.tt_model.prepare_device_inputs_decode(
            tokens,
            start_pos,
            mode="decode",
            page_table=page_table,
            return_tokens=True,
            return_rot_idxs=True,
        )

        # Compile model
        tt_logits = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            cache_idxs=cache_idxs_tt,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            mode="decode",
        )
        logger.info("Done Compiling Model")

        # Capture trace
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        # Run TT model
        tt_inp_emb = self.tt_model.tt_embd(tt_inp)
        tt_inp_emb = ttnn.interleaved_to_sharded(tt_inp_emb, self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"])
        rot_mat = self.tt_model.rope_setup_decode.get_rot_mats(rot_idxs_tt)
        tt_logits = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            cache_idxs=cache_idxs_tt,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            mode="decode",
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Decode Trace")

        return trace_id, tt_inp, rot_idxs_tt, cache_idxs_tt, tt_logits, tt_page_table

    def delete_trace(self, trace_id):
        ttnn.release_trace(self.mesh_device, trace_id)

    def decode_forward_trace(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        trace_id,
        tt_inp,
        rot_idxs_tt,
        cache_idxs_tt,
        tt_logits,
        page_table=None,
        tt_page_table=None,
        read_from_device=True,
    ):
        batch = tokens.shape[0]

        # Update preallocated tensors
        (
            updated_tt_inp,
            start_pos,
            _,
            updated_rot_idxs_tt,
            updated_cache_idxs_tt,
            updated_tt_page_table,
        ) = self.tt_model.prepare_inputs(tokens, start_pos, mode="decode", page_table=page_table)
        ttnn.copy_host_to_device_tensor(updated_tt_inp, tt_inp)
        ttnn.copy_host_to_device_tensor(updated_rot_idxs_tt, rot_idxs_tt)
        ttnn.copy_host_to_device_tensor(updated_cache_idxs_tt, cache_idxs_tt)
        if page_table is not None:
            ttnn.copy_host_to_device_tensor(updated_tt_page_table, tt_page_table)

        # Run TT model
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
        if read_from_device:
            logits = self.read_decode_output(tt_logits, unpadded_batch=batch)
            return logits
        else:
            return tt_logits

    def read_decode_output(self, tt_logits, unpadded_batch=None):
        updated_tt_logits = ttnn.from_device(tt_logits)

        logits = self._process_logits(updated_tt_logits)

        logits = logits.permute(2, 1, 0, 3).squeeze().unsqueeze(1)  # [batch, 1, vocab_size]
        if unpadded_batch is not None:
            logits = logits[:unpadded_batch]  # Remove padded users

        return logits

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        page_table=None,
        kv_cache=None,
        enable_trace=False,
        read_from_device=True,
    ):
        batch = tokens.shape[0]

        if not enable_trace:
            # Get inputs on device
            tt_inp_emb, start_pos, rot_mat, cache_idxs_tt, tt_page_table = self.tt_model.prepare_device_inputs_decode(
                tokens, start_pos, mode="decode", page_table=page_table
            )

            tt_logits = self.tt_model(
                tt_inp_emb,
                rot_mat,
                start_pos,
                cache_idxs=cache_idxs_tt,
                page_table=tt_page_table,
                kv_cache=kv_cache,
                mode="decode",
            )
        else:
            tt_logits = self._easy_trace(tokens, start_pos, page_table, kv_cache)

        if read_from_device:
            return self.read_decode_output(tt_logits, unpadded_batch=batch)
        else:
            return tt_logits

    def _easy_trace(self, tokens, start_pos, page_table=None, kv_cache=None):
        """
        Tracing is easy! Just call this method and we'll handle tracing for you.
        """
        if not hasattr(self, "trace_id"):
            trace_id, tt_inp, rot_idxs_tt, cache_idxs_tt, tt_logits, tt_page_table = self.capture_trace(
                tokens, start_pos, page_table=page_table, kv_cache=kv_cache
            )
            self.trace_id = trace_id
            self.trace_inputs = {
                "tt_inp": tt_inp,
                "rot_idxs_tt": rot_idxs_tt,
                "cache_idxs_tt": cache_idxs_tt,
                "tt_page_table": tt_page_table,
            }
            self.trace_output = tt_logits

        trace_logits_rm = self.decode_forward_trace(
            tokens,
            start_pos,
            self.trace_id,
            self.trace_inputs["tt_inp"],
            self.trace_inputs["rot_idxs_tt"],
            self.trace_inputs["cache_idxs_tt"],
            self.trace_output,
            page_table=page_table,
            tt_page_table=self.trace_inputs["tt_page_table"],
            read_from_device=False,
        )

        return trace_logits_rm

    def prefill_forward_single_user(
        self, tokens: torch.Tensor, start_pos: int, user_id: int, last_token_idx=None, page_table=None, kv_cache=None
    ):
        batch, seq_len = tokens.shape
        assert batch == 1
        assert start_pos == 0, "start_pos must be 0 for prefill_forward_single_user"

        use_chunked_prefill = seq_len > self.tt_model.model_config["MAX_PREFILL_SEQ_LEN"]
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
            chunk_size = get_max_prefill_chunk_size(seq_len, self.tt_model.model_config["MAX_PREFILL_SEQ_LEN"])
            block_size = get_block_size(kv_cache)
            last_token_idx_in_chunk = last_token_idx % chunk_size if last_token_idx is not None else None
            # Calculate which chunk contains the last_token_idx
            last_chunk_start = (last_token_idx // chunk_size) * chunk_size if last_token_idx is not None else None
            page_table_user = page_table[user_id : user_id + 1, :]
            # Pad page table to match number of blocks in seq_len
            num_padding_blocks = num_blocks_in_seq(seq_len, block_size) - page_table_user.shape[1]
            page_table_user_padded = torch.cat(
                [page_table_user, torch.zeros(1, num_padding_blocks, dtype=torch.int32)], dim=-1
            )
            CHUNK_USER_ID = 0

            logits_list = []
            for chunk_start in range(0, seq_len, chunk_size):
                chunk_end = chunk_start + chunk_size
                assert (
                    chunk_end <= seq_len
                ), f"Chunk end should be less than seq_len, got chunk_end={chunk_end} and seq_len={seq_len}"
                chunk_tokens = tokens[:, chunk_start:chunk_end]
                chunk_page_table = page_table_user[:, chunk_start // block_size : chunk_end // block_size]

                (
                    tt_inp_emb,
                    start_pos,
                    rot_mat,
                    _rot_idxs_tt,
                    _cache_idxs_tt,
                    page_table_tt,
                    chunk_page_table_tt,
                ) = self.tt_model.prepare_inputs(
                    chunk_tokens,
                    start_pos=chunk_start,
                    mode="prefill",
                    page_table=page_table_user_padded,
                    chunk_page_table=chunk_page_table,
                )
                tt_logits = self.tt_model(
                    tt_inp_emb,
                    rot_mat,
                    start_pos,
                    user_id=CHUNK_USER_ID,
                    last_token_idx=last_token_idx_in_chunk,
                    page_table=page_table_tt,
                    kv_cache=kv_cache,
                    mode="prefill",
                    chunk_page_table=chunk_page_table_tt,
                    chunk_start_idx=chunk_start,
                )

                logits = self._process_logits(tt_logits)
                logits = logits.squeeze(1)
                ttnn.deallocate(tt_logits)

                if last_token_idx is not None:
                    # If this was the chunk containing last_token_idx, we're done
                    if chunk_start == last_chunk_start:
                        return logits
                else:
                    logits_list.append(logits)

            # Concatenate all logits
            logits = torch.cat(logits_list, dim=-2)
            return logits

        else:
            (
                tt_inp_emb,
                start_pos,
                rot_mat,
                _rot_idxs_tt,
                _cache_idxs_tt,
                tt_page_table,
                _chunk_page_table,
            ) = self.tt_model.prepare_inputs(
                tokens, start_pos=start_pos, valid_seq_len=seq_len, mode="prefill", page_table=page_table
            )

            tt_logits = self.tt_model(
                tt_inp_emb,
                rot_mat,
                start_pos,
                user_id=user_id,
                last_token_idx=last_token_idx,
                page_table=tt_page_table,
                kv_cache=kv_cache,
                mode="prefill",
            )

            del tt_inp_emb
            del rot_mat
            del tt_page_table

            logits = self._process_logits(tt_logits)
            logits = logits.squeeze(1)
            del tt_logits
            return logits

    def prefill_forward(self, tokens: torch.Tensor, start_pos: int, page_table=None, kv_cache=None, prompt_lens=None):
        batch, batch_seq_len = tokens.shape
        output_logits = torch.zeros(batch, 1, self.params.vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch)

        if page_table is not None:
            assert isinstance(
                page_table, torch.Tensor
            ), "page_table must be a torch.Tensor when passing into prefill_forward"

        for user_id in range(batch):
            seq_len = prompt_lens[user_id]
            last_token_idx = seq_len - 1

            prefill_seq_len = get_padded_prefill_len(seq_len)
            prefill_ids = torch.cat(
                [tokens[user_id : user_id + 1, :seq_len], torch.zeros(1, prefill_seq_len - seq_len).long()], dim=-1
            )
            if page_table is not None:
                page_table_user = _get_prefill_user_page_table(page_table, kv_cache, seq_len)

            logger.info(f"Filling kv cache for user {user_id + 1}")

            logits = self.prefill_forward_single_user(
                prefill_ids,
                start_pos,
                user_id,
                last_token_idx=last_token_idx,
                page_table=page_table_user if page_table is not None else None,
                kv_cache=kv_cache,
            )

            # Since we give unpadded_seq_len, only the tile containing the last token is returned
            output_logits[user_id] = logits[:, last_token_idx % 32 : last_token_idx % 32 + 1, :]

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")

        return output_logits

    def _process_logits(self, tt_logits):
        logits = ttnn.to_torch(
            tt_logits, device=self.mesh_device, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=3)
        )
        return logits[..., : self.params.vocab_size].float()

    ## Destructor (used to delete ttnn trace if exists)

    def __del__(self):
        if hasattr(self, "trace_id"):
            self.delete_trace(self.trace_id)

        if hasattr(super(TtLlamaModelForGeneration, self), "__del__"):
            super().__del__()


def _get_prefill_user_page_table(page_table, kv_cache, prefill_len):
    # Ensure page_table is not padded with extra blocks for paged_fill_cache to work properly
    block_size = get_block_size(kv_cache)
    num_blocks = num_blocks_in_seq(prefill_len, block_size)
    return page_table[:, :num_blocks]


def get_padded_prefill_len(seq_len):
    """
    If seq_len is less than 32, pad to 32
    If seq_len is more than 32, pad to whichever is smaller: a power of 2 or a multiple of 1024
    TODO: Generalize for max_mm_seq_len different from 1024
    """
    if seq_len <= 32:
        return 32
    pow_2_pad = nearest_pow_2(seq_len)
    mult_1024_pad = 1024 * math.ceil(seq_len / 1024)
    min_extended_pad = min(pow_2_pad, mult_1024_pad)
    return min_extended_pad


def get_block_size(kv_cache):
    return kv_cache[0][0].shape[2]


def num_blocks_in_seq(seq_len, block_size):
    return math.ceil(seq_len / block_size)


def nearest_pow_2(x):
    return 2 ** math.ceil(math.log2(x))


def get_max_prefill_chunk_size(seq_len, max_prefill_seq_len):
    """
    Determine the largest multiple of 1024 that divides `seq_len` and is less than or equal to `max_prefill_seq_len`.

    **Assumptions**:
    - `seq_len` is a multiple of 1024.
    - `max_prefill_seq_len` is a multiple of 1024.
    """

    if not isinstance(seq_len, int) or not isinstance(max_prefill_seq_len, int):
        raise TypeError("Both seq_len and max_prefill_seq_len must be integers.")
    if seq_len <= 0 or max_prefill_seq_len <= 0:
        raise ValueError("Both seq_len and max_prefill_seq_len must be positive integers.")

    if seq_len % 1024 != 0:
        raise ValueError("seq_len must be a multiple of 1024.")
    if max_prefill_seq_len % 1024 != 0:
        raise ValueError("max_prefill_seq_len must be a multiple of 1024.")

    # Calculate the maximum possible chunk size
    # It cannot exceed either max_prefill_seq_len or seq_len
    max_possible_chunk = min(max_prefill_seq_len, seq_len)

    # Iterate from the largest possible multiple of 1024 down to 1024
    for chunk_size in range(max_possible_chunk, 0, -1024):
        if seq_len % chunk_size == 0:
            return chunk_size

    raise ValueError("No valid chunk size found")
