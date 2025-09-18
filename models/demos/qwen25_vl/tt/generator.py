# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
from models.demos.qwen25_vl.tt.common import get_block_size, get_max_prefill_chunk_size, num_blocks_in_seq
from models.tt_transformers.tt.generator import Generator as TTTGenerator


class Generator:
    def __init__(self, model, model_args, mesh_device, processor=None, tokenizer=None):
        """
        Creating a Qwen2_5_Vision wrapper requires only a mesh_device and model_args.
        With model_args you have the checkpoint location, can specify max batch size
        and max seqlen, and other model specific parameters.

        """
        # favor composition over inheritance: __ is convention for private variables
        self._ttt_generator = TTTGenerator([model], [model_args], mesh_device, processor=processor, tokenizer=tokenizer)

    @property
    def model(self):
        # todo)) change this when implementing data parallelism
        return self._ttt_generator.model[0]

    @property
    def model_args(self):
        # todo)) change this when implementing data parallelism
        return self._ttt_generator.model_args[0]

    @property
    def mesh_device(self):
        return self._ttt_generator.mesh_device

    @property
    def tokenizer(self):
        return self._ttt_generator.tokenizer

    @property
    def processor(self):
        return self._ttt_generator.processor

    def prefill_forward_text(self, tokens: torch.Tensor, rot_mats, page_table=None, kv_cache=None, prompt_lens=None):
        batch, batch_seq_len = tokens.shape[:2]
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

            if page_table is not None:
                page_table_user = self._ttt_generator._get_prefill_user_page_table(page_table, kv_cache, seq_len)

            logits = self.__prefill_forward_single_user_text(
                tokens[user_id : user_id + 1],
                page_table=page_table_user if page_table is not None else None,
                user_id=user_id,
                last_token_idx=last_token_idx,
                rot_mats=rot_mats,
                kv_cache=kv_cache,
            )

            # Since we give unpadded_seq_len, only the tile containing the last token is returned
            output_logits[user_id] = logits

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")

        return output_logits

    def update_cos_sin(self, cos_matrix_pt=None, sin_matrix_pt=None):
        self.model.rope_setup.update_cos_sin(cos_matrix_pt=cos_matrix_pt, sin_matrix_pt=sin_matrix_pt)

    def update_cos_sin_rows(self, rot_mats_seq_ids):
        for i, (cos, sin) in enumerate(rot_mats_seq_ids):
            self.model.rope_setup.cos_matrix_pt[i] = cos[0]
            self.model.rope_setup.sin_matrix_pt[i] = sin[0]
        self.update_cos_sin()

    def decode_forward_text(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
    ):
        return self._ttt_generator.decode_forward_text(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=[kv_cache],
            enable_trace=enable_trace,
            read_from_device=read_from_device,
            sampling_params=sampling_params,
        )

    def __prefill_forward_single_user_text(self, tokens, page_table, user_id, last_token_idx, rot_mats, kv_cache=None):
        seq_len = tokens.shape[1]
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
                    rot_mats_global=chunk_rot_mats_prefill,
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
            prefill_input, rot_mats_prefill, page_table_tt, _ = self.model.prepare_inputs_prefill(
                tokens,
                rot_mats=rot_mats,
                page_table=page_table,
            )

            tt_logits = self.model.ttnn_prefill_forward(
                prefill_input,
                rot_mats_global=[rm[user_id : user_id + 1, ...] for rm in rot_mats_prefill],
                user_id=user_id,
                page_table=page_table_tt,
                get_last_token=(last_token_idx // 32) * 32,
                kv_cache=kv_cache,
            )

            logits = self.model.process_output_prefill(tt_logits, last_token_idx=(last_token_idx % 32))

            # deallocate device tensors that are not needed by decode
            # [INFO] logits is a torch tensor
            ttnn.deallocate(tt_logits)
            ttnn.deallocate(prefill_input)
            if page_table is not None:
                ttnn.deallocate(page_table_tt)

            return logits

    # [INFO] this is called by vLLM
    def read_decode_output(self, tt_out, async_read=False):
        return self._ttt_generator.read_decode_output(tt_out, async_read=async_read)

    # [INFO] this is called by vLLM
    def process_decode_output_host(self, tt_out, is_tokens=False):
        return self._ttt_generator.process_decode_output_host(tt_out, is_tokens=is_tokens)

    ## Destructor (used to delete ttnn trace if exists)

    def __del__(self):
        if hasattr(self, "trace_id"):
            ttnn.release_trace(self.mesh_device, self.trace_id)

        if hasattr(self, "trace_id_text"):
            ttnn.release_trace(self.mesh_device, self.trace_id_text)

        self._ttt_generator.__del__()
