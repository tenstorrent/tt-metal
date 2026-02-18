# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import collections

import torch
from loguru import logger

import ttnn
from models.common.warmup import WarmupForwardMixin
from models.demos.qwen25_vl.tt.common import get_block_size, get_max_prefill_chunk_size, num_blocks_in_seq
from models.tt_transformers.tt.common import copy_host_to_device
from models.tt_transformers.tt.generator import Generator as TTTGenerator


class Generator(WarmupForwardMixin):
    def __init__(self, model, model_args, mesh_device, processor=None, tokenizer=None):
        """
        Creating a Qwen2_5_Vision wrapper requires only a mesh_device and model_args.
        With model_args you have the checkpoint location, can specify max batch size
        and max seqlen, and other model specific parameters.

        """
        # favor composition over inheritance: __ is convention for private variables
        self._ttt_generator = TTTGenerator([model], [model_args], mesh_device, processor=processor, tokenizer=tokenizer)

        # Trace infrastructure for prefill
        # Keyed by padded sequence length string (e.g., "4096")
        self.trace_id_prefill = collections.defaultdict(lambda: None)
        self.trace_inputs_prefill = {}
        self.trace_output_prefill = {}

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

    def _capture_trace_prefill(self, tokens_embd, rot_mats_user, page_table_user, kv_cache):
        """
        Capture a prefill trace for a given sequence length.

        Steps:
        1. Create host tensors via prepare_prefill_inputs_trace
        2. Copy to device and run forward pass (compile)
        3. Copy to device again and capture trace

        The trace includes a to_memory_config copy of each input tensor at the
        start. This ensures the input device buffers (which live in normal device
        memory) are only READ by the trace, not consumed. Without this, the trace
        system may reclaim the input buffer, making subsequent copy_host_to_device
        calls fail with "Buffer must be allocated on device".
        """
        host_inputs = self.model.prepare_prefill_inputs_trace(
            tokens_embd, rot_mats=rot_mats_user, page_table=page_table_user
        )

        # Compile run: copy to device and run forward to compile all programs
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        device_tokens, device_cos, device_sin, device_page_table = device_inputs

        # Create working copies (so inputs survive as read-only buffers)
        work_tokens = ttnn.to_memory_config(device_tokens, ttnn.DRAM_MEMORY_CONFIG)
        work_cos = ttnn.to_memory_config(device_cos, ttnn.DRAM_MEMORY_CONFIG)
        work_sin = ttnn.to_memory_config(device_sin, ttnn.DRAM_MEMORY_CONFIG)

        tt_out = self.model.ttnn_prefill_forward(
            x=work_tokens,
            rot_mats_global=[work_cos, work_sin],
            user_id=0,
            page_table=device_page_table,
            get_last_token=-1,
            kv_cache=kv_cache,
        )
        logger.info("Done compiling prefill model for trace")

        # Trace capture: copy to device again (allocates new device tensors for trace)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        device_tokens, device_cos, device_sin, device_page_table = device_inputs

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        # Create working copies inside trace — these allocate in trace memory,
        # keeping the original device_inputs buffers in normal memory intact.
        work_tokens = ttnn.to_memory_config(device_tokens, ttnn.DRAM_MEMORY_CONFIG)
        work_cos = ttnn.to_memory_config(device_cos, ttnn.DRAM_MEMORY_CONFIG)
        work_sin = ttnn.to_memory_config(device_sin, ttnn.DRAM_MEMORY_CONFIG)
        tt_out = self.model.ttnn_prefill_forward(
            x=work_tokens,
            rot_mats_global=[work_cos, work_sin],
            user_id=0,
            page_table=device_page_table,
            get_last_token=-1,
            kv_cache=kv_cache,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Done capturing prefill trace")

        return trace_id, tt_out, device_inputs

    def _easy_trace_prefill(self, tokens_embd, rot_mats_user, page_table_user, kv_cache, padded_len):
        """
        Capture trace on first call per padded_len, replay on subsequent calls.

        Returns the trace output (raw hidden states, [1, 1, padded_len, hidden]).
        """
        trace_key = str(padded_len)

        if self.trace_id_prefill[trace_key] is None:
            trace_id, tt_out, device_inputs = self._capture_trace_prefill(
                tokens_embd, rot_mats_user, page_table_user, kv_cache
            )
            self.trace_id_prefill[trace_key] = trace_id
            self.trace_inputs_prefill[trace_key] = device_inputs
            self.trace_output_prefill[trace_key] = tt_out

        return self._prefill_forward_trace(
            self.trace_id_prefill[trace_key],
            self.trace_inputs_prefill[trace_key],
            self.trace_output_prefill[trace_key],
            tokens_embd,
            rot_mats_user,
            page_table_user,
        )

    def _prefill_forward_trace(self, trace_id, device_inputs, tt_out, tokens_embd, rot_mats_user, page_table_user):
        """Copy new user data to pre-allocated device tensors and execute the trace."""
        host_inputs = self.model.prepare_prefill_inputs_trace(
            tokens_embd, rot_mats=rot_mats_user, page_table=page_table_user
        )
        copy_host_to_device(host_inputs, device_tensors=device_inputs, mesh_device=self.mesh_device)
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
        return tt_out

    def prefill_forward_text(
        self, tokens: torch.Tensor, rot_mats, page_table=None, kv_cache=None, prompt_lens=None, enable_trace=True
    ):
        batch, batch_seq_len = tokens.shape[:2]
        output_logits = torch.zeros(batch, 1, self.model_args.vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch)

        if page_table is not None:
            assert isinstance(
                page_table, torch.Tensor
            ), "page_table must be a torch.Tensor when passing into prefill_forward"

        # Check if batched prefill is possible:
        # - batch > 1
        # - total tokens fit in device DRAM for intermediate tensors
        #   (hidden state = total_tokens × hidden_dim × 2 bytes must fit in free DRAM)
        # - page table available (needed for per-user KV cache fill)
        # - not requiring chunked prefill (batch_seq_len within chunk limit)
        # - not Galaxy (TG KV cache handling not supported in batched path)
        #
        # Memory limit: each intermediate tensor is total_tokens × hidden_dim × dtype_size.
        # During attention output + all_reduce, ~2 such tensors coexist.
        # Conservative limit: 32k total tokens works safely on N300 for 7B models.
        total_tokens = batch_seq_len * batch
        max_batched_tokens = 32 * 1024
        use_batched_prefill = (
            batch > 1
            and total_tokens <= max_batched_tokens
            and page_table is not None
            and batch_seq_len <= self.model_args.max_prefill_chunk_size
            and not self.model_args.is_galaxy
        )

        if use_batched_prefill:
            logger.info(f"Using batched prefill: {batch} users × {batch_seq_len} tokens = {total_tokens} total tokens")
            return self.__batched_prefill_forward_text(
                tokens, rot_mats, page_table, kv_cache, prompt_lens, batch, batch_seq_len
            )

        # Trace-based prefill is not currently supported for Qwen2.5-VL because the model
        # uses pre-computed embeddings (large bfloat16 tensors) as inputs, which don't fit
        # in the trace region. Standard models use small uint32 token IDs as trace inputs.
        # TODO: Enable trace when the model can accept token IDs or trace_region_size is dynamic.
        use_trace = False

        if batch > 1:
            if use_trace:
                logger.info(
                    f"Using trace-based sequential prefill: {batch} users × {batch_seq_len} tokens"
                    f" = {total_tokens} total tokens"
                )
            else:
                logger.info(
                    f"Using sequential prefill: {batch} users × {batch_seq_len} tokens = {total_tokens} total tokens"
                    + (f" (exceeds {max_batched_tokens} batched limit)" if total_tokens > max_batched_tokens else "")
                )

        block_size = get_block_size(kv_cache) if kv_cache is not None else None
        num_blocks_padded = num_blocks_in_seq(batch_seq_len, block_size) if block_size is not None else None
        out_list = []

        for user_id in range(batch):
            logger.info(f"Prefilling User {user_id + 1}")
            seq_len = int(prompt_lens[user_id])
            last_token_idx = seq_len - 1

            if use_trace:
                # Per-user page table with consistent shape for trace reuse
                pt_user = page_table[user_id : user_id + 1, :num_blocks_padded]

                # Per-user rotation matrices sliced to padded_len
                rot_mats_user = (
                    rot_mats[0][user_id : user_id + 1, :, :batch_seq_len, :],
                    rot_mats[1][user_id : user_id + 1, :, :batch_seq_len, :],
                )

                # Execute via trace (capture on first user, replay on subsequent)
                tt_hidden = self._easy_trace_prefill(
                    tokens[user_id : user_id + 1],
                    rot_mats_user,
                    pt_user,
                    kv_cache,
                    batch_seq_len,
                )

                # Post-process outside trace: slice last-token tile, norm, lm_head
                logits = self.model.process_logits_after_prefill_trace(tt_hidden, last_token_idx)
                out_list.append(logits.cpu(blocking=False))
            else:
                if page_table is not None:
                    page_table_user = self._ttt_generator._get_prefill_user_page_table(page_table, kv_cache, seq_len)
                else:
                    page_table_user = None

                logits = self.__prefill_forward_single_user_text(
                    tokens[user_id : user_id + 1],
                    page_table=page_table_user,
                    user_id=user_id,
                    last_token_idx=last_token_idx,
                    rot_mats=rot_mats,
                    kv_cache=kv_cache,
                )
                output_logits[user_id] = logits

        if use_trace:
            # Process asynchronously-copied outputs
            ttnn.synchronize_device(self.mesh_device)
            for user_id, out in enumerate(out_list):
                seq_len = int(prompt_lens[user_id])
                last_token_idx = seq_len - 1
                output_logits[user_id] = self.model.process_output_prefill(out, last_token_idx=(last_token_idx % 32))

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")

        return output_logits

    def __batched_prefill_forward_text(self, tokens, rot_mats, page_table, kv_cache, prompt_lens, batch, padded_len):
        """
        Process all users in a single forward pass by concatenating their tokens and RoPE.
        This is significantly faster than sequential per-user prefill for batch > 1.
        """
        logger.info(f"Batched prefill: {batch} users, padded_len={padded_len}")
        output_logits = torch.zeros(batch, 1, self.model_args.vocab_size)

        # Build per-user last_token_idx and page tables
        last_token_indices = []
        user_page_tables = []
        block_size = get_block_size(kv_cache)
        num_blocks = num_blocks_in_seq(padded_len, block_size)

        for user_id in range(batch):
            seq_len = int(prompt_lens[user_id])
            last_token_indices.append(seq_len - 1)
            # Get per-user page table (slice by user first, then truncate to needed blocks)
            pt_user = self._ttt_generator._get_prefill_user_page_table(
                page_table[user_id : user_id + 1], kv_cache, seq_len
            )
            # Ensure all page tables have the same number of columns
            if pt_user.shape[1] < num_blocks:
                padding = torch.zeros(1, num_blocks - pt_user.shape[1], dtype=torch.int32)
                pt_user = torch.cat([pt_user, padding], dim=1)
            else:
                pt_user = pt_user[:, :num_blocks]
            user_page_tables.append(pt_user)

        # Stack page tables: [batch, num_blocks]
        batched_page_table = torch.cat(user_page_tables, dim=0)

        # Concatenate tokens from all users along seq dim: [1, batch * padded_len, hidden]
        # tokens shape: [batch, padded_len, hidden]
        concat_tokens = tokens.reshape(1, batch * padded_len, tokens.shape[-1])

        # Concatenate rot_mats along seq dim: (cos[batch, 1, seq, hd], sin[batch, 1, seq, hd])
        # First slice each user's rot_mats to padded_len, then reshape to concatenate
        hd = rot_mats[0].shape[-1]
        cos_per_user = rot_mats[0][:, :, :padded_len, :]  # [batch, 1, padded_len, hd]
        sin_per_user = rot_mats[1][:, :, :padded_len, :]  # [batch, 1, padded_len, hd]
        concat_cos = cos_per_user.reshape(1, 1, batch * padded_len, hd)
        concat_sin = sin_per_user.reshape(1, 1, batch * padded_len, hd)
        concat_rot_mats = (concat_cos, concat_sin)

        # Prepare inputs
        prefill_input, rot_mats_prefill, page_table_tt, _ = self.model.prepare_inputs_prefill(
            concat_tokens,
            rot_mats=concat_rot_mats,
            page_table=batched_page_table,
        )

        # Forward pass with batch_size
        # Use get_last_token=-1 to return full hidden states (we extract per-user logits after)
        tt_hidden = self.model.ttnn_prefill_forward(
            prefill_input,
            rot_mats_global=rot_mats_prefill,
            user_id=list(range(batch)),
            page_table=page_table_tt,
            get_last_token=-1,
            kv_cache=kv_cache,
            batch_size=batch,
        )

        # Extract per-user logits from hidden states [1, 1, batch*padded_len, hidden]
        for user_id in range(batch):
            lt_idx = last_token_indices[user_id]
            offset = user_id * padded_len
            get_last = (lt_idx // 32) * 32 + offset
            # Slice the 32-token tile containing the last token
            tile = ttnn.slice(tt_hidden, (0, 0, get_last, 0), (1, 1, get_last + 32, tt_hidden.shape[-1]))
            # Apply norm + LM head
            tile = self.model.norm(tile, mode="prefill")
            if self.model.model_config["LM_HEAD_INPUT_MEMCFG"].is_sharded():
                tile = ttnn.interleaved_to_sharded(tile, self.model.model_config["LM_HEAD_INPUT_MEMCFG"])
            logits = self.model.lm_head(tile)
            logits = ttnn.to_layout(logits, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            output_logits[user_id] = self.model.process_output_prefill(logits.cpu(), last_token_idx=(lt_idx % 32))

        ttnn.deallocate(tt_hidden)
        ttnn.deallocate(prefill_input)
        if page_table_tt is not None:
            ttnn.deallocate(page_table_tt)

        logger.info(f"Finished batched prefill for {batch} users up to {padded_len} tokens, Starting decode...")

        return output_logits

    def update_cos_sin(self, cos_matrix_pt=None, sin_matrix_pt=None):
        self.model.rope_setup.update_cos_sin(cos_matrix_pt=cos_matrix_pt, sin_matrix_pt=sin_matrix_pt)

    def update_cos_sin_rows(self, rot_mats_seq_ids):
        for i, (cos, sin) in enumerate(rot_mats_seq_ids):
            self.model.rope_setup.cos_matrix_pt[i] = cos[0]
            self.model.rope_setup.sin_matrix_pt[i] = sin[0]
        self.update_cos_sin()

    def update_rope_deltas(self, rope_deltas_list: list):
        # pad rope_deltas_list to the batch size
        rope_deltas_list = rope_deltas_list + [0] * (self.model.rope_setup.batch_size - len(rope_deltas_list))
        # convert to torch tensor
        self.model.rope_setup.rope_deltas = torch.tensor(rope_deltas_list)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
    ):
        return self._ttt_generator.decode_forward(
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
                    rot_mats=rot_mats,
                    start_pos=chunk_start,
                    page_table=page_table_user_padded,
                    chunk_page_table=chunk_page_table,
                )
                tt_logits = self.model.ttnn_prefill_forward(
                    chunk_prefill_input,
                    rot_mats_global=[rm[user_id : user_id + 1, ...] for rm in chunk_rot_mats_prefill],
                    user_id=CHUNK_USER_ID,
                    page_table=page_table_tt,
                    chunk_page_table=chunk_page_table_tt,
                    chunk_start_idx=chunk_start,
                    get_last_token=(last_token_idx_in_chunk // 32) * 32,
                    kv_cache=kv_cache,
                )

                if chunk_start == last_chunk_start:
                    logits = self.model.process_output_prefill(
                        tt_logits.cpu(), last_token_idx=(last_token_idx_in_chunk % 32)
                    )
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

            logits = self.model.process_output_prefill(tt_logits.cpu(), last_token_idx=(last_token_idx % 32))

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

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, non_greedy_decoding_on_device) -> None:
        """
        Pre-compile programs for expected power-of-2 prefill sequence lengths.
        Uses dummy embeddings and rotation matrices to avoid depending on real data.
        """

        warmup_seq_lens = [128, 1024, 2048, 4096]
        max_warmup = min(self.model_args.max_prefill_chunk_size, self.model_args.max_seq_len)
        warmup_seq_lens = [s for s in warmup_seq_lens if s <= max_warmup]

        if not warmup_seq_lens:
            logger.warning("No valid warmup sequence lengths for Qwen2.5-VL prefill")
            return

        logger.info(f"Warming up Qwen2.5-VL prefill for sequence lengths: {warmup_seq_lens}")
        hidden_dim = self.model_args.dim
        head_dim = self.model_args.head_dim

        for seq_len in warmup_seq_lens:
            dummy_tokens = torch.zeros(1, seq_len, hidden_dim, dtype=torch.bfloat16)
            dummy_cos = torch.ones(1, 1, seq_len, head_dim, dtype=torch.bfloat16)
            dummy_sin = torch.zeros(1, 1, seq_len, head_dim, dtype=torch.bfloat16)
            dummy_rot_mats = (dummy_cos, dummy_sin)

            if kv_cache is not None:
                block_size = get_block_size(kv_cache)
                num_blocks = num_blocks_in_seq(seq_len, block_size)
                dummy_page_table = torch.zeros(1, num_blocks, dtype=torch.int32)
            else:
                dummy_page_table = None

            logger.info(f"  Compiling prefill for seq_len={seq_len}")
            self.__prefill_forward_single_user_text(
                dummy_tokens,
                page_table=dummy_page_table,
                user_id=0,
                last_token_idx=seq_len - 1,
                rot_mats=dummy_rot_mats,
                kv_cache=kv_cache,
            )

        logger.info("Qwen2.5-VL prefill warmup complete")

    ## Destructor (used to delete ttnn trace if exists)

    def __del__(self):
        for trace_key, trace_id in self.trace_id_prefill.items():
            if trace_id is not None:
                ttnn.release_trace(self.mesh_device, trace_id)

        if hasattr(self, "trace_id"):
            ttnn.release_trace(self.mesh_device, self.trace_id)

        if hasattr(self, "trace_id_text"):
            ttnn.release_trace(self.mesh_device, self.trace_id_text)

        self._ttt_generator.__del__()
