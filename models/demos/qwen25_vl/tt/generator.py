# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import collections

import torch
from loguru import logger

import ttnn
from models.common.warmup import WarmupForwardMixin
from models.demos.qwen25_vl.tt.common import get_block_size, get_max_prefill_chunk_size, num_blocks_in_seq
from models.tt_transformers.tt.common import copy_host_to_device
from models.tt_transformers.tt.generator import MAX_BATCHED_PREFILL_SEQ_LEN
from models.tt_transformers.tt.generator import Generator as TTTGenerator


class Generator(WarmupForwardMixin):
    def __init__(self, model, model_args, mesh_device, processor=None, tokenizer=None):
        """
        Creating a Qwen2_5_Vision wrapper requires only a mesh_device and model_args.
        With model_args you have the checkpoint location, can specify max batch size
        and max seqlen, and other model specific parameters.

        Supports data parallelism: pass lists of models/model_args for DP > 1.
        """
        if not isinstance(model, list):
            model = [model]
        if not isinstance(model_args, list):
            model_args = [model_args]
        self._ttt_generator = TTTGenerator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)

        # Trace infrastructure for prefill
        # Keyed by "{padded_len}_{model_id}" (e.g., "4096_0")
        self.trace_id_prefill = collections.defaultdict(lambda: None)
        self.trace_inputs_prefill = {}
        self.trace_output_prefill = {}
        self.trace_mesh_prefill = {}

    @property
    def data_parallel(self):
        return self._ttt_generator.data_parallel

    @property
    def model(self):
        return self._ttt_generator.model[0]

    @property
    def model_args(self):
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

    def _get_model_mesh_device(self, model_id=0):
        """Get the mesh device for a specific model instance (submesh for DP>1)."""
        return self._ttt_generator.model[model_id].mesh_device

    def _capture_trace_prefill(self, tokens_embd, rot_mats_user, page_table_user, kv_cache, model_id=0):
        """Capture a prefill trace for a given sequence length on a specific model's submesh."""
        model_inst = self._ttt_generator.model[model_id]
        model_mesh = model_inst.mesh_device

        host_inputs = model_inst.prepare_prefill_inputs_trace(
            tokens_embd, rot_mats=rot_mats_user, page_table=page_table_user
        )

        device_inputs = copy_host_to_device(host_inputs, mesh_device=model_mesh)
        device_tokens, device_cos, device_sin, device_page_table = device_inputs

        work_tokens = ttnn.to_memory_config(device_tokens, ttnn.DRAM_MEMORY_CONFIG)
        work_cos = ttnn.to_memory_config(device_cos, ttnn.DRAM_MEMORY_CONFIG)
        work_sin = ttnn.to_memory_config(device_sin, ttnn.DRAM_MEMORY_CONFIG)

        tt_out = model_inst.ttnn_prefill_forward(
            x=work_tokens,
            rot_mats_global=[work_cos, work_sin],
            user_id=0,
            page_table=device_page_table,
            get_last_token=-1,
            kv_cache=kv_cache,
        )
        logger.info(f"Done compiling prefill model for trace (model_id={model_id})")

        device_inputs = copy_host_to_device(host_inputs, mesh_device=model_mesh)
        device_tokens, device_cos, device_sin, device_page_table = device_inputs

        trace_id = ttnn.begin_trace_capture(model_mesh, cq_id=0)
        work_tokens = ttnn.to_memory_config(device_tokens, ttnn.DRAM_MEMORY_CONFIG)
        work_cos = ttnn.to_memory_config(device_cos, ttnn.DRAM_MEMORY_CONFIG)
        work_sin = ttnn.to_memory_config(device_sin, ttnn.DRAM_MEMORY_CONFIG)
        tt_out = model_inst.ttnn_prefill_forward(
            x=work_tokens,
            rot_mats_global=[work_cos, work_sin],
            user_id=0,
            page_table=device_page_table,
            get_last_token=-1,
            kv_cache=kv_cache,
        )
        ttnn.end_trace_capture(model_mesh, trace_id, cq_id=0)
        logger.info(f"Done capturing prefill trace (model_id={model_id})")

        return trace_id, tt_out, device_inputs

    def _easy_trace_prefill(self, tokens_embd, rot_mats_user, page_table_user, kv_cache, padded_len, model_id=0):
        """Capture trace on first call per (padded_len, model_id), replay on subsequent calls."""
        trace_key = f"{padded_len}_{model_id}"

        if self.trace_id_prefill[trace_key] is None:
            trace_id, tt_out, device_inputs = self._capture_trace_prefill(
                tokens_embd, rot_mats_user, page_table_user, kv_cache, model_id=model_id
            )
            self.trace_id_prefill[trace_key] = trace_id
            self.trace_inputs_prefill[trace_key] = device_inputs
            self.trace_output_prefill[trace_key] = tt_out
            self.trace_mesh_prefill[trace_key] = self._ttt_generator.model[model_id].mesh_device

        return self._prefill_forward_trace(
            self.trace_id_prefill[trace_key],
            self.trace_inputs_prefill[trace_key],
            self.trace_output_prefill[trace_key],
            tokens_embd,
            rot_mats_user,
            page_table_user,
            model_id=model_id,
        )

    def _prefill_forward_trace(
        self, trace_id, device_inputs, tt_out, tokens_embd, rot_mats_user, page_table_user, model_id=0
    ):
        """Copy new user data to pre-allocated device tensors and execute the trace."""
        model_inst = self._ttt_generator.model[model_id]
        model_mesh = model_inst.mesh_device
        host_inputs = model_inst.prepare_prefill_inputs_trace(
            tokens_embd, rot_mats=rot_mats_user, page_table=page_table_user
        )
        copy_host_to_device(host_inputs, device_tensors=device_inputs, mesh_device=model_mesh)
        ttnn.execute_trace(model_mesh, trace_id, cq_id=0, blocking=False)
        return tt_out

    def _unwrap_kv_cache(self, kv_cache):
        """Unwrap single-element kv_cache list for DP=1 (e.g. T3K)."""
        is_dp = self.data_parallel > 1 and isinstance(kv_cache, list)
        if not is_dp and isinstance(kv_cache, list) and len(kv_cache) == 1:
            kv_cache = kv_cache[0]
        return kv_cache, is_dp

    def prefill_forward_text(
        self,
        tokens: torch.Tensor,
        rot_mats,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        enable_trace=True,
        empty_slots=None,
    ):
        batch, batch_seq_len = tokens.shape[:2]
        output_logits = torch.zeros(batch, 1, self.model_args.vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch)
        if not isinstance(prompt_lens, list):
            prompt_lens = prompt_lens.tolist()
        max_batch_size_per_model = self.model_args.max_batch_size

        if empty_slots is None:
            empty_slots = list(range(batch))

        if page_table is not None:
            assert isinstance(
                page_table, torch.Tensor
            ), "page_table must be a torch.Tensor when passing into prefill_forward"

        kv_cache, is_dp_kv_cache = self._unwrap_kv_cache(kv_cache)
        first_kv_cache = kv_cache[0] if is_dp_kv_cache else kv_cache

        max_batch = getattr(self.model_args, "max_batched_prefill_size", 32)
        users_per_group = max_batch_size_per_model if self.data_parallel > 1 else batch
        use_batched_prefill = (
            batch > 1
            and batch_seq_len * min(users_per_group, max_batch) <= MAX_BATCHED_PREFILL_SEQ_LEN
            and not getattr(self.model_args, "disable_batched_prefill", False)
            and page_table is not None
            and first_kv_cache is not None
        )

        if use_batched_prefill:
            total_tokens = batch_seq_len * batch
            logger.info(
                f"Using batched prefill: {batch} users × {batch_seq_len} tokens"
                f" = {total_tokens} total tokens (DP={self.data_parallel})"
            )
            num_dp = max(self.data_parallel, 1)
            for dp_group in range(num_dp):
                group_start = dp_group * max_batch_size_per_model
                group_end = min(group_start + max_batch_size_per_model, batch)
                if group_start >= batch:
                    break
                group_kv_cache = kv_cache[dp_group] if is_dp_kv_cache else kv_cache
                group_size = group_end - group_start
                chunk_size = min(group_size, max_batch)

                for chunk_start in range(0, group_size, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, group_size)
                    abs_start = group_start + chunk_start
                    abs_end = group_start + chunk_end
                    chunk_logits = self.__prefill_forward_batched_text(
                        tokens=tokens[abs_start:abs_end],
                        rot_mats=(rot_mats[0][abs_start:abs_end], rot_mats[1][abs_start:abs_end]),
                        page_table=page_table[abs_start:abs_end],
                        kv_cache=group_kv_cache,
                        prompt_lens=prompt_lens[abs_start:abs_end],
                        prefill_seq_len=batch_seq_len,
                        slot_offset=chunk_start,
                        model_id=dp_group,
                    )
                    output_logits[abs_start:abs_end] = chunk_logits
        else:
            use_trace = enable_trace and page_table is not None
            total_tokens = batch_seq_len * batch
            if batch > 1:
                logger.info(
                    f"Using {'traced ' if use_trace else ''}sequential prefill: {batch} users × {batch_seq_len} tokens"
                    f" = {total_tokens} total tokens (DP={self.data_parallel})"
                )

            block_size = get_block_size(first_kv_cache) if first_kv_cache is not None else None
            num_blocks_padded = num_blocks_in_seq(batch_seq_len, block_size) if block_size is not None else None
            out_list = []

            for idx, user_id in enumerate(empty_slots):
                model_id = user_id // max_batch_size_per_model if self.data_parallel > 1 else 0
                group_user_id = user_id % max_batch_size_per_model if page_table is None else 0
                model_kv_cache = kv_cache[model_id] if is_dp_kv_cache else kv_cache

                logger.info(
                    f"Prefilling User {user_id + 1} slot={user_id}"
                    f" (DP group {model_id}, local user {group_user_id})"
                )
                seq_len = int(prompt_lens[idx])
                last_token_idx = seq_len - 1

                user_rot_mats = (
                    rot_mats[0][idx : idx + 1],
                    rot_mats[1][idx : idx + 1],
                )

                if use_trace:
                    pt_user = page_table[idx : idx + 1, :num_blocks_padded]
                    trace_rot_mats = (
                        user_rot_mats[0][:, :, :batch_seq_len, :],
                        user_rot_mats[1][:, :, :batch_seq_len, :],
                    )
                    tt_hidden = self._easy_trace_prefill(
                        tokens[idx : idx + 1],
                        trace_rot_mats,
                        pt_user,
                        model_kv_cache,
                        batch_seq_len,
                        model_id=model_id,
                    )
                    model_inst = self._ttt_generator.model[model_id]
                    logits = model_inst.process_logits_after_prefill_trace(tt_hidden, last_token_idx)
                    out_list.append((logits.cpu(blocking=False), model_id))
                else:
                    if page_table is not None:
                        page_table_user = self._ttt_generator._get_prefill_user_page_table(
                            page_table[idx : idx + 1], model_kv_cache, seq_len
                        )
                    else:
                        page_table_user = None

                    logits = self.__prefill_forward_single_user_text(
                        tokens[idx : idx + 1],
                        page_table=page_table_user,
                        user_id=group_user_id,
                        last_token_idx=last_token_idx,
                        rot_mats=user_rot_mats,
                        kv_cache=model_kv_cache,
                        model_id=model_id,
                    )
                    output_logits[idx] = logits

            if use_trace:
                used_model_ids = set(mid for _, mid in out_list)
                for mid in used_model_ids:
                    ttnn.synchronize_device(self._ttt_generator.model[mid].mesh_device)
                for idx, (out, mid) in enumerate(out_list):
                    seq_len = int(prompt_lens[idx])
                    last_token_idx = seq_len - 1
                    model_inst = self._ttt_generator.model[mid]
                    output_logits[idx] = model_inst.process_output_prefill(out, last_token_idx=(last_token_idx % 32))

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")
        return output_logits

    def __prefill_forward_batched_text(
        self,
        tokens,
        rot_mats,
        page_table,
        kv_cache,
        prompt_lens,
        prefill_seq_len,
        slot_offset=0,
        model_id=0,
    ):
        model_inst = self._ttt_generator.model[model_id]
        batch_size = tokens.shape[0]
        last_token_idx = [int(pl) - 1 for pl in prompt_lens]
        batch_user_ids = list(range(slot_offset, slot_offset + batch_size))

        page_table_user = self._ttt_generator._get_prefill_user_page_table(
            page_table,
            kv_cache,
            prefill_seq_len,
            trace_enabled=False,
            prefill_seq_len=prefill_seq_len,
            use_batched_prefill=True,
            user_id=batch_user_ids,
        )

        prefill_input, rot_mats_prefill, page_table_tt, _ = model_inst.prepare_inputs_prefill(
            tokens,
            rot_mats=rot_mats,
            page_table=page_table_user,
            batch_size=batch_size,
        )

        tt_out = model_inst.ttnn_prefill_forward(
            prefill_input,
            rot_mats_global=rot_mats_prefill,
            user_id=batch_user_ids,
            page_table=page_table_tt,
            get_last_token=-1,
            kv_cache=kv_cache,
            batch_size=batch_size,
        )

        hidden_dim = tt_out.shape[-1]
        tt_out = ttnn.reshape(tt_out, [batch_size, 1, prefill_seq_len, hidden_dim])

        user_hidden = model_inst.extract_last_tokens_batched_prefill(
            tt_out, last_token_idx, batch_size, prefill_seq_len
        )
        batched_logits = model_inst._apply_norm_and_lm_head(user_hidden)

        output_logits = torch.zeros(batch_size, 1, self.model_args.vocab_size)
        batched_logits_cpu = ttnn.to_torch(
            batched_logits,
            mesh_composer=ttnn.ConcatMeshToTensor(model_inst.mesh_device, dim=-1),
        )
        output_logits[:, 0, :] = batched_logits_cpu[0, 0, :batch_size, : self.model_args.vocab_size]

        ttnn.deallocate(tt_out)
        ttnn.deallocate(prefill_input)
        if page_table_tt is not None:
            ttnn.deallocate(page_table_tt)

        return output_logits

    def update_cos_sin(self, cos_matrix_pt=None, sin_matrix_pt=None):
        for model_inst in self._ttt_generator.model:
            model_inst.rope_setup.update_cos_sin(cos_matrix_pt=cos_matrix_pt, sin_matrix_pt=sin_matrix_pt)

    def update_cos_sin_rows(self, rot_mats_seq_ids):
        batch_per_dp = len(rot_mats_seq_ids) // self.data_parallel
        for dp_id in range(self.data_parallel):
            model_inst = self._ttt_generator.model[dp_id]
            for i in range(batch_per_dp):
                global_idx = dp_id * batch_per_dp + i
                cos, sin = rot_mats_seq_ids[global_idx]
                model_inst.rope_setup.cos_matrix_pt[i] = cos[0]
                model_inst.rope_setup.sin_matrix_pt[i] = sin[0]
            model_inst.rope_setup.update_cos_sin()

    def update_rope_deltas(self, rope_deltas_list: list):
        batch_per_dp = len(rope_deltas_list) // self.data_parallel
        for dp_id in range(self.data_parallel):
            model_inst = self._ttt_generator.model[dp_id]
            dp_deltas = rope_deltas_list[dp_id * batch_per_dp : (dp_id + 1) * batch_per_dp]
            dp_deltas = dp_deltas + [0] * (model_inst.rope_setup.batch_size - len(dp_deltas))
            model_inst.rope_setup.rope_deltas = torch.tensor(dp_deltas)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        reset_batch=False,
        prompt_tokens=None,
        output_tokens=None,
    ):
        if not isinstance(kv_cache, list):
            kv_cache = [kv_cache]
        return self._ttt_generator.decode_forward(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            read_from_device=read_from_device,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

    def __prefill_forward_single_user_text(
        self, tokens, page_table, user_id, last_token_idx, rot_mats, kv_cache=None, model_id=0
    ):
        model_inst = self._ttt_generator.model[model_id]
        model_args_inst = self._ttt_generator.model_args[model_id]
        seq_len = tokens.shape[1]
        use_chunked_prefill = seq_len > model_args_inst.max_prefill_chunk_size
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
            chunk_size = get_max_prefill_chunk_size(seq_len, model_args_inst.max_prefill_chunk_size)
            block_size = get_block_size(kv_cache)
            last_token_idx_in_chunk = last_token_idx % chunk_size
            last_chunk_start = (last_token_idx // chunk_size) * chunk_size
            page_table_user = page_table[user_id : user_id + 1, :]
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
                ) = model_inst.prepare_inputs_prefill(
                    chunk_tokens,
                    rot_mats=rot_mats,
                    start_pos=chunk_start,
                    page_table=page_table_user_padded,
                    chunk_page_table=chunk_page_table,
                )
                tt_logits = model_inst.ttnn_prefill_forward(
                    chunk_prefill_input,
                    rot_mats_global=[rm[0:1, ...] for rm in chunk_rot_mats_prefill],
                    user_id=CHUNK_USER_ID,
                    page_table=page_table_tt,
                    chunk_page_table=chunk_page_table_tt,
                    chunk_start_idx=chunk_start,
                    get_last_token=(last_token_idx_in_chunk // 32) * 32,
                    kv_cache=kv_cache,
                )

                if chunk_start == last_chunk_start:
                    logits = model_inst.process_output_prefill(
                        tt_logits.cpu(), last_token_idx=(last_token_idx_in_chunk % 32)
                    )
                    return logits
                else:
                    del tt_logits
        else:
            prefill_input, rot_mats_prefill, page_table_tt, _ = model_inst.prepare_inputs_prefill(
                tokens,
                rot_mats=rot_mats,
                page_table=page_table,
            )

            tt_logits = model_inst.ttnn_prefill_forward(
                prefill_input,
                rot_mats_global=[rm[0:1, ...] for rm in rot_mats_prefill],
                user_id=user_id,
                page_table=page_table_tt,
                get_last_token=(last_token_idx // 32) * 32,
                kv_cache=kv_cache,
            )

            logits = model_inst.process_output_prefill(tt_logits.cpu(), last_token_idx=(last_token_idx % 32))

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
        result = self._ttt_generator.process_decode_output_host(tt_out, is_tokens=is_tokens)
        if is_tokens:
            # Device sampling produces UInt32 token IDs; PyTorch CPU lacks
            # kernel support for UInt32 indexing, so cast to int32 here
            # before the V0 model runner applies perm_table reordering.
            def _safe_cast(t):
                if (
                    isinstance(t, torch.Tensor)
                    and not t.is_floating_point()
                    and t.dtype not in (torch.int32, torch.int64)
                ):
                    return t.to(torch.int32)
                return t

            if isinstance(result, tuple):
                result = tuple(_safe_cast(t) for t in result)
            else:
                result = _safe_cast(result)
        return result

    def warmup_model_prefill(
        self,
        kv_cache,
        enable_trace,
        can_sample_on_device=None,
        non_greedy_decoding_on_device=None,
        sampling_params=None,
    ) -> None:
        """
        Pre-compile programs for expected power-of-2 prefill sequence lengths.
        Uses dummy embeddings and rotation matrices to avoid depending on real data.
        Warms up all DP model instances.
        """

        warmup_seq_lens = [128, 1024, 2048, 4096, 8192, 16384]
        max_warmup = min(self.model_args.max_prefill_chunk_size, self.model_args.max_seq_len)
        warmup_seq_lens = [s for s in warmup_seq_lens if s <= max_warmup]

        if not warmup_seq_lens:
            logger.warning("No valid warmup sequence lengths for Qwen2.5-VL prefill")
            return

        logger.info(f"Warming up Qwen2.5-VL prefill for sequence lengths: {warmup_seq_lens} (DP={self.data_parallel})")
        hidden_dim = self.model_args.dim
        head_dim = self.model_args.head_dim

        kv_cache, is_dp_kv_cache = self._unwrap_kv_cache(kv_cache)
        for model_id in range(self.data_parallel):
            model_kv_cache = kv_cache[model_id] if is_dp_kv_cache else kv_cache
            logger.info(f"  Warming up DP group {model_id}")
            for seq_len in warmup_seq_lens:
                dummy_tokens = torch.zeros(1, seq_len, hidden_dim, dtype=torch.bfloat16)
                dummy_cos = torch.ones(1, 1, seq_len, head_dim, dtype=torch.bfloat16)
                dummy_sin = torch.zeros(1, 1, seq_len, head_dim, dtype=torch.bfloat16)
                dummy_rot_mats = (dummy_cos, dummy_sin)

                if model_kv_cache is not None:
                    block_size = get_block_size(model_kv_cache)
                    num_blocks = num_blocks_in_seq(seq_len, block_size)
                    dummy_page_table = torch.zeros(1, num_blocks, dtype=torch.int32)
                else:
                    dummy_page_table = None

                logger.info(f"    Compiling prefill for seq_len={seq_len}")
                self.__prefill_forward_single_user_text(
                    dummy_tokens,
                    page_table=dummy_page_table,
                    user_id=0,
                    last_token_idx=seq_len - 1,
                    rot_mats=dummy_rot_mats,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                )

        logger.info("Qwen2.5-VL prefill warmup complete")

    ## Destructor (used to delete ttnn trace if exists)

    def __del__(self):
        for trace_key, trace_id in self.trace_id_prefill.items():
            if trace_id is not None:
                mesh = self.trace_mesh_prefill.get(trace_key, self.mesh_device)
                ttnn.release_trace(mesh, trace_id)

        if hasattr(self, "trace_id"):
            ttnn.release_trace(self.mesh_device, self.trace_id)

        if hasattr(self, "trace_id_text"):
            ttnn.release_trace(self.mesh_device, self.trace_id_text)

        self._ttt_generator.__del__()
