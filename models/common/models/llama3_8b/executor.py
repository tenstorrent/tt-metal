# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Executors for Llama 3.1-8B.

LlamaExecutor         — direct execution (canonical path, no tracing)
TracedLlamaExecutor   — traced execution (sibling, not wrapper)
TeacherForceExecutor  — accuracy measurement via teacher forcing
PerfBenchmarkExecutor — performance measurement (TTFT, tok/s/u)

Scope: text-only, DP=1, paged attention, non-TG (1D).
"""

import time
from collections import defaultdict
from dataclasses import dataclass

import torch
from loguru import logger

import ttnn
from models.common.models.llama3_8b.model import Llama3Transformer1D, _all_gather_rmsnorm_tensor
from models.tt_transformers.tt.common import (
    Mode,
    copy_host_to_device,
    get_block_size,
    get_max_prefill_chunk_size,
    get_padded_prefill_len,
    num_blocks_in_seq,
)

# =============================================================================
# Shared helpers (used by both executors)
# =============================================================================


def _concat_host_output(tt_out, cluster_shape, is_galaxy=False):
    """Concatenate multi-device output into a single host tensor."""
    torch_out_tensors = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(tt_out)]
    row_dim, col_dim = (1, -1)

    rows, cols = cluster_shape
    mesh_shape = [torch_out_tensors[i : i + cols] for i in range(0, len(torch_out_tensors), cols)]
    row_concatenated = [torch.cat(row, dim=col_dim) for row in mesh_shape]
    return torch.cat(row_concatenated, dim=row_dim)


def _process_output_prefill(tt_out, last_token_idx, vocab_size, cluster_shape):
    """Device→host for prefill. Returns logits for the last token."""
    assert tt_out.storage_type() == ttnn.StorageType.HOST, "Expected host tensor"
    return _concat_host_output(tt_out, cluster_shape)[0, 0, last_token_idx, :vocab_size]


def _process_output_decode(tt_out, B, vocab_size, num_devices, cluster_shape):
    """Device→host for decode. Returns logits [B, 1, vocab_size]."""
    if num_devices > 1:
        tt_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
    else:
        tt_out = ttnn.to_torch(tt_out).float()
    return tt_out[:, :, :B, :vocab_size].view(B, 1, -1)


def _process_output_decode_tokens(tt_out, B, cluster_shape):
    """Device→host for decode when sampling on device. Returns token ids [B]."""
    padded_batch_size = 32
    tt_out = ttnn.reshape(tt_out, ttnn.Shape([1, 1, padded_batch_size, 1]))
    return _concat_host_output(tt_out, cluster_shape)[0, 0, :B, 0]


def _get_prefill_user_page_table(page_table, kv_cache, prefill_len, trace_enabled, prefill_seq_len):
    """Slice and pad page table for a single prefill user."""
    block_size = get_block_size(kv_cache)
    num_blocks = num_blocks_in_seq(prefill_len, block_size)
    if trace_enabled:
        num_blocks = num_blocks_in_seq(prefill_seq_len, block_size)
    return page_table[:, :num_blocks]


# =============================================================================
# LlamaExecutor — direct execution
# =============================================================================


class LlamaExecutor:
    """Direct (non-traced) executor for Llama 3.1-8B.

    Handles: input preparation, output processing, chunked prefill,
    KV cache allocation, and on-device sampling.
    TracedLlamaExecutor is a sibling that adds trace capture/replay.
    """

    def __init__(self, model: Llama3Transformer1D, mesh_device: ttnn.MeshDevice, model_args=None):
        self.model = model
        self.mesh_device = mesh_device
        self.model_args = model_args
        self.mode = None

    # =========================================================================
    # KV Cache
    # =========================================================================

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate paged KV cache on device. Returns list[list[ttnn.Tensor]]."""
        cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)
        kv_cache = []
        cache_path = self.model_args.model_cache_path if self.model_args else None

        for layer_num in range(num_layers):
            kv_cache_dtype = ttnn.bfloat8_b
            if self.model_args and self.model_args.optimizations is not None:
                from models.tt_transformers.tt.model_config import TensorGroup

                configured = self.model_args.optimizations.get_tensor_dtype(
                    decoder_id=layer_num, tensor=TensorGroup.KV_CACHE
                )
                if configured is not None:
                    kv_cache_dtype = configured

            kv_tt_i = [
                ttnn.as_tensor(
                    cache_kv,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=kv_cache_dtype,
                    cache_file_name=(
                        cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}" if cache_path else None
                    ),
                )
                for kv in ["k", "v"]
            ]
            kv_cache.append(kv_tt_i)
        self._kv_cache = kv_cache
        self.model.set_kv_cache(kv_cache)
        return kv_cache

    def _assert_kv_cache_identity(self, kv_cache):
        """Verify kv_cache passed to forward is the same object bound at allocation."""
        if kv_cache is not None and hasattr(self, "_kv_cache"):
            assert kv_cache is self._kv_cache, (
                "kv_cache passed to forward differs from the allocated cache. "
                "Call allocate_kv_cache() again after reallocating."
            )

    # =========================================================================
    # Input preparation
    # =========================================================================

    def prepare_prefill_inputs(
        self, tokens, start_pos=0, page_table=None, chunk_page_table=None, trace_enabled=False, last_token_idx=None
    ):
        """Prepare prefill inputs. Returns (tokens_or_embed, rot_mats, page_table_tt, chunk_page_table_tt)."""
        device = None if trace_enabled else self.mesh_device

        assert tokens.dim() == 2, "tokens must be 2D"
        tokens_reshaped = tokens.reshape(1, 1, 1, -1)
        S = tokens_reshaped.shape[-1]
        tokens_tt = ttnn.from_torch(
            tokens_reshaped,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        if not trace_enabled:
            tokens_embd = self.model.embed_prefill(tokens_tt)
        else:
            tokens_embd = None

        rope = self.model.rope_setup
        # Load device weights if not already loaded
        rope.load_device_weights()
        mat_len = rope.cos_matrix.shape[2]
        seq_len = last_token_idx + 1 if last_token_idx is not None else S
        assert mat_len >= seq_len, f"Sequence length {seq_len} exceeds max seq len {mat_len}"

        required_end = start_pos + S
        pad_len = max(0, required_end - mat_len)
        max_seq_len = self.model_args.max_seq_len if self.model_args else mat_len

        prefill_start = 0 if trace_enabled else start_pos
        slice_end = max_seq_len if trace_enabled else min(mat_len, required_end)

        cos_slice = rope.cos_matrix[:, :, prefill_start:slice_end, :]
        sin_slice = rope.sin_matrix[:, :, prefill_start:slice_end, :]

        if pad_len > 0:
            padding = [(0, 0)] * 4
            padding[2] = (0, pad_len)
            cos_slice = ttnn.pad(cos_slice, padding=padding, value=0.0)
            sin_slice = ttnn.pad(sin_slice, padding=padding, value=0.0)

        tt_page_table = None
        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        tt_chunk_page_table = None
        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return (
            tokens_tt if trace_enabled else tokens_embd,
            cos_slice,
            sin_slice,
            tt_page_table,
            tt_chunk_page_table,
        )

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """Prepare decode inputs as host tensors. Returns (tokens, current_pos, rope_idxs, page_table)."""
        B = tokens.shape[0]
        max_batch = self.model_args.max_batch_size if self.model_args else B
        assert B == max_batch, f"Batch size {B} must equal max_batch_size {max_batch}"

        tokens_padded = torch.nn.functional.pad(tokens.view(-1), (0, 32 - len(tokens)), "constant", 0)
        tokens_tt = ttnn.from_torch(
            tokens_padded,
            device=None,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tokens_tt = ttnn.unsqueeze_to_4D(tokens_tt)

        rot_current_pos = torch.maximum(current_pos, torch.tensor(0, dtype=torch.int64))
        rope_idxs = self.model.rope_setup.get_rot_idxs(rot_current_pos, on_host=True)

        cluster_shape = self.model_args.cluster_shape if self.model_args else [1, 1]
        current_pos_tt = ttnn.from_torch(
            current_pos,
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(None, None),
                mesh_shape=cluster_shape,
            ),
        )

        tt_page_table = None
        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=None,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device,
                    dims=(None, None),
                    mesh_shape=cluster_shape,
                ),
            )

        return tokens_tt, current_pos_tt, rope_idxs, tt_page_table

    def prepare_decode_inputs_device(self, tokens, current_pos, page_table=None):
        """Prepare decode inputs on device. Returns device tensors."""
        host_inputs = self.prepare_decode_inputs_host(tokens, current_pos, page_table)
        return copy_host_to_device(host_inputs, mesh_device=self.mesh_device)

    # =========================================================================
    # Prefill
    # =========================================================================

    def prefill_forward(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        sampling_params=None,
        start_pos=None,
        warmup_prefill=True,
        **kwargs,
    ):
        """Per-user prefill loop with chunked prefill + prefix caching."""
        self.mode = Mode.PREFILL
        self._assert_kv_cache_identity(kv_cache)

        if page_table is None:
            enable_trace = False

        batch_size, batch_seq_len = tokens.shape
        max_batch = self.model_args.max_batch_size if self.model_args else batch_size
        vocab_size = self.model.vocab_size
        cluster_shape = self.model_args.cluster_shape if self.model_args else [1, 1]

        output_tensor = torch.zeros(batch_size, 1, vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch_size)
        if empty_slots is None:
            empty_slots = list(range(batch_size))

        prefill_results = []

        for idx, user_id in enumerate(empty_slots):
            seq_len = int(prompt_lens[idx])
            num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
            last_token_idx = seq_len - 1
            prefill_seq_len = get_padded_prefill_len(seq_len - num_cached_tokens)

            logger.info(f"Prefilling User {user_id + 1} up to {seq_len} tokens")

            prefill_ids = torch.cat(
                [
                    tokens[idx : idx + 1, num_cached_tokens:seq_len],
                    torch.zeros(1, prefill_seq_len - (seq_len - num_cached_tokens)).long(),
                ],
                dim=-1,
            )

            page_table_user = (
                _get_prefill_user_page_table(
                    page_table[idx : idx + 1],
                    kv_cache,
                    seq_len,
                    False,
                    prefill_seq_len,
                )
                if page_table is not None
                else None
            )

            logits = self._prefill_single_user(
                prefill_ids,
                page_table=page_table_user,
                user_id=0 if page_table is not None else user_id,
                last_token_idx=last_token_idx,
                num_cached_tokens=num_cached_tokens,
            )

            logits = ttnn.untilize(logits, use_multicore=True)
            prefill_results.append(
                {
                    "idx": idx,
                    "last_token_idx": last_token_idx,
                    "logits": logits.cpu(blocking=False),
                }
            )

        for res in prefill_results:
            ttnn.synchronize_device(self.mesh_device)
            last_relative = res["last_token_idx"] - (int(start_pos[res["idx"]]) if start_pos is not None else 0)
            output_tensor[res["idx"]] = _process_output_prefill(
                res["logits"],
                last_relative % 32,
                vocab_size,
                cluster_shape,
            )

        return output_tensor

    def _prefill_single_user(self, tokens, page_table, user_id, last_token_idx, num_cached_tokens=0):
        """Prefill a single user with chunked prefill support."""
        seq_len = tokens.shape[-1]
        max_chunk = self.model_args.max_prefill_chunk_size if self.model_args else seq_len
        use_chunked = seq_len > max_chunk
        use_prefix_caching = num_cached_tokens > 0

        if use_chunked or use_prefix_caching:
            assert page_table is not None and self._kv_cache is not None
            chunk_size = get_max_prefill_chunk_size(seq_len, max_chunk) if use_chunked else seq_len

            last_token_in_seq = last_token_idx - num_cached_tokens
            block_size = get_block_size(self._kv_cache)
            last_token_in_chunk = last_token_in_seq % chunk_size
            last_chunk_start = (last_token_in_seq // chunk_size) * chunk_size

            page_table_user = page_table[user_id : user_id + 1, :]
            num_pad_blocks = num_blocks_in_seq(seq_len + num_cached_tokens, block_size) - page_table_user.shape[1]
            page_table_padded = torch.cat([page_table_user, torch.zeros(1, num_pad_blocks, dtype=torch.int32)], dim=-1)

            for chunk_start in range(num_cached_tokens, num_cached_tokens + seq_len, chunk_size):
                chunk_end = chunk_start + chunk_size
                chunk_start_rel = chunk_start - num_cached_tokens
                chunk_end_rel = chunk_end - num_cached_tokens

                chunk_tokens = tokens[:, chunk_start_rel:chunk_end_rel]
                chunk_page_table = page_table_padded[:, chunk_start // block_size : chunk_end // block_size]

                prefill_input, cos, sin, page_table_tt, chunk_page_table_tt = self.prepare_prefill_inputs(
                    chunk_tokens,
                    start_pos=chunk_start,
                    page_table=page_table_padded,
                    chunk_page_table=chunk_page_table,
                    last_token_idx=last_token_idx,
                )

                get_last_token = (last_token_in_chunk // 32) * 32
                logits = self.model.prefill_forward(
                    prefill_input,
                    [cos, sin],
                    user_id=0,
                    page_table=page_table_tt,
                    chunk_page_table=chunk_page_table_tt,
                    chunk_start_idx=chunk_start,
                    get_last_token=get_last_token,
                )

                if chunk_start_rel == last_chunk_start:
                    return logits
                else:
                    del logits
        else:
            prefill_input, cos, sin, page_table_tt, _ = self.prepare_prefill_inputs(
                tokens,
                page_table=page_table,
                last_token_idx=last_token_idx,
            )

            get_last_token = (last_token_idx // 32) * 32
            return self.model.prefill_forward(
                prefill_input,
                [cos, sin],
                user_id=user_id,
                page_table=page_table_tt,
                get_last_token=get_last_token,
            )

    # =========================================================================
    # Decode
    # =========================================================================

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        **kwargs,
    ):
        """Single decode step. Returns (logits_or_tokens, log_probs)."""
        self.mode = Mode.DECODE
        self._assert_kv_cache_identity(kv_cache)
        B = tokens.shape[0]
        vocab_size = self.model.vocab_size
        num_devices = self.model.num_devices
        cluster_shape = self.model_args.cluster_shape if self.model_args else [1, 1]

        sampling_on_device = sampling_params is not None

        if sampling_on_device and self.model.sampling is not None:
            from models.common.sampling import broadcast_sampling_params, format_sampling_params

            per_request_params = format_sampling_params(broadcast_sampling_params(sampling_params, 0, slot_len=B), B)
            self.model.sampling.apply_decode_state(
                [per_request_params],
                reset_batch=False,
                prompt_tokens=kwargs.get("prompt_tokens"),
                output_tokens=kwargs.get("output_tokens"),
            )
            self.model.sampling.seed_manager.get_new_values()

        tt_tokens, tt_current_pos, tt_rot_mat_idxs, tt_page_table = self.prepare_decode_inputs_device(
            tokens, start_pos, page_table
        )

        rot_mats = self.model.rope_setup.get_rot_mats(tt_rot_mat_idxs)
        x_embed = self.model.embed_decode(tt_tokens)

        logits = self.model.decode_forward(
            x_embed,
            tt_current_pos,
            rot_mats,
            page_table=tt_page_table,
        )

        if sampling_on_device and self.model.sampling is not None:
            self.model.increment_positions(tt_current_pos, tt_rot_mat_idxs)
            tt_toks, tt_log_probs = self.model.sampling.decode_forward(logits, tt_out_tok=tt_tokens)
            if read_from_device:
                ttnn.synchronize_device(self.mesh_device)
                toks = _process_output_decode_tokens(tt_toks.cpu(), B, cluster_shape)
                return toks, None
            return (tt_toks, tt_log_probs)

        logits = self.model.gather_and_untilize_logits(logits)

        if read_from_device:
            logits_host = logits.cpu()
            ttnn.synchronize_device(self.mesh_device)
            return _process_output_decode(logits_host, B, vocab_size, num_devices, cluster_shape), None
        return (logits, None)


# =============================================================================
# TracedLlamaExecutor — traced execution (sibling)
# =============================================================================


class TracedLlamaExecutor:
    """Traced executor for Llama 3.1-8B. Sibling of LlamaExecutor.

    Same model, same config. Execution uses TTNN trace capture/replay.
    Follows tt_cnn's TracedModelExecutor pattern.
    """

    def __init__(self, model: Llama3Transformer1D, mesh_device: ttnn.MeshDevice, model_args=None):
        self.model = model
        self.mesh_device = mesh_device
        self.model_args = model_args
        self._direct = LlamaExecutor(model, mesh_device, model_args)
        self._cleaned_up = False

        self.trace_id_prefill = defaultdict(lambda: None)
        self.trace_inputs_prefill = defaultdict(lambda: None)
        self.trace_output_prefill = defaultdict(lambda: None)

        self.trace_ids_decode = defaultdict(lambda: None)
        self.trace_inputs_decode = defaultdict(lambda: None)
        self.trace_output_decode = defaultdict(lambda: None)

        self.prev_page_table = None
        self.mode = None
        self.already_warmed_up_prefill = False

    def allocate_kv_cache(self, *args, **kwargs):
        return self._direct.allocate_kv_cache(*args, **kwargs)

    # =========================================================================
    # Warmup
    # =========================================================================

    def warmup_model_prefill(
        self, kv_cache=None, enable_trace=True, can_sample_on_device=False, non_greedy_decoding_on_device=False
    ):
        """Warmup prefill traces for supported sequence lengths."""
        if self.already_warmed_up_prefill:
            return
        self.already_warmed_up_prefill = True

        if not self.model_args:
            return

        kv_cache = getattr(self._direct, "_kv_cache", None)
        is_paged = kv_cache is not None and hasattr(self.model, "layers") and len(self.model.layers) > 0
        if is_paged:
            attn_cfg = self.model.layers[0].attention.config
            is_paged = attn_cfg.paged_attention_config is not None

        supported_seq_lens = self.model_args.get_warmup_prefill_supported_seq_lens()

        for seq_len in supported_seq_lens:
            warmup_tokens = torch.zeros(1, seq_len, dtype=torch.long)
            warmup_prompt_lens = torch.tensor([seq_len], dtype=torch.long)
            warmup_empty_slots = [0]

            warmup_page_table = None
            if is_paged:
                block_size = get_block_size(kv_cache)
                num_blocks = num_blocks_in_seq(seq_len, block_size)
                warmup_page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

            self.prefill_forward(
                warmup_tokens,
                page_table=warmup_page_table,
                kv_cache=kv_cache,
                prompt_lens=warmup_prompt_lens,
                empty_slots=warmup_empty_slots,
                enable_trace=False,
                warmup_prefill=False,
            )

        logger.info("Prefill warmup complete")

    # =========================================================================
    # Prefill (traced)
    # =========================================================================

    def prefill_forward(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        sampling_params=None,
        start_pos=None,
        warmup_prefill=True,
        **kwargs,
    ):
        """Traced prefill: lazy capture on first call per seq_len, replay after."""
        self.mode = Mode.PREFILL
        self._direct._assert_kv_cache_identity(kv_cache)

        if page_table is None:
            enable_trace = False

        if warmup_prefill:
            sampling_supported = getattr(self.model, "sampling", None) is not None
            self.warmup_model_prefill(
                enable_trace=enable_trace,
                can_sample_on_device=sampling_supported,
                non_greedy_decoding_on_device=sampling_supported,
            )

        batch_size, batch_seq_len = tokens.shape
        vocab_size = self.model.vocab_size
        cluster_shape = self.model_args.cluster_shape if self.model_args else [1, 1]
        output_tensor = torch.zeros(batch_size, 1, vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch_size)
        if empty_slots is None:
            empty_slots = list(range(batch_size))

        prefill_results = []

        for idx, user_id in enumerate(empty_slots):
            seq_len = int(prompt_lens[idx])
            num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
            last_token_idx = seq_len - 1
            prefill_seq_len = get_padded_prefill_len(seq_len - num_cached_tokens)

            prefill_ids = torch.cat(
                [
                    tokens[idx : idx + 1, num_cached_tokens:seq_len],
                    torch.zeros(1, prefill_seq_len - (seq_len - num_cached_tokens)).long(),
                ],
                dim=-1,
            )

            page_table_user = (
                _get_prefill_user_page_table(
                    page_table[idx : idx + 1],
                    kv_cache,
                    seq_len,
                    enable_trace,
                    prefill_seq_len,
                )
                if page_table is not None
                else None
            )

            can_trace = (
                enable_trace
                and self.model_args
                and self.model_args.can_enable_trace(prefill_seq_len, num_cached_tokens)
            )

            if can_trace:
                logits = self._easy_trace_prefill(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=0,
                    last_token_idx=last_token_idx,
                    prefill_seq_len=prefill_seq_len,
                )
                logits = self.model.norm.prefill_forward(
                    ttnn.slice(
                        logits,
                        (0, 0, (last_token_idx // 32) * 32, 0),
                        (1, 1, (last_token_idx // 32) * 32 + 32, logits.shape[-1]),
                    )
                )
                logits = _all_gather_rmsnorm_tensor(self.model.norm, logits)
                logits = self.model.lm_head.forward(logits)
                logits = ttnn.to_memory_config(logits, ttnn.DRAM_MEMORY_CONFIG)
            else:
                logits = self._direct._prefill_single_user(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=0 if page_table is not None else user_id,
                    last_token_idx=last_token_idx,
                    num_cached_tokens=num_cached_tokens,
                )

            logits = ttnn.untilize(logits, use_multicore=True)
            prefill_results.append(
                {
                    "idx": idx,
                    "last_token_idx": last_token_idx,
                    "logits": logits.cpu(blocking=False),
                }
            )

        for res in prefill_results:
            ttnn.synchronize_device(self.mesh_device)
            last_relative = res["last_token_idx"] - (int(start_pos[res["idx"]]) if start_pos is not None else 0)
            output_tensor[res["idx"]] = _process_output_prefill(
                res["logits"],
                last_relative % 32,
                vocab_size,
                cluster_shape,
            )

        return output_tensor

    def _easy_trace_prefill(self, tokens, page_table, user_id, last_token_idx, prefill_seq_len):
        """Lazy trace capture for prefill. Captures on first call per seq_len."""
        if self.trace_id_prefill[prefill_seq_len] is None:
            return self._capture_and_run_prefill_trace(
                tokens,
                page_table,
                user_id,
                last_token_idx,
                prefill_seq_len,
            )

        host_inputs = self._direct.prepare_prefill_inputs(
            tokens,
            page_table=page_table,
            trace_enabled=True,
            last_token_idx=last_token_idx,
        )
        copy_host_to_device(
            host_tensors=host_inputs,
            device_tensors=self.trace_inputs_prefill[prefill_seq_len],
        )

        ttnn.execute_trace(self.mesh_device, self.trace_id_prefill[prefill_seq_len], cq_id=0, blocking=False)
        return self.trace_output_prefill[prefill_seq_len]

    def _capture_and_run_prefill_trace(self, tokens, page_table, user_id, last_token_idx, prefill_seq_len):
        """Compile + capture trace for a specific prefill seq_len."""
        self._direct._prefill_single_user(
            tokens,
            page_table=page_table,
            user_id=user_id,
            last_token_idx=last_token_idx,
        )
        logger.info(f"Compiled prefill for seq_len={prefill_seq_len}")

        host_inputs = self._direct.prepare_prefill_inputs(
            tokens,
            page_table=page_table,
            trace_enabled=True,
            last_token_idx=last_token_idx,
        )
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        tokens_embd = self.model.embed_prefill(device_inputs[0])
        rot_mats = [device_inputs[1], device_inputs[2]]
        tt_page_table = device_inputs[3]
        tt_chunk_page_table = device_inputs[4]

        max_seq_len = self.model_args.max_seq_len if self.model_args else prefill_seq_len
        logits = self.model.prefill_forward(
            tokens_embd,
            device_inputs[1],
            user_id=user_id,
            page_table=tt_page_table,
            chunk_page_table=tt_chunk_page_table,
            get_last_token=-1,
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)

        self.trace_id_prefill[prefill_seq_len] = trace_id
        self.trace_inputs_prefill[prefill_seq_len] = device_inputs
        self.trace_output_prefill[prefill_seq_len] = logits

        logger.info(f"Captured prefill trace for seq_len={prefill_seq_len}")
        return logits

    # =========================================================================
    # Decode (traced)
    # =========================================================================

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        **kwargs,
    ):
        """Traced decode: lazy capture on first call, replay after."""
        self.mode = Mode.DECODE
        self._direct._assert_kv_cache_identity(kv_cache)
        sampling_on_device = sampling_params is not None
        B = tokens.shape[0]
        vocab_size = self.model.vocab_size
        num_devices = self.model.num_devices
        cluster_shape = self.model_args.cluster_shape if self.model_args else [1, 1]

        if sampling_on_device and self.model.sampling is not None:
            from models.common.sampling import broadcast_sampling_params, format_sampling_params

            per_request_params = format_sampling_params(broadcast_sampling_params(sampling_params, 0, slot_len=B), B)
            self.model.sampling.apply_decode_state(
                [per_request_params],
                reset_batch=False,
                prompt_tokens=kwargs.get("prompt_tokens"),
                output_tokens=kwargs.get("output_tokens"),
            )
            self.model.sampling.seed_manager.get_new_values()

        if not enable_trace:
            return self._direct.decode_forward(
                tokens,
                start_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=False,
                read_from_device=read_from_device,
                sampling_params=sampling_params,
                **kwargs,
            )

        if not self.trace_ids_decode[sampling_on_device]:
            self._capture_decode_trace(tokens, start_pos, page_table, kv_cache, sampling_on_device)

        reset_inputs = self.prev_page_table is None or (
            page_table is not None and not torch.equal(self.prev_page_table, page_table)
        )
        if reset_inputs:
            host_inputs = self._direct.prepare_decode_inputs_host(tokens, start_pos, page_table)
            copy_host_to_device(
                host_tensors=host_inputs,
                device_tensors=self.trace_inputs_decode[sampling_on_device],
            )
            if page_table is not None:
                self.prev_page_table = page_table.clone()

        ttnn.execute_trace(
            self.mesh_device,
            self.trace_ids_decode[sampling_on_device],
            cq_id=0,
            blocking=False,
        )
        tt_output = self.trace_output_decode[sampling_on_device]

        if read_from_device:
            if sampling_on_device:
                tt_toks, tt_log_probs = tt_output
                toks_host = tt_toks.cpu()
                ttnn.synchronize_device(self.mesh_device)
                return _process_output_decode_tokens(toks_host, B, cluster_shape), None
            else:
                logits, _ = tt_output
                logits_host = logits.cpu()
                ttnn.synchronize_device(self.mesh_device)
                return _process_output_decode(logits_host, B, vocab_size, num_devices, cluster_shape), None

        return tt_output

    def _capture_decode_trace(self, tokens, start_pos, page_table, kv_cache, sampling_on_device):
        """Compile + capture decode trace."""
        self._direct.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,  # passed to _direct for identity assertion
            enable_trace=False,
            read_from_device=False,
            sampling_params=None,
        )
        logger.info("Compiled decode")

        host_inputs = self._direct.prepare_decode_inputs_host(tokens, start_pos, page_table)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        tt_tokens, tt_current_pos, tt_rot_mat_idxs, tt_page_table = device_inputs
        rot_mats = self.model.rope_setup.get_rot_mats(tt_rot_mat_idxs)
        x_embed = self.model.embed_decode(tt_tokens)

        logits = self.model.decode_forward(
            x_embed,
            tt_current_pos,
            rot_mats,
            page_table=tt_page_table,
        )

        if sampling_on_device and self.model.sampling is not None:
            self.model.increment_positions(tt_current_pos, tt_rot_mat_idxs)
            tt_toks, tt_log_probs = self.model.sampling.decode_forward(logits, tt_out_tok=tt_tokens)
            output = (tt_toks, tt_log_probs)
        else:
            logits = self.model.gather_and_untilize_logits(logits)
            output = (logits, None)

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)

        self.trace_ids_decode[sampling_on_device] = trace_id
        self.trace_inputs_decode[sampling_on_device] = device_inputs
        self.trace_output_decode[sampling_on_device] = output

        logger.info("Captured decode trace")

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(self):
        """Release all captured traces."""
        if self._cleaned_up:
            return

        for key, trace_id in list(self.trace_id_prefill.items()):
            if trace_id is not None:
                ttnn.release_trace(self.mesh_device, trace_id)
                self.trace_id_prefill[key] = None
        for key, trace_id in list(self.trace_ids_decode.items()):
            if trace_id is not None:
                ttnn.release_trace(self.mesh_device, trace_id)
                self.trace_ids_decode[key] = None

        self._cleaned_up = True


# =============================================================================
# TeacherForceExecutor
# =============================================================================


@dataclass
class TeacherForceResult:
    """Result from a teacher-forcing evaluation run."""

    predicted_tokens: list[int]
    reference_top5: torch.Tensor  # shape [num_tokens, 5]

    def top1_accuracy(self) -> float:
        matches = sum(1 for i, p in enumerate(self.predicted_tokens) if self.reference_top5[i, 0].item() == p)
        return matches / len(self.predicted_tokens)

    def top5_accuracy(self) -> float:
        matches = sum(1 for i, p in enumerate(self.predicted_tokens) if p in self.reference_top5[i, :])
        return matches / len(self.predicted_tokens)


class TeacherForceExecutor:
    """Accuracy measurement via teacher forcing.

    Takes only a direct (non-traced) LlamaExecutor — tracing is
    incompatible with teacher forcing because inputs change every step.
    """

    def __init__(self, executor: LlamaExecutor):
        if not isinstance(executor, LlamaExecutor):
            raise TypeError(
                f"TeacherForceExecutor requires LlamaExecutor (non-traced), got {type(executor).__name__}. "
                "Teacher forcing is incompatible with tracing because inputs change every step."
            )
        self.executor = executor

    def run(
        self,
        prompt_tokens: torch.Tensor,
        reference_tokens: torch.Tensor,
        top5_tokens: torch.Tensor,
        kv_cache: list,
        page_table: torch.Tensor | None = None,
        max_batch_size: int = 1,
    ) -> TeacherForceResult:
        """Run teacher-forcing evaluation.

        Args:
            prompt_tokens: Prompt token IDs, shape [1, prompt_len].
            reference_tokens: Full reference sequence (prompt + target), shape [total_len].
            top5_tokens: Top-5 reference tokens per position, shape [num_target_tokens, 5].
            kv_cache: Per-layer KV cache from allocate_kv_cache.
            page_table: Page table for paged attention, or None.
            max_batch_size: Maximum batch size (for decode input padding).

        Returns:
            TeacherForceResult with predicted tokens and accuracy metrics.
        """
        prompt_len = prompt_tokens.shape[-1]
        total_len = len(reference_tokens)
        num_target = total_len - prompt_len

        logger.info(f"Teacher forcing: prefilling {prompt_len} tokens")
        prefill_output = self.executor.prefill_forward(
            prompt_tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=torch.tensor([prompt_len]),
            empty_slots=[0],
            enable_trace=False,
            warmup_prefill=False,
        )

        first_logits = prefill_output[0]
        first_token = torch.argmax(first_logits, dim=-1).item()
        predicted_tokens = [first_token]

        logger.info(f"Teacher forcing: decoding {num_target - 1} tokens")
        for step in range(1, num_target):
            gt_token = reference_tokens[prompt_len + step - 1]
            decode_token = torch.full((max_batch_size,), 0, dtype=torch.long)
            decode_token[0] = gt_token

            current_pos = torch.full((max_batch_size,), -1, dtype=torch.long)
            current_pos[0] = prompt_len + step - 1

            logits, _ = self.executor.decode_forward(
                decode_token,
                current_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=False,
                read_from_device=True,
            )

            pred = torch.argmax(logits[0], dim=-1).item()
            predicted_tokens.append(pred)

        return TeacherForceResult(
            predicted_tokens=predicted_tokens,
            reference_top5=top5_tokens[:num_target],
        )


# =============================================================================
# PerfBenchmarkExecutor
# =============================================================================


@dataclass
class PerfBenchmarkResult:
    """Result from a performance benchmark run."""

    prefill_time_s: float
    compile_decode_time_s: float
    decode_times_s: list[float]
    batch_size: int
    num_decode_tokens: int

    @property
    def ttft_ms(self) -> float:
        """Average time-to-first-token per user (ms)."""
        return self.prefill_time_s / self.batch_size * 1000

    @property
    def tok_s_u(self) -> float:
        """Tokens per second per user (steady-state decode)."""
        if not self.decode_times_s:
            return 0.0
        return len(self.decode_times_s) / sum(self.decode_times_s)

    @property
    def tok_s(self) -> float:
        """Total throughput."""
        return self.tok_s_u * self.batch_size

    @property
    def decode_latency_mean_ms(self) -> float:
        if not self.decode_times_s:
            return 0.0
        return (sum(self.decode_times_s) / len(self.decode_times_s)) * 1000

    def meets_target(self, expected: dict, tolerance: float = 0.05) -> dict[str, bool]:
        """Check against expected metrics. Returns {metric: passed}."""
        return {
            "tok_s_u": self.tok_s_u >= expected["tok_s_u"] * (1 - tolerance),
            "ttft_ms": self.ttft_ms <= expected["ttft_ms"] * (1 + tolerance),
        }


class PerfBenchmarkExecutor:
    """Performance measurement (TTFT, tok/s/u).

    Takes any executor (typically TracedLlamaExecutor for realistic numbers).
    Owns the timed prefill + decode loop; returns PerfBenchmarkResult.
    """

    def __init__(self, executor):
        self.executor = executor

    def run(
        self,
        tokens: torch.Tensor,
        kv_cache: list,
        page_table: torch.Tensor | None = None,
        num_decode_tokens: int = 128,
        max_batch_size: int = 1,
        start_pos: list[int] | None = None,
        enable_trace: bool = True,
        sampling_params=None,
    ) -> PerfBenchmarkResult:
        """Timed prefill + decode loop.

        Matches TTTv1 methodology: compile prefill is excluded from TTFT.
        Iteration 0 of decode is the compile iteration (timed separately).
        Returns PerfBenchmarkResult with raw timings + derived metrics.
        """
        batch_size = tokens.shape[0]
        prompt_len = tokens.shape[1]
        max_batch_size = max(max_batch_size, batch_size)

        prefill_kwargs = dict(
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=torch.tensor([prompt_len] * batch_size),
            empty_slots=list(range(batch_size)),
            enable_trace=False,
            start_pos=start_pos,
            sampling_params=sampling_params,
        )

        # Compile prefill: warmup + first run (excluded from TTFT, matches TTTv1)
        self.executor.prefill_forward(tokens, **prefill_kwargs)
        if hasattr(self.executor, "mesh_device"):
            ttnn.synchronize_device(self.executor.mesh_device)

        # Inference prefill: timed run with ops already compiled (this is TTFT)
        t0 = time.perf_counter()
        prefill_output = self.executor.prefill_forward(tokens, **prefill_kwargs)
        if hasattr(self.executor, "mesh_device"):
            ttnn.synchronize_device(self.executor.mesh_device)
        prefill_time = time.perf_counter() - t0

        if isinstance(prefill_output, tuple):
            first_token = prefill_output[0]
        else:
            first_token = torch.argmax(prefill_output, dim=-1)

        current_tokens = torch.zeros(max_batch_size, dtype=torch.long)
        current_tokens[:batch_size] = first_token.view(-1)[:batch_size]

        current_pos = torch.full((max_batch_size,), -1, dtype=torch.long)
        current_pos[:batch_size] = prompt_len

        compile_time = None
        decode_times = []

        for i in range(num_decode_tokens):
            t0 = time.perf_counter()
            logits, _ = self.executor.decode_forward(
                current_tokens,
                current_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                read_from_device=True,
                sampling_params=sampling_params,
            )
            if hasattr(self.executor, "mesh_device"):
                ttnn.synchronize_device(self.executor.mesh_device)
            elapsed = time.perf_counter() - t0

            if i == 0:
                compile_time = elapsed
            else:
                decode_times.append(elapsed)

            if isinstance(logits, torch.Tensor) and logits.dim() >= 2:
                next_tok = torch.argmax(logits[:, -1, :], dim=-1)
            else:
                next_tok = logits
            current_tokens[:batch_size] = next_tok.view(-1)[:batch_size]
            current_pos[:batch_size] += 1

        return PerfBenchmarkResult(
            prefill_time_s=prefill_time,
            compile_decode_time_s=compile_time or 0.0,
            decode_times_s=decode_times,
            batch_size=batch_size,
            num_decode_tokens=num_decode_tokens,
        )
