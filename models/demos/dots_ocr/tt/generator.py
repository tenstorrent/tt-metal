# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time

import torch
from loguru import logger

from models.common.warmup import WarmupForwardMixin
from models.demos.dots_ocr.tt._ttnn_import import get_ttnn
from models.demos.dots_ocr.tt.common import (
    fused_ttnn_embeddings_to_torch,
    get_block_size,
    get_max_prefill_chunk_size,
    num_blocks_in_seq,
)
from models.tt_transformers.tt.generator import Generator as TTTGenerator


class Generator(WarmupForwardMixin):
    """
    Dots OCR generator: TT Transformers prefill/decode path (chunked prefill, rot_mats on host for text).
    """

    def __init__(self, model, model_args, mesh_device, processor=None, tokenizer=None):
        self._ttt = TTTGenerator([model], [model_args], mesh_device, processor=processor, tokenizer=tokenizer)

    @property
    def model(self):
        return self._ttt.model[0]

    @property
    def model_args(self):
        return self._ttt.model_args[0]

    @property
    def mesh_device(self):
        return self._ttt.mesh_device

    @property
    def tokenizer(self):
        return self._ttt.tokenizer

    @property
    def processor(self):
        return self._ttt.processor

    def prefill_forward_text(self, tokens, rot_mats, page_table=None, kv_cache=None, prompt_lens=None):
        ttnn = get_ttnn()
        if ttnn is not None and isinstance(tokens, ttnn.Tensor):
            tokens = fused_ttnn_embeddings_to_torch(tokens, self.mesh_device)
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("prefill_forward_text expects torch or ttnn embedding tensor [B, S, D] after conversion")
        batch, batch_seq_len = tokens.shape[:2]
        output_logits = torch.zeros(batch, 1, self.model_args.vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch)

        if page_table is not None:
            assert isinstance(page_table, torch.Tensor), "page_table must be a torch.Tensor when passing into prefill"

        for user_id in range(batch):
            logger.info(f"Prefilling User {user_id + 1}")
            seq_len = prompt_lens[user_id]
            last_token_idx = seq_len - 1

            if page_table is not None:
                page_table_user = self._ttt._get_prefill_user_page_table(page_table, kv_cache, seq_len)
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

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")
        return output_logits

    def prefill_forward_embeddings(self, input_embeds, rot_mats, *, page_table=None, kv_cache=None, prompt_lens=None):
        """Alias: embeddings are already `[B, S, D]`."""
        return self.prefill_forward_text(
            input_embeds, rot_mats, page_table=page_table, kv_cache=kv_cache, prompt_lens=prompt_lens
        )

    def update_cos_sin(self, cos_matrix_pt=None, sin_matrix_pt=None):
        self.model.rope_setup.update_cos_sin(cos_matrix_pt=cos_matrix_pt, sin_matrix_pt=sin_matrix_pt)

    def update_cos_sin_rows(self, rot_mats_seq_ids):
        for i, (cos, sin) in enumerate(rot_mats_seq_ids):
            self.model.rope_setup.cos_matrix_pt[i] = cos[0]
            self.model.rope_setup.sin_matrix_pt[i] = sin[0]
        self.update_cos_sin()

    def update_rope_deltas(self, rope_deltas_list: list):
        rope_deltas_list = rope_deltas_list + [0] * (self.model.rope_setup.batch_size - len(rope_deltas_list))
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
        return self._ttt.decode_forward(
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
            assert page_table is not None, "page_table must be provided for chunked prefill"
            assert kv_cache is not None, "kv_cache must be provided for chunked prefill"
            assert last_token_idx is not None and last_token_idx < seq_len, "last_token_idx must be valid"
            chunk_size = get_max_prefill_chunk_size(seq_len, self.model_args.max_prefill_chunk_size)
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
                assert chunk_end <= seq_len, f"Chunk end invalid: chunk_end={chunk_end} seq_len={seq_len}"
                chunk_tokens = tokens[:, chunk_start:chunk_end]
                chunk_page_table = page_table_user[:, chunk_start // block_size : chunk_end // block_size]

                (
                    chunk_prefill_input,
                    chunk_rot_mats_prefill_global,
                    chunk_rot_mats_prefill_local,
                    page_table_tt,
                    chunk_page_table_tt,
                ) = self._prepare_inputs_prefill_compat(
                    chunk_tokens,
                    rot_mats=rot_mats,
                    start_pos=chunk_start,
                    page_table=page_table_user_padded,
                    chunk_page_table=chunk_page_table,
                )
                tt_logits = self.model.ttnn_prefill_forward(
                    chunk_prefill_input,
                    rot_mats_global=[rm[user_id : user_id + 1, ...] for rm in chunk_rot_mats_prefill_global],
                    rot_mats_local=(
                        [rm[user_id : user_id + 1, ...] for rm in chunk_rot_mats_prefill_local]
                        if chunk_rot_mats_prefill_local is not None
                        else None
                    ),
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
                del tt_logits
        else:
            ttnn = get_ttnn()
            if ttnn is None:
                raise RuntimeError("ttnn is required for prefill_forward")
            debug_sync = os.getenv("DOTS_PREFILL_DEBUG_SYNC", "").lower() in ("1", "true", "yes", "y")
            t0 = time.time()
            logger.info(f"Prefill(single-user): preparing inputs seq_len={seq_len} user_id={user_id}")

            # tt_transformers prefill kernels (attention/MLP) commonly assume the input sequence
            # length is padded to a multiple of 128. Padding at the end is safe here because
            # causal attention prevents earlier tokens from attending to future (padded) tokens.
            padded_len = ((int(seq_len) + 127) // 128) * 128
            if padded_len != seq_len:
                if tokens.dim() == 3:
                    pad = torch.zeros(tokens.shape[0], padded_len - seq_len, tokens.shape[2], dtype=tokens.dtype)
                    tokens = torch.cat([tokens, pad], dim=1)
                    # Ensure RoPE matrices cover the padded length too *if* caller supplied host rot_mats.
                    if rot_mats is not None:
                        from models.tt_transformers.tt.common import precompute_freqs

                        head_dim = getattr(self.model_args, "head_dim", 128)
                        theta = getattr(self.model_args, "rope_theta", 10000.0)
                        rope_scaling = getattr(self.model_args, "rope_scaling", None)
                        scale_factor = getattr(rope_scaling, "factor", None) if rope_scaling is not None else None
                        orig_context_len = (
                            getattr(rope_scaling, "original_max_position_embeddings", None)
                            if rope_scaling is not None
                            else None
                        )
                        rope_type = (
                            getattr(getattr(rope_scaling, "rope_type", None), "value", "llama3")
                            if rope_scaling
                            else "llama3"
                        )

                        cos_freqs, sin_freqs = precompute_freqs(
                            head_dim,
                            padded_len * 2,
                            theta=theta,
                            scale_factor=scale_factor,
                            orig_context_len=orig_context_len,
                            rope_type=rope_type,
                        )
                        cos_hf = (
                            torch.cat([cos_freqs[:padded_len], cos_freqs[:padded_len]], dim=-1)
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                        sin_hf = (
                            torch.cat([sin_freqs[:padded_len], sin_freqs[:padded_len]], dim=-1)
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                        rot_mats = (cos_hf, sin_hf)
                elif tokens.dim() == 2:
                    # Token-id prefill: pad with zeros (usually <pad> / <unk>, but attention is causal so it won't affect earlier tokens).
                    pad = torch.zeros(tokens.shape[0], padded_len - seq_len, dtype=tokens.dtype)
                    tokens = torch.cat([tokens, pad], dim=1)
                    # If caller supplied host rot_mats, ensure they cover the padded length too.
                    if rot_mats is not None:
                        from models.tt_transformers.tt.common import precompute_freqs

                        head_dim = getattr(self.model_args, "head_dim", 128)
                        theta = getattr(self.model_args, "rope_theta", 10000.0)
                        rope_scaling = getattr(self.model_args, "rope_scaling", None)
                        scale_factor = getattr(rope_scaling, "factor", None) if rope_scaling is not None else None
                        orig_context_len = (
                            getattr(rope_scaling, "original_max_position_embeddings", None)
                            if rope_scaling is not None
                            else None
                        )
                        rope_type = (
                            getattr(getattr(rope_scaling, "rope_type", None), "value", "llama3")
                            if rope_scaling
                            else "llama3"
                        )

                        cos_freqs, sin_freqs = precompute_freqs(
                            head_dim,
                            padded_len * 2,
                            theta=theta,
                            scale_factor=scale_factor,
                            orig_context_len=orig_context_len,
                            rope_type=rope_type,
                        )
                        cos_hf = (
                            torch.cat([cos_freqs[:padded_len], cos_freqs[:padded_len]], dim=-1)
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                        sin_hf = (
                            torch.cat([sin_freqs[:padded_len], sin_freqs[:padded_len]], dim=-1)
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                        rot_mats = (cos_hf, sin_hf)
                else:
                    raise AssertionError(f"Unexpected tokens rank {tokens.dim()} in prefill")

            (
                prefill_input,
                rot_mats_prefill_global,
                rot_mats_prefill_local,
                page_table_tt,
                _,
            ) = self._prepare_inputs_prefill_compat(
                tokens,
                rot_mats=rot_mats,
                page_table=page_table,
            )
            logger.info(
                f"Prefill(single-user): inputs prepared in {(time.time() - t0):.2f}s; launching ttnn_prefill_forward"
            )

            tt_logits = self.model.ttnn_prefill_forward(
                prefill_input,
                rot_mats_global=[rm[user_id : user_id + 1, ...] for rm in rot_mats_prefill_global],
                rot_mats_local=(
                    [rm[user_id : user_id + 1, ...] for rm in rot_mats_prefill_local]
                    if rot_mats_prefill_local is not None
                    else None
                ),
                user_id=user_id,
                page_table=page_table_tt,
                get_last_token=(last_token_idx // 32) * 32,
                kv_cache=kv_cache,
            )
            if debug_sync:
                logger.info("Prefill(single-user): synchronizing mesh device (DOTS_PREFILL_DEBUG_SYNC=1)")
                ttnn.synchronize_device(self.mesh_device)

            logger.info("Prefill(single-user): transferring logits to host (tt_logits.cpu())")
            logits = self.model.process_output_prefill(tt_logits.cpu(), last_token_idx=(last_token_idx % 32))

            ttnn.deallocate(tt_logits)
            ttnn.deallocate(prefill_input)
            if page_table is not None:
                ttnn.deallocate(page_table_tt)

            return logits

    def _prepare_inputs_prefill_compat(self, *args, **kwargs):
        """
        Compatibility wrapper around `prepare_inputs_prefill`.

        `DotsTransformer.prepare_inputs_prefill` (embeddings path) returns 4-tuple:
          (prefill_input, rot_mats_prefill, page_table_tt, chunk_page_table_tt)

        The parent `tt_transformers` implementation (token-id path) returns a 5-tuple:
          (prefill_input, rot_mats_global, rot_mats_local, page_table_tt, chunk_page_table_tt)

        This wrapper always returns a 5-tuple:
          (prefill_input, rot_mats_global, rot_mats_local, page_table_tt, chunk_page_table_tt)
        For embedding-prefill paths that don't have `rot_mats_local`, it returns None for that slot.
        """
        out = self.model.prepare_inputs_prefill(*args, **kwargs)
        if not isinstance(out, (tuple, list)):
            raise TypeError(f"prepare_inputs_prefill returned {type(out).__name__}, expected tuple")
        if len(out) == 4:
            prefill_input, rot_mats_global, page_table_tt, chunk_page_table_tt = out
            return prefill_input, rot_mats_global, None, page_table_tt, chunk_page_table_tt
        if len(out) == 5:
            prefill_input, rot_mats_global, _rot_mats_local, page_table_tt, chunk_page_table_tt = out
            return prefill_input, rot_mats_global, _rot_mats_local, page_table_tt, chunk_page_table_tt
        raise ValueError(f"prepare_inputs_prefill returned {len(out)} values, expected 4 or 5")

    def read_decode_output(self, tt_out, async_read=False):
        return self._ttt.read_decode_output(tt_out, async_read=async_read)

    def process_decode_output_host(self, tt_out, is_tokens=False):
        return self._ttt.process_decode_output_host(tt_out, is_tokens=is_tokens)

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, non_greedy_decoding_on_device) -> None:
        logger.warning("Warmup model prefill not implemented for Dots OCR Generator")

    def __del__(self):
        ttnn = get_ttnn()
        if ttnn is not None:
            if hasattr(self, "trace_id"):
                ttnn.release_trace(self.mesh_device, self.trace_id)
            if hasattr(self, "trace_id_text"):
                ttnn.release_trace(self.mesh_device, self.trace_id_text)
        self._ttt.__del__()
