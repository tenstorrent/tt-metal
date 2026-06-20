# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for text-backbone prefill perf tests."""

from __future__ import annotations

import torch
import ttnn

_PREFILL_BULK_MAX_ISL = 256


def use_paged_kv_for_isl(isl: int) -> bool:
    return isl > _PREFILL_BULK_MAX_ISL


def can_bulk_prefill(isl: int, *, paged_kv: bool = False) -> bool:
    if paged_kv:
        return False
    return isl <= _PREFILL_BULK_MAX_ISL and isl % 128 == 0


def prefill_tile_start(token_index: int) -> int:
    return (int(token_index) // 32) * 32


def prefill_token_range(
    model,
    tokens: torch.Tensor,
    start: int,
    end: int,
    *,
    page_table_tt: ttnn.Tensor | None = None,
) -> None:
    """Per-token KV fill for ``tokens[:, start:end]`` at positions ``start .. end-1``."""
    for i in range(start, end):
        step_token = tokens[:, i]
        tt_tokens, tt_current_pos, tt_rope_idxs, tt_page_table = model.prepare_inputs_decode(
            step_token, torch.tensor([i], dtype=torch.int64)
        )
        page_table = tt_page_table if page_table_tt is None else page_table_tt
        _ = model.inner.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mat_idxs=tt_rope_idxs,
            page_table=page_table,
            kv_cache=None,
        )


def prefill_tokens(
    model,
    tokens: torch.Tensor,
    *,
    page_table_tt: ttnn.Tensor | None = None,
    paged_kv: bool = False,
    start: int = 0,
) -> None:
    """Fill KV for ``tokens`` from ``start`` through seq_len (bulk or per-token)."""
    isl = int(tokens.shape[1])
    if start >= isl:
        return
    if start == 0 and can_bulk_prefill(isl, paged_kv=paged_kv):
        tt_x, rot_mats_global, rot_mats_local, _, _, _ = model.prepare_inputs_prefill(tokens, start_pos=0)
        _ = model.inner.ttnn_prefill_forward(
            tt_x,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            get_last_token=prefill_tile_start(isl - 1),
        )
        return
    prefill_token_range(model, tokens, start, isl, page_table_tt=page_table_tt)


def decode_one_step(model, token: torch.Tensor, pos: int) -> None:
    tt_tokens, tt_current_pos, tt_rope_idxs, tt_page_table = model.prepare_inputs_decode(
        token, torch.tensor([pos], dtype=torch.int64)
    )
    _ = model.inner.ttnn_decode_forward(
        tt_tokens,
        tt_current_pos,
        rot_mat_idxs=tt_rope_idxs,
        page_table=tt_page_table,
        kv_cache=None,
    )
