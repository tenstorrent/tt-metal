# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device generation helpers for DiffusionGemma (#47464).

This module owns the outer block-generation glue that is specific to
DiffusionGemma. It starts with the commit step: once the denoise controller has
chosen the clean argmax canvas, append those tokens to the frozen KV cache using
Gemma4's decode path in ``COMMIT_APPEND`` phase.
"""

from __future__ import annotations

import torch

from models.demos.gemma4.tt.attention.kv_phase import KVCachePhase


def _deallocate_decode_inputs(device_inputs) -> None:
    for value in device_inputs:
        if value is not None and hasattr(value, "deallocate"):
            value.deallocate(True)


def commit_canvas_tokens(
    tt_model,
    canvas_tokens: torch.Tensor,
    *,
    start_pos: int,
    page_table=None,
    page_tables_per_layer=None,
) -> None:
    """Append committed canvas token ids to the model KV cache.

    Args:
        tt_model: Gemma4-backed DiffusionGemma model with ``tt_kv_cache``.
        canvas_tokens: Host token ids ``[batch, canvas_len]``. This matches the
            current W3 controller output, which returns the clean argmax canvas
            on host for trajectory comparison.
        start_pos: Absolute position of ``canvas_tokens[:, 0]``. For block ``N``
            this is ``prompt_len + N * canvas_len``.
        page_table: Optional shared page table for the decode append path.
        page_tables_per_layer: Optional hybrid per-layer page tables.
    """
    if canvas_tokens.dim() != 2:
        raise ValueError("canvas_tokens must have shape [batch, canvas_len]")
    if canvas_tokens.shape[0] != 1:
        raise NotImplementedError("commit_canvas_tokens currently supports batch=1")

    for offset in range(canvas_tokens.shape[1]):
        token = canvas_tokens[:, offset]
        position = torch.tensor([start_pos + offset], dtype=torch.int32)
        device_inputs = tt_model.prepare_inputs_decode(token, position, page_table=page_table)
        logits, _ = tt_model.ttnn_decode_forward(
            device_inputs[0],
            device_inputs[1],
            device_inputs[2],
            device_inputs[3],
            page_tables_per_layer=page_tables_per_layer,
            kv_phase=KVCachePhase.COMMIT_APPEND,
        )
        logits.deallocate(True)
        _deallocate_decode_inputs(device_inputs)
