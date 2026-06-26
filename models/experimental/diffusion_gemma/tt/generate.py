# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device generation helpers for DiffusionGemma (#47464).

This module owns the outer block-generation glue that is specific to
DiffusionGemma. It starts with the commit step: once the denoise controller has
chosen the clean argmax canvas, append those tokens to the frozen KV cache using
Gemma4's decode path in ``COMMIT_APPEND`` phase.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import torch

from models.demos.gemma4.tt.attention.kv_phase import KVCachePhase
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory
from models.experimental.diffusion_gemma.tt.denoise_loop import denoise_block as tt_denoise_block


class GeneratedBlock(NamedTuple):
    committed: torch.Tensor
    next_pos: int
    trajectory: DenoiseTrajectory


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


def _set_q_rope_offset(logits_fn, q_rope_offset: int) -> None:
    if hasattr(logits_fn, "q_rope_offset"):
        logits_fn.q_rope_offset = q_rope_offset


def denoise_and_commit_block(
    tt_model,
    logits_fn,
    init_canvas,
    config: DiffusionConfig,
    *,
    start_pos: int,
    gumbel_noise_fn=None,
    noise_tokens_fn=None,
    page_table=None,
    page_tables_per_layer=None,
    denoise_block_fn: Callable[..., DenoiseTrajectory] = tt_denoise_block,
    commit_fn: Callable[..., None] = commit_canvas_tokens,
) -> GeneratedBlock:
    """Denoise one canvas, commit the clean argmax, and advance position.

    ``start_pos`` is the absolute canvas start for this block. When ``logits_fn``
    is a ``DenoiseLogitsAdapter`` this helper updates its ``q_rope_offset`` so
    canvas RoPE positions advance with each committed block.
    """
    _set_q_rope_offset(logits_fn, start_pos)
    trajectory = denoise_block_fn(
        logits_fn,
        init_canvas,
        config,
        gumbel_noise_fn=gumbel_noise_fn,
        noise_tokens_fn=noise_tokens_fn,
    )
    if trajectory.committed is None:
        raise RuntimeError("denoise trajectory did not produce committed canvas tokens")
    commit_fn(
        tt_model,
        trajectory.committed,
        start_pos=start_pos,
        page_table=page_table,
        page_tables_per_layer=page_tables_per_layer,
    )
    return GeneratedBlock(
        committed=trajectory.committed,
        next_pos=start_pos + trajectory.committed.shape[1],
        trajectory=trajectory,
    )
