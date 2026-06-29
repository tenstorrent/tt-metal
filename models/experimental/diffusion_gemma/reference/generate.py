# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block-autoregressive multi-canvas generation (reference, #47464).

The outer loop that assembles per-block denoise trajectories into a full
generation (plan.md §2.1): for each 256-token block, initialize a random canvas,
denoise it to convergence, commit the clean argmax, and append it to the
generated sequence so the model's prefix grows — then the next block. This is
the text-first e2e algorithm (#47464); on device the growing prefix is the
frozen committed KV managed by the phase state machine (#47474).

The model is injected as ``model(prefix_tokens, canvas_tokens, step) -> logits``
so the loop is testable with a mock now and the real backbone (HF / device)
later. KV / attention / mask mechanics live in the model callback; this loop
only owns the commit-and-extend control flow.
"""

from __future__ import annotations

from typing import Callable, List, NamedTuple, Optional

import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory, NoiseFn, denoise_block

# model(prefix_tokens [B, P], canvas_tokens [B, C], step) -> logits [B, C, vocab]
BlockModel = Callable[..., torch.Tensor]
# per-block noise hook: (block_index) -> a denoise NoiseFn for that block
BlockNoiseFn = Callable[[int], NoiseFn]
# per-block canvas init hook: (block_index, prefix_tokens) -> [B, canvas_len]
BlockCanvasFn = Callable[[int, torch.Tensor], torch.Tensor]


class Generation(NamedTuple):
    generated: torch.Tensor  # [B, num_blocks * canvas_length] committed tokens
    prompt_len: int
    trajectories: List[DenoiseTrajectory]  # one per block


def _validate_replay_tensors(tensors, *, kind: str) -> None:
    if not tensors:
        return
    expected_shape = tuple(tensors[0].shape)
    for tensor in tensors:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{kind} must all have shape {list(expected_shape)}")


def _validate_replay_blocks(blocks, *, kind: str) -> None:
    expected_shape = None
    for block in blocks:
        _validate_replay_tensors(block, kind=kind)
        if not block:
            continue
        block_shape = tuple(block[0].shape)
        if expected_shape is None:
            expected_shape = block_shape
        elif block_shape != expected_shape:
            raise ValueError(f"{kind} must all have shape {list(expected_shape)}")


def make_replay_canvas_init_fn(host_canvases) -> BlockCanvasFn:
    """Create an ``init_canvas_fn`` that replays fixed host canvases by block."""
    canvases = [canvas.clone() for canvas in host_canvases]
    _validate_replay_tensors(canvases, kind="replay canvas")

    def init_canvas_fn(block_idx: int, prefix_tokens: torch.Tensor) -> torch.Tensor:
        del prefix_tokens
        if block_idx < 0 or block_idx >= len(canvases):
            raise IndexError(f"replay canvas block index {block_idx} out of range for {len(canvases)} blocks")
        return canvases[block_idx].clone()

    return init_canvas_fn


def make_replay_noise_fn(host_noise) -> BlockNoiseFn:
    """Create a block/step noise hook that replays fixed host tensors."""
    blocks = [[noise.clone() for noise in block] for block in host_noise]
    _validate_replay_blocks(blocks, kind="replay noise")

    def noise_for_block(block_idx: int) -> NoiseFn:
        if block_idx < 0 or block_idx >= len(blocks):
            raise IndexError(f"replay noise block index {block_idx} out of range for {len(blocks)} blocks")

        def noise_for_step(step: int) -> torch.Tensor:
            if step < 0 or step >= len(blocks[block_idx]):
                raise IndexError(
                    f"replay noise step index {step} out of range for block {block_idx} "
                    f"with {len(blocks[block_idx])} steps"
                )
            return blocks[block_idx][step].clone()

        return noise_for_step

    return noise_for_block


def _validate_generate_args(
    prompt_tokens: torch.Tensor,
    *,
    num_blocks: int,
    canvas_len: int,
    vocab_size: int,
) -> None:
    if prompt_tokens.dim() != 2:
        raise ValueError("prompt_tokens must have shape [batch, seq_len]")
    if num_blocks < 0:
        raise ValueError("num_blocks must be non-negative")
    if canvas_len <= 0:
        raise ValueError("diffusion_config.canvas_length must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")


def _validate_canvas_shape(canvas: torch.Tensor, *, batch: int, canvas_len: int) -> None:
    if canvas.shape != (batch, canvas_len):
        raise ValueError(f"init_canvas_fn must return shape [{batch}, {canvas_len}]")


def generate_blocks(
    model: BlockModel,
    prompt_tokens: torch.Tensor,
    num_blocks: int,
    diffusion_config: DiffusionConfig,
    vocab_size: int,
    *,
    sampler: str = S.SAMPLER_MULTINOMIAL,
    generator: Optional[torch.Generator] = None,
    gumbel_noise_fn: Optional[BlockNoiseFn] = None,
    noise_tokens_fn: Optional[BlockNoiseFn] = None,
    init_canvas_fn: Optional[BlockCanvasFn] = None,
) -> Generation:
    """Generate ``num_blocks`` canvases autoregressively, committing each.

    ``generator`` seeds BOTH the per-block random canvas init AND (threaded into
    ``denoise_block``) the per-step regenerated sample/renoise noise, so a single
    seeded generator reproduces the whole generation. ``sampler`` selects the
    HF-faithful ``multinomial`` (default) or ``gumbel``; pass the ``*_noise_fn``
    hooks to inject the torch run's exact noise (token-for-token, R5). Pass
    ``init_canvas_fn`` to replay exact per-block canvases in device-vs-reference
    acceptance tests.
    """
    canvas_len = diffusion_config.canvas_length
    _validate_generate_args(prompt_tokens, num_blocks=num_blocks, canvas_len=canvas_len, vocab_size=vocab_size)
    batch = prompt_tokens.shape[0]
    prompt_len = prompt_tokens.shape[1]

    prefix = prompt_tokens
    committed_blocks: List[torch.Tensor] = []
    trajectories: List[DenoiseTrajectory] = []

    for block in range(num_blocks):
        init_canvas = (
            init_canvas_fn(block, prefix)
            if init_canvas_fn is not None
            else S.random_canvas((batch, canvas_len), vocab_size, generator=generator)
        )
        _validate_canvas_shape(init_canvas, batch=batch, canvas_len=canvas_len)
        frozen_prefix = prefix  # read-only during this block's denoise

        traj = denoise_block(
            lambda canvas, step: model(frozen_prefix, canvas, step),
            init_canvas,
            diffusion_config,
            vocab_size,
            sampler=sampler,
            gumbel_noise_fn=gumbel_noise_fn(block) if gumbel_noise_fn else None,
            noise_tokens_fn=noise_tokens_fn(block) if noise_tokens_fn else None,
            generator=generator,
        )
        committed = traj.committed  # [B, canvas_len] clean argmax
        committed_blocks.append(committed)
        trajectories.append(traj)
        prefix = torch.cat([prefix, committed], dim=1)  # commit-append -> grow prefix

    generated = torch.cat(committed_blocks, dim=1) if committed_blocks else prompt_tokens.new_zeros(batch, 0)
    return Generation(generated=generated, prompt_len=prompt_len, trajectories=trajectories)
