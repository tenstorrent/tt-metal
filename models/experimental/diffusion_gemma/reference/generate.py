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


class Generation(NamedTuple):
    generated: torch.Tensor  # [B, num_blocks * canvas_length] committed tokens
    prompt_len: int
    trajectories: List[DenoiseTrajectory]  # one per block


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
) -> Generation:
    """Generate ``num_blocks`` canvases autoregressively, committing each.

    ``generator`` seeds BOTH the per-block random canvas init AND (threaded into
    ``denoise_block``) the per-step regenerated sample/renoise noise, so a single
    seeded generator reproduces the whole generation. ``sampler`` selects the
    HF-faithful ``multinomial`` (default) or ``gumbel``; pass the ``*_noise_fn``
    hooks to inject the torch run's exact noise (token-for-token, R5).
    """
    batch = prompt_tokens.shape[0]
    canvas_len = diffusion_config.canvas_length
    prompt_len = prompt_tokens.shape[1]

    prefix = prompt_tokens
    committed_blocks: List[torch.Tensor] = []
    trajectories: List[DenoiseTrajectory] = []

    for block in range(num_blocks):
        init_canvas = S.random_canvas((batch, canvas_len), vocab_size, generator=generator)
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
