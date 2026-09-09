# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The chunked flow-matching loop of MiniMax-Music3 (diffusers ``modular_pipelines/minimax_music3/denoise.py``).

Per 200-frame window ``k`` (``MiniMaxMusic3ChunkDenoiseStep``), with ``previous_latent`` /
``previous_condition`` carried from window ``k - 1``:

1. ``ChunkConditionStep``: ``condition = condition_encoder(frame_hiddens[:, start:end])``; with a carry,
   ``overlap = min(previous_latent.shape[-1], L)`` and ``condition[:, :overlap] = previous_condition[:, :overlap]``.
2. ``ChunkPrepareLatentsStep``: ``latents = noise`` (``[1, 128, L]``; the caller draws it, so the golden noise
   can be injected), ``noise_prompt = noise[..., :overlap]``.
3. ``ChunkSetTimestepsStep``: ``sigmas = linspace(1, 1/N, N)`` -> ``tt/scheduler.py``.
4. ``ChunkDenoiseInner``: for every ``t``: blend the overlap
   ``latents[..., :overlap] = (1 - (1 - 1e-6) t) noise_prompt + t previous_latent[..., :overlap]``, run the DiT
   once for both CFG rows (row 0 conditional, row 1 zero condition), ``v = v_u + 1.7 (v_c - v_u)``, Euler step.
5. ``ChunkUpdateStep``: restore ``latents[..., :overlap] = previous_latent[..., :overlap]``; carry
   ``latents[..., L - 344 : L - 172]`` and the same window of ``condition`` to the next chunk (clamped for short
   windows exactly like the reference).

Everything but the DiT and the condition conv runs on the host in fp32, as in the reference (the latents,
the CFG combine and the Euler update are ``[1, 128, L]`` axpys). The DiT input is bf16 on device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch

from models.autoports.minimaxai_minimax_music3.tt.condition_encoder import ConditionEncoder
from models.autoports.minimaxai_minimax_music3.tt.flow_transformer import BATCH, FlowTransformer
from models.autoports.minimaxai_minimax_music3.tt.scheduler import FlowMatchEulerScheduler

CHUNK_FRAMES = 200
CHUNK_HOP = 100
OVERLAP_LATENT_LENGTH = 172
DIT_CFG_SCALE = 1.7
DEFAULT_STEPS = 30
BLEND_EPS = 1e-6


@dataclass
class ChunkResult:
    latents: torch.Tensor  # [1, 128, L] denoised, uncropped (overlap restored)
    condition: torch.Tensor  # [1, L, 2048] as fed to the DiT (spliced)
    previous_latent: torch.Tensor  # carry for the next chunk
    previous_condition: torch.Tensor
    overlap: int
    step_log: List[dict] = field(default_factory=list)


def chunk_starts_for(num_frames: int) -> List[int]:
    """Window starts exactly as ``before_denoise.py`` computes them (``[0]`` up to 200 frames, else
    ``range(0, frames - 100, 100)``: the last window may be shorter than 200 frames)."""
    if num_frames <= CHUNK_FRAMES:
        return [0]
    return list(range(0, num_frames - CHUNK_HOP, CHUNK_HOP))


class ChunkDenoiser:
    def __init__(
        self,
        transformer: FlowTransformer,
        condition_encoder: ConditionEncoder,
        *,
        num_inference_steps: int = DEFAULT_STEPS,
        cfg_scale: float = DIT_CFG_SCALE,
    ):
        self.transformer = transformer
        self.condition_encoder = condition_encoder
        self.num_inference_steps = num_inference_steps
        self.cfg_scale = cfg_scale

    # ------------------------------------------------------------------ one window
    def denoise_chunk(
        self,
        frame_hiddens_chunk: torch.Tensor,
        previous_latent: Optional[torch.Tensor],
        previous_condition: Optional[torch.Tensor],
        noise: torch.Tensor,
        steps: Optional[int] = None,
        *,
        condition: Optional[torch.Tensor] = None,
        on_step: Optional[Callable[[int, float, torch.Tensor], None]] = None,
    ) -> ChunkResult:
        """One window. ``frame_hiddens_chunk [1, F, 32768]`` (F <= 200), ``noise [1, 128, L]`` with
        ``L = latent_length(F)``; ``condition`` (raw encoder output ``[1, L, 2048]``) may be passed to skip the encoder.
        ``on_step(i, t, latents)`` sees the latents after every Euler update (for drift logging)."""
        steps = steps or self.num_inference_steps
        if condition is None:
            condition = self.condition_encoder(frame_hiddens_chunk)
        condition = condition.float().clone()
        num_latents = condition.shape[1]
        assert tuple(noise.shape) == (1, 128, num_latents), (tuple(noise.shape), num_latents)

        overlap = 0
        if previous_latent is not None:
            overlap = min(previous_latent.shape[-1], num_latents)
            condition[:, :overlap] = previous_condition[:, :overlap]

        latents = noise.float().clone()
        noise_prompt = latents[..., :overlap].clone() if overlap > 0 else None

        scheduler = FlowMatchEulerScheduler(steps)
        cond_both = torch.cat([condition, torch.zeros_like(condition)], dim=0)  # [2, L, 2048]
        cond_proj = self.transformer.prepare_condition(cond_both)
        log: List[dict] = []
        try:
            for i, t in enumerate(scheduler.timesteps):
                if overlap > 0:
                    tv = float(t)
                    latents[..., :overlap] = (1.0 - (1.0 - BLEND_EPS) * tv) * noise_prompt + tv * previous_latent[
                        ..., :overlap
                    ]
                timestep = t.reshape(1).expand(BATCH)
                velocity = self.transformer(latents.expand(BATCH, -1, -1).contiguous(), timestep, cond_proj=cond_proj)
                v_cond, v_uncond = velocity[0:1], velocity[1:2]
                velocity = v_uncond + self.cfg_scale * (v_cond - v_uncond)
                latents = scheduler.step(velocity, t, latents)
                log.append({"step": i, "t": float(t), "latent_rms": float(latents.pow(2).mean().sqrt())})
                if on_step is not None:
                    on_step(i, float(t), latents)
        finally:
            import ttnn

            ttnn.deallocate(cond_proj)

        if overlap > 0:
            latents[..., :overlap] = previous_latent[..., :overlap]
        overlap_start = max(0, num_latents - 2 * OVERLAP_LATENT_LENGTH)
        overlap_end = max(overlap_start, num_latents - OVERLAP_LATENT_LENGTH)
        return ChunkResult(
            latents=latents,
            condition=condition,
            previous_latent=latents[..., overlap_start:overlap_end].clone(),
            previous_condition=condition[:, overlap_start:overlap_end].clone(),
            overlap=overlap,
            step_log=log,
        )

    # ------------------------------------------------------------------ all windows
    def denoise(
        self,
        frame_hiddens: torch.Tensor,
        noises: Sequence[torch.Tensor],
        *,
        chunk_starts: Optional[Sequence[int]] = None,
        steps: Optional[int] = None,
    ) -> List[ChunkResult]:
        """``frame_hiddens [1, frames, 32768]`` and one ``[1, 128, L_k]`` noise per window -> per-window results."""
        num_frames = frame_hiddens.shape[1]
        starts = list(chunk_starts) if chunk_starts is not None else chunk_starts_for(num_frames)
        assert len(noises) == len(starts), (len(noises), len(starts))
        results: List[ChunkResult] = []
        prev_lat = prev_cond = None
        for k, start in enumerate(starts):
            end = min(start + CHUNK_FRAMES, num_frames)
            r = self.denoise_chunk(frame_hiddens[:, start:end], prev_lat, prev_cond, noises[k], steps)
            prev_lat, prev_cond = r.previous_latent, r.previous_condition
            results.append(r)
        return results
