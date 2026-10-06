# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 with a lightx2v Turbo distillation adapter fused into the transformer.

The adapter is a plain low-rank delta over the attention and feed-forward weights, so the generation
path is the base pipeline's: same packing, schedulers, VAE and audio decode. What changes is the
transformer's weights, the step count the caller asks for, and for the 768p files the video shift.
Those three travel together here rather than as switches on the base class.

A Turbo file publishes no sampling contract in its header. The caller supplies the step count, and
`num_inference_steps` counts sigma grid points, so a 4-forward adapter runs at 5 and an 8-forward one
at 9. The model card's shift for the file goes in `video_shift`; the 768p variants were distilled at
6 against the checkpoint's 12, and a wrong shift is a valid schedule over the wrong grid.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

import ttnn

from ...experimental.lora.h3_adapter_loader import H3AdapterHandle, load_h3_adapter_into
from ...experimental.lora.promote import lora_modules
from .pipeline_minimax_h3 import AUDIO_SHIFT, VIDEO_SHIFT, MiniMaxH3Pipeline
from .weights_minimax_h3 import LORA_PATH_ENV, resolve_adapter_settings

#: Forward counts the published adapters were distilled for; `num_inference_steps` is one more.
TURBO_NUM_FORWARDS = (4, 8)


class MiniMaxH3TurboPipeline(MiniMaxH3Pipeline):
    """`MiniMaxH3Pipeline` whose transformer carries a lightx2v Turbo adapter.

    `lora_strength` multiplies the adapter's own `alpha / rank`, which the loader reads from the file;
    1.0 runs the adapter as trained.
    """

    def __init__(
        self,
        *,
        lora_path: str | os.PathLike,
        lora_strength: float = 1.0,
        **kwargs,
    ) -> None:
        # Bound onto the built transformer after the base weights load, never fused into the
        # checkpoint, so the weight cache stays adapter-independent and one cached copy serves every
        # adapter and strength.
        self.lora_path: Path = resolve_adapter_settings(
            lora_path=lora_path,
            lora_strength=lora_strength,
            default_video_shift=VIDEO_SHIFT,
            default_audio_shift=AUDIO_SHIFT,
        ).lora_path
        self.lora_strength = float(lora_strength)
        self._adapter: H3AdapterHandle | None = None
        super().__init__(**kwargs)

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        weights_dir: str | os.PathLike | None = None,
        lora_path: str | os.PathLike | None = None,
        lora_strength: float | None = None,
        video_shift: float | None = None,
        audio_shift: float | None = None,
        **kwargs,
    ) -> MiniMaxH3TurboPipeline:
        """`lora_path`, `lora_strength`, `video_shift` and `audio_shift` fall back to `MINIMAX_H3_LORA_PATH`,
        `MINIMAX_H3_LORA_STRENGTH`, `MINIMAX_H3_VIDEO_SHIFT` and `MINIMAX_H3_AUDIO_SHIFT`.

        Everything else is `MiniMaxH3Pipeline.create_pipeline`'s.
        """
        settings = resolve_adapter_settings(
            lora_path=lora_path,
            lora_strength=lora_strength,
            video_shift=video_shift,
            audio_shift=audio_shift,
            default_video_shift=VIDEO_SHIFT,
            default_audio_shift=AUDIO_SHIFT,
        )
        if settings.lora_path is None:
            raise ValueError(f"the Turbo pipeline needs an adapter: pass lora_path= or set {LORA_PATH_ENV}")
        return super().create_pipeline(
            mesh_device=mesh_device,
            weights_dir=weights_dir,
            video_shift=settings.video_shift,
            audio_shift=settings.audio_shift,
            lora_path=settings.lora_path,
            lora_strength=settings.lora_strength,
            **kwargs,
        )

    @property
    def adapter(self) -> H3AdapterHandle | None:
        """What is bound on the transformer, or None before its first residency."""
        return self._adapter

    def _prepare_transformer(self):
        transformer = super()._prepare_transformer()
        if self._adapter is None:
            self._adapter = load_h3_adapter_into(
                transformer,
                str(self.lora_path),
                scale=self.lora_strength,
                name=self.lora_path.name,
            )
            logger.info(
                f"turbo adapter {self._adapter.name}: {len(self._adapter)} targets, strength {self.lora_strength:g}"
            )
        else:
            # `coresident=False` evicts the transformer between stages and `cache.load_model` brings
            # back the cached *base* weights, so the fused delta has to be merged again. A no-op while
            # the weights stayed resident.
            for module in lora_modules(transformer):
                module.reapply_after_load()
        return transformer


__all__ = ["TURBO_NUM_FORWARDS", "MiniMaxH3TurboPipeline"]
