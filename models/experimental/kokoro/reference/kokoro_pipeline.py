# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro-82M reference pipeline wrapper.

This keeps the demo and future TT bring-up code stable, while still using the
upstream `kokoro.pipeline.KPipeline` implementation for:
- G2P / chunking
- voicepack download + selection
- calling into the underlying `KModel`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Union

import torch

from .kokoro_config import KokoroConfig
from .kokoro_model import KokoroModelReference


@dataclass(frozen=True)
class KokoroChunk:
    graphemes: str
    phonemes: str
    audio: Optional[torch.FloatTensor]


class KokoroPipelineReference:
    def __init__(
        self,
        lang_code: str = "a",
        repo_id: str = KokoroConfig.repo_id,  # type: ignore[attr-defined]
        model: Union[KokoroModelReference, bool] = True,
        device: Optional[str] = None,
    ):
        from kokoro import KPipeline  # upstream

        if isinstance(model, KokoroModelReference):
            # Reuse existing KModel to avoid duplicate weight loads.
            upstream_model = model.kmodel
        else:
            upstream_model = model

        self.repo_id = repo_id
        self.lang_code = lang_code
        self.pipeline = KPipeline(lang_code=lang_code, repo_id=repo_id, model=upstream_model, device=device)

    @property
    def model_repo_id(self) -> str:
        # upstream stores repo_id too
        return getattr(self.pipeline, "repo_id", self.repo_id)

    def generate(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
    ) -> Generator[KokoroChunk, None, None]:
        for gs, ps, audio in self.pipeline(text, voice=voice, speed=speed):
            # audio is torch.FloatTensor in upstream Result.audio
            if audio is not None and not isinstance(audio, torch.Tensor):
                audio = torch.as_tensor(audio)
            yield KokoroChunk(graphemes=gs, phonemes=ps, audio=audio)


def load_reference_pipeline(
    lang_code: str = "a",
    repo_id: str = KokoroConfig.repo_id,  # type: ignore[attr-defined]
    device: Optional[str] = None,
) -> KokoroPipelineReference:
    return KokoroPipelineReference(lang_code=lang_code, repo_id=repo_id, model=True, device=device)
