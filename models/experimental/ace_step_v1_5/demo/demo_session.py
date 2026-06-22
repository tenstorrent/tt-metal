# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""In-process state for ``run_prompt_to_wav.py`` (TTNN devices, handlers, preprocess cache)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CachedPreprocess:
    """Host-side condition tensors reused when prompt/duration/seed/lyrics match."""

    prompt: str
    duration_sec: float
    seed: int
    lyrics: str
    instrumental: bool
    frames: int
    enc_hs: Any
    enc_mask: Any
    ctx_lat: Any
    null_emb: Any

    def matches(
        self,
        *,
        prompt: str,
        duration_sec: float,
        seed: int,
        lyrics: str,
        instrumental: bool,
    ) -> bool:
        return (
            self.prompt == str(prompt)
            and float(self.duration_sec) == float(duration_sec)
            and int(self.seed) == int(seed)
            and self.lyrics == str(lyrics)
            and bool(self.instrumental) == bool(instrumental)
        )


@dataclass
class AceStepDemoSession:
    """TTNN objects and handlers for one ``run_prompt_to_wav.py`` invocation."""

    preprocess_dev: Any = None
    dit_dev: Any = None
    dit_handler: Any = None
    llm_handler: Any = None
    qwen_tt_encoder: Any = None
    audio_code_detokenizer: Any = None
    condition_encoder: Any = None
    pipe: Any = None
    tt_vae: Any = None
    trace_state: Any = None
    dit_frames: int | None = None
    dit_pipe_key: tuple[int, bool] | None = None
    torch_dit_pipe: Any = None
    torch_dit_pipe_key: tuple[str, tuple[float, ...]] | None = None
    vae_init_key: tuple[int, bool] | None = None
    cached_preprocess: CachedPreprocess | None = None
    session_perf: Any = None

    def __post_init__(self) -> None:
        if self.session_perf is None:
            from models.experimental.ace_step_v1_5.utils.ace_step_perf_log import SessionPerfState

            self.session_perf = SessionPerfState()

    def can_reuse_preprocess(
        self,
        *,
        prompt: str,
        duration_sec: float,
        seed: int,
        lyrics: str,
        instrumental: bool,
    ) -> bool:
        cached = self.cached_preprocess
        return cached is not None and cached.matches(
            prompt=prompt,
            duration_sec=duration_sec,
            seed=seed,
            lyrics=lyrics,
            instrumental=instrumental,
        )

    def store_preprocess(
        self,
        *,
        prompt: str,
        duration_sec: float,
        seed: int,
        lyrics: str,
        instrumental: bool,
        frames: int,
        enc_hs: Any,
        enc_mask: Any,
        ctx_lat: Any,
        null_emb: Any,
    ) -> None:
        self.cached_preprocess = CachedPreprocess(
            prompt=str(prompt),
            duration_sec=float(duration_sec),
            seed=int(seed),
            lyrics=str(lyrics),
            instrumental=bool(instrumental),
            frames=int(frames),
            enc_hs=enc_hs,
            enc_mask=enc_mask,
            ctx_lat=ctx_lat,
            null_emb=null_emb,
        )

    def clear_preprocess_device(self, ttnn_mod: Any) -> None:
        from models.experimental.ace_step_v1_5.utils.tt_device import close_ace_step_device

        if self.preprocess_dev is not None:
            try:
                close_ace_step_device(ttnn_mod, self.preprocess_dev)
            except Exception:
                pass
        self.preprocess_dev = None
        self.qwen_tt_encoder = None
        self.audio_code_detokenizer = None
        self.condition_encoder = None

    def release(self, ttnn_mod: Any) -> None:
        from models.experimental.ace_step_v1_5.utils.tt_device import close_ace_step_device

        if self.trace_state is not None:
            dev = self.dit_dev or self.preprocess_dev
            if dev is not None:
                try:
                    self.trace_state.release(dev)
                except Exception:
                    pass
            self.trace_state = None

        for dev in (self.dit_dev, self.preprocess_dev):
            if dev is not None:
                try:
                    close_ace_step_device(ttnn_mod, dev)
                except Exception:
                    pass
        self.dit_dev = None
        self.preprocess_dev = None
        self.pipe = None
        self.tt_vae = None
        self.dit_frames = None
        self.dit_pipe_key = None
        self.torch_dit_pipe = None
        self.torch_dit_pipe_key = None
        self.vae_init_key = None
        self.qwen_tt_encoder = None
        self.audio_code_detokenizer = None
        self.condition_encoder = None


__all__ = [
    "AceStepDemoSession",
    "CachedPreprocess",
]
