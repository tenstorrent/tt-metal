"""Request/response schemas. `SpeechRequest` is wire-compatible with the SGLang-Omni `/v1/audio/speech` contract the
MiniMax model card documents (input = lyrics, instructions = caption, max_new_tokens = audio frames at 25 fps)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from models.autoports.minimaxai_minimax_music3.config import DIT_NUM_STEPS, MAX_AUDIO_FRAMES


class SpeechRequest(BaseModel):
    model: str = "MiniMaxAI/MiniMax-Music3"
    input: str = Field(..., description="Lyrics. Structure tags such as [Verse] / [Chorus] on their own lines.")
    instructions: str = Field(..., description="Music description (genre, BPM, key, vocals, arrangement).")
    response_format: Literal["wav", "flac", "mp3", "pcm"] = "wav"
    seed: Optional[int] = None
    max_new_tokens: int = Field(750, ge=1, le=MAX_AUDIO_FRAMES, description="Maximum audio frames (25 per second).")
    audio_duration: Optional[float] = Field(
        None, gt=0, description="Alternative to max_new_tokens: seconds (upper bound)."
    )
    stream: bool = False
    sample_rate: Literal[32000, 44100] = 32000
    num_inference_steps: int = Field(DIT_NUM_STEPS, ge=1, le=100)
    # accepted for OpenAI-client compatibility, ignored (there is no voice / speed / temperature in this model)
    voice: Optional[str] = None
    speed: Optional[float] = None
    temperature: Optional[float] = None

    @field_validator("input", "instructions")
    @classmethod
    def _non_empty(cls, v: str):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("must be a non-empty string")
        return v

    @property
    def frames(self) -> int:
        if self.audio_duration is not None:
            return max(1, min(int(self.audio_duration * 25.0), MAX_AUDIO_FRAMES))
        return self.max_new_tokens


class JobSubmitted(BaseModel):
    id: str
    status: str
    queue_position: int


class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "error", "cancelled"]
    frames_done: int = 0
    frames_max: int = 0
    stage: str = ""
    elapsed_s: float = 0.0
    stats: Optional[dict] = None
    error: Optional[str] = None
    audio_url: Optional[str] = None
