"""Request/response schemas. `ServeTTSRequest` is wire-compatible with fish-speech's (fish_speech/utils/schema.py)."""
from __future__ import annotations

import base64
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field, conint, field_validator


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

    @field_validator("audio", mode="before")
    @classmethod
    def _decode_b64(cls, v):
        if isinstance(v, str):
            try:
                return base64.b64decode(v)
            except Exception:
                return v.encode()
        return v


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=1000, strict=True)] = 200
    format: Literal["wav", "pcm", "mp3", "opus", "flac"] = "wav"
    latency: Literal["normal", "balanced"] = "normal"
    references: List[ServeReferenceAudio] = []
    reference_id: Optional[str] = None
    seed: Optional[int] = None
    use_memory_cache: Literal["on", "off"] = "off"
    normalize: bool = True
    streaming: bool = False
    max_new_tokens: int = 1024
    top_p: Annotated[float, Field(ge=0.1, le=1.0)] = 0.8
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0)] = 1.1  # accepted for compatibility; inert upstream too
    temperature: Annotated[float, Field(ge=0.1, le=1.0)] = 0.8
    # extensions
    top_k: int = 30
    greedy: bool = False


class OpenAISpeechRequest(BaseModel):
    model: str = "fishaudio/s2-pro"
    input: str
    voice: str = "default"
    response_format: Literal["mp3", "wav", "pcm", "opus", "flac", "aac"] = "mp3"
    speed: float = 1.0
    stream_format: Optional[str] = None
    instructions: Optional[str] = None
    seed: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    mesh: str
    device_name: str
    codec_device: str
    busy: bool
    queue_depth: int
    uptime_s: float
    requests_served: int
    impl: dict


class AddReferenceResponse(BaseModel):
    success: bool
    message: str
    reference_id: str


class ListReferencesResponse(BaseModel):
    success: bool
    reference_ids: List[str]
    message: str = "Success"


class DeleteReferenceResponse(BaseModel):
    success: bool
    message: str
    reference_id: str
