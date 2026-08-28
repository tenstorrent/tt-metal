# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pydantic v2 request/response schemas for the FLUX.2 image server.

Validators reject shapes the pipeline would otherwise reject with a bare ``assert``
(which surfaces as an opaque 500), so a bad request yields a clean 422 instead.

This module imports no ttnn / device code, so it loads on any host.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    """Body for ``POST /generate``.

    ``height``/``width`` default to ``None`` and are filled from the server's warmed
    shape. Supplying a different resolution is rejected rather than silently
    recompiling: the pipeline builds its RoPE tables and captures its trace for one
    shape at construction time.
    """

    prompt: str = Field(..., min_length=1, description="Text prompt to generate from.")
    num_inference_steps: int | None = Field(default=None, ge=1, le=200)
    guidance_scale: float | None = Field(default=None, ge=0.0)
    seed: int | None = None
    height: int | None = Field(default=None, ge=16)
    width: int | None = Field(default=None, ge=16)

    @field_validator("height", "width")
    @classmethod
    def _validate_multiple_of_16(cls, v: int | None) -> int | None:
        # vae_scale_factor (8) * patch_size (2): the latent grid must divide evenly.
        if v is not None and v % 16 != 0:
            raise ValueError(f"must be divisible by 16 (got {v})")
        return v


class JobResponse(BaseModel):
    """Status payload for ``POST /generate`` (202) and ``GET /jobs/{job_id}``."""

    job_id: str
    status: str
    created_at: float
    completed_at: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Payload for ``GET /health``."""

    status: str  # "initializing" | "ready"
    ready: bool
    model: str
    mesh_shape: list[int]
    height: int
    width: int
