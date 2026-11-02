# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field


class SystemOneRequest(BaseModel):
    """Request of POST /v1/systemone; the questions are checked by prompting.validate_questions."""

    state: str | dict | list | None = ""
    questions: dict[str, dict] = Field(min_length=1)
    model: str = "strands-decider-latest"  # ignored; the server answers with its own model name
    images: list[str] = []
