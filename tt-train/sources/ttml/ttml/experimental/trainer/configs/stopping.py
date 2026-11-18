# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StoppingCriteria(BaseModel):
    """When a run ends."""

    model_config = ConfigDict(extra="forbid")

    # What `limit` counts. "epoch": passes over the train data. "step": optimizer steps, over as many passes as
    # they take.
    type: Literal["epoch", "step"]
    # The run ends once it has made this many.
    limit: int = Field(gt=0)
