# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class DeviceConfig(BaseModel):
    """The devices a run trains on."""

    model_config = ConfigDict(extra="forbid")

    # The size of each dimension of the device mesh, such as [rows, columns]. A single chip, [1, 1], is the only
    # mesh a run trains on so far.
    mesh: list[PositiveInt] = Field(default_factory=lambda: [1, 1], min_length=1)
