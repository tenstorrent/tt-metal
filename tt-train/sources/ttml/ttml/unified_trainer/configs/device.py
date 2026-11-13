# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict, PositiveInt

# A single chip.
SINGLE_CHIP_MESH = (1, 1)


class DeviceConfig(BaseModel):
    """The devices a run trains on."""

    model_config = ConfigDict(extra="forbid")

    # Rows and columns of the device mesh. A single chip, [1, 1], is the only mesh a run trains on so far.
    mesh: tuple[PositiveInt, PositiveInt] = SINGLE_CHIP_MESH
