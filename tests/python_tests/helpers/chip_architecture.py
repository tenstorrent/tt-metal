# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum

from ttexalens.tt_exalens_lib import check_context


class ChipArchitecture(Enum):
    BLACKHOLE = "blackhole"
    WORMHOLE = "wormhole"
    QUASAR = "quasar"

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, arch_str):
        if arch_str.lower() == "blackhole":
            return cls.BLACKHOLE
        elif arch_str.lower() == "wormhole":
            return cls.WORMHOLE
        elif arch_str.lower() == "quasar":
            return cls.QUASAR
        else:
            raise ValueError(f"Unknown architecture: {arch_str}")


def get_chip_architecture():
    chip_architecture = os.getenv("CHIP_ARCH")
    if not chip_architecture:
        context = check_context()
        chip_architecture = context.devices[0]._arch
        if chip_architecture == "wormhole_b0":
            chip_architecture = "wormhole"
        os.environ["CHIP_ARCH"] = chip_architecture

    return ChipArchitecture.from_string(chip_architecture)
