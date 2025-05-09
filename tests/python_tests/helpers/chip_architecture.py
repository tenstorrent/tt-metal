# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum


class ChipArchitecture(Enum):
    BLACKHOLE = "blackhole"
    WORMHOLE = "wormhole"

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, arch_str):
        if arch_str.lower() == "blackhole":
            return cls.BLACKHOLE
        elif arch_str.lower() == "wormhole":
            return cls.WORMHOLE
        else:
            raise ValueError(f"Unknown architecture: {arch_str}")


def get_chip_architecture():
    chip_architecture = os.getenv("CHIP_ARCH")
    if not chip_architecture:
        return None
    return ChipArchitecture.from_string(chip_architecture.strip())
