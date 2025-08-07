# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum

from ttexalens.tt_exalens_lib import check_context


class ChipArchitecture(Enum):
    BLACKHOLE = "blackhole"
    WORMHOLE = "wormhole"

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, arch_str):
        if arch_str.lower() == "blackhole":
            return cls.BLACKHOLE
        elif arch_str.lower() == "wormhole_b0":
            return cls.WORMHOLE
        else:
            raise ValueError(f"Unknown architecture: {arch_str}")


def get_chip_architecture():
    context = check_context()
    if not context.devices:
        raise RuntimeError(
            "No devices found. Please ensure a device is connected and tt-smi is working correctly."
        )
    architecture = ChipArchitecture.from_string(context.devices[0]._arch)
    os.environ["CHIP_ARCH"] = architecture.value
    return architecture
