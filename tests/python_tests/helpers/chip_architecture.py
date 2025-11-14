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
    def _get_string_to_enum_map(cls):
        if not hasattr(cls, "_cached_string_map"):
            cls._cached_string_map = {
                "blackhole": cls.BLACKHOLE,
                "quasar": cls.QUASAR,
                "wormhole": cls.WORMHOLE,
            }
        return cls._cached_string_map

    @classmethod
    def from_string(cls, arch_str):
        arch_lower = arch_str.lower()
        enum_value = cls._get_string_to_enum_map().get(arch_lower)
        if enum_value is None:
            raise ValueError(f"Unknown architecture: {arch_str}")
        return enum_value


# Cache for chip architecture
_cached_chip_architecture = None


def get_chip_architecture():
    global _cached_chip_architecture

    if _cached_chip_architecture is not None:
        return _cached_chip_architecture

    chip_architecture = os.getenv("CHIP_ARCH")
    if not chip_architecture:
        context = check_context()
        chip_architecture = context.devices[0]._arch
        if chip_architecture == "wormhole_b0":
            chip_architecture = "wormhole"
        os.environ["CHIP_ARCH"] = chip_architecture

    _cached_chip_architecture = ChipArchitecture.from_string(chip_architecture)
    return _cached_chip_architecture
