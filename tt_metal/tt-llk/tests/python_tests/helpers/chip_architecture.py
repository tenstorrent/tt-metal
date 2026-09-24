# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum


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
                "wormhole_b0": cls.WORMHOLE,
            }
        return cls._cached_string_map

    @classmethod
    def from_string(cls, arch_str):
        arch_lower = arch_str.lower()
        enum_value = cls._get_string_to_enum_map().get(arch_lower)
        if enum_value is None:
            raise ValueError(f"Unknown architecture: {arch_str}")
        return enum_value


def _env_flag_enabled(name):
    return os.getenv(name, "").strip().lower() in ("1", "true")


def is_4row_arch():
    """Which Quasar part the build targets. Only build configuration should ask this.

    Tests select coverage through the capability helpers below instead, mirroring
    tt_llk_quasar/common/inc/ckernel_arch_config.h: a capability is shared by every part that has
    it, so a new part changes one definition here rather than every test that tested the part.
    """
    return _env_flag_enabled("TT_METAL_QUASAR_FOUR_ROW")


def quasar_has_mx_formats():
    """The selected Quasar part supports MX block formats, including the MxFp4_2x register formats."""
    return not is_4row_arch()


def quasar_has_int8_2x():
    """The selected Quasar part supports the 2x-packed Int8_2x / UInt8_2x source-register formats."""
    return is_4row_arch()


def quasar_mx_formats(*formats):
    """Return ``formats`` as a list when the selected part supports MX formats, else an empty list."""
    return list(formats) if quasar_has_mx_formats() else []


# Cache for chip architecture
_cached_chip_architecture = None


def get_chip_architecture():
    global _cached_chip_architecture

    if _cached_chip_architecture is not None:
        return _cached_chip_architecture

    chip_architecture = os.getenv("CHIP_ARCH")
    if not chip_architecture:
        from ttexalens.tt_exalens_lib import check_context

        context = check_context()
        chip_architecture = str(context.devices[0]._arch)

    _cached_chip_architecture = ChipArchitecture.from_string(chip_architecture)
    # Always write the LLK name back. Several CLIs take --arch $CHIP_ARCH and
    # only accept wormhole|blackhole|quasar; leaving wormhole_b0 in the
    # environment is what produces "invalid choice: 'wormhole_b0'".
    os.environ["CHIP_ARCH"] = _cached_chip_architecture.value
    return _cached_chip_architecture
