# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from pathlib import Path


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


QUASAR_ARCH_ROOT = Path(__file__).resolve().parents[3] / "tt_llk_quasar" / "arch"


def quasar_arch_variant():
    variant = os.getenv("QUASAR_ARCH_VARIANT", "").strip()
    if variant and not (QUASAR_ARCH_ROOT / variant).is_dir():
        known = sorted(p.name for p in QUASAR_ARCH_ROOT.iterdir() if p.is_dir())
        raise ValueError(
            f"QUASAR_ARCH_VARIANT={variant!r} is not a directory under {QUASAR_ARCH_ROOT}; known variants: {known}"
        )
    return variant


def is_4row_arch():
    return quasar_arch_variant() == "quasar_4row"


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
