# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import sys
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

    @classmethod
    def from_umd_arch(cls, umd_arch):
        """Map a tt_umd.ARCH (or its member name, e.g. a soc descriptor's arch_name)."""
        import tt_umd

        if isinstance(umd_arch, str):
            umd_arch = tt_umd.ARCH.__members__.get(
                umd_arch.upper(), tt_umd.ARCH.Invalid
            )
        enum_value = {
            tt_umd.ARCH.WORMHOLE_B0: cls.WORMHOLE,
            tt_umd.ARCH.BLACKHOLE: cls.BLACKHOLE,
            tt_umd.ARCH.QUASAR: cls.QUASAR,
        }.get(umd_arch)
        if enum_value is None:
            raise ValueError(f"Unsupported UMD architecture: {umd_arch}")
        return enum_value


QUASAR_ARCH_ROOT = Path(__file__).resolve().parents[3] / "tt_llk_quasar" / "arch"


# Other spellings tt-metal accepts for TT_METAL_QUASAR_VARIANT. Empty means the base Quasar part.
QUASAR_VARIANT_ALIASES = {
    "quasar": "",
    "2.0.0": "",
    "2.0.1": "horizon",
    "2.0.2": "trinity",
}


def quasar_variant():
    raw = os.getenv("TT_METAL_QUASAR_VARIANT", "").strip().lower()
    variant = QUASAR_VARIANT_ALIASES.get(raw, raw)
    if variant and not (QUASAR_ARCH_ROOT / variant).is_dir():
        known = sorted(p.name for p in QUASAR_ARCH_ROOT.iterdir() if p.is_dir())
        raise ValueError(
            f"TT_METAL_QUASAR_VARIANT={raw!r} has no directory under {QUASAR_ARCH_ROOT}; known variants: {known}"
        )
    return variant


def is_4row_arch():
    return quasar_variant() == "trinity"


# Cache for chip architecture
_cached_chip_architecture = None


def cli_flag_given(flag):
    """Whether a pytest flag is on the command line or in PYTEST_ADDOPTS.

    For code that runs at import, before pytest has parsed its options. Flags that only come from
    the ini file's addopts are not seen here; pytest_configure checks the parsed options.
    """
    return flag in sys.argv or flag in shlex.split(os.environ.get("PYTEST_ADDOPTS", ""))


def simulator_architecture():
    """Read arch_name from the soc_descriptor.yaml UMD loads for the simulator, if one is configured.

    The RTL simulator path is a directory holding the descriptor; for ttsim it sits next to the .so.
    """
    sim_path = os.environ.get("TT_METAL_SIMULATOR") or os.environ.get(
        "TT_UMD_SIMULATOR_PATH"
    )
    if not sim_path:
        return None
    sim_path = Path(sim_path).expanduser()
    soc_descriptor = (
        sim_path if sim_path.is_dir() else sim_path.parent
    ) / "soc_descriptor.yaml"
    if not soc_descriptor.is_file():
        return None

    import yaml

    with open(soc_descriptor) as f:
        return ChipArchitecture.from_umd_arch(yaml.safe_load(f)["arch_name"])


def _pcie_architecture():
    """Arch of the first local PCIe device, read from PCI config space without opening the device."""
    import tt_umd

    devices = tt_umd.PCIDevice.enumerate_devices_info()
    if not devices:
        return None
    return ChipArchitecture.from_umd_arch(devices[min(devices)].get_arch())


def _detect_chip_architecture():
    # Same gate the pytest plugin uses to pick the simulator over silicon. xdist workers never get
    # here: they inherit the CHIP_ARCH the controller writes back.
    if cli_flag_given("--run-simulator") and (arch := simulator_architecture()):
        return arch
    # Silicon first, so a stray simulator env var does not override the card in the machine.
    arch = _pcie_architecture() or simulator_architecture()
    if arch is None:
        raise RuntimeError(
            "Could not detect the chip architecture: no Tenstorrent PCIe device and no simulator "
            "soc_descriptor.yaml. Set CHIP_ARCH (wormhole / blackhole / quasar)."
        )
    return arch


def get_chip_architecture():
    global _cached_chip_architecture

    if _cached_chip_architecture is not None:
        return _cached_chip_architecture

    chip_architecture = os.getenv("CHIP_ARCH")
    if chip_architecture:
        _cached_chip_architecture = ChipArchitecture.from_string(chip_architecture)
    else:
        _cached_chip_architecture = _detect_chip_architecture()
    # Always write the LLK name back. Several CLIs take --arch $CHIP_ARCH and
    # only accept wormhole|blackhole|quasar; leaving wormhole_b0 in the
    # environment is what produces "invalid choice: 'wormhole_b0'".
    os.environ["CHIP_ARCH"] = _cached_chip_architecture.value
    return _cached_chip_architecture
