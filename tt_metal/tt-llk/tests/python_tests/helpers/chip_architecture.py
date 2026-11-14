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
        """Map a tt_umd.ARCH from a PCIe device. Descriptor arch_name strings use from_string."""
        import tt_umd

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
    from .target_config import simulator_path

    sim_path = simulator_path()
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
        loaded = yaml.safe_load(f)
    arch_name = loaded.get("arch_name") if isinstance(loaded, dict) else None
    if not isinstance(arch_name, str):
        raise RuntimeError(
            f"{soc_descriptor} has no arch_name string (got {arch_name!r})"
        )
    try:
        return ChipArchitecture.from_string(arch_name)
    except ValueError as exc:
        raise RuntimeError(
            f"{soc_descriptor} has unknown arch_name {arch_name!r}"
        ) from exc


def _pcie_architecture():
    """Arch of UMD logical device 0, the card the harness runs on.

    ``enumerate_devices()`` is BDF order and honors ``TT_VISIBLE_DEVICES``, so index 0 is
    exalens device 0. That is not the lowest ``/dev/tenstorrent/N``. Reading the arch opens
    that device and issues GET_DEVICE_INFO; it does not initialize the chip or start tt-exalens.
    """
    import tt_umd

    device_ids = tt_umd.PCIDevice.enumerate_devices()
    if not device_ids:
        return None
    info = tt_umd.PCIDevice.read_device_info(f"/dev/tenstorrent/{device_ids[0]}")
    return ChipArchitecture.from_umd_arch(info.get_arch())


def _detect_chip_architecture():
    # ``--run-simulator`` on the command line or in PYTEST_ADDOPTS, including next to
    # ``--compile-producer``, so a producer on a card host compiles for the simulator.
    # The plugin's simulator gate is a different check: it drops ``--compile-producer``
    # (that step never talks to a device) and treats xdist workers as simulator runs.
    # Workers never get here; they inherit the CHIP_ARCH the controller writes back.
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
