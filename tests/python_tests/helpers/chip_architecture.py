# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
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


def set_chip_architecture():
    def _identify_chip_architecture(output):
        if "Blackhole" in output:
            return ChipArchitecture.BLACKHOLE
        elif "Wormhole" in output:
            return ChipArchitecture.WORMHOLE
        return None

    try:
        result = subprocess.run(
            ["tt-smi", "-ls"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "tt-smi command not found. Please ensure tt-smi is installed and in PATH."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"tt-smi failed with error: {e.stderr}")

    architecture = _identify_chip_architecture(result.stdout)
    if not architecture:
        raise RuntimeError(
            "Unable to detect architecture from tt-smi output. Please verify device is connected and tt-smi is working correctly."
        )
    os.environ["CHIP_ARCH"] = architecture.value
    return architecture


def get_chip_architecture():
    chip_architecture = os.getenv("CHIP_ARCH")
    if not chip_architecture:
        return set_chip_architecture()
    return ChipArchitecture.from_string(chip_architecture.strip())
