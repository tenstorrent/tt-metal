# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import importlib

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture

_arch = get_chip_architecture()

_ARCH_NAME = {
    ChipArchitecture.QUASAR: "quasar",
    ChipArchitecture.BLACKHOLE: "blackhole",
    ChipArchitecture.WORMHOLE: "wormhole",
}[_arch]

unpack_common = importlib.import_module(f"fuser.{_ARCH_NAME}.unpacker.common")
fpu_common = importlib.import_module(f"fuser.{_ARCH_NAME}.fpu.common")
pack_common = importlib.import_module(f"fuser.{_ARCH_NAME}.packer.common")


def _get_parser():
    return importlib.import_module(f"fuser.{_ARCH_NAME}.parser")
