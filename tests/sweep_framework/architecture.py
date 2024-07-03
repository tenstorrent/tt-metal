# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class Arch(Enum):
    INVALID = -1
    GRAYSKULL = 0
    WORMHOLE = 1
    WORMHOLE_B0 = 2
    BLACKHOLE = 3


def str_to_arch(arch: str):
    if arch == "grayskull":
        return Arch.GRAYSKULL
    elif arch == "wormhole":
        return Arch.WORMHOLE
    elif arch == "wormhole_b0":
        return Arch.WORMHOLE_B0
    elif arch == "blackhole":
        return Arch.BLACKHOLE
    else:
        return Arch.INVALID


def tt_smi_path(arch: Arch):
    if arch == Arch.GRAYSKULL:
        return "/home/software/syseng/gs/tt-smi"
    elif arch == Arch.WORMHOLE or arch == Arch.WORMHOLE_B0:
        return "/home/software/syseng/wh/tt-smi"
    else:
        return "/home/software/syseng/bh/tt-smi"
