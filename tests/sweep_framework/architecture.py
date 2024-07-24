# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


def tt_smi_path(arch: str):
    if arch == "grayskull":
        return "/home/software/syseng/gs/tt-smi"
    elif arch == "wormhole" or arch == "wormhole_b0":
        return "/home/software/syseng/wh/tt-smi"
    else:
        return "/home/software/syseng/bh/tt-smi"
