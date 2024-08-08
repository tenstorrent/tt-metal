# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


def tt_smi_command(arch: str):
    if arch == "grayskull":
        return ["/home/software/syseng/gs/tt-smi", "-tr", "0"]
    elif arch == "wormhole_b0":
        return ["/home/software/syseng/wh/tt-smi", "-wr", "all", "wait"]
    else:
        raise Exception("Blackhole TT-SMI Reset Not Supported")
