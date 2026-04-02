#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    system_info

Description:
    Reports host system information: OS, kernel, machine type, and versions of
    KMD, UMD and tt-exalens packages.

Owner:
    macimovic
"""

import platform
from dataclasses import dataclass

from triage import ScriptConfig, triage_field, run_script
from ttexalens.context import Context
from triage_hw_utils import get_kmd_version, get_pkg_version

script_config = ScriptConfig()


@dataclass
class SystemInfoRow:
    os: str = triage_field("OS")
    os_version: str = triage_field("Version")
    kernel: str = triage_field("Kernel")
    machine: str = triage_field("Machine")
    kmd: str = triage_field("KMD")
    umd: str = triage_field("UMD")
    tt_exalens: str = triage_field("tt-exalens")


def get_os_version():
    os_version = None

    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("VERSION_ID="):
                    os_version = line.strip().split("=", 1)[1].strip('"')
    except FileNotFoundError:
        pass

    return os_version


def run(args, context: Context):
    uname = platform.uname()

    return [
        SystemInfoRow(
            os=uname.system,
            os_version=get_os_version() or "unknown",
            kernel=uname.release,
            machine=uname.machine,
            kmd=get_kmd_version(),
            umd=get_pkg_version("tt-umd"),
            tt_exalens=get_pkg_version("tt-exalens"),
        )
    ]


if __name__ == "__main__":
    run_script()
