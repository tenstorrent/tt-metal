#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    additional_extra_provider

Description:
    Test fixture for --additional-scripts-directory. A data provider in a second additional scripts
    directory, so check_additional_directory can depend across two additional directories.

Owner:
    tt-vjovanovic
"""

from triage import ScriptConfig, run_script
from ttexalens.context import Context

script_config = ScriptConfig(data_provider=True)


def run(args, context: Context) -> str:
    return "additional_extra_provider"


if __name__ == "__main__":
    run_script()
