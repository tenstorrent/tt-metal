#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    test_output

Description:
    This script does nothing. It is used to test the triage output testing framework.
"""

from triage import ScriptConfig, run_script
from ttexalens.context import Context

script_config = ScriptConfig(
    disabled=True,
)


def run(args, context: Context):
    pass


if __name__ == "__main__":
    run_script()
