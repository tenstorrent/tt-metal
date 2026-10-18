#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    broken_dependency_check

Description:
    Test fixture: declares a dependency on a script that does not exist anywhere on the search path.
    Loading it must report which script wanted what, and where it looked.

Owner:
    tt-vjovanovic
"""

from triage import ScriptConfig, run_script
from ttexalens.context import Context

script_config = ScriptConfig(depends=["no_such_provider"])


def run(args, context: Context) -> None:
    raise AssertionError("broken_dependency_check must never run: its dependency cannot be resolved")


if __name__ == "__main__":
    run_script()
