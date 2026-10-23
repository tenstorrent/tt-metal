#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_additional_directory

Description:
    Test fixture for --additional-scripts-directory. Lives in an additional scripts directory and
    depends on a script in each place a bare dependency name can resolve: system_info (built-in,
    in tools/triage), additional_provider (its own directory) and additional_extra_provider (a
    second additional directory). Prints a message the test asserts on.

Owner:
    tt-vjovanovic
"""

from dataclasses import dataclass

from additional_provider import run as get_provider
from triage import ScriptConfig, run_script, triage_field
from ttexalens.context import Context

SUCCESS_MESSAGE = "check_additional_directory ran from an additional scripts directory"

script_config = ScriptConfig(
    depends=["system_info", "additional_provider", "additional_extra_provider"],
)


@dataclass
class AdditionalDirectoryRow:
    message: str = triage_field("Message")
    dependencies: str = triage_field("Dependencies")


def run(args, context: Context) -> list[AdditionalDirectoryRow]:
    # additional_provider is a sibling, so a module-scope import of it works anywhere. This one
    # lives in a *different* scripts directory, so it is imported here instead: by the time run()
    # is called triage has put that directory on sys.path. At module scope it would be too early
    # when this file is executed directly, and making that work is the script's own job.
    from additional_extra_provider import run as get_extra_provider

    dependencies = ", ".join([get_provider(args, context), get_extra_provider(args, context)])
    print(f"{SUCCESS_MESSAGE} (dependencies: {dependencies})")
    return [AdditionalDirectoryRow(message=SUCCESS_MESSAGE, dependencies=dependencies)]


if __name__ == "__main__":
    run_script()
