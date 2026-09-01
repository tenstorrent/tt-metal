#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_extra_directory

Description:
    Test fixture: depends only on a script sitting next to it, and imports nothing from any other
    directory. That makes it loadable with no --additional-scripts-directory at all, which is how a
    script run directly from a consuming repo finds the scripts beside it.

Owner:
    tt-vjovanovic
"""

from dataclasses import dataclass

from additional_extra_provider import run as get_extra_provider
from triage import ScriptConfig, run_script, triage_field
from ttexalens.context import Context

script_config = ScriptConfig(depends=["additional_extra_provider"])


@dataclass
class ExtraDirectoryRow:
    dependency: str = triage_field("Dependency")


def run(args, context: Context) -> list[ExtraDirectoryRow]:
    return [ExtraDirectoryRow(dependency=get_extra_provider(args, context))]


if __name__ == "__main__":
    run_script()
