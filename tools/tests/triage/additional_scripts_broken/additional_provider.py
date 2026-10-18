#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    additional_provider

Description:
    Test fixture: deliberately reuses the file name of additional_scripts/additional_provider.py.
    Only one file can own the module name, so discovery must drop this copy rather than run it in
    place of the other one.

Owner:
    tt-vjovanovic
"""

from triage import ScriptConfig, run_script
from ttexalens.context import Context

script_config = ScriptConfig(data_provider=True)


def run(args, context: Context) -> str:
    raise AssertionError("the shadowed copy of additional_provider must never run")


if __name__ == "__main__":
    run_script()
