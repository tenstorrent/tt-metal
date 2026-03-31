#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_configuration [--scope=<scope>]

Options:
    --scope=<scope>     Filter by scope: environment, rtOptions, ttnnConfig. Shows all if not specified.

Description:
    Dumps configuration data from Inspector: environment variables, runtime options,
    and TTNN configuration. Groups output by scope.

Owner:
    onenezicTT
"""

from dataclasses import dataclass
from inspector_data import run as get_inspector_data, InspectorException
from triage import ScriptConfig, ScriptPriority, log_check, triage_field, run_script

script_config = ScriptConfig(
    depends=["inspector_data"],
    priority=ScriptPriority.HIGH,
)


@dataclass
class ConfigEntry:
    name: str = triage_field("Name")
    value: str = triage_field("Value")
    scope: str = triage_field("Scope")


def run(args, context):
    scope_filter = args["--scope"]

    try:
        inspector = get_inspector_data(args, context)
        result = inspector.getConfiguration()
    except InspectorException as e:
        log_check(False, f"Failed to get configuration from Inspector: {e}")
        return None

    entries = result.entries

    # Group by scope
    by_scope = {}
    for entry in entries:
        scope = entry.scope
        by_scope.setdefault(scope, []).append(entry)

    all_entries = []
    for scope_name, items in sorted(by_scope.items()):
        if scope_filter and scope_name != scope_filter:
            continue
        for item in sorted(items, key=lambda x: x.name):
            all_entries.append(
                ConfigEntry(
                    name=item.name,
                    value=item.value,
                    scope=scope_name,
                )
            )

    return all_entries


if __name__ == "__main__":
    run_script()
