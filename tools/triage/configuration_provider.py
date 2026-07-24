#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    configuration_provider

Description:
    Data provider that fetches Inspector configuration (environment variables,
    runtime options, TTNN configuration) once and caches it for other scripts.
    Provides lookups by (scope, name).

Owner:
    onenezicTT
"""

from dataclasses import dataclass
from triage import triage_singleton, ScriptConfig, run_script
from inspector_data import run as get_inspector_data

script_config = ScriptConfig(
    data_provider=True,
    depends=["inspector_data"],
)


@dataclass(frozen=True)
class ConfigValue:
    scope: str
    name: str
    value: str


class ConfigurationProvider:
    """Cached view of Inspector configuration entries."""

    def __init__(self, inspector):
        self._entries = [
            ConfigValue(entry.scope, entry.name, entry.value) for entry in inspector.getConfiguration().entries
        ]

    def all(self) -> list[ConfigValue]:
        return self._entries

    def get(self, name: str, default: str | None = None) -> str | None:
        # Names are unique across scopes (environment/rtOptions/ttnnConfig use disjoint conventions).
        for entry in self._entries:
            if entry.name == name:
                return entry.value
        return default

    def get_bool(self, name: str, default: bool = False) -> bool:
        value = self.get(name)
        return default if value is None else value.lower() == "true"


@triage_singleton
def run(args, context) -> ConfigurationProvider:
    return ConfigurationProvider(get_inspector_data(args, context))


if __name__ == "__main__":
    run_script()
