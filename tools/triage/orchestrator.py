#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Script discovery, dep wiring, and topological ordering."""

from __future__ import annotations

import heapq
import os
from collections import defaultdict

import utils
from triage_script import TriageScript

# Runtime modules living next to the scripts — skipped during discovery.
_NON_SCRIPT_BASENAMES = frozenset(
    {"triage.py", "triage_script.py", "orchestrator.py", "runner.py", "process_group.py", "aggregator.py"}
)


class ScriptOrchestrator:
    def __init__(self, directory: str):
        self.directory = os.path.abspath(directory)

    def discover(self) -> dict[str, TriageScript]:
        """Scan the directory for triage scripts. No dep wiring, no ordering."""
        scripts: dict[str, TriageScript] = {}
        for fname in os.listdir(self.directory):
            if not fname.endswith(".py") or fname in _NON_SCRIPT_BASENAMES:
                continue
            script_path = os.path.join(self.directory, fname)
            try:
                script = TriageScript.load(script_path)
                if script.config.disabled:
                    continue
            except Exception:
                continue
            scripts[script_path] = script
        return scripts

    def queue_for_full_triage(self) -> tuple[dict[str, TriageScript], list[TriageScript]]:
        scripts = self.discover()
        self._wire_deps(scripts)
        return scripts, self._resolve_order(scripts)

    def queue_for_target(self, script_path: str) -> tuple[dict[str, TriageScript], list[TriageScript]]:
        scripts = self._load_with_deps(script_path)
        return scripts, self._resolve_order(scripts)

    @staticmethod
    def _wire_deps(scripts: dict[str, TriageScript]) -> None:
        for script in scripts.values():
            for dep in script.config.depends:
                if dep in scripts:
                    script.depends.append(scripts[dep])
                else:
                    utils.ERROR(f"Dependency {dep} for script {script.name} not found.")
                    script.failed = True
                    script.failure_message = f"Dependency {dep} not found."

    @staticmethod
    def _load_with_deps(script_path: str) -> dict[str, TriageScript]:
        scripts: dict[str, TriageScript] = {}
        loading: list[str] = []
        script = TriageScript.load(script_path)
        scripts[script_path] = script
        loading.extend(script.config.depends)
        while loading:
            loading_path = loading.pop(0)
            if loading_path not in scripts:
                script = TriageScript.load(loading_path)
                scripts[loading_path] = script
                loading.extend(script.config.depends)
        # Wire transitive deps onto each script's `.depends` list.
        for script in scripts.values():
            for dep in script.config.depends:
                assert dep in scripts, f"Dependency {dep} for script {script.name} not found."
                script.depends.append(scripts[dep])
        return scripts

    @staticmethod
    def _resolve_order(scripts: dict[str, TriageScript]) -> list[TriageScript]:
        """Topological sort by dep edges, ties broken by priority (HIGH first)."""
        dependents: dict[str, list[str]] = defaultdict(list)
        missing: dict[str, int] = defaultdict(int)
        for path, script in scripts.items():
            missing[path] = len(script.config.depends)
            for dep in script.config.depends:
                dependents[dep].append(path)

        # Min-heap keyed by negative priority (HIGH priority pops first).
        heap: list[tuple[int, str, TriageScript]] = []
        for path, script in scripts.items():
            if missing[path] == 0:
                heapq.heappush(heap, (-script.config.priority.value, path, script))

        result: list[TriageScript] = []
        while heap:
            _, path, script = heapq.heappop(heap)
            result.append(script)
            for dep_path in dependents[path]:
                missing[dep_path] -= 1
                if missing[dep_path] == 0:
                    dep_script = scripts[dep_path]
                    heapq.heappush(heap, (-dep_script.config.priority.value, dep_path, dep_script))

        if len(result) != len(scripts):
            remaining = set(scripts.keys()) - {s.path for s in result}
            raise ValueError(
                f"Bad dependency detected in scripts: {', '.join(remaining)}\n"
                f"  Circular dependency, dependency on disabled or non-existing script is not allowed.\n"
                f"  Please check if all dependencies are met and scripts are enabled."
            )
        return result
