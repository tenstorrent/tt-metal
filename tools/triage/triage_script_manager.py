#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from copy import deepcopy
import heapq
import importlib
import inspect
import os
import re
import sys
from time import time
from typing import Any

import utils
from triage import (
    ScriptArguments,
    ScriptConfig,
    TriageScript,
    TTTriageError,
    create_progress,
    serialize_result,
)
from ttexalens.context import Context


class TriageScriptManager:
    """Owns script discovery, dep resolution, execution, and reporting for one triage run."""

    def __init__(self) -> None:
        # Script registry keyed by absolute path. Populated by load_*().
        self.scripts: dict[str, TriageScript] = {}
        # Topologically ordered execution queue. Populated by resolve_dependencies().
        self.script_queue: list[TriageScript] = []

    def load_application_scripts(self, application_path: str) -> None:
        """Discover triage scripts by enumerating .py files in the directory.
        Disabled or invalid scripts are silently dropped."""
        own_basename = "triage.py"  # the entrypoint's loader, not a triage script itself
        manager_basename = os.path.basename(__file__)
        script_files = [
            f for f in os.listdir(application_path) if f.endswith(".py") and f != own_basename and f != manager_basename
        ]
        for script_file in script_files:
            script_path = os.path.join(application_path, script_file)
            try:
                triage_script = self._load_script(script_path)
                if triage_script.config.disabled:
                    utils.DEBUG(f"Script {script_path} is disabled, skipping...")
                    continue
            except Exception as e:
                utils.DEBUG(f"Failed to load script {script_path}: {e}")
                continue
            self.scripts[script_path] = triage_script

    def load_script_with_dependencies(self, script_path: str) -> None:
        """Load one script plus its transitive deps via BFS. Used by run_script()."""
        script_path = os.path.abspath(script_path)
        loading: list[str] = []
        self.scripts[script_path] = self._load_script(script_path)
        loading.extend(self.scripts[script_path].config.depends)
        while loading:
            dep_path = loading.pop(0)
            if dep_path in self.scripts:
                continue
            self.scripts[dep_path] = self._load_script(dep_path)
            loading.extend(self.scripts[dep_path].config.depends)

    @staticmethod
    def _load_script(script_path: str) -> TriageScript:
        """Import a single script module and construct a TriageScript from it.
        Validates that the module has script_config, an Owner: tag, and run(args, context)."""
        script_path = os.path.abspath(script_path)
        base_path = os.path.dirname(script_path)
        appended = False
        if base_path not in sys.path:
            sys.path.append(base_path)
            appended = True
        try:
            script_name = os.path.splitext(os.path.basename(script_path))[0]
            script_module = importlib.import_module(script_name)

            script_config: ScriptConfig = script_module.script_config
            if script_config is None:
                raise ValueError(f"Script {script_path} does not have script_config.")

            if not script_module.__doc__:
                raise ValueError(f"Script {script_path} must have a docstring, see relevant scripts for examples.\n")

            if not re.search(r"^Owner:\s*\S+", script_module.__doc__, re.MULTILINE):
                raise ValueError(
                    f"Script {script_path} docstring must include an 'Owner:' field with the corresponding owner of the script.\n"
                )

            run_method = script_module.run if hasattr(script_module, "run") and callable(script_module.run) else None
            if run_method is not None:
                signature = inspect.signature(run_method)
                if (
                    len(signature.parameters) != 2
                    or "args" not in signature.parameters
                    or "context" not in signature.parameters
                ):
                    run_method = None
            if run_method is None:
                raise ValueError(
                    f"Script {script_path} does not have a valid run method with two arguments (args, context)."
                )

            triage_script = TriageScript(
                name=os.path.basename(script_path),
                path=script_path,
                config=deepcopy(script_config),
                module=script_module,
                run_method=run_method,
                documentation=script_module.__doc__,
            )

            # Normalize string `depends` entries to absolute file paths so the
            # registry keying lines up with what _load_script returns.
            if triage_script.config.depends is None:
                triage_script.config.depends = []
            else:
                triage_script.config.depends = [
                    dep if isinstance(dep, str) and dep.endswith(".py") else f"{dep}.py"
                    for dep in triage_script.config.depends
                ]
                triage_script.config.depends = [os.path.join(base_path, dep) for dep in triage_script.config.depends]

            return triage_script
        finally:
            if appended:
                sys.path.remove(base_path)

    def resolve_dependencies(self) -> None:
        """Build the DAG and topo-sort into script_queue.
        Scripts whose declared deps aren't in the registry are dropped (with a clear log
        line) so they never enter the queue. Drops cascade: if A depends on B and B was
        dropped, A is dropped too. By the time this returns, every script in script_queue
        has all its deps wired up; failed/skipped are reserved for runtime outcomes."""
        # Iteratively drop scripts whose deps cannot be resolved, until stable.
        # Drops cascade: dropping A causes scripts that depended on A to drop too.
        while True:
            missing_by_path: dict[str, list[str]] = {}
            for path, script in self.scripts.items():
                unresolved = [dep for dep in script.config.depends if dep not in self.scripts]
                if unresolved:
                    missing_by_path[path] = unresolved
            if not missing_by_path:
                break
            for path, unresolved in missing_by_path.items():
                script = self.scripts[path]
                missing = ", ".join(os.path.basename(d) for d in unresolved)
                utils.ERROR(f"Dropping '{script.name}': missing dependencies: {missing}")
                del self.scripts[path]

        # Every surviving script has all its deps in the registry; wire them up.
        for script in self.scripts.values():
            script.depends = [self.scripts[dep] for dep in script.config.depends]

        self.script_queue = _resolve_execution_order(self.scripts)

    def execute_all(
        self,
        args: ScriptArguments,
        context: Context,
        print_script_times: bool = False,
    ) -> tuple[float, float]:
        """Run every script in topological order with skip/fail propagation.
        Returns (total_time, serialization_time) for printing if --print-script-times is on."""
        total_time = 0.0
        serialization_time = 0.0

        with create_progress() as progress:
            scripts_task = progress.add_task("Script execution", total=len(self.script_queue))
            try:
                for script in self.script_queue:
                    progress.update(scripts_task, description=f"Running {script.name}")

                    if self._propagate_skip_or_fail(script):
                        progress.advance(scripts_task)
                        continue

                    start_time = time()
                    result = script.run(args=args, context=context)
                    end_time = time()
                    total_time += end_time - start_time
                    execution_time = f" [{end_time - start_time:.2f}s]" if print_script_times else ""

                    if script.skipped:
                        print()
                        utils.INFO(f"{script.name}{execution_time}:")
                        utils.INFO(f"  skipped: {script.status_message}")
                    elif script.config.data_provider:
                        if result is None:
                            print()
                            utils.INFO(f"{script.name}{execution_time}:")
                            if script.status_message is not None:
                                utils.ERROR(f"  Data provider script failed: {script.status_message}")
                            else:
                                utils.ERROR(f"  Data provider script did not return any data.")
                        elif execution_time:
                            print()
                            utils.INFO(f"{script.name}{execution_time}:")
                            utils.INFO("  pass")
                    else:
                        ser_start = time()
                        serialize_result(script, result, execution_time)
                        ser_end = time()
                        total_time += ser_end - ser_start
                        serialization_time += ser_end - ser_start

                    progress.advance(scripts_task)
            finally:
                progress.remove_task(scripts_task)

        return total_time, serialization_time

    def execute_script(
        self,
        script_path: str,
        args: ScriptArguments,
        context: Context,
    ) -> Any:
        """Run the loaded subgraph with log_error=False (raises on failure).
        Returns the result of `script_path` specifically -- not the last script in
        topological order, even though those happen to coincide today."""
        results: dict[str, Any] = {}
        for script in self.script_queue:
            if self._propagate_skip_in_subgraph(script):
                results[script.path] = None
                continue
            if not all(not dep.failed for dep in script.depends):
                raise TTTriageError(f"{script.name}: Cannot run script due to failed dependencies.")
            result = script.run(args=args, context=context, log_error=False)
            results[script.path] = result
            if script.skipped:
                continue
            if script.config.data_provider and result is None:
                raise TTTriageError(f"{script.name}: Data provider script did not return any data.")
        return results.get(os.path.abspath(script_path))

    def build_summary(self) -> str:
        """One line per script: FAIL, SKIP, or pass (data providers' pass status is omitted)."""
        lines = []
        for script in self.script_queue:
            if script.failed:
                lines.append(f"{script.name}: FAIL - {script.status_message or 'unknown error'}")
            elif script.skipped:
                lines.append(f"{script.name}: SKIP - {script.status_message or 'no reason given'}")
            elif not script.config.data_provider:
                lines.append(f"{script.name}: pass")
        return "\n".join(lines) if lines else "No triage scripts executed."

    def _propagate_skip_or_fail(self, script: TriageScript) -> bool:
        """Mark `script` skipped/failed if any dep was. Skip wins over fail.
        Returns True when the script should not be invoked. Used by execute_all."""
        skipped_dep = next((dep for dep in script.depends if dep.skipped), None)
        if skipped_dep is not None:
            script.status = "skipped"
            script.status_message = f"dependency '{skipped_dep.name}' was skipped"
            return True
        if not all(not dep.failed for dep in script.depends):
            utils.INFO(f"{script.name}:")
            utils.WARN(f"  Cannot run script due to failed dependencies.")
            script.status = "failed"
            script.status_message = "Cannot run script due to failed dependencies."
            return True
        return False

    def _propagate_skip_in_subgraph(self, script: TriageScript) -> bool:
        """Skip-only propagation for execute_subgraph (which raises on fail elsewhere)."""
        skipped_dep = next((dep for dep in script.depends if dep.skipped), None)
        if skipped_dep is not None:
            script.status = "skipped"
            script.status_message = f"dependency '{skipped_dep.name}' was skipped"
            return True
        return False


def _resolve_execution_order(scripts: dict[str, TriageScript]) -> list[TriageScript]:
    """Topological sort with ScriptPriority as the tiebreaker (HIGH first)."""
    script_dependents: dict[str, list[str]] = defaultdict(list)
    script_missing_dependencies: dict[str, int] = defaultdict(int)

    for path, script in scripts.items():
        script_missing_dependencies[path] = len(script.config.depends)
        for dep in script.config.depends:
            script_dependents[dep].append(path)

    heap: list[tuple[int, str, TriageScript]] = []
    for path, script in scripts.items():
        if script_missing_dependencies[path] == 0:
            heapq.heappush(heap, (-script.config.priority.value, path, script))

    result: list[TriageScript] = []
    while heap:
        _, path, script = heapq.heappop(heap)
        result.append(script)
        for dep_path in script_dependents[path]:
            script_missing_dependencies[dep_path] -= 1
            if script_missing_dependencies[dep_path] == 0:
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
