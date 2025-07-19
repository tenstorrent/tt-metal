#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TODO: Write documentation
"""

# Check if tt-exalens is installed
import inspect
import os
from utils import ORANGE, RED, RST, GREEN

try:
    from ttexalens.tt_exalens_init import init_ttexalens
except ImportError as e:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    install_script = os.path.join(os.path.dirname(script_dir), "install_debugger.sh")
    print(f"Module '{e}' not found. Please install tt-exalens by running: {GREEN}")
    print(f"  {install_script}{RST}")
    exit(1)

# Import necessary libraries
from copy import deepcopy
from dataclasses import dataclass, field
import importlib
import sys
from ttexalens.context import Context
from typing import Any, Callable, TypeVar
from types import ModuleType


@dataclass
class ScriptConfig:
    data_provider: bool = False
    disabled: bool = False
    depends: list[str] = field(default_factory=list)


class ScriptArguments:
    def __init__(self, args: dict[str, Any]):
        self.args = args

    def __getitem__(self, item: str) -> Any:
        return self.args.get(item, None)


# Type variables for the decorator
T = TypeVar('T')


def triage_cache(run_method: Callable[[ScriptArguments, Context], T], /) -> Callable[[ScriptArguments, Context], T]:
    # Check that run method has two arguments (args, context)
    assert callable(run_method), "run_method must be a callable function."
    signature = inspect.signature(run_method)
    assert len(signature.parameters) == 2 and 'args' in signature.parameters and 'context' in signature.parameters, "run_method must have two arguments (args, context)."

    # Create simple cache
    cache: dict[tuple[int, int], T] = {}
    def cache_wrapper(args: ScriptArguments, context: Context) -> T:
        cache_key = (id(args), id(context))
        if cache_key not in cache:
            cache[cache_key] = run_method(args, context)
        return cache[cache_key]

    return cache_wrapper

@dataclass
class TriageScript:
    name: str
    path: str
    config: ScriptConfig
    module: ModuleType
    run_method: Callable[..., Any]
    depends: list["TriageScript"] = field(default_factory=list)
    failed: bool = False
    failure_message: str | None = None

    def run(self, args: ScriptArguments, context: Context, log_error: bool = True) -> Any:
        try:
            result = self.run_method(args=args, context=context)
            if self.config.data_provider and result is None:
                if log_error:
                    self.failed = True
                    self.failure_message = "Data provider script did not return any data."
                else:
                    raise TTTriageError("Data provider script did not return any data.")
            return result
        except Exception as e:
            if log_error:
                self.failed = True
                self.failure_message = str(e)
                return None
            else:
                raise

    @staticmethod
    def load(script_path: str) -> "TriageScript":
        script_path = os.path.abspath(script_path)
        base_path = os.path.dirname(script_path)
        appended = False
        if not base_path in sys.path:
            sys.path.append(base_path)
            appended = True
        try:
            script_name = os.path.splitext(os.path.basename(script_path))[0]
            script_module = importlib.import_module(script_name)

            # Check if script has a configuration
            script_config: ScriptConfig = script_module.script_config
            if script_config is None:
                # This script does not have a configuration, which means it is not tt-triage script, skipping...
                raise ValueError(f"Script {script_path} does not have script_config.")

            # Check if script has a run method with two arguments (args and context)
            run_method = script_module.run if hasattr(script_module, 'run') and callable(script_module.run) else None
            if run_method is not None:
                signature = inspect.signature(run_method)
                if len(signature.parameters) != 2 or 'args' not in signature.parameters or 'context' not in signature.parameters:
                    run_method = None
            if run_method is None:
                raise ValueError(f"Script {script_path} does not have a valid run method with two arguments (args, context).")

            triage_script = TriageScript(
                name=os.path.basename(script_path),
                path=script_path,
                config=deepcopy(script_config),
                module=script_module,
                run_method=run_method,
            )

            if triage_script.config.depends is None:
                # If script does not have dependencies, set it to empty list
                triage_script.config.depends = []
            else:
                triage_script.config.depends = [dep if isinstance(dep, str) and dep.endswith(".py") else f"{dep}.py" for dep in triage_script.config.depends]
                triage_script.config.depends = [os.path.join(base_path, dep) for dep in triage_script.config.depends]

            return triage_script
        finally:
            if appended:
                sys.path.remove(base_path)

    @staticmethod
    def load_all(script_path: str) -> dict[str, "TriageScript"]:
        scripts: dict[str, TriageScript] = {}
        loading: list[str] = []
        script = TriageScript.load(script_path)
        scripts[script_path] = script

        # Load all dependencies
        loading.extend(script.config.depends)
        while len(loading) > 0:
            loading_script = loading.pop(0)
            if loading_script not in scripts:
                script = TriageScript.load(loading_script)
                scripts[loading_script] = script
                loading.extend(script.config.depends)

        # Update dependencies in scripts
        for script in scripts.values():
            for dep in script.config.depends:
                assert dep in scripts, f"Dependency {dep} for script {script.name} not found."
                script.depends.append(scripts[dep])
        return scripts


def resolve_execution_order(scripts: dict[str, TriageScript]) -> list[TriageScript]:
    used_scripts: set[str] = set()
    script_queue: list[TriageScript] = []
    while len(scripts) > len(script_queue):
        deployed_scripts: int = 0
        for script_path, script in scripts.items():
            if script_path in used_scripts:
                continue

            # Check if all dependencies are met
            if all(dep in used_scripts for dep in script.config.depends):
                # Add script to the queue
                script_queue.append(script)
                used_scripts.add(script_path)
                deployed_scripts += 1

        # Check circular dependency
        if deployed_scripts == 0:
            # If no scripts were deployed, it means there is a circular dependency or disabled script dependency
            remaining_scripts = set(scripts.keys()) - used_scripts
            raise ValueError(
                f"{RED}Bad dependency detected in scripts:{RST} {', '.join(remaining_scripts)}\n"
                f"{RED}  Circular dependency, dependency on disabled or non-existing script is not allowed.{RST}\n"
                f"{RED}  Please check if all dependencies are met and scripts are enabled.{RST}"
            )
    return script_queue


def parse_arguments(scripts: dict[str, TriageScript]) -> ScriptArguments:
    # TODO: Implement argument parsing for scripts
    from docopt import docopt, parse_defaults, parse_pattern, formal_usage, printable_usage, parse_argv, extras, Required, TokenStream, DocoptExit, Option, AnyOptions
    import sys

    combined_options = []
    combined_pattern: Required = Required(*[Required(*[])])

    for script in scripts.values():
        if hasattr(script.module, '__doc__') and script.module.__doc__:
            try:
                script_options = parse_defaults(script.module.__doc__)
                combined_options.extend(script_options)

                usage = printable_usage(script.module.__doc__)
                pattern = parse_pattern(formal_usage(usage), script_options)
                combined_pattern.children[0].children.extend(pattern.children[0].children)
            except BaseException as e:
                print(f"Error parsing arguments for script {script.name}: {e}")
                continue

    argv = parse_argv(TokenStream(sys.argv[1:], DocoptExit), list(combined_options), options_first = False)
    pattern_options = set(combined_pattern.flat(Option))
    for ao in combined_pattern.flat(AnyOptions):
        ao.children = list(set(combined_options) - pattern_options)
    matched, left, collected = combined_pattern.fix().match(argv)
    if matched and left == []:  # better error message if left?
        return ScriptArguments(dict((a.name, a.value) for a in (combined_pattern.flat() + collected)))
    return ScriptArguments({})


def run_script(script_path: str | None = None, args: ScriptArguments | None = None, context: Context | None = None) -> Any:
    # Resolve script path
    if script_path is None:
        # Check if previous call on callstack is a TriageScript
        stack = inspect.stack()
        if stack is None or len(stack) < 2:
            raise ValueError("No script path provided and no caller found in callstack.")
        script_path = stack[1].filename
    else:
        if not script_path.endswith(".py"):
            script_path = script_path + ".py"
        application_path = os.path.dirname(__file__)
        if not os.path.isabs(script_path):
            script_path = os.path.join(application_path, script_path)
        script_path = os.path.abspath(script_path)
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script {script_path} does not exist.")

    # Load script and its dependencies
    scripts = TriageScript.load_all(script_path)

    # Find execution order of scripts
    script_queue = resolve_execution_order(scripts)

    # Parse arguments
    if args is None:
        args = parse_arguments(scripts)

    # Initialize context if not provided
    if context is None:
        context = init_ttexalens()

    # Run scripts in order
    result: Any = None
    for script in script_queue:
        if not all(not dep.failed for dep in script.depends):
            raise TTTriageError(f"Cannot run script {script.name} due to failed dependencies.")
        else:
            result = script.run(args=args, context=context, log_error=False)
            if script.config.data_provider and result is None:
                raise TTTriageError(f"Data provider script {script.name} did not return any data.")
    if scripts[script_path].config.data_provider:
        print(result)


class TTTriageError(Exception):
    """Base class for TT Triage errors."""
    pass


def main():
    # Initialize tt-exalens
    init_ttexalens(use_noc1=False) # TODO: Add command line argument to select NOC

    # Enumerate all scripts in application directory
    application_path = os.path.dirname(__file__)
    script_files = [f for f in os.listdir(application_path) if f.endswith('.py') and f != os.path.basename(__file__)]

    # Load tt-triage scripts
    # TODO: do we need to check for subdirectories?
    scripts: dict[str, TriageScript] = {}
    base_path = application_path
    for script in script_files:
        # TODO: Do this prints only in verbose mode
        script_path = os.path.join(base_path, script)
        try:
            triage_script = TriageScript.load(script_path)
            if triage_script.config.disabled:
                print(f"{ORANGE}Script {script_path} is disabled{RST}")
                continue
        except Exception as e:
            print(f"{RED}Failed to load script {script_path}: {e}{RST}")
            continue
        scripts[script_path] = triage_script

    # Resolve dependencies
    for script in scripts.values():
        for dep in script.config.depends:
            if dep in scripts:
                script.depends.append(scripts[dep])
            else:
                print(f"{RED}Dependency {dep} for script {script.name} not found.{RST}")
                script.failed = True
                script.failure_message = f"Dependency {dep} not found."

    # Find dependency graph of script execution
    script_queue = resolve_execution_order(scripts)

    # Parse common command line arguments
    args = parse_arguments(scripts)
    context = init_ttexalens()

    # Execute scripts
    for script in script_queue:
        if not all(not dep.failed for dep in script.depends):
            print(f"{RED}Cannot run script {script.name} due to failed dependencies.{RST}")
        else:
            result = script.run(args=args, context=context, log_error=False)
            if script.config.data_provider and result is None:
                print(f"{RED}Data provider script {script.name} did not return any data.{RST}")

if __name__ == "__main__":
    main()


# TODO:
# Inspector data - first data provider
# Devices to check - second data provider (depends on inspector data)
# Dispatcher data - third data provider (depends on devices to check and inspector data)
# Callstack printing - (callstack might be data provider, but printing it is not)
# Binary integrity validation - (state checker, depends on devices, dispatcher data)
# All other state checkers that are currently present in tt-triage.py
