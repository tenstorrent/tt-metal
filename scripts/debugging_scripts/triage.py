#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TODO: Write documentation
"""

# Check if tt-exalens is installed
import os
from utils import RED, RST, GREEN

try:
    from ttexalens.tt_exalens_init import init_ttexalens
except ImportError as e:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    install_script = os.path.join(os.path.dirname(script_dir), "install_debugger.sh")
    print(f"Module '{e}' not found. Please install tt-exalens by running: {GREEN}")
    print(f"  {install_script}{RST}")
    exit(1)

# Import necessary libraries
import importlib
from dataclasses import dataclass
from types import ModuleType


@dataclass
class TriageScript:
    data_provider: bool = False
    disabled: bool = False
    depends: list[str] = []
    module: ModuleType = None


class TTTriageError(Exception):
    """Base class for TT Triage errors."""
    pass


def main():
    # Initialize tt-exalens
    init_ttexalens(use_noc1=False) # TODO: Add command line argument to select NOC

    # Enumerate all scripts in application directory
    application_path = os.path.dirname(__file__)
    script_files = [f for f in os.listdir(application_path) if f.endswith('.py') and f != os.path.basename(__file__)]

    # Load tt-triage script configuration
    scripts: dict[str, TriageScript] = {}
    for script in script_files:
        # Import each script as a module
        script_path = os.path.join(application_path, script)
        try:
            script_module = importlib.import_module(script_path)
        except Exception as e:
            # Print call stack
            print(f"{RED}Failed to import script {script}:{RST} {e}")
            continue

        # Check if script has a configuration
        script_config: TriageScript = script_module.triage_config
        if script_config is None or script_config.disabled:
            # This script does not have a configuration, which means it is not tt-triage script, skipping...
            continue
        script_config.module = script_module
        if script_module.depends is None:
            # If script does not have dependencies, set it to empty list
            script_config.depends = []
        else:
            script_config.depends = [dep if isinstance(dep, str) and dep.endswith(".py") else f"{dep}.py" for dep in script_module.depends]

        # Add script to the list of scripts
        scripts[script] = script_config

    # Find dependency graph of script execution
    used_scripts: set[str] = set()
    script_queue: list[str] = []
    while len(scripts) > len(script_queue):
        deployed_scripts: int = 0
        for script_name, script_config in scripts.items():
            if script_name in used_scripts:
                continue

            # Check if all dependencies are met
            if all(dep in used_scripts for dep in script_config.depends):
                # Add script to the queue
                script_queue.append(script_name)
                used_scripts.add(script_name)
                deployed_scripts += 1

        # Check circular dependency
        if deployed_scripts == 0:
            # If no scripts were deployed, it means there is a circular dependency or disabled script dependency
            remaining_scripts = set(scripts.keys()) - used_scripts
            print(f"{RED}Bad dependency detected in scripts:{RST} {', '.join(remaining_scripts)}")
            print(f"{RED}  Circular dependency, dependency on disabled or non-existing script is not allowed.{RST}")
            print(f"{RED}  Please check if all dependencies are met and scripts are enabled.{RST}")
            exit(1)

    # TODO: Parse common command line arguments

    # TODO: Execute scripts
    for script_name in script_queue:
        script = scripts[script_name]

        # TODO: Parse command line arguments for the script

if __name__ == "__main__":
    main()


# TODO:
# Inspector data - first data provider
# Devices to check - second data provider (depends on inspector data)
# Dispatcher data - third data provider (depends on devices to check and inspector data)
# Callstack printing - (callstack might be data provider, but printing it is not)
# Binary integrity validation - (state checker, depends on devices, dispatcher data)
# All other state checkers that are currently present in tt-triage.py
