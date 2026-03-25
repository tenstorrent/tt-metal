#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    triage [--initialize-with-noc1] [--remote-exalens] [--remote-server=<remote-server>] [--remote-port=<remote-port>] [--verbosity=<verbosity>] [--run=<script>]... [--skip-version-check] [--print-script-times] [-v ...] [--disable-colors] [--disable-progress] [--triage-summary-path=<path>]

Options:
    --remote-exalens                 Connect to remote exalens server.
    --remote-server=<remote-server>  Specify the remote server to connect to. [default: localhost]
    --remote-port=<remote-port>      Specify the remote server port. [default: 5555]
    --initialize-with-noc1           Initialize debugger context with NOC1 enabled. [default: False]
    --verbosity=<verbosity>          Choose output verbosity. 1: ERROR, 2: WARN, 3: INFO, 4: VERBOSE, 5: DEBUG. [default: 3]
    --run=<script>                   Run specific script(s) by name. If not provided, all scripts will be run. [default: all]
    --skip-version-check             Do not enforce debugger version check. [default: False]
    --print-script-times             Print the execution time of each script. [default: False]
    -v                               Increase verbosity level (can be repeated: -v, -vv, -vvv).
                                     Controls which columns/fields are displayed:
                                     Level 0 (default): Essential fields (Kernel ID:Name, Go Message, Subdevice, Preload, Waypoint, PC, Callstack)
                                     Level 1 (-v): Include detailed dispatcher fields (Firmware/Kernel Path, Host Assigned ID, Kernel Offset, Previous Kernel)
                                     Level 2 (-vv): Include internal debug fields (RD PTR, Base, Offset, Kernel XIP Path)
    --disable-colors                 Disable colored output. [default: False]
    --disable-progress               Disable progress bars. [default: False]
    --triage-summary-path=<path>     Write a triage summary file to the given path (used by CI for hang reports).

Description:
    Diagnoses Tenstorrent AI hardware by performing comprehensive health checks on ARC processors, NOC connectivity, L1 memory, and RISC-V cores.
    Identifies running kernels and provides callstack information to troubleshoot failed operations.
    Example use with tt-metal:
        export TT_METAL_HOME=~/work/tt-metal
        ./build_metal.sh --build-programming-examples
        build/programming_examples/matmul_multi_core
        triage

Owner:
    tt-vjovanovic
"""

# Check if tt-exalens is installed
from collections import defaultdict
from enum import Enum
import heapq
import inspect
import os
import shutil
import threading
from time import time
import traceback
import utils
from collections.abc import Iterable
from pathlib import Path
import re

_triage_requirements_path = str(Path(__file__).resolve().parent / "requirements.txt")

try:
    from ttexalens.tt_exalens_init import init_ttexalens, init_ttexalens_remote
    import capnp
except ImportError as e:
    RST = "\033[0m" if utils.should_use_color() else ""
    GREEN = "\033[32m" if utils.should_use_color() else ""  # For instructions
    pip_cmd = "uv pip" if shutil.which("uv") is not None else "pip"
    print(f"Module '{e.name}' not found. Please install requirements by running:")
    print(f"  {GREEN}{pip_cmd} install -r {_triage_requirements_path}{RST}")
    exit(1)


# Import necessary libraries
from copy import deepcopy
from dataclasses import dataclass, field
import importlib
import importlib.metadata as importlib_metadata
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn
from rich.table import Table
import sys
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.elf import ElfVariable
from ttexalens.umd_device import TimeoutDeviceRegisterError
from typing import Any, Callable, Iterable, TypeVar
from types import ModuleType


class ScriptPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass
class ScriptConfig:
    data_provider: bool = False
    disabled: bool = False
    depends: list[str] = field(default_factory=list)
    priority: ScriptPriority = ScriptPriority.MEDIUM


class ScriptArguments:
    def __init__(self, args: dict[str, Any]):
        self.args = args

    def __getitem__(self, item: str) -> Any:
        return self.args.get(item, None)


# Type variables for the decorator
T = TypeVar("T")


def triage_singleton(run_method: Callable[[ScriptArguments, Context], T], /) -> Callable[[ScriptArguments, Context], T]:
    # Check that run method has two arguments (args, context)
    assert callable(run_method), "run_method must be a callable function."
    signature = inspect.signature(run_method)
    assert (
        len(signature.parameters) == 2 and "args" in signature.parameters and "context" in signature.parameters
    ), "run_method must have two arguments (args, context)."

    # Create simple cache
    cache: dict[tuple[int, int], T] = {}

    def cache_wrapper(args: ScriptArguments, context: Context) -> T:
        cache_key = (id(args), id(context))
        if cache_key not in cache:
            cache[cache_key] = run_method(args, context)
        return cache[cache_key]

    return cache_wrapper


# Data serialization
def serialize_collection(items: Iterable[Any], separator: str = ", ") -> str:
    return separator.join(default_serializer(item) for item in items)


def default_serializer(value) -> str:
    """Default serializer for fields."""
    if value is None:
        return "N/A"
    if isinstance(value, Device):
        return str(value.id)
    elif isinstance(value, OnChipCoordinate):
        return value.to_user_str()
    elif isinstance(value, ElfVariable):
        try:
            return serialize_collection(value.as_list())
        except:
            return str(value)
    elif isinstance(value, Iterable) and not isinstance(value, str):
        return serialize_collection(value)
    else:
        return str(value)


def hex_serializer(value: int | None) -> str:
    if value is None:
        return "N/A"
    return hex(value)


def collection_serializer(separator: str):
    def serializer(item):
        return serialize_collection(item, separator)

    return serializer


def triage_field(serialized_name: str | None = None, serializer: Callable[[Any], str] | None = None, verbose: int = 0):
    if serializer is None:
        serializer = default_serializer
    return field(
        metadata={"recurse": False, "serialized_name": serialized_name, "serializer": serializer, "verbose": verbose}
    )


def recurse_field(verbose: int = 0):
    return field(metadata={"recurse": True, "verbose": verbose})


# Module-level flag to control verbose field output
# Level 0 (default): Essential fields for triage
# Level 1 (-v): Include detailed fields
# Level 2 (-vv): Include internal debug fields
_verbose_level = 0


def set_verbose_level(level: int):
    """Set the global verbose level for field serialization."""
    global _verbose_level
    _verbose_level = level


def get_verbose_level() -> int:
    """Get the current verbose level for field serialization."""
    global _verbose_level
    return _verbose_level


@dataclass
class TriageScript:
    name: str
    path: str
    config: ScriptConfig
    module: ModuleType
    run_method: Callable[..., Any]
    documentation: str
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
        except TimeoutDeviceRegisterError:
            raise
        except ValueError as e:
            if log_error:
                self.failed = True
                self.failure_message = f"{e}"
                return None
            else:
                raise
        except Exception as e:
            if log_error:
                self.failed = True
                self.failure_message = traceback.format_exc()
                return None
            else:
                raise

    @staticmethod
    def load(script_path: str) -> "TriageScript":
        script_path_obj = Path(script_path).resolve()
        script_path = str(script_path_obj)
        base_path = str(script_path_obj.parent)
        appended = False
        if base_path not in sys.path:
            sys.path.append(base_path)
            appended = True
        try:
            script_name = script_path_obj.stem
            script_module = importlib.import_module(script_name)

            # Check if script has a configuration
            script_config: ScriptConfig = script_module.script_config
            if script_config is None:
                # This script does not have a configuration, which means it is not tt-triage script, skipping...
                raise ValueError(f"Script {script_path} does not have script_config.")

            # Check if script has a docstring and an owner
            if not script_module.__doc__:
                raise ValueError(f"Script {script_path} must have a docstring, see relevant scripts for examples.\n")

            if not re.search(r"^Owner:\s*\S+", script_module.__doc__, re.MULTILINE):
                raise ValueError(
                    f"Script {script_path} docstring must include an 'Owner:' field with the corresponding owner of the script.\n"
                )

            # Check if script has a run method with two arguments (args and context)
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
                name=script_path_obj.name,
                path=script_path,
                config=deepcopy(script_config),
                module=script_module,
                run_method=run_method,
                documentation=script_module.__doc__,
            )

            if triage_script.config.depends is None:
                # If script does not have dependencies, set it to empty list
                triage_script.config.depends = []
            else:
                triage_script.config.depends = [
                    dep if isinstance(dep, str) and dep.endswith(".py") else f"{dep}.py"
                    for dep in triage_script.config.depends
                ]
                triage_script.config.depends = [str(Path(base_path) / dep) for dep in triage_script.config.depends]

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


def summarize_failure_message(message: str | None) -> str:
    """Extract a concise single-line summary from a traceback or error message.

    Uses a heuristic that scans lines in reverse and returns the first line that
    does not look like a Python traceback frame (i.e. does not start with
    "Traceback", "File ", "During handling of the above exception", or "^").

    Limitation: lines containing these substrings as part of the actual error
    message (e.g. ``RuntimeError: File descriptor exhausted``) will be skipped
    incorrectly. The function falls back to the last line if all lines match
    the filter.
    """
    if not message:
        return "No failure details available."

    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        return "No failure details available."

    for line in reversed(lines):
        if line.startswith("Traceback"):
            continue
        if line.startswith("File "):
            continue
        if line.startswith("During handling of the above exception"):
            continue
        if line == "^":
            continue
        return line
    return lines[-1]


def add_contextual_failure_hint(script_name: str | None, summary: str, message: str | None) -> str:
    """Append script-aware hints to otherwise generic exception summaries."""
    script_name = script_name or ""
    full_message = message or ""

    if (
        script_name == "check_arc.py"
        and "TypeError: 'NoneType' object is not iterable" in summary
        and "heartbeat_samples" in full_message
    ):
        return (
            f"{summary} (check_arc could not collect heartbeat samples; this usually means upstream device data "
            "collection failed. Check prior failures from run_checks.py / metal_device_id_mapping.py / inspector data.)"
        )

    return summary


def format_failure_message_lines(message: str | None, indent: str = "    ") -> list[str]:
    if not message:
        return [f"{indent}No additional details available."]
    return [f"{indent}{line}" if line.strip() else indent for line in message.splitlines()]


def get_failed_dependencies(script: "TriageScript") -> list["TriageScript"]:
    return [dep for dep in script.depends if dep.failed]


def build_dependency_failure_lines(script: "TriageScript") -> list[str]:
    failed_dependencies = get_failed_dependencies(script)
    if not failed_dependencies:
        return ["No failed dependencies were recorded."]

    lines = ["Failed dependencies:"]
    for failed_dep in failed_dependencies:
        summary = summarize_failure_message(failed_dep.failure_message)
        summary = add_contextual_failure_hint(failed_dep.name, summary, failed_dep.failure_message)
        lines.append(f"- {failed_dep.name}: {summary}")
    lines.append(
        "Action: fix dependency failures above. For inspector-related failures, verify --inspector-log-path or "
        "--inspector-rpc-host/--inspector-rpc-port and that TT_METAL_INSPECTOR=1 and TT_METAL_INSPECTOR_RPC=1. "
        "In multi-rank runs, ensure TT_METAL_LOGS_PATH contains <hostname>_rank_N directories and that "
        "the Inspector RPC port (base_port + rank) is correctly matched between C++ and triage tools."
    )
    return lines


def resolve_execution_order(scripts: dict[str, TriageScript]) -> list[TriageScript]:
    # Build script dependents graph and script missing dependencies map
    script_dependents = defaultdict(list)  # dep_path -> list of scripts depending on it
    script_missing_dependencies = defaultdict(int)  # script_path -> number of unmet dependencies

    for path, script in scripts.items():
        script_missing_dependencies[path] = len(script.config.depends)
        for dep in script.config.depends:
            script_dependents[dep].append(path)

    # Min-heap for runnable scripts: (-priority, script name, script object)
    # Negative priority because heapq is a min-heap
    heap = []

    # Initialize heap with scripts with in-degree 0 (no unmet dependencies)
    for path, script in scripts.items():
        if script_missing_dependencies[path] == 0:
            heapq.heappush(heap, (-script.config.priority.value, path, script))

    result = []

    while heap:
        # Pop the highest priority ready script
        _, path, script = heapq.heappop(heap)
        result.append(script)

        # Decrease in-degree of dependent scripts
        for dep_path in script_dependents[path]:
            script_missing_dependencies[dep_path] -= 1
            if script_missing_dependencies[dep_path] == 0:
                dep_script = scripts[dep_path]
                heapq.heappush(heap, (-dep_script.config.priority.value, dep_path, dep_script))

    # If some scripts remain with non-zero in-degree, we have a cycle
    if len(result) != len(scripts):
        remaining_scripts = set(scripts.keys()) - {s.config.name for s in result}
        raise ValueError(
            f"Bad dependency detected in scripts: {', '.join(remaining_scripts)}\n"
            f"  Circular dependency, dependency on disabled or non-existing script is not allowed.\n"
            f"  Please check if all dependencies are met and scripts are enabled."
        )

    return result


# Purposely uninitialized global console object to ensure proper initialization only once later
console: Console = None  # type: ignore[assignment]
progress_disabled: bool = False


def init_console_and_verbosity(args: ScriptArguments) -> None:
    global console
    global progress_disabled

    if console is not None:
        return

    # When redirecting to file, use a larger width to avoid wrapping.
    # When in a terminal, let Rich auto-detect the terminal width.
    # Similarly, if verbosity is increased, use larger width to avoid wrapping.
    width = None if sys.stdout.isatty() and _verbose_level == 0 else 10000
    console = Console(theme=utils.create_console_theme(args["--disable-colors"]), highlight=False, width=width)
    progress_disabled = bool(args["--disable-progress"])

    # Set verbose level from -v count (controls which columns are displayed)
    verbose_level = args["-v"] or 0
    set_verbose_level(verbose_level)

    # Setting verbosity level
    try:
        verbosity = int(args["--verbosity"])
        utils.Verbosity.set(verbosity)
    except:
        utils.WARN("Verbosity level must be an integer. Falling back to default value.")
    utils.VERBOSE(f"Verbosity level: {utils.Verbosity.get().name} ({utils.Verbosity.get().value})")


def create_progress() -> Progress:
    global console
    global progress_disabled

    return Progress(
        SpinnerColumn(),
        TimeElapsedColumn(),
        BarColumn(),
        TimeRemainingColumn(),
        TextColumn("[progress.tasks]{task.completed}/{task.total}[/] [progress.description]{task.description}[/]"),
        console=console,
        transient=True,
        disable=progress_disabled,
    )


def process_arguments(args: ScriptArguments) -> None:
    init_console_and_verbosity(args)


def parse_arguments(
    scripts: dict[str, TriageScript] = {},
    script_path: str | None = None,
    argv: list[str] | None = None,
    only_triage_script_args=False,
) -> ScriptArguments:
    from docopt import (
        parse_defaults,
        parse_pattern,
        formal_usage,
        printable_usage,
        parse_argv,
        Required,
        TokenStream,
        DocoptExit,
        Option,
        AnyOptions,
    )
    import sys

    docs: dict[str, str] = {}
    assert __doc__ is not None, "Help message must be provided in the script docstring."
    my_name = Path(__file__).stem
    docs[my_name] = __doc__
    for script in scripts.values():
        docs[script.name] = script.documentation

    combined_options = []
    combined_pattern: Required = Required(*[Required(*[])])

    for script_name, doc in docs.items():
        try:
            script_options = parse_defaults(doc)
            combined_options.extend(script_options)

            usage = printable_usage(doc)
            pattern = parse_pattern(formal_usage(usage), script_options)
            combined_pattern.children[0].children.extend(pattern.children[0].children)
        except BaseException as e:
            utils.ERROR(f"Error parsing arguments for script {script_name}: {e}")
            continue

    # Deduplicate options if some scripts define the same option
    seen_options: set[str | None] = set()
    unique_options: list[Option] = []
    for opt in combined_options:
        key = opt.long or opt.short
        if key not in seen_options:
            seen_options.add(key)
            unique_options.append(opt)

    if argv is None:
        argv = sys.argv[1:]
    parsed_argv = parse_argv(TokenStream(argv, DocoptExit), list(unique_options), options_first=False)
    pattern_options = set(combined_pattern.flat(Option))
    for ao in combined_pattern.flat(AnyOptions):
        ao.children = list(set(unique_options) - pattern_options)
    matched, left, collected = combined_pattern.fix().match(parsed_argv)
    if only_triage_script_args or (matched and left == []):
        arguments = ScriptArguments(dict((a.name, a.value) for a in (combined_pattern.flat() + collected)))
        process_arguments(arguments)
        return arguments

    detailed_help = any([a.name == "--help" or a.name == "-h" or a.name == "/?" for a in left])
    doc = __doc__ if script_path is None else scripts[script_path].documentation
    if detailed_help:
        help_message = doc
        if script_path is None:
            help_message += "\n\nYou can also use arguments of available scripts:\n"
        else:
            help_message += "\n\nYou can also use arguments of dependent scripts:\n"
        for script in scripts.values():
            if script.path != script_path:
                script_options = parse_defaults(script.documentation)
                if len(script_options) > 0:
                    help_message += f"\n{script.documentation}\n"
        print(help_message)
        sys.exit(0)
    else:
        help_message = printable_usage(doc)
        for script in scripts.values():
            if script.path != script_path:
                script_options = parse_defaults(script.documentation)
                if len(script_options) > 0:
                    usage = printable_usage(script.documentation)
                    help_message += " " + " ".join(usage.split()[2:])

    DocoptExit.usage = help_message
    raise DocoptExit()


FAILURE_CHECKS_LOCK = threading.Lock()
FAILURE_CHECKS: list[str] = []


def log_check(success: bool, message: str) -> None:
    global FAILURE_CHECKS, FAILURE_CHECKS_LOCK
    if not success:
        with FAILURE_CHECKS_LOCK:
            FAILURE_CHECKS.append(message)


def log_check_device(device: Device, success: bool, message: str) -> None:
    formatted_message = f"Device {device.id}: {message}"
    log_check(success, formatted_message)


def log_check_location(location: OnChipCoordinate, success: bool, message: str) -> None:
    device = location.device
    block_type = location.noc_block.block_type
    location_str = location.to_user_str()
    formatted_message = f"{block_type} [{location_str}]: {message}"
    log_check_device(device, success, formatted_message)


def log_check_risc(risc_name: str, location: OnChipCoordinate, success: bool, message: str) -> None:
    formatted_message = f"{risc_name}: {message}"
    log_check_location(location, success, formatted_message)


WARNING_CHECKS_LOCK = threading.Lock()
WARNING_CHECKS: list[str] = []


def log_warning(message: str) -> None:
    global WARNING_CHECKS, WARNING_CHECKS_LOCK
    with WARNING_CHECKS_LOCK:
        WARNING_CHECKS.append(message)


def log_warning_device(device: Device, message: str) -> None:
    log_warning(f"Device {device.id}: {message}")


def log_warning_location(location: OnChipCoordinate, message: str) -> None:
    device = location.device
    block_type = location.noc_block.block_type
    location_str = location.to_user_str()
    log_warning_device(device, f"{block_type} [{location_str}]: {message}")


def log_warning_risc(risc_name: str, location: OnChipCoordinate, message: str) -> None:
    log_warning_location(location, f"{risc_name}: {message}")


def serialize_result(script: TriageScript | None, result, execution_time: str = ""):
    from dataclasses import fields, is_dataclass

    if script is not None:
        print()
        utils.INFO(f"{script.name}{execution_time}:")

    global FAILURE_CHECKS, FAILURE_CHECKS_LOCK, WARNING_CHECKS, WARNING_CHECKS_LOCK
    with FAILURE_CHECKS_LOCK:
        failures = FAILURE_CHECKS
        FAILURE_CHECKS = []
    with WARNING_CHECKS_LOCK:
        warnings = WARNING_CHECKS
        WARNING_CHECKS = []
    if result is None:
        if len(failures) > 0 or script.failed:
            utils.ERROR("  fail")
            for failure in failures:
                utils.ERROR(f"    {failure}")
            if script.failed:
                summary = summarize_failure_message(script.failure_message)
                summary = add_contextual_failure_hint(script.name, summary, script.failure_message)
                utils.ERROR(f"    Summary: {summary}")
                if script.failure_message and script.failure_message.strip() != summary:
                    utils.ERROR("    Details:")
                    for detail_line in format_failure_message_lines(script.failure_message, indent="      "):
                        utils.ERROR(detail_line)

                import textwrap

                docstring_indented = textwrap.indent(script.documentation.strip(), "    ")
                utils.ERROR(f"  Script help:\n{docstring_indented}")
        else:
            utils.INFO("  pass")
            for warning in warnings:
                utils.WARN(f"    {warning}")
        return

    for failure in failures:
        utils.ERROR(f"  {failure}")

    for warning in warnings:
        utils.WARN(f"  {warning}")

    if isinstance(result, list) and len(result) == 0:
        utils.ERROR("  No results found.")

    if not (is_dataclass(result) or (isinstance(result, list) and all(is_dataclass(item) for item in result))):
        utils.INFO(f"  {result}")
    else:
        if not isinstance(result, list):
            result = [result]

        def generate_header(table: Table, obj, flds):
            for field in flds:
                metadata = field.metadata
                # Skip field if it requires higher verbosity level
                if metadata.get("verbose", 0) > _verbose_level:
                    continue
                if "dont_serialize" in metadata and metadata["dont_serialize"]:
                    continue
                elif "recurse" in metadata and metadata["recurse"]:
                    value = getattr(obj, field.name)
                    assert is_dataclass(value)
                    generate_header(table, value, fields(value))
                elif "serialized_name" in metadata:
                    justify = metadata.get("justify", "left")
                    table.add_column(metadata.get("serialized_name", field.name), justify=justify)

        def generate_row(row: list[str], obj, flds):
            for field in flds:
                metadata = field.metadata
                # Skip field if it requires higher verbosity level
                if metadata.get("verbose", 0) > _verbose_level:
                    continue
                if "dont_serialize" in metadata and metadata["dont_serialize"]:
                    continue
                elif "recurse" in metadata and metadata["recurse"]:
                    value = getattr(obj, field.name)
                    assert is_dataclass(value)
                    generate_row(row, value, fields(value))
                elif "additional_fields" in metadata:
                    assert all(hasattr(obj, additional_field) for additional_field in metadata["additional_fields"])
                    all_values = [getattr(obj, field.name)]
                    all_values.extend(
                        [getattr(obj, additional_field) for additional_field in metadata["additional_fields"]]
                    )
                    assert "serializer" in metadata, "Serializer must be provided for combined field."
                    row.append(metadata["serializer"](all_values))
                elif "serializer" in metadata:
                    row.append(metadata["serializer"](getattr(obj, field.name)))

        table = Table()

        # Create table header
        generate_header(table, result[0], fields(result[0]))
        for item in result:
            row: list[str] = []
            generate_row(row, item, fields(item))
            table.add_row(*row)
        console.print(table)


def _enforce_dependencies(args: ScriptArguments) -> None:
    """Enforce approved `ttexalens` version unless skipped.

    Reads the `tt-exalens` requirement from `requirements.txt` in the same
    directory as this script. Compares the specifier to the installed
    `tt-exalens` version and exits on mismatch unless `--skip-version-check`
    is provided. If the requirement is missing or the file is unreadable, a
    warning is printed and the check is skipped.
    """
    from packaging.requirements import Requirement
    from packaging.version import Version

    try:
        skip_check = bool(args["--skip-version-check"])
    except Exception:
        skip_check = False

    try:
        with open(_triage_requirements_path, "r", encoding="utf-8") as f:
            req_lines = f.read().splitlines()
    except FileNotFoundError:
        utils.WARN(
            f"requirements.txt not found. Skipping debugger version check. Expected at: {_triage_requirements_path}"
        )
        return
    except Exception as e:
        utils.WARN(f"Failed to read requirements.txt: {e}. Skipping debugger version check.")
        return

    # Find the tt-exalens requirement line
    tt_exalens_req = None
    for line in req_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            req = Requirement(line)
            if re.sub(r"[-_]", "-", req.name).lower() == "tt-exalens":
                tt_exalens_req = req
                break
        except Exception:
            continue

    if tt_exalens_req is None:
        utils.WARN(
            f"tt-exalens not found in requirements.txt ({_triage_requirements_path}). Skipping debugger version check."
        )
        return

    # Get installed version
    try:
        installed_version_str = importlib_metadata.version("tt-exalens")
        installed_version = Version(installed_version_str)
        utils.DEBUG(f"Installed tt-exalens version: {installed_version_str}")
    except importlib_metadata.PackageNotFoundError:
        pip_cmd = "uv pip" if shutil.which("uv") is not None else "pip"
        install_cmd = f"{pip_cmd} install -r {_triage_requirements_path}"
        utils.WARN(f"Required debugger component is not installed. Please run: {install_cmd}")
        console.print(f"Module 'tt-exalens' not found. Please install tt-exalens by running:")
        console.print(f"  [command]{install_cmd}[/]")
        exit(1)

    # Check if installed version satisfies the requirement
    if installed_version not in tt_exalens_req.specifier:
        pip_cmd = "uv pip" if shutil.which("uv") is not None else "pip"
        install_cmd = f"{pip_cmd} install -r {_triage_requirements_path}"
        message = f"Debugger version mismatch.\n  Installed: {installed_version_str}\n  Required:  {tt_exalens_req}"
        if skip_check:
            utils.WARN(message)
            utils.WARN("Proceeding due to --skip-version-check")
        else:
            console.print(message)
            console.print(f"Please install tt-exalens by running:")
            console.print(f"  [command]{install_cmd}[/]")
            console.print(f"Or disable this check by running with [command]--skip-version-check[/] argument.")
            exit(1)


def _patch_risc_debug() -> None:
    """
    Blackhole and Wormhole HW bug: HALT → READ/WRITE → CONTINUE breaks cores.
    This patches both risc_debug and debug_hardware instances so that
    cont() is a no-op and ensure_halted() only halts, never continues.

    Also patches halt() to record which cores triage halted, so that
    check_broken_components can verify they are still halted at the end.

    More info at tt-exalens:#908
    """

    from ttexalens.hardware.baby_risc_debug import BabyRiscDebugHardware
    from triage_session import get_triage_session

    def is_affected_by_cont_bug(device) -> bool:
        return device.is_wormhole() or device.is_blackhole()

    original_hw_cont = BabyRiscDebugHardware.cont
    original_hw_continue_without_debug = BabyRiscDebugHardware.continue_without_debug
    original_hw_halt = BabyRiscDebugHardware.halt

    BabyRiscDebugHardware.cont = lambda self: (
        None if is_affected_by_cont_bug(self.risc_info.noc_block.device) else original_hw_cont(self)
    )
    BabyRiscDebugHardware.continue_without_debug = lambda self: (
        None if is_affected_by_cont_bug(self.risc_info.noc_block.device) else original_hw_continue_without_debug(self)
    )

    def patched_halt(self):
        session = get_triage_session()
        location = self.risc_info.noc_block.location
        risc_name = self.risc_info.risc_name
        already_halted_by_triage = session.is_halted_core(location, risc_name)
        if not already_halted_by_triage:
            original_hw_halt(self)
            session.add_halted_core(location, risc_name)

    BabyRiscDebugHardware.halt = patched_halt


def _init_ttexalens(args: ScriptArguments) -> Context:
    """Initialize the ttexalens context."""
    if args["--remote-exalens"]:
        context = init_ttexalens_remote(ip_address=args["--remote-server"], port=args["--remote-port"])
    else:
        context = init_ttexalens(use_noc1=args["--initialize-with-noc1"])

    _patch_risc_debug()
    return context


def run_script(
    script_path: str | None = None,
    args: ScriptArguments | None = None,
    context: Context | None = None,
    argv: list[str] | None = None,
    return_result: bool = False,
) -> Any:
    force_exit = False

    # Resolve script path
    if script_path is None:
        # Check if previous call on callstack is a TriageScript
        stack = inspect.stack()
        if stack is None or len(stack) < 2:
            raise ValueError("No script path provided and no caller found in callstack.")
        script_path = stack[1].filename
        force_exit = True
    else:
        if not script_path.endswith(".py"):
            script_path = script_path + ".py"
        script_path_obj = Path(script_path)
        if not script_path_obj.is_absolute():
            application_path = Path(__file__).parent
            script_path_obj = application_path / script_path
        script_path_obj = script_path_obj.resolve()
        script_path = str(script_path_obj)
        if not script_path_obj.exists():
            raise FileNotFoundError(f"Script {script_path} does not exist.")

    # Load script and its dependencies
    scripts = TriageScript.load_all(script_path)

    # Find execution order of scripts
    script_queue = resolve_execution_order(scripts)

    # Parse arguments
    if args is None:
        args = parse_arguments(scripts, script_path, argv)

    # Initialize context if not provided
    if context is None:
        _enforce_dependencies(args)
        context = _init_ttexalens(args)

    # Run scripts in order
    result: Any = None
    for script in script_queue:
        if not all(not dep.failed for dep in script.depends):
            dependency_lines = build_dependency_failure_lines(script)
            dependency_message = "\n".join(dependency_lines)
            raise TTTriageError(f"{script.name}: Cannot run script due to failed dependencies.\n{dependency_message}")
        else:
            result = script.run(args=args, context=context, log_error=False)
            if script.config.data_provider and result is None:
                raise TTTriageError(f"{script.name}: Data provider script did not return any data.")
    script = scripts[script_path] if script_path in scripts else None
    if return_result:
        return result
    serialize_result(script, result)

    if force_exit:
        # Remove nanobind leak check to avoid false positives on exit
        os._exit(0)


class TTTriageError(Exception):
    """Base class for TT Triage errors."""

    pass


def _build_triage_summary(script_queue: list[TriageScript]) -> str:
    summary_lines = []
    for script in script_queue:
        if script.failed:
            summary_lines.append(f"{script.name}: FAIL - {script.failure_message or 'unknown error'}")
        elif not script.config.data_provider:
            summary_lines.append(f"{script.name}: pass")
    return "\n".join(summary_lines) if summary_lines else "No triage scripts executed."


def main():
    triage_start = time()

    # Parse only tt-triage script arguments first to initialize logging and console
    parse_arguments(only_triage_script_args=True)

    # Enumerate all scripts in application directory
    application_path = Path(__file__).resolve().parent
    this_file_name = Path(__file__).name
    script_files = [f.name for f in application_path.iterdir() if f.suffix == ".py" and f.name != this_file_name]

    # To avoid multiple imports of this script, we add it to sys.modules
    my_name = Path(__file__).stem
    if my_name not in sys.modules:
        sys.modules[my_name] = sys.modules["__main__"]

    # Load tt-triage scripts
    # TODO: do we need to check for subdirectories?
    scripts: dict[str, TriageScript] = {}
    for script in script_files:
        script_path = str(application_path / script)
        try:
            triage_script = TriageScript.load(script_path)
            if triage_script.config.disabled:
                utils.DEBUG(f"Script {script_path} is disabled, skipping...")
                continue
        except Exception as e:
            utils.DEBUG(f"Failed to load script {script_path}: {e}")
            continue
        scripts[script_path] = triage_script

    # Resolve dependencies
    for script in scripts.values():
        for dep in script.config.depends:
            if dep in scripts:
                script.depends.append(scripts[dep])
            else:
                utils.ERROR(f"Dependency {dep} for script {script.name} not found.")
                script.failed = True
                script.failure_message = f"Dependency {dep} not found."

    # Find dependency graph of script execution
    script_queue = resolve_execution_order(scripts)

    # Parse common command line arguments
    args = parse_arguments(scripts)

    # Enforce debugger dependencies, then initialize
    _enforce_dependencies(args)
    context = _init_ttexalens(args)

    with create_progress() as progress:
        scripts_task = progress.add_task("Script execution", total=len(script_queue))

        # Check if we should run specific scripts
        if args["--run"] is not None and (len(args["--run"]) != 1 or args["--run"][0] != "all"):
            progress.update(scripts_task, total=len(args["--run"]))
            for script_name in args["--run"]:
                progress.update(scripts_task, description=f"Running {script_name}")
                run_script(script_name, args, context)
                progress.advance(scripts_task)
        else:
            # Execute all scripts
            triage_init_end = time()
            if args["--print-script-times"]:
                utils.INFO(f"Triage initialization time: {triage_init_end - triage_start:.2f}s")
            total_time = triage_init_end - triage_start
            serialization_time = 0.0
            progress.update(scripts_task, total=len(script_queue))
            for script in script_queue:
                progress.update(scripts_task, description=f"Running {script.name}")
                if not all(not dep.failed for dep in script.depends):
                    dependency_lines = build_dependency_failure_lines(script)
                    utils.INFO(f"{script.name}:")
                    utils.WARN(f"  Cannot run script due to failed dependencies.")
                    for dependency_line in dependency_lines:
                        utils.WARN(f"  {dependency_line}")
                    script.failed = True
                    script.failure_message = "Cannot run script due to failed dependencies.\n" + "\n".join(
                        dependency_lines
                    )
                else:
                    start_time = time()
                    result = script.run(args=args, context=context)
                    end_time = time()
                    total_time += end_time - start_time
                    execution_time = f" [{end_time - start_time:.2f}s]" if args["--print-script-times"] else ""
                    if script.config.data_provider:
                        if result is None:
                            print()
                            utils.INFO(f"{script.name}{execution_time}:")
                            if script.failure_message is not None:
                                summary = summarize_failure_message(script.failure_message)
                                summary = add_contextual_failure_hint(script.name, summary, script.failure_message)
                                utils.ERROR(f"  Data provider script failed: {summary}")
                                if script.failure_message.strip() != summary:
                                    utils.ERROR("  Details:")
                                    for detail_line in format_failure_message_lines(script.failure_message, "    "):
                                        utils.ERROR(detail_line)
                            else:
                                utils.ERROR(f"  Data provider script did not return any data.")
                        elif execution_time:
                            print()
                            utils.INFO(f"{script.name}{execution_time}:")
                            utils.INFO("  pass")
                    else:
                        start_time = time()
                        serialize_result(script, result, execution_time)
                        end_time = time()
                        total_time += end_time - start_time
                        serialization_time += end_time - start_time
                progress.advance(scripts_task)
            if args["--print-script-times"]:
                print()
                utils.INFO(f"Total serialization time: {serialization_time:.2f}s")
                utils.INFO(f"Total execution time: {total_time:.2f}s")
        progress.remove_task(scripts_task)

    triage_summary_path = args["--triage-summary-path"]
    if triage_summary_path:
        try:
            Path(triage_summary_path).parent.mkdir(parents=True, exist_ok=True)
            Path(triage_summary_path).write_text(_build_triage_summary(script_queue))
            utils.INFO(f"Triage summary written to {triage_summary_path}")
        except Exception as e:
            utils.WARN(f"Failed to write triage summary: {e}")

    # Remove nanobind leak check to avoid false positives on exit
    os._exit(0)


if __name__ == "__main__":
    main()
