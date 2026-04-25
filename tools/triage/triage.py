#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    triage [--initialize-with-noc1] [--remote-exalens] [--remote-server=<remote-server>] [--remote-port=<remote-port>] [--verbosity=<verbosity>] [--run=<script>]... [--skip-version-check] [--print-script-times] [-v ...] [--disable-colors] [--disable-progress] [--disable-elf-cache] [--triage-summary-path=<path>]

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
    --disable-elf-cache              Re-parse ELF files on every access instead of caching. [default: False]
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
    from mpi4py import MPI
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
                name=os.path.basename(script_path),
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
    my_name = os.path.splitext(os.path.basename(__file__))[0]
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
    # Triage scripts invoked from the test harness use run_script(..., return_result=True),
    # which skips serialize_result() — the only place WARNING_CHECKS is otherwise flushed.
    # Without this immediate print, every log_warning_* on that path is silently dropped,
    # which is exactly why TT_TRIAGE_VERBOSE_SKIPS=1 produced no output in CI.
    #
    # We mirror to stdout (not stderr) on purpose: stderr is asserted-empty by
    # tools/tests/triage/test_triage.py::test_triage_executes_no_errors, which is run
    # in CI with TT_TRIAGE_VERBOSE_SKIPS=1. Verbose skip warnings are informational
    # ("this DRAM has no drisc", "this worker is DONE", "halt didn't take, retried")
    # and must not be conflated with hard errors. Both stdout and stderr are captured
    # by `pytest -s -v` and end up in the build log, so the diagnostic value motivating
    # TT_TRIAGE_VERBOSE_SKIPS in the workflow yaml is preserved either way.
    if verbose_skips_enabled():
        import sys

        print(f"[triage warning] {message}", file=sys.stdout, flush=True)


def log_warning_device(device: Device, message: str) -> None:
    log_warning(f"Device {device.id}: {message}")


def log_warning_location(location: OnChipCoordinate, message: str) -> None:
    device = location.device
    block_type = location.noc_block.block_type
    location_str = location.to_user_str()
    log_warning_device(device, f"{block_type} [{location_str}]: {message}")


def log_warning_risc(risc_name: str, location: OnChipCoordinate, message: str) -> None:
    log_warning_location(location, f"{risc_name}: {message}")


def verbose_skips_enabled() -> bool:
    """Triage per-core skip-diagnostic verbosity toggle.

    Enabled via TT_TRIAGE_VERBOSE_SKIPS=1. Off by default because a healthy chip
    produces dozens of "boring" skip reasons per run (DONE cores, cores not in an
    ebreak/spin loop, etc.). Off → we still log the *interesting* skips (failed
    halts, unexpected exceptions, missing kernel metadata on a running core).
    On → we log every single skip with its reason + full exception tracebacks,
    so CI logs on a hung chip tell us exactly which core we bailed on and why.
    """
    return os.environ.get("TT_TRIAGE_VERBOSE_SKIPS", "0") not in ("", "0", "false", "False")


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
                utils.ERROR(f"    {script.failure_message}")

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


def _dump_rt_profiler_heartbeat(device, emit) -> None:
    """
    Best-effort probe: on halt failure, find workers whose NOC master counters have
    STOPPED advancing. That is the signature of a wedged core waiting on a NOC
    transaction that will never come back — in particular, the RT profiler NCRISC
    stuck inside socket_reserve_pages() when the host D2H receiver is dead.

    Healthy idle workers keep moving counters at a low rate, so a two-sample probe
    across a short delay is plenty to tell "stalled" from "busy". We specifically do
    NOT use is_halted() as the filter because most workers on a live-but-hung chip are
    still running firmware; is_halted() comes back false for everyone and is noise.
    """
    import time as _time

    try:
        block_locations = device.get_block_locations()
    except Exception as e:  # noqa: BLE001
        emit(f"[halt-diag]   rt-probe: get_block_locations failed: {type(e).__name__}: {e}")
        return

    # Collect register stores for each worker-like block on both NOCs.
    probes = []  # list of (loc, noc_id, register_store)
    for loc in block_locations:
        try:
            noc_block = device.get_block(loc)
        except Exception:
            continue
        block_name = type(noc_block).__name__.lower()
        if "worker" not in block_name and "tensix" not in block_name:
            continue
        for noc_id in (0, 1):
            try:
                rs = noc_block.get_register_store(noc_id=noc_id)
            except Exception:
                continue
            probes.append((loc, noc_id, rs))

    _COUNTERS = ("NIU_MST_CMD_ACCEPTED", "NIU_MST_WR_ACK_RECEIVED", "NIU_MST_RD_RESP_RECEIVED")

    def _read_all():
        out = []
        for loc, noc_id, rs in probes:
            vals = []
            for reg in _COUNTERS:
                try:
                    vals.append(rs.read_register(reg))
                except Exception:
                    vals.append(None)
            # Also read outstanding ID bucket for human context.
            try:
                outst = rs.read_register("NIU_MST_REQS_OUTSTANDING_ID")
            except Exception:
                outst = None
            out.append((loc, noc_id, vals, outst))
        return out

    sample1 = _read_all()
    _time.sleep(0.02)  # 20ms is enough for a healthy worker to tick counters
    sample2 = _read_all()

    reported = 0
    for (loc1, noc1, v1, o1), (loc2, noc2, v2, o2) in zip(sample1, sample2):
        assert loc1 == loc2 and noc1 == noc2
        # Skip entries where any read failed.
        if any(x is None for x in v1) or any(x is None for x in v2):
            continue
        outstanding_accepted_not_acked = v2[0] - v2[1] - v2[2]
        # Stalled := NOC master counters didn't move at all across 20ms AND the core
        # has accepted more commands than it has seen responses for (i.e. there's
        # something outstanding). This fires cleanly on a core wedged in
        # socket_reserve_pages / noc_async_write_barrier and should NOT fire on
        # idle-firmware cores that keep polling over NOC.
        stalled = v1 == v2
        has_outstanding = outstanding_accepted_not_acked > 0 or (o2 is not None and o2 > 0)
        if stalled and has_outstanding:
            reported += 1
            emit(
                f"[halt-diag]   stalled-core: {loc2} NOC{noc2} "
                f"accepted={v2[0]} wr_ack={v2[1]} rd_resp={v2[2]} "
                f"outstanding_id={o2} accepted-acked={outstanding_accepted_not_acked}"
            )
            if reported >= 16:
                emit("[halt-diag]   rt-probe: truncating after 16 stalled cores")
                return

    if reported == 0:
        emit("[halt-diag]   rt-probe: no worker cores with stalled NOC masters found")


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
    from ttexalens.hardware.risc_debug import RiscHaltError
    from triage_session import get_triage_session

    def is_affected_by_cont_bug(device) -> bool:
        return device.is_wormhole() or device.is_blackhole()

    original_hw_cont = BabyRiscDebugHardware.cont
    original_hw_continue_without_debug = BabyRiscDebugHardware.continue_without_debug
    original_hw_halt = BabyRiscDebugHardware.halt

    BabyRiscDebugHardware.cont = (
        lambda self: None if is_affected_by_cont_bug(self.risc_info.noc_block.device) else original_hw_cont(self)
    )
    BabyRiscDebugHardware.continue_without_debug = (
        lambda self: None
        if is_affected_by_cont_bug(self.risc_info.noc_block.device)
        else original_hw_continue_without_debug(self)
    )

    # Registers queried when a halt fails. Split into master/slave and a short label so the
    # log is compact. We read these on both NOC0 and NOC1 for the failing core. The key
    # invariant is: after halt fails, if MST_CMD_ACCEPTED - (MST_WR_ACK_RECEIVED +
    # MST_RD_RESP_RECEIVED + MST_ATOMIC_RESP_RECEIVED) > 0, the core has outstanding NOC
    # master transactions and the hardware halt command is likely blocked behind them.
    _HALT_DIAG_NIU_REGS = (
        ("MST_CMD_ACCEPTED", "NIU_MST_CMD_ACCEPTED"),
        ("MST_WR_ACK_RECEIVED", "NIU_MST_WR_ACK_RECEIVED"),
        ("MST_RD_RESP_RECEIVED", "NIU_MST_RD_RESP_RECEIVED"),
        ("MST_ATOMIC_RESP_RECEIVED", "NIU_MST_ATOMIC_RESP_RECEIVED"),
        ("MST_REQS_OUTSTANDING_ID", "NIU_MST_REQS_OUTSTANDING_ID"),
        ("MST_WRITE_REQS_OUTGOING_ID", "NIU_MST_WRITE_REQS_OUTGOING_ID"),
        ("SLV_REQ_ACCEPTED", "NIU_SLV_REQ_ACCEPTED"),
        ("SLV_WR_ACK_SENT", "NIU_SLV_WR_ACK_SENT"),
        ("SLV_RD_RESP_SENT", "NIU_SLV_RD_RESP_SENT"),
    )

    def _safe_read_register(register_store, reg_name: str):
        try:
            return register_store.read_register(reg_name)
        except Exception as e:  # noqa: BLE001
            return f"<read failed: {type(e).__name__}: {e}>"

    def _format_niu_counters(noc_block, noc_id: int) -> str:
        try:
            rs = noc_block.get_register_store(noc_id=noc_id)
        except Exception as e:  # noqa: BLE001
            return f"NOC{noc_id}: <register store unavailable: {type(e).__name__}: {e}>"
        parts = [f"NOC{noc_id}:"]
        for label, reg in _HALT_DIAG_NIU_REGS:
            val = _safe_read_register(rs, reg)
            if isinstance(val, int):
                parts.append(f"  {label}=0x{val:08x} ({val})")
            else:
                parts.append(f"  {label}={val}")
        return "\n".join(parts)

    def _dump_halt_failure_diagnostics(self, exc: Exception) -> None:
        """
        Called from patched_halt() when the underlying ttexalens halt failed. Dumps
        enough state to tell the "RT profiler NCRISC wedged on dead D2H socket and the
        dispatch core NOC is choked" story apart from other hang modes. Safe to call
        under the broken condition: every read is try/except'd and no write is issued.
        """
        location = self.risc_info.noc_block.location
        risc_name = self.risc_info.risc_name
        device = self.risc_info.noc_block.device

        lines = [f"[halt-diag] halt failed on {risc_name} @ {location} (device {device._id})"]
        lines.append(f"[halt-diag]   reason: {type(exc).__name__}: {exc}")

        # 1. Status register: decoded (halted / watchpoint / ebreak). We avoid reaching
        #    through the name-mangled raw read so the diagnostic doesn't itself depend on
        #    internal ttexalens attribute naming.
        try:
            status = self.read_status()
            lines.append(
                "[halt-diag]   status: halted={0} pc_wp={1} mem_wp={2} ebreak={3} wp_hits={4}".format(
                    status.is_halted,
                    status.is_pc_watchpoint_hit,
                    status.is_memory_watchpoint_hit,
                    status.is_ebreak_hit,
                    list(getattr(status, "watchpoints_hit", []) or []),
                )
            )
        except Exception as e:  # noqa: BLE001
            lines.append(f"[halt-diag]   status: <read failed: {type(e).__name__}: {e}>")

        # 2. Reset state.
        try:
            in_reset = self.is_in_reset()
        except Exception as e:  # noqa: BLE001
            in_reset = f"<read failed: {type(e).__name__}: {e}>"
        lines.append(f"[halt-diag]   is_in_reset={in_reset}")

        # 3. NIU counters on NOC0 + NOC1. Outstanding transactions here are the single
        #    most useful signal we have: if master counters show mismatched accepts vs
        #    responses, the core is stuck waiting on a NOC round-trip.
        noc_block = self.risc_info.noc_block
        for noc_id in (0, 1):
            try:
                lines.append(
                    "[halt-diag]   " + _format_niu_counters(noc_block, noc_id).replace("\n", "\n[halt-diag]   ")
                )
            except Exception as e:  # noqa: BLE001
                lines.append(f"[halt-diag]   NOC{noc_id}: <niu read failed: {type(e).__name__}: {e}>")

        # 4. RT-profiler heartbeat probe — full-chip stall detection. Off by default
        #    because it produces false positives on devices that just happen to be idle
        #    (counters don't move ≠ core is wedged, if there was no real work in flight).
        #    Enable with TT_TRIAGE_HALT_RT_PROBE=1 when investigating a specific hang.
        if os.environ.get("TT_TRIAGE_HALT_RT_PROBE") == "1":
            try:
                _dump_rt_profiler_heartbeat(device, lines.append)
            except Exception as e:  # noqa: BLE001
                lines.append(f"[halt-diag]   rt-probe failed: {type(e).__name__}: {e}")

        for line in lines:
            utils.WARN(line)

    # Number of times we retry the raw debug-NIU HALT command before giving up.
    # On Blackhole/Wormhole, halt occasionally needs to be re-issued because the
    # HW transiently drops halt when the core has outstanding NOC transactions
    # (same root cause as the "HALT -> READ/WRITE -> CONTINUE breaks cores" bug).
    # Each attempt is one NIU write + one status read, so this is cheap.
    _HALT_MAX_RETRIES = 5

    def _sticky_halt(hw) -> bool:
        """
        Issue HALT and verify it took, retrying a few times. Returns True if the
        core ended up halted, False otherwise. Does not raise. Used both on the
        first halt and to silently re-arm halt when reads/writes observe the
        core has spuriously fallen out of halt.
        """
        for _ in range(_HALT_MAX_RETRIES):
            if hw.is_halted():
                return True
            hw._halt_command()
        return hw.is_halted()

    def patched_halt(self):
        session = get_triage_session()
        location = self.risc_info.noc_block.location
        risc_name = self.risc_info.risc_name
        if session.is_halted_core(location, risc_name):
            # Already accounted as halted-by-triage. Re-arm halt in case HW
            # dropped it between the original halt and now -- callers expect a
            # halted core after halt() returns.
            _sticky_halt(self)
            return
        if not _sticky_halt(self):
            exc = RiscHaltError(risc_name, location)
            try:
                _dump_halt_failure_diagnostics(self, exc)
            except Exception as diag_exc:  # noqa: BLE001
                utils.WARN(
                    f"[halt-diag] diagnostic dump itself raised "
                    f"{type(diag_exc).__name__}: {diag_exc}; re-raising original halt error"
                )
            raise exc
        session.add_halted_core(location, risc_name)

    BabyRiscDebugHardware.halt = patched_halt

    # Patch read_memory / write_memory to silently re-halt when the HW has
    # spuriously dropped halt since the enclosing ensure_halted() context began,
    # AND to suppress the inner assert_halted() check inside the original
    # read/write. The assert is racy on Blackhole: even after we succeed at
    # re-halting, the few NOC ops between our re-halt and the assert give the
    # HW another chance to drop halt, surfacing as
    # "Skipping (ValueError): brisc is not halted". The debug NIU still reads
    # back valid bytes from L1/private memory regardless of the core's halt
    # state, so for triage purposes (snapshotting state on a hung chip) the
    # assert is over-strict. Only apply this on architectures exhibiting the
    # cont() bug; others fall through unchanged.
    original_hw_read_memory = BabyRiscDebugHardware.read_memory
    original_hw_write_memory = BabyRiscDebugHardware.write_memory

    def _read_with_assert_suppressed(hw, addr):
        # original_hw_read_memory does:
        #   if self.enable_asserts: self.assert_halted()
        #   ... NIU reads ...
        # Toggle enable_asserts off for the duration of the call so the
        # assert_halted does not raise ValueError mid-loop while we are doing
        # our best to keep the core halted around it.
        saved = hw.enable_asserts
        try:
            hw.enable_asserts = False
            return original_hw_read_memory(hw, addr)
        finally:
            hw.enable_asserts = saved

    def _write_with_assert_suppressed(hw, addr, value):
        saved = hw.enable_asserts
        try:
            hw.enable_asserts = False
            return original_hw_write_memory(hw, addr, value)
        finally:
            hw.enable_asserts = saved

    def patched_read_memory(self, addr):
        if not is_affected_by_cont_bug(self.risc_info.noc_block.device):
            return original_hw_read_memory(self, addr)
        if not self.is_halted():
            _sticky_halt(self)
        return _read_with_assert_suppressed(self, addr)

    def patched_write_memory(self, addr, value):
        if not is_affected_by_cont_bug(self.risc_info.noc_block.device):
            return original_hw_write_memory(self, addr, value)
        if not self.is_halted():
            _sticky_halt(self)
        return _write_with_assert_suppressed(self, addr, value)

    BabyRiscDebugHardware.read_memory = patched_read_memory
    BabyRiscDebugHardware.write_memory = patched_write_memory


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
        args = parse_arguments(scripts, script_path, argv)

    # Initialize context if not provided
    if context is None:
        _enforce_dependencies(args)
        context = _init_ttexalens(args)

    # Run scripts in order
    result: Any = None
    for script in script_queue:
        if not all(not dep.failed for dep in script.depends):
            raise TTTriageError(f"{script.name}: Cannot run script due to failed dependencies.")
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
    application_path = os.path.abspath(os.path.dirname(__file__))
    script_files = [f for f in os.listdir(application_path) if f.endswith(".py") and f != os.path.basename(__file__)]

    # To avoid multiple imports of this script, we add it to sys.modules
    my_name = os.path.splitext(os.path.basename(__file__))[0]
    if my_name not in sys.modules:
        sys.modules[my_name] = sys.modules["__main__"]

    # Load tt-triage scripts
    # TODO: do we need to check for subdirectories?
    scripts: dict[str, TriageScript] = {}
    base_path = application_path
    for script in script_files:
        script_path = os.path.join(base_path, script)
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
                    utils.INFO(f"{script.name}:")
                    utils.WARN(f"  Cannot run script due to failed dependencies.")
                    script.failed = True
                    script.failure_message = "Cannot run script due to failed dependencies."
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
                                utils.ERROR(f"  Data provider script failed: {script.failure_message}")
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
            os.makedirs(os.path.dirname(triage_summary_path), exist_ok=True)
            with open(triage_summary_path, "w") as f:
                f.write(_build_triage_summary(script_queue))
            utils.INFO(f"Triage summary written to {triage_summary_path}")
        except Exception as e:
            utils.WARN(f"Failed to write triage summary: {e}")

    from elfs_cache import run as get_elfs_cache

    get_elfs_cache(args, context).log_stats()

    # Remove nanobind leak check to avoid false positives on exit
    os._exit(0)


if __name__ == "__main__":
    main()
