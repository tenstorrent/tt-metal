#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    triage [--initialize-with-noc1] [--remote-exalens] [--remote-server=<remote-server>] [--remote-port=<remote-port>] [--verbosity=<verbosity>] [--run=<script>]... [--skip-version-check]

Options:
    --remote-exalens                 Connect to remote exalens server.
    --remote-server=<remote-server>  Specify the remote server to connect to. [default: localhost]
    --remote-port=<remote-port>      Specify the remote server port. [default: 5555]
    --initialize-with-noc1           Initialize debugger context with NOC1 enabled. [default: False]
    --verbosity=<verbosity>          Choose output verbosity. 1: ERROR, 2: WARN, 3: INFO, 4: VERBOSE, 5: DEBUG. [default: 3]
    --run=<script>                   Run specific script(s) by name. If not provided, all scripts will be run. [default: all]
    --skip-version-check    Do not enforce debugger version check. [default: False]

Description:
    Diagnoses Tenstorrent AI hardware by performing comprehensive health checks on ARC processors, NOC connectivity, L1 memory, and RISC-V cores.
    Identifies running kernels and provides callstack information to troubleshoot failed operations.
    Example use with tt-metal:
        export TT_METAL_HOME=~/work/tt-metal
        ./build_metal.sh --build-programming-examples
        build/programming_examples/matmul_multi_core
        triage
"""

# Check if tt-exalens is installed
import inspect
import os
import utils
from collections.abc import Iterable

try:
    from ttexalens.tt_exalens_init import init_ttexalens, init_ttexalens_remote
except ImportError as e:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    install_script = os.path.join(os.path.dirname(script_dir), "install_debugger.sh")
    print(f"Module '{e}' not found. Please install tt-exalens by running:")
    print(f"  {utils.GREEN}{install_script}{utils.RST}")
    exit(1)

# Import necessary libraries
from copy import deepcopy
from dataclasses import dataclass, field
import importlib
import importlib.metadata as importlib_metadata
import sys
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.coordinate import OnChipCoordinate
from typing import Any, Callable, Iterable, TypeVar
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
        return str(value._id)
    elif isinstance(value, OnChipCoordinate):
        return value.to_user_str()
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


def triage_field(serialized_name: str | None = None, serializer: Callable[[Any], str] | None = None):
    if serializer is None:
        serializer = default_serializer
    return field(metadata={"recurse": False, "serialized_name": serialized_name, "serializer": serializer})


def combined_field(
    additional_fields: str | list[str] | None = None, serialized_name: str | None = None, serializer=None
):
    if additional_fields is None and serialized_name is None and serializer is None:
        return field(metadata={"recurse": False, "dont_serialize": True})
    assert (
        additional_fields is not None and serialized_name is not None
    ), "additional_fields and serialized_name must be provided."
    if serializer is None:
        serializer = default_serializer
    # TODO: If serializer accepts single value, it should be wrapped around method that converts arguments to list and passes them to serializer
    if not isinstance(additional_fields, list):
        additional_fields = [additional_fields]
    return field(
        metadata={
            "recurse": False,
            "additional_fields": additional_fields,
            "serialized_name": serialized_name,
            "serializer": serializer,
        }
    )


def recurse_field():
    return field(metadata={"recurse": True})


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
                f"Bad dependency detected in scripts: {', '.join(remaining_scripts)}\n"
                f"  Circular dependency, dependency on disabled or non-existing script is not allowed.\n"
                f"  Please check if all dependencies are met and scripts are enabled."
            )
    return script_queue


def parse_arguments(scripts: dict[str, TriageScript], script_path: str | None = None) -> ScriptArguments:
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
        if hasattr(script.module, "__doc__") and script.module.__doc__:
            docs[script.name] = script.module.__doc__

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

    argv = parse_argv(TokenStream(sys.argv[1:], DocoptExit), list(combined_options), options_first=False)
    pattern_options = set(combined_pattern.flat(Option))
    for ao in combined_pattern.flat(AnyOptions):
        ao.children = list(set(combined_options) - pattern_options)
    matched, left, collected = combined_pattern.fix().match(argv)
    if matched and left == []:
        return ScriptArguments(dict((a.name, a.value) for a in (combined_pattern.flat() + collected)))

    detailed_help = any([a.name == "--help" or a.name == "-h" or a.name == "/?" for a in left])
    doc = __doc__ if script_path is None else scripts[script_path].module.__doc__
    if doc is None:
        doc = __doc__
    if detailed_help:
        help_message = doc
        if script_path is None:
            help_message += "\n\nYou can also use arguments of available scripts:\n"
        else:
            help_message += "\n\nYou can also use arguments of dependent scripts:\n"
        for script in scripts.values():
            if script.path != script_path and hasattr(script.module, "__doc__") and script.module.__doc__:
                script_options = parse_defaults(script.module.__doc__)
                if len(script_options) > 0:
                    help_message += f"\n{script.module.__doc__}\n"
    else:
        help_message = printable_usage(doc)
        for script in scripts.values():
            if script.path != script_path and hasattr(script.module, "__doc__") and script.module.__doc__:
                script_options = parse_defaults(script.module.__doc__)
                if len(script_options) > 0:
                    usage = printable_usage(script.module.__doc__)
                    help_message += " " + " ".join(usage.split()[2:])

    DocoptExit.usage = help_message
    raise DocoptExit()


FAILURE_CHECKS: list[str] = []


def log_check(success: bool, message: str) -> None:
    global FAILURE_CHECKS
    if not success:
        FAILURE_CHECKS.append(message)


def serialize_result(script: TriageScript | None, result):
    from dataclasses import fields, is_dataclass

    if script is not None:
        print()
        utils.INFO(f"{script.name}:")

    global FAILURE_CHECKS
    failures = FAILURE_CHECKS
    FAILURE_CHECKS = []
    if result is None:
        if len(failures) > 0:
            utils.ERROR("  fail")
            for failure in failures:
                utils.ERROR(f"    {failure}")
        else:
            utils.INFO("  pass")
        return

    for failure in failures:
        utils.ERROR(f"  {failure}")

    if isinstance(result, list) and len(result) == 0:
        utils.ERROR("  No results found.")

    if not (is_dataclass(result) or (isinstance(result, list) and all(is_dataclass(item) for item in result))):
        utils.INFO(f"  {result}")
    else:
        if not isinstance(result, list):
            result = [result]

        def generate_header(header: list[str], obj, flds):
            for field in flds:
                metadata = field.metadata
                if "dont_serialize" in metadata and metadata["dont_serialize"]:
                    continue
                elif "recurse" in metadata and metadata["recurse"]:
                    value = getattr(obj, field.name)
                    assert is_dataclass(value)
                    generate_header(header, value, fields(value))
                elif "serialized_name" in metadata:
                    header.append(metadata["serialized_name"] if metadata["serialized_name"] else field.name)

        def generate_row(row: list[str], obj, flds):
            for field in flds:
                metadata = field.metadata
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
                else:
                    row.append(metadata["serializer"](getattr(obj, field.name)))

        # Create table header
        header = []
        generate_header(header, result[0], fields(result[0]))
        table = [header]
        for item in result:
            row = []
            generate_row(row, item, fields(item))
            multilined_row = [r.splitlines() for r in row]
            multirow = max([len(r) for r in multilined_row])
            if multirow == 1:
                table.append(row)
            else:
                # If multirow, add empty rows for each line in the row
                for i in range(multirow):
                    multirow_row = []
                    for lines in multilined_row:
                        if i < len(lines):
                            multirow_row.append(lines[i])
                        else:
                            multirow_row.append("")
                    table.append(multirow_row)

        from tabulate import tabulate
        from utils import DEFAULT_TABLE_FORMAT

        print(tabulate(table, headers="firstrow", tablefmt=DEFAULT_TABLE_FORMAT))


def _enforce_dependencies(args: ScriptArguments) -> None:
    """Enforce approved `ttexalens` version unless skipped.

    Reads a single-line SHA from `ttexalens_ref.txt` in the parent
    directory of this script (next to `install_debugger.sh`). Compares it to the
    installed `ttexalens` version's dev hash and raises `TTTriageError` on
    mismatch unless `--skip-version-check` is provided. If the ref file
    is missing or empty, a warning is printed and the check is skipped.
    """
    # Skip flag for dependency checks
    try:
        skip_check = bool(args.get("--skip-version-check", False)) if args else False
    except Exception:
        skip_check = False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.abspath(os.path.join(os.path.dirname(script_dir), "ttexalens_ref.txt"))

    try:
        with open(ref_path, "r", encoding="utf-8") as f:
            approved_ref = f.read().strip()
    except FileNotFoundError:
        utils.WARN("ttexalens_ref.txt not found. Skipping debugger version check. " f"Expected at: {ref_path}")
        return
    except Exception as e:
        utils.WARN(f"Failed to read ttexalens_ref.txt: {e}. Skipping debugger version check.")
        return

    if not approved_ref:
        utils.WARN("ttexalens_ref.txt is empty. Skipping debugger version check.")
        return

    # Get installed version string
    try:
        installed_version = importlib_metadata.version("ttexalens")
        utils.DEBUG(f"Installed ttexalens version: {installed_version}")
    except importlib_metadata.PackageNotFoundError:
        utils.WARN(
            "Required debugger component is not installed. Please run scripts/install_debugger.sh to install debugger dependencies."
        )
        raise TTTriageError("Debugger dependency is not installed")

    # Expected version format from setup.py: 0.1.<date>+dev.<short_hash>
    installed_hash: str | None = None
    if "+dev." in installed_version:
        try:
            installed_hash = installed_version.split("+dev.", 1)[1]
        except Exception:
            installed_hash = None

    expected_hash: str | None = approved_ref

    # Match by prefix to allow short-vs-long
    match_ok = False
    if installed_hash and expected_hash:
        if expected_hash.startswith(installed_hash) or installed_hash.startswith(expected_hash):
            match_ok = True

    if not match_ok:
        message = (
            "Debugger version mismatch.\n"
            f"  Installed: {installed_version} (hash: {installed_hash or 'unknown'})\n"
            f"  Approved:  hash: {approved_ref}\n"
            "Use scripts/install_debugger.sh to install the approved version, or run with --skip-version-check"
        )
        if skip_check:
            utils.WARN(message)
            utils.WARN("Proceeding due to --skip-version-check")
        else:
            raise TTTriageError(message)


def _init_ttexalens(args: ScriptArguments | None = None) -> Context | None:
    """Initialize the ttexalens context."""
    if args is None:
        return None
    if args["--remote-exalens"]:
        return init_ttexalens_remote(ip_address=args["--remote-server"], port=args["--remote-port"])
    return init_ttexalens(use_noc1=args["--initialize-with-noc1"])


def run_script(
    script_path: str | None = None, args: ScriptArguments | None = None, context: Context | None = None
) -> Any:
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
        args = parse_arguments(scripts, script_path)

        # Setting verbosity level
        try:
            verbosity = int(args["--verbosity"])
            utils.Verbosity.set(verbosity)
        except:
            utils.WARN("Verbosity level must be an integer. Falling back to default value.")
        utils.VERBOSE(f"Verbosity level: {utils.Verbosity.get().name} ({utils.Verbosity.get().value})")

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
    serialize_result(script, result)


class TTTriageError(Exception):
    """Base class for TT Triage errors."""

    pass


def main():
    # Enumerate all scripts in application directory
    application_path = os.path.dirname(__file__)
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

    # Setting verbosity level
    try:
        verbosity = int(args["--verbosity"])
        utils.Verbosity.set(verbosity)
    except:
        utils.WARN("Verbosity level must be an integer. Falling back to default value.")
    utils.VERBOSE(f"Verbosity level: {utils.Verbosity.get().name} ({utils.Verbosity.get().value})")

    # Enforce debugger dependencies, then initialize
    _enforce_dependencies(args)
    context = _init_ttexalens(args)

    # Check if we should run specific scripts
    if args["--run"] is not None and (len(args["--run"]) != 1 or args["--run"][0] != "all"):
        for script_name in args["--run"]:
            run_script(script_name, args, context)
    else:
        # Execute all scripts
        for script in script_queue:
            if not all(not dep.failed for dep in script.depends):
                utils.INFO(f"{script.name}:")
                utils.WARN(f"  Cannot run script due to failed dependencies.")
                script.failed = True
                script.failure_message = "Cannot run script due to failed dependencies."
            else:
                result = script.run(args=args, context=context)
                if script.config.data_provider and result is None:
                    utils.INFO(f"{script.name}:")
                    if script.failure_message is not None:
                        utils.ERROR(f"  Data provider script failed: {script.failure_message}")
                    else:
                        utils.ERROR(f"  Data provider script did not return any data.")
                if not script.config.data_provider:
                    serialize_result(script, result)


if __name__ == "__main__":
    main()
