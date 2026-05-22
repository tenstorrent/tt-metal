#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    triage [--initialize-with-noc1] [--remote-exalens] [--remote-server=<remote-server>] [--remote-port=<remote-port>] [--verbosity=<verbosity>] [--run=<script>]... [--skip-version-check] [--print-script-times] [-v ...] [--disable-colors] [--disable-progress] [--disable-elf-cache] [--triage-summary-path=<path>] [--llm-output] [--llm-output-path=<path>]

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
    --llm-output                     Replace Rich tables on the console with a machine-readable report (CSV-formatted tables). Easier and cheaper for LLMs (and grep/CI) to consume. Implies --disable-colors.
    --llm-output-path=<path>         Additionally write the machine-readable report to <path>. Can be combined with --llm-output; without it, Rich output still goes to the console.

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
import inspect
import os
import shutil
import threading
from time import time
import utils
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
import importlib.metadata as importlib_metadata
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn
import sys
from ttexalens.context import Context
from ttexalens.device import Device
from ttexalens.coordinate import OnChipCoordinate
from typing import Any

# Script-API surface lives in triage_script.py; re-export so existing scripts
# can keep `from triage import ScriptConfig, triage_field, ...`.
from triage_script import (
    ScriptArguments,
    ScriptConfig,
    ScriptPriority,
    TriageScript,
    TTTriageError,
    collection_serializer,
    default_serializer,
    hex_serializer,
    recurse_field,
    serialize_collection,
    triage_field,
    triage_singleton,
)
from orchestrator import ScriptOrchestrator
from process_group import get_process_group, make_process_group
from runner import create_runner


def _raise_open_file_limit(desired: int = 65536) -> None:
    """
    Raise the open file limit for the current process to the desired value if possible.
    This is necessary to avoid hitting the open file limit when processing many ELF files with the elf cache enabled.
    If the file limit is already at or above the desired value, this function does nothing.
    """
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = desired if hard == resource.RLIM_INFINITY else min(desired, hard)
        if new_soft <= soft:
            return
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (ImportError, OSError, ValueError) as e:
        utils.WARN(
            f"Failed to raise open file limit: {e}. This may cause issues when processing many ELF files. Consider increasing the limit manually (ulimit -n {desired})."
        )


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


# Purposely uninitialized global console object to ensure proper initialization only once later
console: Console = None  # type: ignore[assignment]
progress_disabled: bool = False


def init_console_and_verbosity(args: ScriptArguments) -> None:
    global console
    global progress_disabled

    if console is not None:
        return

    disable_colors = bool(args["--disable-colors"]) or bool(args["--llm-output"])
    # When redirecting to file, use a larger width to avoid wrapping.
    # When in a terminal, let Rich auto-detect the terminal width.
    width = None if sys.stdout.isatty() and _verbose_level == 0 else 10000
    console = Console(theme=utils.create_console_theme(disable_colors), highlight=False, width=width)

    pg = get_process_group()
    progress_disabled = pg.is_multi or bool(args["--disable-progress"]) or bool(args["--llm-output"])

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
    _raise_open_file_limit()


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


def log_warning_device(device: Device, message: str) -> None:
    log_warning(f"Device {device.id}: {message}")


def log_warning_location(location: OnChipCoordinate, message: str) -> None:
    device = location.device
    block_type = location.noc_block.block_type
    location_str = location.to_user_str()
    log_warning_device(device, f"{block_type} [{location_str}]: {message}")


def log_warning_risc(risc_name: str, location: OnChipCoordinate, message: str) -> None:
    log_warning_location(location, f"{risc_name}: {message}")


_output_serializer: Any = None


def get_output_serializer() -> Any:
    """Return the active serializer, defaulting on first use (Null on non-root, Rich otherwise)."""
    global _output_serializer
    if _output_serializer is None:
        if not get_process_group().is_root:
            from serializers import NullSerializer

            _output_serializer = NullSerializer()
        else:
            from serializers import RichSerializer

            _output_serializer = RichSerializer(console, utils, get_verbose_level)
    return _output_serializer


def set_output_serializer(serializer: Any) -> None:
    global _output_serializer
    _output_serializer = serializer


def init_output_serializer(args: ScriptArguments) -> None:
    """Build the active serializer(s) based on CLI args.

    Combinations:
      neither flag                      -> RichSerializer on the console
      --llm-output                      -> CsvSerializer on the console (replaces Rich)
      --llm-output-path=<path>          -> Rich on console + CsvSerializer to file
      --llm-output --llm-output-path=.. -> CsvSerializer on console + CsvSerializer to file
    """
    from serializers import ConsoleSink, CsvSerializer, FileSink, MultiSerializer, NullSerializer, RichSerializer

    if not get_process_group().is_root:
        set_output_serializer(NullSerializer())
        return

    console_sink = ConsoleSink(console)
    serializers: list[Any] = []

    if args["--llm-output"]:
        serializers.append(CsvSerializer(console_sink, get_verbose_level))
    else:
        serializers.append(RichSerializer(console_sink, utils, get_verbose_level))

    csv_path = args["--llm-output-path"]
    if csv_path:
        try:
            file_sink = FileSink(csv_path)
            serializers.append(CsvSerializer(file_sink, get_verbose_level))
        except OSError as e:
            utils.WARN(f"Failed to open --llm-output-path={csv_path!r}: {e}. File output will be skipped.")

    set_output_serializer(serializers[0] if len(serializers) == 1 else MultiSerializer(serializers))


def serialize_result(script: TriageScript | None, result, execution_time: str = ""):
    global FAILURE_CHECKS, FAILURE_CHECKS_LOCK, WARNING_CHECKS, WARNING_CHECKS_LOCK
    with FAILURE_CHECKS_LOCK:
        failures = FAILURE_CHECKS
        FAILURE_CHECKS = []
    with WARNING_CHECKS_LOCK:
        warnings = WARNING_CHECKS
        WARNING_CHECKS = []

    get_output_serializer().emit(
        script_name=script.name if script is not None else None,
        execution_time=execution_time,
        result=result,
        failures=failures,
        warnings=warnings,
        script_failed=script.failed if script is not None else False,
        failure_message=script.failure_message if script is not None else None,
        documentation=script.documentation if script is not None else None,
    )


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

    BabyRiscDebugHardware.cont = (
        lambda self: None if is_affected_by_cont_bug(self.risc_info.noc_block.device) else original_hw_cont(self)
    )
    BabyRiscDebugHardware.continue_without_debug = (
        lambda self: None
        if is_affected_by_cont_bug(self.risc_info.noc_block.device)
        else original_hw_continue_without_debug(self)
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
    force_exit = script_path is None
    pg = make_process_group()

    try:
        if script_path is None:
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

        orchestrator = ScriptOrchestrator(os.path.dirname(script_path))
        scripts, script_queue = orchestrator.queue_for_target(script_path)

        if args is None:
            all_scripts = orchestrator.discover()
            for path, script in scripts.items():
                all_scripts.setdefault(path, script)
            args = parse_arguments(all_scripts, script_path, argv)

        if context is None:
            _enforce_dependencies(args)
            context = _init_ttexalens(args)

        runner = create_runner(args, context, pg, target_paths={script_path})
        runner.run_all(script_queue)
        runner.finalize()

        if return_result:
            return runner.last_result

        init_output_serializer(args)
        runner.render_last_result()
    except BaseException:
        # Programmatic callers expect to see the exception; only firewall when we own the exit.
        if not force_exit:
            raise
        pg.report_fatal()

    if force_exit:
        get_output_serializer().close()
        pg.shutdown()
        os._exit(0)


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
    process_group = make_process_group()

    try:
        parse_arguments(only_triage_script_args=True)

        application_path = os.path.abspath(os.path.dirname(__file__))

        # Avoid double-importing this module when it's __main__.
        my_name = os.path.splitext(os.path.basename(__file__))[0]
        if my_name not in sys.modules:
            sys.modules[my_name] = sys.modules["__main__"]

        orchestrator = ScriptOrchestrator(application_path)
        scripts, script_queue = orchestrator.queue_for_full_triage()

        args = parse_arguments(scripts)
        init_output_serializer(args)
        _enforce_dependencies(args)
        context = _init_ttexalens(args)

        with create_progress() as progress:
            scripts_task = progress.add_task("Script execution", total=len(script_queue))

            if args["--run"] is not None and (len(args["--run"]) != 1 or args["--run"][0] != "all"):
                progress.update(scripts_task, total=len(args["--run"]))
                for script_name in args["--run"]:
                    progress.update(scripts_task, description=f"Running {script_name}")
                    run_script(script_name, args, context)
                    progress.advance(scripts_task)
            else:
                triage_init_end = time()
                if args["--print-script-times"]:
                    utils.INFO(f"Triage initialization time: {triage_init_end - triage_start:.2f}s")
                runner = create_runner(args, context, process_group, progress=progress, progress_task=scripts_task)
                runner.run_all(script_queue)
                runner.finalize()
                runner.print_totals(triage_init_end - triage_start)
            progress.remove_task(scripts_task)

        if process_group.is_root:
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

        get_output_serializer().close()
    except BaseException:
        process_group.report_fatal()

    process_group.shutdown()
    # Single-process re-raises (Python exits non-zero); MPI logs and falls through.
    os._exit(0)


if __name__ == "__main__":
    main()
