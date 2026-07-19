# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import atexit
import datetime
import json
import logging
import os
import re
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

# ttsim runs in-process (no ExalensServer). Its init must complete before any helpers.* import,
# because helpers.chip_architecture.get_chip_architecture() reaches check_context() and this
# conftest calls it at module-load time via the skip_for_* markers defined further down.
# TT_METAL_SIMULATOR is the canonical env var (matches tt-metal runtime and the ttsim README);
# TT_UMD_SIMULATOR_PATH is kept as an alias for the existing RTL-simulator workflow.
# Gate on --run-simulator so the env var being set doesn't force a ttsim init on silicon runs,
# and skip --compile-producer (it only compiles ELFs and never talks to a device). xdist workers
# inherit env vars but not the controller's argv, so trust the env var when PYTEST_XDIST_WORKER
# is set.
_SIMULATOR_PATH = os.environ.get("TT_METAL_SIMULATOR") or os.environ.get(
    "TT_UMD_SIMULATOR_PATH"
)
_IS_XDIST_WORKER = "PYTEST_XDIST_WORKER" in os.environ
_SHOULD_RUN_SIMULATOR = _IS_XDIST_WORKER or (
    "--run-simulator" in sys.argv and "--compile-producer" not in sys.argv
)
if _SHOULD_RUN_SIMULATOR and _SIMULATOR_PATH and _SIMULATOR_PATH.endswith(".so"):
    from ttexalens import tt_exalens_init as _tt_exalens_init

    _tt_exalens_init.init_ttexalens(simulation_directory=_SIMULATOR_PATH)

from ttexalens import umd_device as _umd_device

_umd_device.UmdDevice.can_use_dma = False

import helpers.order_processing as order_processing
import helpers.utils as utils_module
import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import LLKAssertException
from helpers.exalens_server import ExalensServer
from helpers.format_config import InputOutputFormat
from helpers.logger import configure_logger, logger
from helpers.perf import PerfConfig, PerfReport, combine_perf_reports
from helpers.test_config import BuildMode, TestConfig, process_coverage_run_artefacts
from ttexalens import check_context, tt_exalens_init
from ttexalens.tt_exalens_lib import get_tensix_state

_exalens_server: Optional[ExalensServer] = None


# This is a workaround for this issue: https://github.com/tenstorrent/tt-exalens/issues/958
# In a nutshell, everything except Tensix GPRs is accessible over NoC, thus ignoring that allows us to dump
# most of the Tensix state, without causing any runtime issues.
def override_gprs_used_by_tensix_dump():
    context = check_context()
    for device_id in context.devices.keys():
        context.devices[
            device_id
        ].get_tensix_registers_description().general_purpose_registers = []


@atexit.register
def _stop_exalens_server():
    """atexit handler to ensure the tt-exalens server is stopped on process exit."""
    global _exalens_server
    if _exalens_server is not None:
        _exalens_server.stop()
        _exalens_server = None


def _fatal_signal_handler(signum, frame):
    """Convert fatal signals into KeyboardInterrupt.

    If raised during _wait_until_ready, the existing except KeyboardInterrupt
    block will wait for the server to become ready before stopping it, preventing
    orphaned emulator sessions. Outside the wait loop, it propagates like Ctrl+C
    and pytest handles teardown normally (via pytest_sessionfinish / atexit).
    """
    raise KeyboardInterrupt


# Ensure the tt-exalens server is stopped on SIGTERM/SIGQUIT so the emulator
# session is released. Without this, `kill` or Ctrl+\ would terminate the
# process immediately, leaving the emulator slot orphaned.
signal.signal(signal.SIGTERM, _fatal_signal_handler)
signal.signal(signal.SIGQUIT, _fatal_signal_handler)


def init_llk_home():
    if "LLK_HOME" in os.environ:
        return
    os.environ["LLK_HOME"] = str(Path(__file__).resolve().parents[2])


# Default LLK_HOME environment variable
init_llk_home()


@pytest.fixture()
def regenerate_cpp(request):
    return not request.config.getoption("--skip-codegen")


# Default seed for deterministic stimuli. Override via LLK_TEST_SEED to reproduce
# a specific run or to sweep different random inputs.
_LLK_TEST_SEED = os.environ.get("LLK_TEST_SEED")
try:
    _DEFAULT_TORCH_SEED = int(_LLK_TEST_SEED, 0) if _LLK_TEST_SEED is not None else 42
except ValueError as e:
    raise pytest.UsageError(
        f"LLK_TEST_SEED must be an integer, got {_LLK_TEST_SEED!r}"
    ) from e


@pytest.fixture(autouse=True)
def _seed_torch_rng():
    """Lock torch's global RNG before every test to avoid flaky failures.

    Stimuli that don't set an explicit StimuliSpec.seed fall back to torch's
    global generator, so seeding here makes both generate_stimuli() and any
    direct torch.rand/randn/randint/uniform_ calls reproducible.
    """
    torch.manual_seed(_DEFAULT_TORCH_SEED)
    yield


# Define the possible custom command line options
def pytest_addoption(parser):
    parser.addoption(
        "--run-simulator", action="store_true", help="Run tests using the simulator."
    )
    parser.addoption(
        "--port",
        action="store",
        type=int,
        default=5555,
        help="Integer number of the server port.",
    )
    parser.addoption(
        "--reset-simulator-per-test",
        action="store_true",
        default=False,
        help="Restart the tt-exalens server after each test. "
        "Only effective with --run-simulator.",
    )

    parser.addoption(
        "--coverage",
        action="store_true",
        help="Enables coverage *.info file generation for every test variant run",
    )

    parser.addoption(
        "--compile-producer",
        action="store_true",
        help="Only compile *.elf(s) for every test variant selected and store them on path specified",
    )

    parser.addoption(
        "--compile-consumer",
        action="store_true",
        help="Consume pre-compiled *.elf(s) for every test variant selected, from pre-specified path, and execute specified variants",
    )

    parser.addoption(
        "--detailed-artefacts",
        action="store_true",
        help="Insert few more compilation flags to produce binary artefacts suitable for debugging",
    )

    parser.addoption(
        "--skip-codegen",
        action="store_true",
        default=False,
        help="Skip C++ code generation for fused tests and use existing files",
    )

    parser.addoption(
        "--no-debug-symbols",
        action="store_true",
        default=False,
        help="Compile without debug symbols (-g flag) to save disk space",
    )

    parser.addoption(
        "--logging-level",
        action="store",
        default=None,
        help="Set loguru log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL). "
        "Overrides LOGURU_LEVEL env var. Default: INFO",
    )

    parser.addoption(
        "--speed-of-light",
        action="store_true",
        default=False,
        help="Should tests be compiled with everything runtime, converted to compile-time",
    )

    parser.addoption(
        "--record-test-order",
        action="store",
        nargs="?",
        const="_USE_DEFAULT_PATH",  # Value when flag is used without argument
        default=None,  # Value when flag is not used at all
        help="Path to where the test order, per runner should be stored to, default path is the same folder as LLK repo",
    )

    parser.addoption(
        "--test-order-file",
        action="store",
        default=None,
        help="Path to file containing ordered list of tests to run",
    )

    parser.addoption(
        "--rewind-runner",
        action="store",
        default=None,
        help="Runner name to be selected from test order file",
    )

    parser.addoption(
        "--stimuli-only",
        action="store",
        nargs="?",
        const="_USE_DEFAULT_PATH",  # Value when flag is used without argument
        default=None,  # Value when flag is not used at all
        help="Path to folder where stimuli should be stored",
    )

    parser.addoption(
        "--use-stimuli",
        action="store",
        nargs="?",
        const="_USE_DEFAULT_PATH",  # Value when flag is used without argument
        default=None,  # Value when flag is not used at all
        help="Path to folder where stimuli should be loaded from",
    )

    parser.addoption(
        "--enable-perf-counters",
        action="store_true",
        default=False,
        help="Enable hardware performance counter collection during perf tests",
    )

    parser.addoption(
        "--dump-raw-counters",
        action="store_true",
        default=False,
        help="Print raw hardware counter values to console (implies --enable-perf-counters)",
    )

    parser.addoption(
        "--dump-raw-metrics",
        action="store_true",
        default=False,
        help="Print derived efficiency metrics to console (implies --enable-perf-counters)",
    )

    parser.addoption(
        "--dump-csv-counters",
        action="store_true",
        default=False,
        help="Export raw hardware counter values to a separate .counters.csv file (implies --enable-perf-counters)",
    )

    parser.addoption(
        "--disable-sfploadmacro",
        action="store_true",
        default=False,
        help="Compile kernels with -DDISABLE_SFPLOADMACRO so SFPLOADMACRO-based SFPU "
        "kernels fall back to their plain sfpi/TTI calculate path (equivalent to "
        "setting TT_METAL_DISABLE_SFPLOADMACRO=1).",
    )

    parser.addoption(
        "--op",
        action="append",
        default=[],
        metavar="OP",
        help="Run only tests for the given SFPU op(s), by MathOperation name "
        "(case-insensitive, exact). Repeatable: --op=exp --op=log.",
    )


_RECORD_TEST_ORDER: bool = False
_UNIFIED_ORDER_FILE: str = "DEFAULT"


def pytest_configure(config):
    # Configure loguru log level from CLI option or environment variable.
    log_level = config.getoption("--logging-level", default=None)
    configure_logger(level=log_level)

    # Enable pytest's live logging when --logging-level is set.
    # Loguru propagates to stdlib logging, and pytest's log_cli displays
    # those messages in the terminal - integrating cleanly with pytest-sugar.
    if log_level is not None:
        config.option.log_cli_level = log_level
        config.option.log_cli = True

    # Let the CLI flag drive the compile define; test_config reads the env var
    # when assembling per-variant compile options.
    if config.getoption("--disable-sfploadmacro", default=False):
        os.environ["TT_METAL_DISABLE_SFPLOADMACRO"] = "1"

    config.coverage_enabled = config.getoption("--coverage", default=False)
    TestConfig.DUMP_RAW_COUNTERS = config.getoption(
        "--dump-raw-counters", default=False
    )
    TestConfig.DUMP_RAW_METRICS = config.getoption("--dump-raw-metrics", default=False)
    TestConfig.DUMP_CSV_COUNTERS = config.getoption(
        "--dump-csv-counters", default=False
    )
    # --dump-raw-counters, --dump-raw-metrics, or --dump-csv-counters imply --enable-perf-counters
    TestConfig.ENABLE_PERF_COUNTERS = (
        config.getoption("--enable-perf-counters", default=False)
        or TestConfig.DUMP_RAW_COUNTERS
        or TestConfig.DUMP_RAW_METRICS
        or TestConfig.DUMP_CSV_COUNTERS
    )

    # Device print is enabled on debug or trace.
    resolved_log_level = (
        config.getoption("--logging-level", default=None)
        or os.getenv("LOGURU_LEVEL", "INFO")
    ).upper()
    TestConfig.DEVICE_PRINT_ENABLED = (
        resolved_log_level in ("DEBUG", "TRACE") and not config.coverage_enabled
    )

    TestConfig.setup_build(
        Path(os.environ["LLK_HOME"]),
        config.getoption("--coverage", default=False),
        config.getoption("--detailed-artefacts", default=False),
        config.getoption("--no-debug-symbols", default=False),
        config.getoption("--speed-of-light", default=False),
    )

    worker_id = getattr(config, "workerinput", {}).get("workerid", "master")

    if worker_id != "master":
        import torch

        torch.set_num_threads(1)

    TestConfig.setup_mode(
        worker_id,
        config.getoption("--compile-consumer", default=False),
        config.getoption("--compile-producer", default=False),
        config.getoption("--stimuli-only"),
        config.getoption("--use-stimuli"),
    )

    # Create directories from all processes - lock in create_directories handles race
    TestConfig.create_build_directories()

    TestConfig.TEST_TARGET.update_from_pytest_config(config)

    global _RECORD_TEST_ORDER, _UNIFIED_ORDER_FILE

    if _RECORD_TEST_ORDER := config.getoption("--record-test-order"):
        if _RECORD_TEST_ORDER == "_USE_DEFAULT_PATH":
            current_time = datetime.datetime.now()
            _UNIFIED_ORDER_FILE = (
                TestConfig.LLK_ROOT
                / f"../run_order_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}_{current_time.second}.json"
            )
        else:
            _UNIFIED_ORDER_FILE = _RECORD_TEST_ORDER
        _RECORD_TEST_ORDER = True
        utils_module._RECORD_TEST_ORDER = True

    is_ttsim = _SIMULATOR_PATH and _SIMULATOR_PATH.endswith(".so")
    if (
        (is_ttsim or not TestConfig.TEST_TARGET.run_simulator)
        and TestConfig.ARCH != ChipArchitecture.QUASAR
        and TestConfig.BUILD_MODE != BuildMode.PRODUCE
    ):
        override_gprs_used_by_tensix_dump()

    log_file = "pytest_errors.log"
    if not hasattr(config, "workerinput"):  # executed only by master pytest runner
        # Refresh order folder with setup_files function
        order_processing.setup_files(TestConfig.ARTEFACTS_DIR / "order_records", True)
        if os.path.exists(log_file):
            os.remove(log_file)

    else:
        # Workers only need to set their local versions of ORDER_FOLDER_PATH
        order_processing.setup_files(TestConfig.ARTEFACTS_DIR / "order_records")

    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if TestConfig.BUILD_MODE != BuildMode.PRODUCE:
        if TestConfig.TEST_TARGET.run_simulator:
            if _SIMULATOR_PATH is None:
                pytest.exit(
                    "ERROR: --run-simulator requires TT_METAL_SIMULATOR "
                    "(or TT_UMD_SIMULATOR_PATH) environment variable to be set.",
                    returncode=1,
                )

            if _SIMULATOR_PATH.endswith(".so"):
                # ttsim: already initialized at module import above; runs in-process, no server.
                # --reset-simulator-per-test restarts the ExalensServer, which ttsim doesn't use,
                # so it would be a silent no-op. Fail fast to avoid confusing false-green runs.
                if TestConfig.TEST_TARGET.reset_simulator_per_test:
                    pytest.exit(
                        "ERROR: --reset-simulator-per-test is not supported with ttsim. "
                        "Re-run without it.",
                        returncode=1,
                    )
            elif not hasattr(config, "workerinput"):
                # RTL simulator: only the controller process manages the server; xdist workers
                # just connect to the already-running instance.
                global _exalens_server
                _exalens_server = ExalensServer(
                    simulator_path=_SIMULATOR_PATH,
                    port=TestConfig.TEST_TARGET.simulator_port,
                )
        else:
            tt_exalens_init.init_ttexalens()


def pytest_ignore_collect(collection_path, config):
    # Skip collecting the quasar/ dir on non-quasar arch — those tests are
    # deselected there anyway, so there's no need to collect them.
    if (
        get_chip_architecture() != ChipArchitecture.QUASAR
        and "quasar" in collection_path.parts
    ):
        return True
    return None


def _collapse_runtime_only_variants(config, items):
    """Keep only one test per unique compile key, dropping runtime only duplicates.

    Tests decorated with ``@parametrize`` that use ``runtime()`` markers carry a
    ``compile_key_fn`` on their ``runtime_axes`` pytest mark.  That function extracts
    the compile time subset of each item's params.  Items that share the same compile
    key produce identical ELFs, so only the first is kept for the compile-producer pass.
    """
    from helpers.param_config import RUNTIME_AXES_MARK

    seen = set()
    keep = []
    deselected = []
    for item in items:
        marker = item.get_closest_marker(RUNTIME_AXES_MARK)
        if marker is None:
            keep.append(item)
            continue
        compile_key_fn = marker.kwargs["compile_key_fn"]
        key = (item.nodeid.split("[")[0], repr(compile_key_fn(item.callspec.params)))
        if key not in seen:
            seen.add(key)
            keep.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = keep


def _item_op_names(item) -> set:
    """Return the op name(s) a test covers, lowercased.

    Reads the MathOperation from the test's parameters, falling back to the op name in the test id.
    """
    from helpers.llk_params import MathOperation

    names = set()
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        for val in callspec.params.values():
            if isinstance(val, MathOperation):
                names.add(val.name.lower())
    if not names:
        names.update(
            m.lower() for m in re.findall(r"MathOperation\.(\w+)", item.nodeid)
        )
    return names


def _select_tests_by_op(config, items):
    """Run only the tests for the op(s) passed with --op.

    Each op is a MathOperation name, matched case-insensitively and exactly. An
    unknown name raises an error; a valid op with no matching test selects
    nothing. Does nothing without --op.
    """
    requested = config.getoption("--op") or []
    if not requested:
        return

    from helpers.llk_params import MathOperation

    valid = {op.name.lower() for op in MathOperation}
    wanted = set()
    for raw in requested:
        key = raw.lower()
        if key not in valid:
            raise pytest.UsageError(
                f"--op {raw!r}: not a known SFPU op. Expected a MathOperation "
                f"name (case-insensitive), e.g. Exp, Reciprocal, Gelu."
            )
        wanted.add(key)

    selected, deselected = [], []
    for item in items:
        (selected if _item_op_names(item) & wanted else deselected).append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
    logger.info(
        f"--op kept {len(selected)} test(s) for op(s): {', '.join(sorted(wanted))}"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    _select_tests_by_op(config, items)

    if TestConfig.BUILD_MODE == BuildMode.PRODUCE and not TestConfig.SPEED_OF_LIGHT:
        _collapse_runtime_only_variants(config, items)

    test_order_file = config.getoption("--test-order-file")

    if not test_order_file:
        return

    temp_runner_name = config.getoption("--rewind-runner")
    if temp_runner_name is None:
        raise ValueError(
            "If you want to execute tests in the same order you also need to provide which exact runner you want to rewind using --rewind-runner='your_runner_name' argument to pytest"
        )

    if not os.path.exists(test_order_file):
        raise FileNotFoundError(f"Test order file not found: {test_order_file}")

    with open(test_order_file, "r") as fp:
        test_order_dict = json.load(fp)

    try:
        temp_runner_list = test_order_dict[temp_runner_name]
    except KeyError:
        raise KeyError(
            f"Test order file {test_order_file}, doesn't have a run entry for the runner name {temp_runner_name} you provided"
        )

    node_ids_to_rewind = [variant_run["test"] for variant_run in temp_runner_list]

    items_dict = {item.nodeid: item for item in items}
    items[:] = [
        items_dict[nodeid] for nodeid in node_ids_to_rewind if nodeid in items_dict
    ]

    logger.info(
        f"Executing {len(items)} variants as they were executed on runner {temp_runner_name} on run recorded to file {test_order_file}"
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Suppress the short test summary info section."""

    # Override the method that writes the short test summary
    def no_summary(*args, **kwargs):
        pass

    if TestConfig.CHIP_ARCH != ChipArchitecture.QUASAR:
        terminalreporter.short_test_summary = no_summary
    terminalreporter.showfspath = False


def _stringify_params(params):
    parts = []
    for name, value in params.items():
        # todo: handle FormatConfig?
        if name == "test_name":
            continue
        elif isinstance(value, InputOutputFormat):
            parts.append(f"{name}.input={value.input}")
            parts.append(f"{name}.output={value.output}")
        elif isinstance(value, str):
            parts.append(f'{name}="{value}"')
        elif hasattr(value, "repr"):
            parts.append(f"{name}={value.repr()}")
        else:
            parts.append(f"{name}={str(value)}")

    return f"[{' | '.join(parts)}]"


# Match the ttsim error preamble printed by ttsim_error() before _Exit(1):
#   [<clk>] ERROR: <Category>: <function>: <details>
_TTSIM_ERR_RE = re.compile(r"^\[\d+\] ERROR: (\w+): (\w+): (.*)$", re.MULTILINE)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    # pytest-forked appends a crashed child's captured streams to report.sections
    # using lowercase headers ("captured stdout"/"captured stderr"), but pytest's
    # TestReport.capstdout/capstderr properties only match headers that start with
    # "Captured stdout"/"Captured stderr" (capital C). The junit XML writer reads
    # via those properties, so without renaming, <system-out> stays empty for
    # forked crashes. Rename in place so the ttsim "ERROR: ..." line lands in the
    # XML and any junit-XML-aware viewer (junit2html, CI, etc.) can render it.
    sections = getattr(report, "sections", None)
    if sections:
        report.sections = [
            (
                ("Captured stdout call", content)
                if name == "captured stdout"
                else (
                    ("Captured stderr call", content)
                    if name == "captured stderr"
                    else (name, content)
                )
            )
            for name, content in sections
        ]

    # Rewrite the headline of forked-crash reports from the useless
    #   ":-1: running the test CRASHED with signal 0"
    # to the actual ttsim category/function so it shows up in the test summary,
    # in <error message="..."> in the junit XML, and in CI annotations.
    #
    # We also normalize report.when from pytest-forked's sentinel "???" to "call".
    # Without this, pytest-sugar (and other reporters that count by phase) treat
    # forked-crash reports as setup-phase errors and miss them in the live
    # pass/fail/skip footer; the junit writer also classifies them as <error>
    # rather than <failure>. Setting when="call" makes the report indistinguish-
    # able from a normal call-phase failure, which is what we actually want:
    # the test ran, ttsim hit an unimplemented path mid-execution, the test
    # failed.
    if report.outcome == "failed" and "CRASHED with signal" in str(
        report.longrepr or ""
    ):
        report.when = "call"
        m = _TTSIM_ERR_RE.search(report.capstdout or "")
        if m:
            cat, func, msg = m.group(1), m.group(2), m.group(3).strip()
            report.longrepr = f"[ttsim:{cat}] {func}: {msg}"
            props = list(getattr(report, "user_properties", []))
            props.append(("ttsim_category", cat))
            props.append(("ttsim_func", func))
            report.user_properties = props


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    global _RECORD_TEST_ORDER

    # Execute all other hooks to obtain the report object
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and not report.skipped and _RECORD_TEST_ORDER:
        worker_id = getattr(item.config, "workerinput", {}).get("workerid", "master")

        order_processing.append_record(
            f"{worker_id}.jsonl",
            {
                "test": item.nodeid,
                "status": report.outcome,
                "state": asdict(
                    get_tensix_state(TestConfig.TENSIX_LOCATION, device_id=0)
                ),
            },
        )

    if hasattr(item, "callspec") and item.callspec:
        report.test_params = _stringify_params(item.callspec.params)
    else:
        report.test_params = None

    if report.skipped and report.when == "call":
        skip_reason = (
            str(report.longrepr[2])
            if hasattr(report.longrepr, "__getitem__") and len(report.longrepr) > 2
            else str(report.longrepr)
        )

        if (
            TestConfig.SKIP_JUST_FOR_COMPILE_MARKER in skip_reason
            or TestConfig.SKIP_JUST_FOR_STIMULI_MARKER in skip_reason
        ):
            report.outcome = "passed"

    if report.failed and report.when == "call":
        if hasattr(report, "longrepr") and report.longrepr:
            if hasattr(call, "excinfo") and call.excinfo:
                exc_type = call.excinfo.type

                test_file_and_func = report.nodeid.split("[")[0]

                stack_trace = []
                current_working_directory = Path.cwd()
                for entry in call.excinfo.traceback:
                    path_str = str(entry.path)
                    # Skip pytest and pluggy internal frames
                    if "_pytest" in path_str or "pluggy" in path_str:
                        continue

                    try:
                        file_path = entry.path.relative_to(current_working_directory)
                    except (ValueError, AttributeError):
                        file_path = entry.path
                    line_number = entry.lineno + 1

                    stack_trace.append(f"  {file_path}:{line_number}")

                stack_trace_str = "\n".join(stack_trace) if stack_trace else ""

                exc_msg = str(call.excinfo.value) if call.excinfo.value.args else ""

                if exc_type == LLKAssertException:
                    report.longrepr = f"LLK ASSERT HIT {test_file_and_func}{report.test_params} {exc_msg}"
                elif exc_type == TimeoutError:
                    report.longrepr = f"TENSIX TIMED OUT {test_file_and_func}{report.test_params} {exc_msg}"
                    # Log the timeout error
                    logger.error(
                        f"TENSIX TIMED OUT {test_file_and_func}{report.test_params} {exc_msg}"
                    )
                elif exc_type == AssertionError:
                    # If we want to record test ordering, we already now from order report if test failed, thus to de-clutter logs,
                    # we will mark test as if it passed to speed the whole execution up
                    if _RECORD_TEST_ORDER:
                        logger.error(
                            f"⨯ AssertionError during order recording: {test_file_and_func}{report.test_params} {exc_msg}"
                        )
                        report.outcome = "passed"

                    # Handle assertion failures
                    report.longrepr = (
                        f"⨯ {test_file_and_func}{report.test_params} {exc_msg}"
                    )
                else:
                    report.longrepr = (
                        f"⨯ {test_file_and_func}{report.test_params} Error type: {exc_type.__name__}\n"
                        f"{exc_msg}\n"
                        f"Python Call trace:\n{stack_trace_str}"
                    )

    return report


_reset_simulator_pending = False


def pytest_runtest_teardown(item, nextitem):
    """Mark that a restart is needed before the next test."""
    if not TestConfig.TEST_TARGET.reset_simulator_per_test:
        return
    if nextitem is None:
        return
    if hasattr(item.config, "workerinput"):
        return

    global _reset_simulator_pending
    if _exalens_server is not None:
        _reset_simulator_pending = True


def pytest_runtest_setup(item):
    """Start the server on the first test, or restart between tests if requested."""
    global _exalens_server, _reset_simulator_pending

    if _exalens_server is None:
        return

    if not _exalens_server.running and not _exalens_server.ever_started:
        _exalens_server.start()
        tt_exalens_init.init_ttexalens_remote(
            port=TestConfig.TEST_TARGET.simulator_port
        )
    elif not _exalens_server.running:
        logger.error("tt-exalens server is no longer running unexpectedly.")
        pytest.exit(returncode=1)
    elif _reset_simulator_pending:
        _reset_simulator_pending = False
        tt_exalens_init.cleanup_global_context()
        _exalens_server.restart()
        tt_exalens_init.init_ttexalens_remote(
            port=TestConfig.TEST_TARGET.simulator_port
        )


def pytest_sessionstart(session):
    if hasattr(session.config, "workerinput"):
        return


@pytest.fixture(scope="module", autouse=True)
def counter_report(request, worker_id):
    """Separate report for raw hardware counter CSV data (--dump-csv-counters)."""
    if not TestConfig.DUMP_CSV_COUNTERS:
        PerfConfig.COUNTER_REPORT = None
        yield None
        return

    test_module = request.path.stem
    temp_report = PerfReport()
    PerfConfig.COUNTER_REPORT = temp_report

    try:
        yield temp_report
    except Exception as e:
        logger.warning("Counter report: Unexpected error, saving anyway: {}", e)

    PerfConfig.COUNTER_REPORT = None

    if TestConfig.MODE == TestMode.PRODUCE:
        return

    if PerfConfig.TEST_COUNTER == 0:
        return

    temp_report.assert_single_schema(
        context=f"{test_module} counters (worker {worker_id})"
    )

    counters_path = TestConfig.PERF_DATA_DIR / f"{test_module}.{worker_id}.counters.csv"

    if counters_path.exists():
        counters_path.unlink()

    temp_report.dump_csv(counters_path)


@pytest.fixture(scope="module", autouse=True)
def perf_report(request, worker_id):

    test_module = request.path.stem

    temp_report = PerfReport()

    try:
        yield temp_report
    except Exception as e:
        logger.warning("Perf: Unexpected error, saving report anyway: {}", e)

    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        return

    if PerfConfig.TEST_COUNTER == 0:
        return

    # Fail loud before writing: a single CSV must hold exactly one column schema.
    # More than one means two unrelated tests/ops share this module (split them
    # into separate files) or one test emits inconsistent columns across its sweep.
    temp_report.assert_single_schema(context=f"{test_module} (worker {worker_id})")

    raw_path = TestConfig.PERF_DATA_DIR / f"{test_module}.{worker_id}.csv"
    post_path = TestConfig.PERF_DATA_DIR / f"{test_module}.{worker_id}.post.csv"

    if raw_path.exists():
        raw_path.unlink()

    if post_path.exists():
        post_path.unlink()

    temp_report.dump_csv(raw_path)
    temp_report.post_process()
    temp_report.dump_csv(post_path)


def pytest_sessionfinish(session):
    if hasattr(session.config, "workerinput"):
        return

    if TestConfig.BUILD_MODE != BuildMode.PRODUCE:
        combine_perf_reports()

        # This was set by pytest CLI argument in pytest_configure call
        if _RECORD_TEST_ORDER:
            order_processing.unify_files(_UNIFIED_ORDER_FILE)

        if TestConfig.WITH_COVERAGE:
            process_coverage_run_artefacts()

    _stop_exalens_server()


# Skip decorators for specific architectures
# These decorators can be used to skip tests based on the architecture
# For example, if you want to skip a test for the "wormhole" architecture,
# decorate the test with @skip_for_wormhole.

skip_for_wormhole = pytest.mark.skipif(
    get_chip_architecture() == ChipArchitecture.WORMHOLE,
    reason="Test is not supported on Wormhole architecture",
)

skip_for_blackhole = pytest.mark.skipif(
    get_chip_architecture() == ChipArchitecture.BLACKHOLE,
    reason="Test is not supported on Blackhole architecture",
)

skip_for_quasar = pytest.mark.skipif(
    get_chip_architecture() == ChipArchitecture.QUASAR,
    reason="Test is not supported on Quasar architecture",
)

skip_for_coverage = pytest.mark.skipif(
    "config.coverage_enabled",
    reason="Coverage shouldn't be ran with this test",
)
