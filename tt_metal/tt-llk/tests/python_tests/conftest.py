# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import atexit
import datetime
import json
import logging
import os
import signal
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import helpers.order_processing as order_processing
import helpers.utils as utils_module
import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import LLKAssertException, _send_arc_message
from helpers.exalens_server import ExalensServer
from helpers.format_config import InputOutputFormat
from helpers.logger import configure_logger, logger
from helpers.perf import PerfConfig, PerfReport, combine_perf_reports
from helpers.target_config import TestTargetConfig, initialize_test_target_from_pytest
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


def check_hardware_headers():
    """Check if hardware-specific headers have been downloaded for the current architecture."""

    arch_name = TestConfig.ARCH.value
    header_dir = TestConfig.LLK_ROOT / "tests" / "hw_specific" / arch_name / "inc"

    required_headers = [
        "core_config.h",
        "cfg_defines.h",
        "dev_mem_map.h",
        "tensix.h",
        "tensix_types.h",
    ]
    required_headers_quasar = [
        "core_config.h",
        "cfg_defines.h",
        "dev_mem_map.h",
        "t6_debug_map.h",
        "t6_mop_config_map.h",
        "tensix_types.h",
        "tensix.h",
        "tt_t6_trisc_map.h",
    ]

    # Quasar has a somewhat different set of headers
    if TestConfig.ARCH == ChipArchitecture.QUASAR:
        required_headers = required_headers_quasar

    # Check if header directory exists
    if not header_dir.exists():
        pytest.exit(
            f"ERROR: Hardware-specific header directory not found: {header_dir}\n\n"
            f"SOLUTION: Run the setup script to download required headers:\n"
            f"  cd {TestConfig.LLK_ROOT}/tests\n"
            f"  ./setup_testing_env.sh\n",
            returncode=1,
        )

    # Check for required headers
    missing_headers = []
    for header in required_headers:
        if not (header_dir / header).exists():
            missing_headers.append(header)

    if missing_headers:
        pytest.exit(
            f"ERROR: Missing required hardware headers for {arch_name}:\n"
            + "\n".join(f"  {header}" for header in missing_headers)
            + "\n\n"
            f"SOLUTION: Run the setup script to download missing headers:\n"
            f"  cd {TestConfig.LLK_ROOT}/tests\n"
            f"  ./setup_testing_env.sh\n",
            returncode=1,
        )


@pytest.fixture()
def regenerate_cpp(request):
    return not request.config.getoption("--skip-codegen")


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

    config.coverage_enabled = config.getoption("--coverage", default=False)

    TestConfig.setup_build(
        Path(os.environ["LLK_HOME"]),
        config.getoption("--coverage", default=False),
        config.getoption("--detailed-artefacts", default=False),
        config.getoption("--no-debug-symbols", default=False),
        config.getoption("--speed-of-light", default=False),
    )

    TestConfig.setup_mode(
        # Pass worker id here, so TestConfig can calculate Tensix tile it will run on
        getattr(config, "workerinput", {}).get("workerid", "master"),
        config.getoption("--compile-consumer", default=False),
        config.getoption("--compile-producer", default=False),
        config.getoption("--stimuli-only"),
        config.getoption("--use-stimuli"),
    )

    # Create directories from all processes - lock in create_directories handles race
    TestConfig.create_build_directories()

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

    if (
        TestConfig.ARCH != ChipArchitecture.QUASAR
        and not TestConfig.BUILD_MODE == BuildMode.PRODUCE
    ):
        override_gprs_used_by_tensix_dump()

    log_file = "pytest_errors.log"
    if not hasattr(config, "workerinput"):  # executed only by master pytest runner
        # Refresh order folder with setup_files function
        order_processing.setup_files(TestConfig.ARTEFACTS_DIR / "order_records", True)
        check_hardware_headers()
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

    initialize_test_target_from_pytest(config)
    test_target = TestTargetConfig()

    if TestConfig.BUILD_MODE != BuildMode.PRODUCE:
        if test_target.run_simulator:
            simulator_path = os.environ.get("TT_UMD_SIMULATOR_PATH")

            if simulator_path is None:
                pytest.exit(
                    "ERROR: --run-simulator requires TT_UMD_SIMULATOR_PATH "
                    "environment variable to be set.",
                    returncode=1,
                )

            # Only the controller process manages the server; xdist workers
            # just connect to the already-running instance.
            if not hasattr(config, "workerinput"):
                global _exalens_server
                _exalens_server = ExalensServer(
                    simulator_path=simulator_path,
                    port=test_target.simulator_port,
                )
        else:
            tt_exalens_init.init_ttexalens(use_4B_mode=False)


def pytest_collection_modifyitems(config, items):
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
    test_target = TestTargetConfig()
    if not test_target.reset_simulator_per_test:
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

    test_target = TestTargetConfig()

    if not _exalens_server.running and not _exalens_server.ever_started:
        _exalens_server.start()
        tt_exalens_init.init_ttexalens_remote(
            port=test_target.simulator_port, use_4B_mode=False
        )
    elif not _exalens_server.running:
        logger.error("tt-exalens server is no longer running unexpectedly.")
        pytest.exit(returncode=1)
    elif _reset_simulator_pending:
        _reset_simulator_pending = False
        tt_exalens_init.cleanup_global_context()
        _exalens_server.restart()
        tt_exalens_init.init_ttexalens_remote(
            port=test_target.simulator_port, use_4B_mode=False
        )


def pytest_sessionstart(session):
    if hasattr(session.config, "workerinput"):
        return

    test_target = TestTargetConfig()
    if not test_target.run_simulator and TestConfig.BUILD_MODE != BuildMode.PRODUCE:
        _send_arc_message("GO_BUSY", test_target.device_id)


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

    test_target = TestTargetConfig()
    if not test_target.run_simulator and TestConfig.BUILD_MODE != BuildMode.PRODUCE:
        _send_arc_message("GO_IDLE", test_target.device_id)

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
