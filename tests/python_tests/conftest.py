# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import glob
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import LLKAssertException, _send_arc_message
from helpers.format_config import InputOutputFormat
from helpers.logger import configure_logger, logger
from helpers.perf import PerfConfig, PerfReport, combine_perf_reports
from helpers.target_config import TestTargetConfig, initialize_test_target_from_pytest
from helpers.test_config import TestConfig, TestMode, process_coverage_run_artefacts
from ttexalens import tt_exalens_init


class ExalensServer:
    """Manages the tt-exalens server lifecycle for simulator-based test runs.

    Starts tt-exalens as a subprocess, waits for it to become ready by polling
    its output for the readiness pattern, and provides graceful shutdown.
    """

    READY_PATTERN = "[4B MODE]"
    READY_TIMEOUT_S = 600
    POLL_INTERVAL_S = 2

    def __init__(self, simulator_path: str, port: int):
        self._simulator_path = simulator_path
        self._port = port
        self._process: Optional[subprocess.Popen] = None
        self._log_path: Optional[str] = None
        self._emu_logs_baseline: set = set()
        self._log_read_offset = 0
        self._started_before = False

    def start(self) -> None:
        self._emu_logs_baseline = set(glob.glob(self.EMU_LOG_PATTERN))
        if not os.path.isdir(self._simulator_path):
            logger.error(
                "Simulator build path does not exist: {}", self._simulator_path
            )
            pytest.exit(returncode=1)

        if not shutil.which("tt-exalens"):
            logger.error("tt-exalens not found in PATH")
            pytest.exit(returncode=1)

        missing_vars = [
            v
            for v in ("NNG_SOCKET_ADDR", "NNG_SOCKET_LOCAL_PORT")
            if v not in os.environ
        ]
        if missing_vars:
            logger.error(
                "Required environment variable(s) not set: {}",
                ", ".join(missing_vars),
            )
            pytest.exit(returncode=1)

        self._log_path = os.path.join(os.getcwd(), "tt-exalens.log")
        if self._started_before:
            self._log_read_offset = os.path.getsize(self._log_path)
            log_file = open(self._log_path, "a")
        else:
            self._log_read_offset = 0
            log_file = open(self._log_path, "w")
            self._started_before = True

        logger.info(
            "Starting tt-exalens server (port={}, simulator={}, "
            "NNG_SOCKET_ADDR={}, NNG_SOCKET_LOCAL_PORT={})...",
            self._port,
            self._simulator_path,
            os.environ.get("NNG_SOCKET_ADDR", "<not set>"),
            os.environ.get("NNG_SOCKET_LOCAL_PORT", "<not set>"),
        )
        logger.info("tt-exalens output: {}", self._log_path)

        self._process = subprocess.Popen(
            [
                "tt-exalens",
                f"--port={self._port}",
                "--server",
                "-s",
                self._simulator_path,
            ],
            stdin=subprocess.PIPE,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_file.close()

        self._wait_until_ready()

    EMU_LOG_PATTERN = "emu_*_.log"

    def _wait_until_ready(self) -> None:
        logger.info(
            "Waiting for tt-exalens to become ready (timeout: {}s)...",
            self.READY_TIMEOUT_S,
        )
        shutdown_requested = False
        elapsed = 0
        while elapsed < self.READY_TIMEOUT_S:
            try:
                if self._process.poll() is not None:
                    log_tail = self._read_log_tail(50)
                    logger.error(
                        "tt-exalens exited prematurely (code {}).\nLog output:\n{}",
                        self._process.returncode,
                        log_tail,
                    )
                    pytest.exit(returncode=1)

                if self._log_contains_ready_pattern():
                    logger.info(
                        "tt-exalens ready (PID {}, took ~{}s)",
                        self._process.pid,
                        elapsed,
                    )
                    if shutdown_requested:
                        logger.info(
                            "Gracefully stopping tt-exalens to release emulator..."
                        )
                        self.stop()
                        pytest.exit(
                            "Interrupted by user during tt-exalens startup.",
                            returncode=1,
                        )
                    return

                emu_errors = self._check_emulator_log()
                if emu_errors:
                    logger.error(
                        "Emulator reported errors during tt-exalens startup:\n{}",
                        emu_errors,
                    )
                    self.stop()
                    pytest.exit(returncode=1)

                time.sleep(self.POLL_INTERVAL_S)
            except KeyboardInterrupt:
                if not shutdown_requested:
                    shutdown_requested = True
                    logger.warning(
                        "Ctrl+C received — waiting for tt-exalens to become ready "
                        "before shutting down (to release emulator resources)..."
                    )

            elapsed += self.POLL_INTERVAL_S
            if elapsed % 10 == 0:
                logger.info("    ... still waiting ({}s elapsed)", elapsed)

        log_tail = self._read_log_tail(50)
        if shutdown_requested:
            logger.error(
                "tt-exalens did not become ready after Ctrl+C; "
                "giving up after {}s.\nLog output:\n{}",
                self.READY_TIMEOUT_S,
                log_tail,
            )
        else:
            logger.error(
                "tt-exalens did not become ready within {}s.\nLog output:\n{}",
                self.READY_TIMEOUT_S,
                log_tail,
            )
        self.stop()
        pytest.exit(returncode=1)

    EMU_ERROR_PATTERN = "zServer : ERROR"

    def _check_emulator_log(self) -> Optional[str]:
        """Check emulator logs created after start() for zServer ERROR lines."""
        new_logs = set(glob.glob(self.EMU_LOG_PATTERN)) - self._emu_logs_baseline
        if not new_logs:
            return None

        latest = max(new_logs, key=os.path.getmtime)
        error_lines = []
        try:
            with open(latest, "r") as f:
                for line in f:
                    if self.EMU_ERROR_PATTERN in line:
                        error_lines.append(line.rstrip())
        except OSError:
            return None

        if error_lines:
            return f"(from {latest})\n" + "\n".join(error_lines)
        return None

    def _log_contains_ready_pattern(self) -> bool:
        if not self._log_path or not os.path.exists(self._log_path):
            return False
        try:
            with open(self._log_path, "r") as f:
                f.seek(self._log_read_offset)
                new_data = f.read()
                self._log_read_offset = f.tell()
                return self.READY_PATTERN in new_data
        except OSError:
            return False

    def _read_log_tail(self, lines: int = 30) -> str:
        if not self._log_path or not os.path.exists(self._log_path):
            return "<no log available>"
        try:
            with open(self._log_path, "r") as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except OSError:
            return "<failed to read log>"

    def stop(self) -> None:
        if self._process is None:
            return

        if self._process.poll() is None:
            logger.info("Stopping tt-exalens (PID {})...", self._process.pid)
            try:
                self._process.stdin.write(b"exit\n")
                self._process.stdin.flush()
                self._process.stdin.close()
            except OSError:
                pass

            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("tt-exalens did not exit gracefully, sending SIGTERM...")
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("tt-exalens did not terminate, sending SIGKILL...")
                    self._process.kill()
                    self._process.wait()

            logger.info("tt-exalens stopped.")

        self._process = None

    def restart(self) -> None:
        logger.info("Restarting tt-exalens server...")
        self.stop()
        self.start()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def ever_started(self) -> bool:
        return self._started_before


_exalens_server: Optional[ExalensServer] = None


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
def workers_tensix_coordinates(worker_id):
    if worker_id == "master":
        return "0,0"
    row, col = divmod(int(worker_id[2:]), 8)
    return f"{row},{col}"


@pytest.fixture
def regenerate_cpp(request):
    return not request.config.getoption("--skip-codegen")


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
    compile_producer = config.getoption("--compile-producer", default=False)
    compile_consumer = config.getoption("--compile-consumer", default=False)
    TestConfig.setup_mode(compile_consumer, compile_producer)

    with_coverage = config.getoption("--coverage", default=False)
    detailed_artefacts = config.getoption("--detailed-artefacts", default=False)
    no_debug_symbols = config.getoption("--no-debug-symbols", default=False)
    speed_of_light = config.getoption("--speed-of-light", default=False)

    TestConfig.setup_build(
        Path(os.environ["LLK_HOME"]),
        with_coverage,
        detailed_artefacts,
        no_debug_symbols,
        speed_of_light,
    )

    # Create directories from all processes - lock in create_directories handles race
    TestConfig.create_build_directories()

    log_file = "pytest_errors.log"
    if not hasattr(config, "workerinput"):
        check_hardware_headers()
        if os.path.exists(log_file):
            os.remove(log_file)

    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    initialize_test_target_from_pytest(config)
    test_target = TestTargetConfig()

    if TestConfig.MODE != TestMode.PRODUCE:
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

    if not os.path.exists(test_order_file):
        raise FileNotFoundError(f"Test order file not found: {test_order_file}")

    with open(test_order_file, "r") as f:
        ordered_tests = [line.strip() for line in f if line.strip()]

    items_dict = {item.nodeid: item for item in items}
    items[:] = [items_dict[nodeid] for nodeid in ordered_tests if nodeid in items_dict]


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
    # Execute all other hooks to obtain the report object
    outcome = yield
    report = outcome.get_result()

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

        if TestConfig.SKIP_JUST_FOR_COMPILE_MARKER in skip_reason:
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

                if exc_type == LLKAssertException:
                    # Pretty print
                    exc_msg = str(call.excinfo.value) if call.excinfo.value.args else ""
                    error_message = f"LLK ASSERT HIT {test_file_and_func}{report.test_params} {exc_msg}"
                    report.longrepr = error_message
                elif exc_type == TimeoutError:
                    exc_msg = str(call.excinfo.value) if call.excinfo.value.args else ""
                    error_message = f"TENSIX TIMED OUT {test_file_and_func}{report.test_params} {exc_msg}"
                    report.longrepr = error_message
                elif exc_type == AssertionError:
                    # Handle assertion failures
                    exc_msg = str(call.excinfo.value) if call.excinfo.value.args else ""
                    error_message = (
                        f"⨯ {test_file_and_func}{report.test_params} {exc_msg}"
                    )
                    report.longrepr = error_message
                else:
                    exc_msg = str(call.excinfo.value) if call.excinfo.value.args else ""
                    error_message = (
                        f"⨯ {test_file_and_func}{report.test_params} Error type: {exc_type.__name__}\n"
                        f"{exc_msg}\n"
                        f"Python Call trace:\n{stack_trace_str}"
                    )
                    report.longrepr = error_message

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
    if not test_target.run_simulator and not TestConfig.MODE == TestMode.PRODUCE:
        _send_arc_message("GO_BUSY", test_target.device_id)


@pytest.fixture(scope="module", autouse=True)
def perf_report(request, worker_id):

    test_module = request.path.stem

    temp_report = PerfReport()

    try:
        yield temp_report
    except Exception as e:
        logger.warning("Perf: Unexpected error, saving report anyway: {}", e)

    if TestConfig.MODE == TestMode.PRODUCE:
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
    if not test_target.run_simulator and not TestConfig.MODE == TestMode.PRODUCE:
        _send_arc_message("GO_IDLE", test_target.device_id)

    if TestConfig.MODE != TestMode.PRODUCE:
        combine_perf_reports()
        if TestConfig.WITH_COVERAGE:
            process_coverage_run_artefacts()

    global _exalens_server
    if _exalens_server is not None:
        _exalens_server.stop()
        _exalens_server = None


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
        "--test-order-file",
        action="store",
        default=None,
        help="Path to file containing ordered list of tests to run",
    )


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
