# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
from pathlib import Path

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import _send_arc_message
from helpers.format_config import InputOutputFormat
from helpers.profiler import ProfilerConfig
from helpers.target_config import TestTargetConfig, initialize_test_target_from_pytest
from helpers.test_config import TestConfig, TestMode, process_coverage_run_artefacts
from ttexalens import tt_exalens_init
from ttexalens.util import TTException


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
        "cfg_defines.h",
        "dev_mem_map.h",
        "tensix.h",
        "tensix_types.h",
    ]
    required_headers_quasar = [
        "cfg_defines.h",
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


from helpers.perf import PerfReport, combine_perf_reports


@pytest.fixture(scope="module", autouse=True)
def perf_report(request, worker_id):

    test_module = request.path.stem

    temp_report = PerfReport()

    try:
        yield temp_report
    except Exception as e:
        print("Perf: Unexpected error, Saving report anyway", e)

    if TestConfig.MODE == TestMode.PRODUCE:
        return

    if ProfilerConfig.TEST_COUNTER == 0:
        return

    temp_report.dump_csv(f"{test_module}.{worker_id}.csv")
    temp_report.post_process()
    temp_report.dump_csv(f"{test_module}.{worker_id}.post.csv")


@pytest.fixture
def regenerate_cpp(request):
    return not request.config.getoption("--skip-codegen")


def pytest_configure(config):
    compile_producer = config.getoption("--compile-producer", default=False)
    compile_consumer = config.getoption("--compile-consumer", default=False)
    TestConfig.setup_mode(compile_consumer, compile_producer)

    with_coverage = config.getoption("--coverage", default=False)
    detailed_artefacts = config.getoption("--detailed-artefacts", default=False)
    TestConfig.setup_build(
        Path(os.environ["LLK_HOME"]), with_coverage, detailed_artefacts
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
            tt_exalens_init.init_ttexalens_remote(
                port=test_target.simulator_port, use_4B_mode=False
            )
        else:
            tt_exalens_init.init_ttexalens(use_4B_mode=False)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Suppress the short test summary info section."""

    # Override the method that writes the short test summary
    def no_summary(*args, **kwargs):
        pass

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

                if exc_type == AssertionError:
                    # Handle assertion failures
                    exc_msg = str(call.excinfo.value) if call.excinfo.value.args else ""
                    error_message = (
                        f"⨯ {test_file_and_func}{report.test_params} {exc_msg}"
                    )
                    report.longrepr = error_message
                elif exc_type == TTException:
                    # Handle Our custom TTExceptions
                    exc_msg = str(call.excinfo.value) if call.excinfo.value.args else ""
                    error_message = (
                        f"⨯ {test_file_and_func}{report.test_params}\n"
                        f"TTException: {exc_msg}\n"
                        f"Call trace:\n{stack_trace_str}"
                    )
                    report.longrepr = error_message
                elif exc_type == RuntimeError:
                    exc_msg = str(call.excinfo.value) if call.excinfo.value.args else ""
                    error_message = (
                        f"⨯ {test_file_and_func}{report.test_params} RuntimeError\n"
                        f"{exc_msg}\n"
                        f"Python Call trace:\n{stack_trace_str}"
                    )
                    report.longrepr = error_message

    return report


def pytest_sessionstart(session):
    if hasattr(session.config, "workerinput"):
        return

    test_target = TestTargetConfig()
    if not test_target.run_simulator and not TestConfig.MODE == TestMode.PRODUCE:
        _send_arc_message("GO_BUSY", test_target.device_id)


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


# Define the possible custom command line options
def pytest_addoption(parser):
    parser.addoption(
        "--run_simulator", action="store_true", help="Run tests using the simulator."
    )
    parser.addoption(
        "--port",
        action="store",
        type=int,
        default=5555,
        help="Integer number of the server port.",
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
    "--coverage" in sys.argv or any("coverage" in arg for arg in sys.argv),
    reason="Coverage shouldn't be ran with this test",
)
