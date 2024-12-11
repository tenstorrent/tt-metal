import subprocess
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config


def profile_results():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    devices_data = import_log_run_stats(setup)
    deviceID = list(devices_data["devices"].keys())[0]
    total_cycle = devices_data["devices"][deviceID]["cores"]["DEVICE"]["analysis"]["device_fw_duration"]["stats"][
        "Average"
    ]
    return total_cycle


def test_cpp_unit_test():
    command = (
        "TT_METAL_DEVICE_PROFILER=1 ./build_Release_tracy/test/ttnn/unit_tests_ttnn_ccl "
        + "--gtest_filter=WorkerCclCommandProcessingKernels.ChainOfCommandProcessorsWithVaryingDataReadOrders_LocalOnly0"
    )
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    cycles = profile_results()
    print("Total Cycles spent : ", cycles)

    assert result.returncode == 0, f"Test failed with error: {result.stderr}"
