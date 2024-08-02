import os
import sys
import glob
import pytest

from models.demos.t3000.mixtral8x7b.scripts.op_perf_results import main as calculate_op_perf_results
from tt_metal.tools.profiler.process_model_log import run_device_profiler


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "test, expected_throughput",
    [
        ["decode_32", 26],
        # ["decode_128", 26],  # FIXME: #9028
        ["decode_1k", 24],
        ["decode_2k", 23],
        ["prefill_128", 1000],
        ["prefill_1k", 2550],
        ["prefill_2k", 2714],
    ],
    ids=[
        "decode_32",
        # "decode_128",
        "decode_1k",
        "decode_2k",
        "prefill_128",
        "prefill_1k",
        "prefill_2k",
    ],
)
def test_perf_device_bare_metal(test, expected_throughput):
    """
    Test the performance of the device in bare metal mode.
    Args:
        test (str): The test configuration.
        expected_throughput (int): The expected throughput in tokens per second.
    Raises:
        AssertionError: If the measured throughput is less than the expected throughput.
    """
    # Prepare the command and other arguments to run the profiler
    subdir = f"Mixtral8x7b"
    args = f"wormhole_b0-True-{test}"
    llm_mode, seq_len = test.split("_")
    seq_len = seq_len.replace("1k", "1024").replace("2k", "2048")
    signpost = f"{llm_mode} perf run"  # The name of the signpost used for profiling.
    command = (
        f"pytest models/demos/t3000/mixtral8x7b/tests/test_mixtral_perf.py::test_mixtral_model_{llm_mode}_perf[{args}]"
    )
    env_vars = os.environ.copy()

    # Run the profiler to get the ops performance results
    output_logs_dir = run_device_profiler(command, subdir, env_vars)

    # Get the latest date dir
    latest_date_dir = max(os.listdir(output_logs_dir))
    latest_date_dir = os.path.join(output_logs_dir, latest_date_dir)

    # Get latest dir with ops_perf_results and extract the filename
    ops_perf_filename = glob.glob(f"{latest_date_dir}/ops_perf_results*.csv", recursive=True)[0]

    # Prepare the arguments to calculate the ops performance results
    if llm_mode == "prefill":
        sys.argv = [
            "op_perf_results.py",
            f"{ops_perf_filename}",
            "--signpost",
            f"{signpost}",
            f"--prefill",
            "--seqlen",
            f"{seq_len}",
        ]
    else:
        sys.argv = [
            "op_perf_results.py",
            f"{ops_perf_filename}",
            "--signpost",
            f"{signpost}",
            "--seqlen",
            f"{seq_len}",
        ]

    # Calculate the ops performance results
    tokens_per_sec = calculate_op_perf_results()
    if llm_mode == "prefill":
        assert tokens_per_sec >= expected_throughput
    else:  # In decode mode the script will actually measure tokens/s/user (32 users). We want this, so this makes sure the assert will use the correct nomenclature
        tokens_per_sec_user = tokens_per_sec
        assert tokens_per_sec_user >= expected_throughput
