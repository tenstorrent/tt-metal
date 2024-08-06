import os
import sys
import glob
import pytest

from models.demos.t3000.mixtral8x7b.scripts.op_perf_results import main as calculate_op_perf_results
from tt_metal.tools.profiler.process_model_log import run_device_profiler


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "test, skip_first, skip_last, expected_throughput",
    [
        ["prefill_seq128_bfp8_layers1", 3, 7, 1900],
        ["prefill_seq2048_bfp8_layers1", 3, 9, 3400],
    ],
)
def test_perf_device_bare_metal(test, skip_first, skip_last, expected_throughput):
    """
    Test the performance of the device in bare metal mode.

    Args:
        test (str): The test configuration.
        skip_first (int): The number of starting ops not in layer of LLM to skip.
        skip_last (int): The number of final ops not in layer of LLM to skip.
        expected_throughput (int): The expected throughput in tokens per second.
    Raises:
        AssertionError: If the measured throughput is less than the expected throughput.
    """
    # Prepare the command and other arguments to run the profiler
    subdir = f"falcon_40b"
    args = f"wormhole_b0-True-{subdir}-{test}-8chips"
    llm_mode, seq_len, *_ = test.split("_")
    seq_len = seq_len.replace("seq", "")
    command = f"pytest models/demos/t3000/falcon40b/tests/test_perf_falcon.py::test_device_perf_bare_metal[{args}]"
    env_vars = os.environ.copy()

    # Run the profiler to get the ops performance results
    output_logs_dir = run_device_profiler(command, subdir, env_vars)

    # Get the latest date dir
    latest_date_dir = max(os.listdir(output_logs_dir))
    latest_date_dir = os.path.join(output_logs_dir, latest_date_dir)

    # Get latest dir with ops_perf_results and extract the filename
    ops_perf_filename = glob.glob(f"{latest_date_dir}/ops_perf_results*.csv", recursive=True)[0]

    # Prepare the arguments to calculate the ops performance results
    sys.argv = [
        "op_perf_results.py",
        f"{ops_perf_filename}",
        "--signpost",
        "PERF_RUN",
        "--skip-first",
        f"{skip_first}",
        "--skip-last",
        f"{skip_last}",
        f"--{llm_mode}",
        "--seqlen",
        f"{seq_len}",
        "--estimate-full-model",
        "60",
    ]

    # Calculate the ops performance results
    tokens_per_sec = calculate_op_perf_results()
    assert tokens_per_sec >= expected_throughput
