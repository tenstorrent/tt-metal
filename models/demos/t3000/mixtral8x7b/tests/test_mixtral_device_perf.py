import os
import sys
import glob
import pytest

from models.demos.t3000.mixtral8x7b.scripts.op_perf_results import main as calculate_op_perf_results
from tt_metal.tools.profiler.process_model_log import run_device_profiler


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "test, signpost, llm_mode, expected_throughput",
    [
        # ["32-150-0.085", "decode" ,"Model perf run", 26],
        # ["128-150-0.085", "decode", "Model perf run", 26],  # FIXME: #9028
        # ["1024-150-0.085", "decode", "Model perf run", 26],
        # ["2048-150-0.085", "decode", "Model perf run", 26],
    ],
)
def test_perf_device_bare_metal(test, llm_mode, signpost, expected_throughput):
    """
    Test the performance of the device in bare metal mode.
    Args:
        test (str): The test configuration.
        signpost (str): The name of the signpost used for profiling.
        expected_throughput (int): The expected throughput in tokens per second.
    Raises:
        AssertionError: If the measured throughput is less than the expected throughput.
    """
    # Prepare the command and other arguments to run the profiler
    subdir = f"Mixtral8x7b"
    args = f"wormhole_b0-True-{test}"
    seq_len, *_ = test.split("-")
    # seq_len = seq_len.replace("seq", "")
    command = f"pytest models/demos/t3000/mixtral8x7b/tests/test_mixtral_perf.py::test_mixtral_model_perf[{args}]"
    env_vars = os.environ.copy()

    # Run the profiler to get the ops performance results
    output_logs_dir = run_device_profiler(command, subdir, env_vars)
    print(f"Miguel: output_logs_dir = {output_logs_dir}")

    # Get the latest date dir
    latest_date_dir = max(os.listdir(output_logs_dir))
    latest_date_dir = os.path.join(output_logs_dir, latest_date_dir)

    # Get latest dir with ops_perf_results and extract the filename
    ops_perf_filename = glob.glob(f"{latest_date_dir}/ops_perf_results*.csv", recursive=True)[0]

    # TODO All llm_mode == prefill below
    # Prepare the arguments to calculate the ops performance results
    sys.argv = [
        "op_perf_results.py",
        f"{ops_perf_filename}",
        "--signpost",
        f"{signpost}",
        # "--skip-first",
        # f"{skip_first}",
        # "--skip-last",
        # f"{skip_last}",
        # f"--{llm_mode}",
        "--seqlen",
        f"{seq_len}",
        # "--estimate-full-model",
        # "60",
    ]

    # Calculate the ops performance results
    tokens_per_sec = calculate_op_perf_results()
    assert tokens_per_sec >= expected_throughput
