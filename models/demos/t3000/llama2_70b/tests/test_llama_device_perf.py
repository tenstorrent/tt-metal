# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from models.utility_functions import skip_for_grayskull
from models.demos.t3000.llama2_70b.tt.llama_common import setup_llama_env, check_mesh_device
from models.demos.t3000.llama2_70b.tests.test_llama_model import run_test_LlamaModel_inference
from models.demos.t3000.llama2_70b.tests.test_llama_model_t3000 import N_LAYERS_TO_PCC
from models.demos.t3000.llama2_70b.tests.test_llama_model import DEVICE_PERF_START_SIGNPOST
from models.demos.t3000.mixtral8x7b.scripts.op_perf_results import main as calculate_op_perf_results
from tt_metal.tools.profiler.process_model_log import run_device_profiler, get_latest_ops_log_filename
from models.perf.device_perf_utils import check_device_perf


@pytest.mark.parametrize(
    "llama_version",
    (("llama3"),),
)
@pytest.mark.parametrize("n_layers", (1,), ids=("1L",))
@pytest.mark.parametrize(
    "batch, seq_len, generation_start_pos",
    (
        # Decode, batch 16
        (16, 1, 127),
        (16, 1, 2047),
        (16, 1, 4095),
        (16, 1, 8191),
        # Decode, batch 32
        (32, 1, 127),
        (32, 1, 2047),
        (32, 1, 4095),
        # Prefill
        (1, 128, 0),
        (1, 2048, 0),
        (1, 4096, 0),
        (1, 8192, 0),
    ),
    ids=(
        "decode_128_batch16",
        "decode_2048_batch16",
        "decode_4096_batch16",
        "decode_8192_batch16",
        "decode_128_batch32",
        "decode_2048_batch32",
        "decode_4096_batch32",
        "prefill_128",
        "prefill_2048",
        "prefill_4096",
        "prefill_8192",
    ),
)
@skip_for_grayskull()
def test_run_device_perf_llama(
    batch,
    seq_len,
    generation_start_pos,
    n_layers,
    t3k_mesh_device,
    llama_version,
    use_program_cache,
):
    max_batch_size = batch if seq_len == 1 else 16  # max_batch_size is 16 for prefill
    max_context_len = {16: 8192, 32: 4096}[max_batch_size]  # set max context depending on max batch

    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
        max_batch_size=max_batch_size,
        max_context_len=max_context_len,
    )

    check_mesh_device(t3k_mesh_device, model_config)

    run_test_LlamaModel_inference(
        t3k_mesh_device,
        batch,
        seq_len,
        N_LAYERS_TO_PCC[n_layers],
        model_config,
        n_layers,
        llama_version,
        ckpt_dir,
        tokenizer_path,
        cache_path,
        generation_start_pos=generation_start_pos,
        device_perf=True,
    )


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "test_id, expected_throughput",
    (
        ("decode_128_batch16", 16.9),  # Issue #9028
        ("decode_2048_batch16", 0),  # Issue #9028
        ("decode_4096_batch16", 0),  # Issue #9028
        ("decode_8192_batch16", 0),  # Issue #9028
        ("decode_128_batch32", 16.6),
        ("decode_2048_batch32", 14.1),
        ("decode_4096_batch32", 12.8),
        ("prefill_128", 713),
        ("prefill_2048", 1036),
        ("prefill_4096", 1024),
        ("prefill_8192", 989),
    ),
)
@skip_for_grayskull()
def test_device_perf_llama(
    test_id,
    expected_throughput,  # t/s for prefill, t/s/u for decode
    is_ci_env,
):
    if is_ci_env:
        if test_id in ["decode_128_batch16", "decode_2048_batch16", "decode_4096_batch16", "decode_8192_batch16"]:
            pytest.skip("Skipping on CI due to Issue #9028")

    margin = 0.03
    subdir = "llama3-70b"
    command = (
        f"pytest models/demos/t3000/llama2_70b/tests/test_llama_device_perf.py::test_run_device_perf_llama -k {test_id}"
    )

    # Run profiler
    run_device_profiler(command, output_logs_subdir=subdir)

    # Prepare the arguments to calculate the ops performance results
    ops_perf_filename = get_latest_ops_log_filename(subdir)
    llm_mode, seq_len, *_ = test_id.split("_")
    if llm_mode == "decode":
        skip_first = 3  # embeddings, i2s (embeddings), i2s (rot-mat)
        skip_last = 3  # all-gather, rms-norm, lm-head
    else:
        skip_first = 1  # embeddings
        skip_last = 5  # ln pre-all-gather, all-gather, ln post-all-gather, all-gather, matmul
    n_layers_total = 80
    sys.argv = [
        "op_perf_results.py",
        f"{ops_perf_filename}",
        "--signpost",
        DEVICE_PERF_START_SIGNPOST,
        "--skip-first",
        f"{skip_first}",
        "--skip-last",
        f"{skip_last}",
        "--seqlen",
        f"{seq_len}",
        "--estimate-full-model",
        f"{n_layers_total}",
    ]
    if llm_mode == "prefill":
        sys.argv.append("--prefill")

    # Calculate the ops performance results using the system arguments above
    measured_throughput = calculate_op_perf_results()  # t/s for prefill, t/s/u for decode

    check_device_perf(
        {"throughput": measured_throughput}, margin, {"throughput": expected_throughput}, assert_on_fail=True
    )
