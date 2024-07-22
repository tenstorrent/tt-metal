# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.falcon7b_common.tests.run_falcon_end_to_end import (
    DECODE_CONFIG_TO_PCC,
    PREFILL_CONFIG_TO_PCC,
    DeviceSetup,
    run_test_FalconCausalLM_end_to_end,
)
from models.demos.falcon7b_common.tt.model_config import get_model_config
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.utility_functions import disable_compilation_reports, disable_persistent_kernel_cache


@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize(
    "llm_mode, num_layers, batch, seq_len, kv_cache_len, model_config_str",
    (
        ("prefill", 32, 1, 128, 0, "BFLOAT16-DRAM"),
        ("prefill", 32, 1, 1024, 0, "BFLOAT16-DRAM"),
        ("prefill", 32, 1, 2048, 0, "BFLOAT16-DRAM"),
    ),
    ids=[
        "prefill_seq128_bfloat16-dram",
        "prefill_seq1024_bfloat16-dram",
        "prefill_seq2048_bfloat16-dram",
    ],
)
def test_device_perf_wh_bare_metal(
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    device,
):
    model_config = get_model_config(model_config_str, seq_len, batch)
    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    if llm_mode == "prefill":
        expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc = PREFILL_CONFIG_TO_PCC[
            DeviceSetup.WORMHOLE_B0
        ][model_config_str][seq_len]
    else:
        expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc = DECODE_CONFIG_TO_PCC[DeviceSetup.WORMHOLE_B0][
            model_config_str
        ][kv_cache_len]

    run_test_FalconCausalLM_end_to_end(
        [device],
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        num_layers,
        [expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc],
        model_config,
        model_config_str,
        tt_cache_path,
        model_location_generator,
        device_perf=True,
    )


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("seq_len, samples", [(128, 2060), (2048, 2680)])
def test_device_perf(seq_len, samples):
    margin = 0.03
    num_iterations = 1
    model_config = "BFLOAT16-DRAM".lower()
    command = f"pytest models/demos/falcon7b_common/tests/test_falcon_device_perf.py::test_device_perf_wh_bare_metal -k prefill_seq{seq_len}_{model_config}"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    subdir = "falcon7b"

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, seq_len)

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: samples}

    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)

    prep_device_perf_report(
        model_name=f"falcon7b_prefill_{model_config}_seq{seq_len}",
        batch_size=1,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )
