# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.falcon7b.tests.test_falcon_end_to_end import run_test_FalconCausalLM_end_to_end
from models.demos.falcon7b.tt.model_config import get_model_config
from models.utility_functions import disable_compilation_reports, disable_persistent_kernel_cache, skip_for_grayskull


@pytest.mark.parametrize(
    "llm_mode, num_layers, batch, seq_len, kv_cache_len, model_config_str, expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc",
    (
        ("prefill", 32, 1, 32, 0, "BFLOAT16-DRAM", 0.97, 0.95, 0.95),
        ("prefill", 32, 1, 128, 0, "BFLOAT16-DRAM", 0.97, 0.99, 0.97),
        ("prefill", 32, 1, 1024, 0, "BFLOAT16-DRAM", 0.99, 0.99, 0.98),
        (
            "prefill",
            32,
            1,
            2048,
            0,
            "BFLOAT16-DRAM",
            0.99,
            0.99,
            0.98,
        ),  # CI machines don't have enough RAM memory to run this test atm; to reduce memory usage (#8349)
    ),
    ids=[
        "prefill_seq32",
        "prefill_seq128",
        "prefill_seq1024",
        "prefill_seq2048",
    ],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@skip_for_grayskull()
def test_FalconCausalLM_end_to_end_with_program_cache(
    device,
    use_program_cache,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    expected_output_pcc,
    expected_k_cache_pcc,
    expected_v_cache_pcc,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
):
    model_config = get_model_config(model_config_str, seq_len)
    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    run_test_FalconCausalLM_end_to_end(
        [device],
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        num_layers,
        expected_output_pcc,
        expected_k_cache_pcc,
        expected_v_cache_pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
