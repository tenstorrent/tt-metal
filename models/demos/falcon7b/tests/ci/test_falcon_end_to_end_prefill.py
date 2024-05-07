# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.falcon7b.tests.test_falcon_end_to_end import run_test_FalconCausalLM_end_to_end
from models.demos.falcon7b.tt.model_config import get_model_config
from models.utility_functions import disable_compilation_reports, disable_persistent_kernel_cache


@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 1, 32, 0),
        ("prefill", 1, 128, 0),
    ),
    ids=["prefill_seq32", "prefill_seq128"],
)
@pytest.mark.parametrize(
    "num_layers, out_pcc, cache_pcc",
    ((32, 0.97, 0.95),),
    ids=["layers_32"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
def test_FalconCausalLM_end_to_end_with_program_cache(
    device,
    use_program_cache,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    out_pcc,
    cache_pcc,
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
        out_pcc,
        cache_pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
