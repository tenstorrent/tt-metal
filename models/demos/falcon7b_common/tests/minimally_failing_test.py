# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.falcon7b_common.tests.run_falcon_end_to_end import (
    DECODE_CONFIG_TO_PCC,
    PREFILL_CONFIG_TO_PCC,
    DeviceSetup,
    run_test_FalconCausalLM_end_to_end,
)
from models.demos.falcon7b_common.tt.model_config import (
    get_model_config,
)

from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    is_e75,
    skip_for_grayskull,
    is_wormhole_b0,
    is_blackhole,
)


@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
class TestParametrized:
    @pytest.mark.models_performance_bare_metal
    @pytest.mark.parametrize(
        "llm_mode, num_layers, batch, seq_len, kv_cache_len, model_config_str, expected_inference_time",
        (("prefill", 32, 1, 128, 0, "BFLOAT16-DRAM", 0.31),),
        ids=[
            "prefill_seq128_bf16_dram",
        ],
    )
    @pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
    def test_perf_gs_bare_metal(
        self,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        expected_inference_time,
        num_layers,
        model_config_str,
        model_location_generator,
        get_tt_cache_path,
        device,
        use_program_cache,
    ):
        if is_e75(device) and batch == 32:
            pytest.skip("Falcon batch 32 is not supported on E75")

        if model_config_str == "BFLOAT16-L1_SHARDED":
            pytest.skip("Sharded config is not supported on GS")

        model_config = get_model_config(model_config_str, seq_len, batch)
        tt_cache_path = get_tt_cache_path(
            model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
        )

        disable_persistent_kernel_cache()
        disable_compilation_reports()

        if llm_mode == "prefill":
            expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc = PREFILL_CONFIG_TO_PCC[
                DeviceSetup.GRAYSKULL
            ][model_config_str][seq_len]
        else:
            expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc = DECODE_CONFIG_TO_PCC[
                DeviceSetup.GRAYSKULL
            ][model_config_str][kv_cache_len]

        run_test_FalconCausalLM_end_to_end(
            device,
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
            e2e_perf=True,
            expected_inference_time=expected_inference_time,
        )
