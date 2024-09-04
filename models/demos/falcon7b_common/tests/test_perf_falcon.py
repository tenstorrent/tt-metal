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
    skip_for_wormhole_b0,
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
        (
            ("prefill", 32, 1, 128, 0, "BFLOAT16-DRAM", 0.31),
            ("prefill", 32, 1, 128, 0, "BFLOAT16-L1", 0.29),
            ("prefill", 32, 1, 256, 0, "BFLOAT16-DRAM", 0.43),
            ("prefill", 32, 1, 256, 0, "BFLOAT16-L1", 0.34),
            ("decode", 32, 32, 1, 128, "BFLOAT16-DRAM", 0.28),
            ("decode", 32, 32, 1, 128, "BFLOAT16-L1", 0.28),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-DRAM", 0.37),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-L1", 0.31),
            ("decode", 32, 32, 1, 2047, "BFLOAT16-DRAM", 0.40),
            ("decode", 32, 32, 1, 2047, "BFLOAT16-L1", 0.35),
        ),
        ids=[
            "prefill_seq128_bf16_dram",
            "prefill_seq128_bf16_l1",
            "prefill_seq256_bf16_dram",
            "prefill_seq256_bf16_l1",
            "decode_batch32_128_bf16_dram",
            "decode_batch32_128_bf16_l1",
            "decode_batch32_1024_bf16_dram",
            "decode_batch32_1024_bf16_l1",
            "decode_batch32_2047_bf16_dram",
            "decode_batch32_2047_bf16_l1",
        ],
    )
    @skip_for_wormhole_b0()
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

    def run_perf_wh_bare_metal(
        self,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        expected_inference_time,
        num_layers,
        expected_pccs,
        model_config_str,
        model_location_generator,
        get_tt_cache_path,
        mesh_device,
        async_mode,
    ):
        if model_config_str == "BFLOAT16-L1_SHARDED" and llm_mode == "prefill":
            pytest.skip(f"prefill does not support L1_SHARDED")

        model_config = get_model_config(model_config_str, seq_len, batch)
        tt_cache_path = get_tt_cache_path(
            model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
        )

        disable_persistent_kernel_cache()
        disable_compilation_reports()

        run_test_FalconCausalLM_end_to_end(
            mesh_device,
            model_version,
            llm_mode,
            batch,
            seq_len,
            kv_cache_len,
            num_layers,
            expected_pccs,
            model_config,
            model_config_str,
            tt_cache_path,
            model_location_generator,
            e2e_perf=True,
            expected_inference_time=expected_inference_time,
            async_mode=async_mode,
        )

    @pytest.mark.models_performance_bare_metal
    @pytest.mark.parametrize(
        "llm_mode, num_layers, batch, seq_len, kv_cache_len, model_config_str, expected_inference_time",
        (
            ("prefill", 32, 1, 128, 0, "BFLOAT16-DRAM", 0.1),
            ("prefill", 32, 1, 1024, 0, "BFLOAT16-DRAM", 0.5),
            ("prefill", 32, 1, 2048, 0, "BFLOAT16-DRAM", 1.1),
            ("decode", 32, 32, 1, 128, "BFLOAT16-DRAM", 0.15),
            ("decode", 32, 32, 1, 128, "BFLOAT16-L1", 0.15),
            ("decode", 32, 32, 1, 128, "BFLOAT16-L1_SHARDED", 0.1),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-DRAM", 0.4),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-L1", 0.35),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-L1_SHARDED", 0.1),
            ("decode", 32, 32, 1, 2047, "BFLOAT16-DRAM", 0.75),
            ("decode", 32, 32, 1, 2047, "BFLOAT16-L1", 0.6),
            ("decode", 32, 32, 1, 2047, "BFLOAT16-L1_SHARDED", 0.11),
        ),
        ids=[
            "prefill_seq128_bf16_dram",
            "prefill_seq1024_bf16_dram",
            "prefill_seq2048_bf16_dram",
            "decode_batch32_128_bf16_dram",
            "decode_batch32_128_bf16_l1",
            "decode_batch32_128_bf16_l1_sharded",
            "decode_batch32_1024_bf16_dram",
            "decode_batch32_1024_bf16_l1",
            "decode_batch32_1024_bf16_l1_sharded",
            "decode_batch32_2047_bf16_dram",
            "decode_batch32_2047_bf16_l1",
            "decode_batch32_2047_bf16_l1_sharded",
        ],
    )
    @pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True, ids=["noasync", "async"])
    @pytest.mark.parametrize("mesh_device", (1,), indirect=True)
    @skip_for_grayskull()
    def test_perf_wh_bare_metal(
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
        mesh_device,
        use_program_cache,
        enable_async_mode,
    ):
        if enable_async_mode:
            if llm_mode == "decode" and not (kv_cache_len == 2047):
                pytest.skip(
                    f"Skipping {llm_mode} with {kv_cache_len} in async mode. Config is supported but provides redundant testing."
                )

        if llm_mode == "prefill":
            expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc = PREFILL_CONFIG_TO_PCC[
                DeviceSetup.WORMHOLE_B0
            ][model_config_str][seq_len]
        else:
            expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc = DECODE_CONFIG_TO_PCC[
                DeviceSetup.WORMHOLE_B0
            ][model_config_str][kv_cache_len]

        self.run_perf_wh_bare_metal(
            model_version,
            llm_mode,
            batch,
            seq_len,
            kv_cache_len,
            expected_inference_time,
            num_layers,
            [expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc],
            model_config_str,
            model_location_generator,
            get_tt_cache_path,
            mesh_device,
            enable_async_mode,
        )

    @pytest.mark.model_perf_t3000
    @pytest.mark.parametrize(
        "llm_mode, mesh_device, num_layers, batch, seq_len, kv_cache_len, model_config_str, expected_inference_time, enable_async_mode",
        (
            ("prefill", 4, 32, 1, 128, 0, "BFLOAT16-DRAM", 0.071, False),
            ("prefill", 4, 32, 1, 256, 0, "BFLOAT16-DRAM", 0.142, False),
            ("prefill", 4, 32, 1, 1024, 0, "BFLOAT16-DRAM", 0.42, False),
            ("prefill", 4, 32, 1, 2048, 0, "BFLOAT16-DRAM", 1.02, False),
            ("decode", 4, 32, 32, 1, 128, "BFLOAT16-L1_SHARDED", 0.061, False),
            ("decode", 4, 32, 32, 1, 1024, "BFLOAT16-L1_SHARDED", 0.067, False),
            ("decode", 4, 32, 32, 1, 2047, "BFLOAT16-L1_SHARDED", 0.073, False),
            ("prefill", 4, 32, 1, 128, 0, "BFLOAT16-DRAM", 0.070, True),  # Issue 9422
            ("prefill", 4, 32, 1, 256, 0, "BFLOAT16-DRAM", 0.142, True),
            ("prefill", 4, 32, 1, 1024, 0, "BFLOAT16-DRAM", 0.41, True),
            ("prefill", 4, 32, 1, 2048, 0, "BFLOAT16-DRAM", 0.98, True),
            ("decode", 4, 32, 32, 1, 128, "BFLOAT16-L1_SHARDED", 0.059, True),
            ("decode", 4, 32, 32, 1, 1024, "BFLOAT16-L1_SHARDED", 0.065, True),
            ("decode", 4, 32, 32, 1, 2047, "BFLOAT16-L1_SHARDED", 0.071, True),
        ),
        ids=[
            "prefill_seq128",
            "prefill_seq256",
            "prefill_seq1024",
            "prefill_seq2048",
            "decode_batch32_128",
            "decode_batch32_1024",
            "decode_batch32_2047",
            "prefill_seq128_async",
            "prefill_seq256_async",
            "prefill_seq1024_async",
            "prefill_seq2048_async",
            "decode_batch32_128_async",
            "decode_batch32_1024_async",
            "decode_batch32_2047_async",
        ],
        indirect=["mesh_device", "enable_async_mode"],
    )
    @skip_for_grayskull()
    def test_perf_t3000_bare_metal(
        self,
        use_program_cache,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        expected_inference_time,
        enable_async_mode,
        num_layers,
        model_config_str,
        model_location_generator,
        get_tt_cache_path,
        mesh_device,
    ):
        if llm_mode == "prefill":
            expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc = PREFILL_CONFIG_TO_PCC[DeviceSetup.T3000][
                model_config_str
            ][seq_len]
        else:
            expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc = DECODE_CONFIG_TO_PCC[DeviceSetup.T3000][
                model_config_str
            ][kv_cache_len]

        self.run_perf_wh_bare_metal(
            model_version,
            llm_mode,
            batch,
            seq_len,
            kv_cache_len,
            expected_inference_time,
            num_layers,
            [expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc],
            model_config_str,
            model_location_generator,
            get_tt_cache_path,
            mesh_device,
            enable_async_mode,
        )
