# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.falcon7b.tests.run_falcon_end_to_end import run_test_FalconCausalLM_end_to_end
from models.demos.falcon7b.tt.model_config import (
    get_model_config,
)

from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    is_e75,
    skip_for_grayskull,
    skip_for_wormhole_b0,
    get_devices_for_t3000,
)


@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
class TestParametrized:
    @pytest.mark.models_performance_bare_metal
    @pytest.mark.parametrize(
        "llm_mode, num_layers, batch, seq_len, kv_cache_len, model_config_str, expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc, expected_inference_time",
        (
            ("prefill", 32, 1, 128, 0, "BFLOAT16-DRAM", 0.85, 0.97, 0.86, 0.31),
            ("prefill", 32, 1, 128, 0, "BFLOAT16-L1", 0.85, 0.97, 0.86, 0.29),
            ("prefill", 32, 1, 256, 0, "BFLOAT16-DRAM", 0.90, 0.97, 0.87, 0.43),
            ("prefill", 32, 1, 256, 0, "BFLOAT16-L1", 0.90, 0.97, 0.87, 0.34),
            ("decode", 32, 32, 1, 128, "BFLOAT16-DRAM", 0.63, 0.80, 0.84, 0.28),
            ("decode", 32, 32, 1, 128, "BFLOAT16-L1", 0.63, 0.80, 0.84, 0.28),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-DRAM", 0.56, 0.86, 0.88, 0.37),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-L1", 0.56, 0.86, 0.88, 0.31),
            ("decode", 32, 32, 1, 2047, "BFLOAT16-DRAM", 0.55, 0.91, 0.89, 0.40),
            ("decode", 32, 32, 1, 2047, "BFLOAT16-L1", 0.55, 0.91, 0.89, 0.35),
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
        expected_output_pcc,
        expected_k_cache_pcc,
        expected_v_cache_pcc,
        request,
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
            [expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc],
            model_config,
            model_config_str,
            tt_cache_path,
            model_location_generator,
            expected_inference_time,
            e2e_perf=True,
        )

    def run_perf_wh_bare_metal(
        self,
        model_version,
        num_devices,
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
        all_devices,
        async_mode,
    ):
        if model_config_str == "BFLOAT16-L1_SHARDED" and llm_mode == "prefill":
            pytest.skip(f"prefill does not support L1_SHARDED")
        if num_devices > 1:
            devices = get_devices_for_t3000(all_devices, num_devices)
        else:
            devices = [all_devices]
        # Enable Async Mode
        for device in devices:
            device.enable_async(async_mode)
        model_config = get_model_config(model_config_str, seq_len)
        tt_cache_path = get_tt_cache_path(
            model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
        )

        disable_persistent_kernel_cache()
        disable_compilation_reports()

        run_test_FalconCausalLM_end_to_end(
            devices,
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
            expected_inference_time,
            async_mode,
            e2e_perf=True,
        )

    @pytest.mark.models_performance_bare_metal
    @pytest.mark.parametrize(
        "llm_mode, num_layers, batch, seq_len, kv_cache_len, model_config_str, expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc, expected_inference_time",
        (
            ("prefill", 32, 1, 128, 0, "BFLOAT16-DRAM", 0.97, 0.99, 0.97, 0.1),
            ("prefill", 32, 1, 1024, 0, "BFLOAT16-DRAM", 0.99, 0.99, 0.98, 0.5),
            ("prefill", 32, 1, 2048, 0, "BFLOAT16-DRAM", 0.99, 0.99, 0.98, 1.1),
            ("decode", 32, 32, 1, 128, "BFLOAT16-DRAM", 0.91, 0.92, 0.93, 0.15),
            ("decode", 32, 32, 1, 128, "BFLOAT16-L1", 0.91, 0.92, 0.93, 0.15),
            ("decode", 32, 32, 1, 128, "BFLOAT16-L1_SHARDED", 0.92, 0.95, 0.95, 0.1),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-DRAM", 0.86, 0.92, 0.92, 0.4),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-L1", 0.86, 0.92, 0.92, 0.35),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-L1_SHARDED", 0.87, 0.94, 0.94, 0.1),
            ("decode", 32, 32, 1, 2047, "BFLOAT16-DRAM", 0.88, 0.93, 0.93, 0.75),
            ("decode", 32, 32, 1, 2047, "BFLOAT16-L1", 0.88, 0.93, 0.93, 0.6),
            ("decode", 32, 32, 1, 2047, "BFLOAT16-L1_SHARDED", 0.88, 0.92, 0.93, 0.11),
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
    @pytest.mark.parametrize("async_mode", (False, True))
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
        expected_output_pcc,
        expected_k_cache_pcc,
        expected_v_cache_pcc,
        request,
        model_config_str,
        model_location_generator,
        get_tt_cache_path,
        device,
        use_program_cache,
        async_mode,
    ):
        if async_mode:
            if llm_mode == "decode" and not (kv_cache_len == 2047):
                pytest.skip(
                    f"Skipping {llm_mode} with {kv_cache_len} in async mode. Config is supported but provides redundant testing."
                )
        self.run_perf_wh_bare_metal(
            model_version,
            1,
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
            device,
            async_mode,
        )

    @pytest.mark.model_perf_t3000
    @pytest.mark.parametrize(
        "llm_mode, num_devices, num_layers, batch, seq_len, kv_cache_len, model_config_str, expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc, expected_inference_time, async_mode",
        (
            ("prefill", 4, 32, 1, 128, 0, "BFLOAT16-DRAM", 0.98, 0.99, 0.97, 0.1, False),
            ("prefill", 4, 32, 1, 256, 0, "BFLOAT16-DRAM", 0.99, 0.99, 0.97, 0.18, False),
            ("prefill", 4, 32, 1, 1024, 0, "BFLOAT16-DRAM", 0.99, 0.99, 0.98, 0.5, False),
            ("prefill", 4, 32, 1, 2048, 0, "BFLOAT16-DRAM", 0.99, 0.99, 0.98, 1.1, False),
            ("decode", 4, 32, 32, 1, 128, "BFLOAT16-L1_SHARDED", 0.89, 0.94, 0.94, 0.09, False),
            ("decode", 4, 32, 32, 1, 1024, "BFLOAT16-L1_SHARDED", 0.86, 0.90, 0.91, 0.09, False),
            ("decode", 4, 32, 32, 1, 2047, "BFLOAT16-L1_SHARDED", 0.77, 0.69, 0.72, 0.1, False),
            ("prefill", 4, 32, 1, 128, 0, "BFLOAT16-DRAM", 0.98, 0.99, 0.97, 0.11, True),  # Issue 9422
            ("prefill", 4, 32, 1, 256, 0, "BFLOAT16-DRAM", 0.99, 0.99, 0.97, 0.18, True),
            ("prefill", 4, 32, 1, 1024, 0, "BFLOAT16-DRAM", 0.99, 0.99, 0.98, 0.5, True),
            ("prefill", 4, 32, 1, 2048, 0, "BFLOAT16-DRAM", 0.99, 0.99, 0.98, 1.1, True),
            ("decode", 4, 32, 32, 1, 128, "BFLOAT16-L1_SHARDED", 0.89, 0.94, 0.94, 0.09, True),
            ("decode", 4, 32, 32, 1, 1024, "BFLOAT16-L1_SHARDED", 0.86, 0.90, 0.91, 0.09, True),
            ("decode", 4, 32, 32, 1, 2047, "BFLOAT16-L1_SHARDED", 0.77, 0.69, 0.72, 0.09, True),
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
    )
    @skip_for_grayskull()
    def test_perf_t3000_bare_metal(
        self,
        use_program_cache,
        model_version,
        num_devices,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        expected_inference_time,
        async_mode,
        num_layers,
        expected_output_pcc,
        expected_k_cache_pcc,
        expected_v_cache_pcc,
        request,
        model_config_str,
        model_location_generator,
        get_tt_cache_path,
        all_devices,
    ):
        self.run_perf_wh_bare_metal(
            model_version,
            num_devices,
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
            all_devices,
            async_mode,
        )
