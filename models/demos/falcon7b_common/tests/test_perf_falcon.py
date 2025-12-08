# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from models.demos.falcon7b_common.tests.run_falcon_end_to_end import (
    DECODE_CONFIG_TO_PCC,
    PREFILL_CONFIG_TO_PCC,
    DeviceSetup,
    run_test_FalconCausalLM_end_to_end,
)
from models.demos.falcon7b_common.tt.model_config import get_model_config
from models.tt_transformers.tt.common import get_hf_tt_cache_path


@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
class TestParametrized:
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
        mesh_device,
    ):
        if model_config_str == "BFLOAT16-L1_SHARDED" and llm_mode == "prefill":
            pytest.skip(f"prefill does not support L1_SHARDED")

        model_config = get_model_config(model_config_str, seq_len, batch)
        tt_cache_path = Path(get_hf_tt_cache_path(model_version))

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
            e2e_perf=True,
            expected_inference_time=expected_inference_time,
        )

    @pytest.mark.models_performance_bare_metal
    @pytest.mark.parametrize(
        "llm_mode, num_layers, batch, seq_len, kv_cache_len, model_config_str, expected_inference_time",
        (
            ("prefill", 32, 1, 128, 0, "BFLOAT16-DRAM", 0.064),
            ("prefill", 32, 1, 1024, 0, "BFLOAT16-DRAM", 0.41),
            ("prefill", 32, 1, 2048, 0, "BFLOAT16-DRAM", 0.89),
            ("decode", 32, 32, 1, 128, "BFLOAT16-L1_SHARDED", 0.063),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-L1_SHARDED", 0.065),
            ("decode", 32, 32, 1, 2047, "BFLOAT16-L1_SHARDED", 0.064),
        ),
        ids=[
            "prefill_seq128_bf16_dram",
            "prefill_seq1024_bf16_dram",
            "prefill_seq2048_bf16_dram",
            "decode_batch32_128_bf16_l1_sharded",
            "decode_batch32_1024_bf16_l1_sharded",
            "decode_batch32_2047_bf16_l1_sharded",
        ],
    )
    @pytest.mark.parametrize("mesh_device", (1,), indirect=True)
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
        mesh_device,
    ):
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
            mesh_device,
        )

    @pytest.mark.model_perf_t3000
    @pytest.mark.parametrize(
        "llm_mode, mesh_device, num_layers, batch, seq_len, kv_cache_len, model_config_str, expected_inference_time",
        (
            ("prefill", 4, 32, 1, 128, 0, "BFLOAT16-DRAM", 0.070),
            ("prefill", 4, 32, 1, 256, 0, "BFLOAT16-DRAM", 0.142),
            ("prefill", 4, 32, 1, 1024, 0, "BFLOAT16-DRAM", 0.41),
            ("prefill", 4, 32, 1, 2048, 0, "BFLOAT16-DRAM", 0.98),
            ("decode", 4, 32, 32, 1, 128, "BFLOAT16-L1_SHARDED", 0.059),
            ("decode", 4, 32, 32, 1, 1024, "BFLOAT16-L1_SHARDED", 0.065),
            ("decode", 4, 32, 32, 1, 2047, "BFLOAT16-L1_SHARDED", 0.071),
        ),
        ids=[
            "prefill_seq128",
            "prefill_seq256",
            "prefill_seq1024",
            "prefill_seq2048",
            "decode_batch32_128",
            "decode_batch32_1024",
            "decode_batch32_2047",
        ],
        indirect=["mesh_device"],
    )
    def test_perf_t3000_bare_metal(
        self,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        expected_inference_time,
        num_layers,
        model_config_str,
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
            mesh_device,
        )
