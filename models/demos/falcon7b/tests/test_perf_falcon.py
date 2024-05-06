# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import numpy as np
from sklearn.metrics import top_k_accuracy_score

import tt_lib
from models.demos.falcon7b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from models.demos.falcon7b.tt.falcon_causallm import TtFalconCausalLM

# TODO: Remove this?
from models.demos.falcon7b.tt.falcon_common import (
    PytorchFalconCausalLM,
)

from models.demos.falcon7b.tt.model_config import (
    get_model_config,
)
from models.demos.falcon7b.tests.test_utils import get_rand_falcon_inputs, concat_device_out_layer_present
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    get_atol_rtol_pcc,
)

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    disable_compilation_reports,
    is_e75,
    is_wormhole_b0,
    skip_for_grayskull,
    skip_for_wormhole_b0,
    get_devices_for_t3000,
)
from models.perf.perf_utils import prep_perf_report


def get_inputs_on_device(llm_mode, tt_FalconCausalLM, model_input, kv_cache_len, seq_len, batch, kv_len):
    if llm_mode == "prefill":
        tt_input_ids, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(
                    llm_mode, model_input[i::batch], kv_cache_len, num_input_tokens=seq_len
                )
                for i in range(batch)
            ]
        )
    elif llm_mode == "decode":
        tt_input_ids, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )
    return tt_input_ids, tt_attention_mask


# TODO: Replace this with actual Falcon application-level tests
def run_test_FalconCausalLM_end_to_end(
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
    async_mode=False,
):
    # Clear global profiler state before starting measurements
    profiler.clear()

    num_devices = len(devices)
    global_batch = batch * num_devices
    model_name = model_location_generator(model_version, model_subdir="Falcon")

    profiler.start("hugging_face_model_setup")
    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()
    pytorch_FalconCausalLM = PytorchFalconCausalLM(hugging_face_reference_model, num_layers)
    profiler.end("hugging_face_model_setup")

    # Prepare input ------------------------------------------------------------------------
    torch.manual_seed(0)
    base_url = ""
    max_position_embeddings = 2048
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    use_cache = True

    if True:
        model_input = torch.arange(seq_len * global_batch).reshape(global_batch, seq_len)
    else:
        # batch identical sequences for debugging
        model_input = torch.stack([torch.arange(seq_len)] * global_batch).reshape(global_batch, seq_len)

    # Generate dummy kv_cache --------------------------------------------------------------
    (
        past_key_values,
        tt_layer_past,
        kv_len,
    ) = get_rand_falcon_inputs(
        llm_mode,
        seq_len,
        batch,
        kv_cache_len,
        devices,
        global_batch,
        head_dim,
        max_position_embeddings,
        configuration,
        num_layers=num_layers,
        generate_attention_inputs=False,
    )

    profiler.start("TtFalcon_model_setup")
    tt_FalconCausalLM = TtFalconCausalLM(
        devices,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_position_embeddings,
        model_config,
        tt_cache_path,
        seq_len,
    )
    profiler.end("TtFalcon_model_setup")

    profiler.start("processing_of_input")
    # TODO: Generate attention_mask on device
    tt_input_ids, tt_attention_mask = get_inputs_on_device(
        llm_mode, tt_FalconCausalLM, model_input, kv_cache_len, seq_len, batch, kv_len
    )
    profiler.end("processing_of_input")

    # First run to fill compile cache ----------------------------------------------------
    logger.info(f"Running Falcon model once to fill caches -> disable profiler")
    profiler.disable()

    # Use force enable to only record this profiler call while others are disabled
    profiler.start("first_model_run_with_compile", force_enable=True)
    if llm_mode == "prefill":
        tt_outs = []
        # Device transfer time is included in model run time for prefill
        tt_input_ids, tt_attention_mask = get_inputs_on_device(
            llm_mode, tt_FalconCausalLM, model_input, kv_cache_len, seq_len, batch, kv_len
        )
        for user_id in range(batch):
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_ids=tt_input_ids[user_id],
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask[user_id],
                user_id=user_id,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=use_cache,
            )
            tt_outs.append(tt_out)
        tt_out = tt_outs

    elif llm_mode == "decode":
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_ids=tt_input_ids,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
    for device in devices:
        tt_lib.device.Synchronize(device)
    profiler.end("first_model_run_with_compile", force_enable=True)
    del tt_out
    del tt_layer_past
    del tt_layer_present
    del tt_input_ids
    del tt_attention_mask

    # Re-generate dummy kv_cache ------------------------------------------------------------
    (
        past_key_values,
        tt_layer_past,
        kv_len,
    ) = get_rand_falcon_inputs(
        llm_mode,
        seq_len,
        batch,
        kv_cache_len,
        devices,
        global_batch,
        head_dim,
        max_position_embeddings,
        configuration,
        num_layers=num_layers,
        generate_attention_inputs=False,
    )

    # Prepare reference output -----------------------------------------------------------

    profiler.start("hugging_face_reference_model")
    pytorch_out, pytorch_layer_present = pytorch_FalconCausalLM(
        input_ids=model_input, past_key_values=past_key_values, use_cache=use_cache
    )
    profiler.end("hugging_face_reference_model")

    # Second run for perf ----------------------------------------------------------------

    logger.info(f"Enable profiler and enable binary and compile cache")
    profiler.enable()
    enable_persistent_kernel_cache()

    # Regenerate input ids and attention_mask on device
    tt_input_ids, tt_attention_mask = get_inputs_on_device(
        llm_mode, tt_FalconCausalLM, model_input, kv_cache_len, seq_len, batch, kv_len
    )

    profiler.start(f"model_run_for_inference")
    if llm_mode == "prefill":
        tt_outs = []
        # Device transfer time is included in model run time for prefill
        tt_input_ids, tt_attention_mask = get_inputs_on_device(
            llm_mode, tt_FalconCausalLM, model_input, kv_cache_len, seq_len, batch, kv_len
        )
        for user_id in range(batch):
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_ids=tt_input_ids[user_id],
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask[user_id],
                user_id=user_id,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=use_cache,
            )
            tt_outs.append(tt_out)

    elif llm_mode == "decode":
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_ids=tt_input_ids,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
    for device in devices:
        tt_lib.device.Synchronize(device)
    profiler.end(f"model_run_for_inference")

    if llm_mode == "prefill":
        tt_out_tmp = torch.zeros(global_batch, seq_len, configuration.vocab_size)  # Output tensor to overwrite
        for user_id, tt_out in enumerate(tt_outs):
            # Get outputs from all devices
            tt_out_tmp[user_id::batch] = torch.concat(
                [tt2torch_tensor(tt_out[i]).squeeze(1) for i in range(num_devices)]
            )
        tt_out = tt_out_tmp
    elif llm_mode == "decode":
        for i in range(num_devices):
            tt_out[i] = tt2torch_tensor(tt_out[i]).squeeze(1).transpose(0, 1)
        tt_out = torch.concat(tt_out)

    # check outputs ----------------------------------------------------------------------
    does_pass = True
    tt_out_tmp = tt_out.type(pytorch_out.dtype)
    _, _, device_pcc, pcc_str = get_atol_rtol_pcc(pytorch_out, tt_out_tmp)
    logger.info(f"Output: {pcc_str}")
    if device_pcc < expected_pccs[0]:
        does_pass = False
        logger.warning(f"Output PCC {device_pcc} is lower than {expected_pccs[0]}")
    if device_pcc > (expected_pccs[0] + 0.01):
        does_pass = False
        logger.warning(f"Output PCC {device_pcc} is higher than {expected_pccs[0]}. Please update the expected PCC")

    reference_logits = pytorch_out.view(global_batch * seq_len, -1).float().detach().numpy()
    eval_logits = tt_out.view(global_batch * seq_len, -1).float().detach().numpy()
    reference_top1 = np.argmax(reference_logits, axis=-1)
    top1_acc = top_k_accuracy_score(reference_top1, eval_logits, k=1, labels=np.arange(eval_logits.shape[-1]))
    top5_acc = top_k_accuracy_score(reference_top1, eval_logits, k=5, labels=np.arange(eval_logits.shape[-1]))
    logger.info(f"Top-1 Accuracy: {top1_acc}")
    logger.info(f"Top-5 Accuracy: {top5_acc}")

    device_pcc_k = 1.0
    device_pcc_v = 1.0
    for i in range(num_layers):
        if llm_mode == "prefill":
            pytorch_layer_pres = (pytorch_layer_present[i][0].squeeze(1), pytorch_layer_present[i][1].squeeze(1))
            tt_layer_pres = concat_device_out_layer_present(num_devices, tt_layer_present[i], kv_len)
        elif llm_mode == "decode":
            pytorch_layer_pres = (
                pytorch_layer_present[i][0].squeeze(1)[:, kv_cache_len, :],
                pytorch_layer_present[i][1].squeeze(1)[:, kv_cache_len, :],
            )
            tt_layer_pres = concat_device_out_layer_present(
                num_devices, tt_layer_present[i], kv_cache_len, end_idx_only=True
            )
        tt_layer_pres_0 = tt_layer_pres[0].type(pytorch_layer_pres[0].dtype)
        _, _, device_pcc, pcc_str = get_atol_rtol_pcc(pytorch_layer_pres[0], tt_layer_pres_0)
        logger.info(f"K Cache Layer {i}: {pcc_str}")
        device_pcc_k = min(device_pcc_k, device_pcc)

        tt_layer_pres_1 = tt_layer_pres[1].type(pytorch_layer_pres[1].dtype)
        _, _, device_pcc, pcc_str = get_atol_rtol_pcc(pytorch_layer_pres[1], tt_layer_pres_1)
        logger.info(f"V Cache Layer {i}: {pcc_str}")
        device_pcc_v = min(device_pcc_v, device_pcc)

    logger.info(f"Device PCC K: {device_pcc_k}")
    logger.info(f"Device PCC V: {device_pcc_v}")

    if device_pcc_k < expected_pccs[1]:
        does_pass = False
        logger.warning(f"K Cache PCC {device_pcc_k} is lower than {expected_pccs[1]}")
    if device_pcc_k > (expected_pccs[1] + 0.01):
        does_pass = False
        logger.warning(f"K Cache PCC {device_pcc_k} is higher than {expected_pccs[1]}. Please update the expected PCC")

    if device_pcc_v < expected_pccs[2]:
        does_pass = False
        logger.warning(f"V Cache PCC {device_pcc_v} is lower than {expected_pccs[2]}")
    if device_pcc_v > (expected_pccs[2] + 0.01):
        does_pass = False
        logger.warning(f"V Cache PCC {device_pcc_v} is higher than {expected_pccs[2]}. Please update the expected PCC")

    profiler.print()

    comment = f"num_devices={num_devices}_kv_cache_len={kv_cache_len}_seq_len={seq_len}_num_layers={num_layers}_config={model_config_str}_async={async_mode}"
    cpu_time = profiler.get("hugging_face_reference_model")
    first_iter_time = profiler.get("first_model_run_with_compile")
    second_iter_time = profiler.get("model_run_for_inference")
    expected_compile_time = 44
    prep_perf_report(
        model_name=f"Falcon_{llm_mode}_{comment}",
        batch_size=batch,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comment,
        inference_time_cpu=cpu_time,
    )

    compile_time = first_iter_time - second_iter_time
    logger.info(f"falcon {comment} inference time: {second_iter_time}")
    logger.info(f"falcon {comment} compile time: {compile_time}")

    if does_pass:
        logger.info("Falcon PCC Check Passed!")
    else:
        logger.warning("Falcon PCC Check Failed!")
        assert (
            does_pass
        ), f"Output PCC, k_cache_pcc, or v_cache_pcc is either lower or higher than {expected_pccs}. See earlier warnings for more details."


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
            ("prefill", 32, 1, 128, 0, "BFLOAT16-DRAM", 0.85, 0.97, 0.86, 0.33),
            ("prefill", 32, 1, 128, 0, "BFLOAT16-L1", 0.85, 0.97, 0.86, 0.31),
            ("prefill", 32, 1, 256, 0, "BFLOAT16-DRAM", 0.90, 0.97, 0.87, 0.48),
            ("prefill", 32, 1, 256, 0, "BFLOAT16-L1", 0.90, 0.97, 0.87, 0.39),
            ("decode", 32, 32, 1, 128, "BFLOAT16-DRAM", 0.63, 0.80, 0.84, 0.30),
            ("decode", 32, 32, 1, 128, "BFLOAT16-L1", 0.63, 0.80, 0.84, 0.30),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-DRAM", 0.56, 0.86, 0.88, 0.40),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-L1", 0.56, 0.86, 0.88, 0.34),
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

        model_config = get_model_config(model_config_str)
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
        if model_config_str == "BFLOAT16-L1_SHARDED" and kv_cache_len == 2047:
            pytest.skip(f"kv_cache_len={kv_cache_len} does not fit with L1_SHARDED")
        if model_config_str == "BFLOAT16-L1_SHARDED" and llm_mode == "prefill":
            pytest.skip(f"prefill does not support L1_SHARDED")
        if num_devices > 1:
            devices = get_devices_for_t3000(all_devices, num_devices)
        else:
            devices = [all_devices]
        # Enable Async Mode
        for device in devices:
            device.enable_async(async_mode)
        model_config = get_model_config(model_config_str)
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
        )

    @pytest.mark.models_performance_bare_metal
    @pytest.mark.parametrize(
        "llm_mode, num_layers, batch, seq_len, kv_cache_len, model_config_str, expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc, expected_inference_time",
        (
            ("prefill", 32, 1, 128, 0, "BFLOAT16-DRAM", 0.97, 0.99, 0.96, 0.17),
            ("prefill", 32, 1, 128, 0, "BFLOAT16-L1", 0.97, 0.99, 0.96, 0.17),
            ("prefill", 32, 1, 256, 0, "BFLOAT16-DRAM", 0.98, 0.99, 0.96, 0.2),
            ("prefill", 32, 1, 256, 0, "BFLOAT16-L1", 0.98, 0.99, 0.96, 0.2),
            ("decode", 32, 32, 1, 128, "BFLOAT16-DRAM", 0.91, 0.92, 0.93, 0.15),
            ("decode", 32, 32, 1, 128, "BFLOAT16-L1", 0.91, 0.92, 0.93, 0.15),
            ("decode", 32, 32, 1, 128, "BFLOAT16-L1_SHARDED", 0.92, 0.95, 0.95, 0.1),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-DRAM", 0.86, 0.92, 0.92, 0.4),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-L1", 0.86, 0.92, 0.92, 0.35),
            ("decode", 32, 32, 1, 1024, "BFLOAT16-L1_SHARDED", 0.85, 0.93, 0.94, 0.1),
            ("decode", 32, 32, 1, 2047, "BFLOAT16-DRAM", 0.88, 0.93, 0.93, 0.75),
            ("decode", 32, 32, 1, 2047, "BFLOAT16-L1", 0.88, 0.93, 0.93, 0.6),
        ),
        ids=[
            "prefill_seq128_bf16_dram",
            "prefill_seq128_bf16_l1",
            "prefill_seq256_bf16_dram",
            "prefill_seq256_bf16_l1",
            "decode_batch32_128_bf16_dram",
            "decode_batch32_128_bf16_l1",
            "decode_batch32_128_bf16_l1_sharded",
            "decode_batch32_1024_bf16_dram",
            "decode_batch32_1024_bf16_l1",
            "decode_batch32_1024_bf16_l1_sharded",
            "decode_batch32_2047_bf16_dram",
            "decode_batch32_2047_bf16_l1",
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
            if llm_mode == "prefill" and seq_len == 128:
                pytest.skip(
                    f"Skipping {llm_mode} with {seq_len} in async mode. Config is supported but provides redundant testing."
                )
            if llm_mode == "decode" and not (kv_cache_len == 2047):
                if not (model_config_str == "BFLOAT16-L1_SHARDED" and kv_cache_len == 1024):
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

    @pytest.mark.models_performance_bare_metal_multi_device
    @pytest.mark.parametrize(
        "llm_mode, num_devices, num_layers, batch, seq_len, kv_cache_len, model_config_str, expected_output_pcc, expected_k_cache_pcc, expected_v_cache_pcc, expected_inference_time, async_mode",
        (
            ("prefill", 4, 32, 1, 256, 0, "BFLOAT16-DRAM", 0.98, 0.99, 0.96, 0.225, False),  # Issue 7816 Inference time
            ("decode", 4, 32, 32, 1, 1024, "BFLOAT16-L1_SHARDED", 0.87, 0.91, 0.91, 0.27, False),
            ("prefill", 4, 32, 1, 256, 0, "BFLOAT16-DRAM", 0.98, 0.99, 0.96, 0.18, True),
            ("decode", 4, 32, 32, 1, 1024, "BFLOAT16-L1_SHARDED", 0.87, 0.91, 0.91, 0.10, True),
        ),
        ids=[
            "prefill_seq256",
            "decode_batch32_1024",
            "prefill_seq256_async",
            "decode_batch32_1024_async",
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


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len, expected_inference_time",
    (
        ("prefill", 1, 128, 0, 0.4),
        ("decode", 32, 1, 128, 0.3),
        # ("prefill", 1, 256, 0, 0.40),
        # ("decode", 32, 1, 1024, 0.36),
        # ("decode", 32, 1, 2047, 0.47),
    ),
    ids=[
        "prefill_seq128",
        "decode_batch32",
    ],  # "prefill_seq256","decode_batch32_1024", "decode_batch32_2047"],
)
@pytest.mark.parametrize(
    "num_layers, expected_pcc",
    ((32, 0.89),),
    ids=["layers_32"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-L1",))
def test_perf_virtual_machine(
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    expected_inference_time,
    num_layers,
    expected_pcc,
    request,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    device,
    use_program_cache,
):
    if is_e75(device) and batch == 32:
        pytest.skip("Falcon batch 32 is not supported on E75")

    model_config = get_model_config(model_config_str)
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
        [expected_pcc, expected_pcc, expected_pcc],
        model_config,
        model_config_str,
        tt_cache_path,
        model_location_generator,
        expected_inference_time,
    )
