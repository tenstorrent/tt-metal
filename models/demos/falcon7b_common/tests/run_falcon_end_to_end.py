# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from enum import Enum
import torch
from loguru import logger
import numpy as np
from sklearn.metrics import top_k_accuracy_score

from models.demos.falcon7b_common.tt.falcon_causallm import TtFalconCausalLM

from models.demos.falcon7b_common.tt.falcon_common import (
    PytorchFalconCausalLM,
)

from models.demos.falcon7b_common.tests.test_utils import (
    get_rand_falcon_inputs,
    concat_device_out_layer_present,
    load_hf_model,
    synchronize_devices,
    get_num_devices,
    dump_device_profiler,
)

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    get_atol_rtol_pcc,
)

from models.utility_functions import (
    tt_tensors_to_torch_tensors,
    profiler,
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


class DeviceSetup(Enum):
    GRAYSKULL = 0
    WORMHOLE_B0 = 1
    T3000 = 2


# CONFIG_TO_PCC[arch][model_config_str][seq_len] = (output_pcc, k_cache_pcc, v_cache_pcc)
PREFILL_CONFIG_TO_PCC = {
    DeviceSetup.GRAYSKULL: {
        "BFLOAT16-DRAM": {
            128: (0.88, 0.97, 0.88),
            256: (0.92, 0.97, 0.88),
        },
        "BFLOAT16-L1": {
            128: (0.88, 0.97, 0.88),
            256: (0.92, 0.97, 0.88),
        },
    },
    DeviceSetup.WORMHOLE_B0: {
        "BFLOAT16-DRAM": {
            128: (0.99, 0.99, 0.99),
            256: (0.99, 0.99, 0.99),
            1024: (0.99, 0.99, 0.99),
            2048: (0.99, 0.99, 0.99),
        }
    },
    DeviceSetup.T3000: {
        "BFLOAT16-DRAM": {
            128: (0.99, 0.99, 0.99),
            256: (0.99, 0.99, 0.98),
            1024: (0.99, 0.99, 0.99),
            2048: (0.99, 0.99, 0.99),
        }
    },
}

# CONFIG_TO_PCC[arch][model_config_str][kv_cache_len] = (output_pcc, k_cache_pcc, v_cache_pcc)
DECODE_CONFIG_TO_PCC = {
    DeviceSetup.GRAYSKULL: {
        "BFLOAT16-DRAM": {128: (0.65, 0.77, 0.79), 1024: (0.59, 0.85, 0.85), 2047: (0.55, 0.95, 0.94)},
        "BFLOAT16-L1": {128: (0.65, 0.77, 0.79), 1024: (0.59, 0.85, 0.85), 2047: (0.55, 0.95, 0.94)},
    },
    DeviceSetup.WORMHOLE_B0: {
        "BFLOAT16-DRAM": {128: (0.89, 0.92, 0.91), 1024: (0.92, 0.94, 0.95), 2047: (0.95, 0.96, 0.97)},
        "BFLOAT16-L1": {128: (0.89, 0.92, 0.91), 1024: (0.92, 0.94, 0.95), 2047: (0.95, 0.96, 0.97)},
        "BFLOAT16-L1_SHARDED": {128: (0.90, 0.91, 0.91), 1024: (0.93, 0.94, 0.96), 2047: (0.92, 0.93, 0.94)},
    },
    DeviceSetup.T3000: {
        "BFLOAT16-L1_SHARDED": {128: (0.85, 0.89, 0.90), 1024: (0.90, 0.92, 0.93), 2047: (0.95, 0.91, 0.89)}
    },
}


def run_test_FalconCausalLM_end_to_end(
    mesh_device,  # can be ttnn.Device or ttnn.MeshDevice
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
    e2e_perf=False,
    expected_inference_time=None,
    device_perf=False,
    async_mode=False,
):
    assert not (e2e_perf and device_perf), "Cannot run both e2e and device perf test at the same time"
    if e2e_perf:
        assert expected_inference_time is not None, "Expected inference time is required for e2e perf test"

    if device_perf:  # Enable tracy signpost support in device perf runs only
        from tracy import signpost

    # Clear global profiler state before starting measurements
    if e2e_perf:
        profiler.clear()
    else:
        profiler.disable()

    num_devices = get_num_devices(mesh_device)
    global_batch = batch * num_devices

    profiler.start("hugging_face_model_setup")
    hugging_face_reference_model, state_dict = load_hf_model(model_location_generator, model_version)
    configuration = hugging_face_reference_model.config
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

    profiler.start("TtFalcon_model_setup")
    tt_FalconCausalLM = TtFalconCausalLM(
        mesh_device,
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
        mesh_device,
        global_batch,
        head_dim,
        max_position_embeddings,
        configuration,
        model_config,
        num_layers=num_layers,
        generate_attention_inputs=False,
    )

    if not device_perf:
        # Do warmp up run unless testing device perf -------------------------------------------

        profiler.start("processing_of_input")
        # TODO: Generate attention_mask on device
        tt_input_ids, tt_attention_mask = get_inputs_on_device(
            llm_mode, tt_FalconCausalLM, model_input, kv_cache_len, seq_len, batch, kv_len
        )
        profiler.end("processing_of_input")

        # First run to fill compile cache ----------------------------------------------------
        logger.info(f"Running Falcon model once to fill caches")
        profiler.disable()

        # Use force enable to only record this profiler call while others are disabled
        profiler.start("first_model_run_with_compile", force_enable=e2e_perf)
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
        synchronize_devices(mesh_device)
        profiler.end("first_model_run_with_compile", force_enable=e2e_perf)

        # Dump device profiler data before second run to avoid exceeding profiler memory limits when using tracy
        dump_device_profiler(mesh_device)

        del tt_out
        del tt_layer_past
        del tt_layer_present
        del tt_input_ids
        del tt_attention_mask

    # Generate dummy kv_cache ------------------------------------------------------------
    (
        past_key_values,
        tt_layer_past,
        kv_len,
    ) = get_rand_falcon_inputs(
        llm_mode,
        seq_len,
        batch,
        kv_cache_len,
        mesh_device,
        global_batch,
        head_dim,
        max_position_embeddings,
        configuration,
        model_config,
        num_layers=num_layers,
        generate_attention_inputs=False,
    )

    # Prepare reference output -----------------------------------------------------------

    profiler.start("hugging_face_reference_model")
    pytorch_out, pytorch_layer_present = pytorch_FalconCausalLM(
        input_ids=model_input, past_key_values=past_key_values, use_cache=use_cache
    )
    profiler.end("hugging_face_reference_model")

    # Run model --------------------------------------------------------------------------

    if e2e_perf:
        logger.info(f"Enable profiler")
        profiler.enable()

    # Regenerate input ids and attention_mask on device
    tt_input_ids, tt_attention_mask = get_inputs_on_device(
        llm_mode, tt_FalconCausalLM, model_input, kv_cache_len, seq_len, batch, kv_len
    )

    profiler.start(f"model_run_for_inference")
    if device_perf:
        signpost("start")  # start device perf measurement

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
                device_perf_run=device_perf,
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
            device_perf_run=device_perf,
        )
    synchronize_devices(mesh_device)
    profiler.end(f"model_run_for_inference")

    if llm_mode == "prefill":
        tt_out_tmp = torch.zeros(global_batch, seq_len, configuration.vocab_size)  # Output tensor to overwrite
        for user_id, tt_out in enumerate(tt_outs):
            # Get outputs from all devices
            tt_out_tmp[user_id::batch] = tt_tensors_to_torch_tensors(tt_out, mesh_device, concat_dim=0).squeeze(1)
        tt_out = tt_out_tmp
    elif llm_mode == "decode":
        tt_out = tt_tensors_to_torch_tensors(tt_out, mesh_device, concat_dim=2).squeeze(1).transpose(0, 1)

    if device_perf:
        signpost("stop")  # stop device perf measurement

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
            tt_layer_pres = concat_device_out_layer_present(mesh_device, tt_layer_present[i], kv_len)
        elif llm_mode == "decode":
            pytorch_layer_pres = (
                pytorch_layer_present[i][0].squeeze(1)[:, kv_cache_len, :],
                pytorch_layer_present[i][1].squeeze(1)[:, kv_cache_len, :],
            )
            tt_layer_pres = concat_device_out_layer_present(
                mesh_device, tt_layer_present[i], kv_cache_len, end_idx_only=True
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

    if e2e_perf:
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
