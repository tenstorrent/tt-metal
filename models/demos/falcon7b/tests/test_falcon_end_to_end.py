# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import tt_lib
from loguru import logger
from models.demos.falcon7b.reference.hf_modeling_falcon import FalconForCausalLM
from models.demos.falcon7b.tests.test_utils import concat_device_out_layer_present, get_rand_falcon_inputs
from models.demos.falcon7b.tt.falcon_causallm import TtFalconCausalLM

# TODO: Remove this?
from models.demos.falcon7b.tt.falcon_common import PytorchFalconCausalLM
from models.demos.falcon7b.tt.model_config import get_model_config
from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    is_e75,
    profiler,
    skip_for_wormhole_b0,
    torch2tt_tensor,
    tt2torch_tensor,
)
from sklearn.metrics import top_k_accuracy_score
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


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
    pcc,
    model_config,
    tt_cache_path,
    model_location_generator,
):
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
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output: {output_pcc}")

    reference_logits = pytorch_out.view(global_batch * seq_len, -1).float().detach().numpy()
    eval_logits = tt_out.view(global_batch * seq_len, -1).float().detach().numpy()
    reference_top1 = np.argmax(reference_logits, axis=-1)
    top1_acc = top_k_accuracy_score(reference_top1, eval_logits, k=1, labels=np.arange(eval_logits.shape[-1]))
    top5_acc = top_k_accuracy_score(reference_top1, eval_logits, k=5, labels=np.arange(eval_logits.shape[-1]))
    logger.info(f"Top-1 Accuracy: {top1_acc}")
    logger.info(f"Top-5 Accuracy: {top5_acc}")

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

        does_pass2, output_pcc = comp_pcc(pytorch_layer_pres[0], tt_layer_pres[0], pcc)
        logger.info(f"K Cache Layer {i}: {output_pcc}")

        does_pass = does_pass and does_pass2

        does_pass2, output_pcc = comp_pcc(pytorch_layer_pres[1], tt_layer_pres[1], pcc)
        logger.info(f"V Cache Layer {i}: {output_pcc}")

        does_pass = does_pass and does_pass2

    profiler.print()


@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 2, 128, 0),
        ("prefill", 2, 1024, 0),
        ("decode", 32, 1, 128),
        ("decode", 32, 1, 1024),
    ),
    ids=["prefill_seq128", "prefill_seq1024", "decode_batch32", "decode_batch32_1024"],
)
@pytest.mark.parametrize(
    "num_layers, pcc",
    ((2, 0.98), (32, 0.86)),
    ids=["layers_2", "layers_32"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
@skip_for_wormhole_b0(reason_str="Hangs way too often, issue #4425")
def test_FalconCausalLM_end_to_end_with_program_cache(
    device,
    use_program_cache,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    pcc,
    request,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
):
    if is_e75(device) and batch == 32:
        pytest.skip("Falcon batch 32 is unsupported on E75")

    is_low_card_setup = tt_lib.device.GetNumPCIeDevices() <= 1 or device.arch() == tt_lib.device.Arch.GRAYSKULL

    current_low_card_gs_only_working_config = not (
        model_config_str != "BFLOAT16-L1"
        or llm_mode != "decode"
        or batch != 32
        or kv_cache_len != 128
        or num_layers != 32
    )

    if (
        is_low_card_setup
        and (model_config_str != "BFLOAT16-L1" or llm_mode != "prefill" or num_layers != 32)
        and not current_low_card_gs_only_working_config
    ):
        pytest.skip(
            "Single-card falcon for both archs must run with config: BFLOAT16-L1-falcon_7b-layers_32-prefill_seq128"
        )

    # gs only
    if is_low_card_setup and device.arch() != tt_lib.device.Arch.GRAYSKULL and current_low_card_gs_only_working_config:
        pytest.skip(
            "Single-card falcon cannot run this config on non-Grayskull: BFLOAT16-L1-falcon_7b-layers_32-decode_batch32"
        )

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
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
