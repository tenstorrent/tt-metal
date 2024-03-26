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
    get_tt_cache_path,
)
from models.demos.falcon7b.tests.test_utils import get_rand_falcon_inputs, concat_device_out_layer_present
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
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


def generate_embeddings(llm_mode, tt_FalconCausalLM, model_input, kv_cache_len, seq_len, batch, kv_len):
    if llm_mode == "prefill":
        tt_embeddings, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(
                    llm_mode, model_input[i::batch], kv_cache_len, num_input_tokens=seq_len
                )
                for i in range(batch)
            ]
        )
    elif llm_mode == "decode":
        tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )
    return tt_embeddings, tt_attention_mask


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
    expected_inference_time,
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

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders

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
    )
    profiler.end("TtFalcon_model_setup")

    profiler.start("processing_of_input")
    # TODO: Generate embeddings and attention_mask on device
    tt_embeddings, tt_attention_mask = generate_embeddings(
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
        # Embedding time is included in model run time for prefill
        tt_embeddings, tt_attention_mask = generate_embeddings(
            llm_mode, tt_FalconCausalLM, model_input, kv_cache_len, seq_len, batch, kv_len
        )
        for user_id in range(batch):
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_embeddings=tt_embeddings[user_id],
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
            input_embeddings=tt_embeddings,
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
    del tt_embeddings
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

    # Regenerate embeddings and attention_mask
    tt_embeddings, tt_attention_mask = generate_embeddings(
        llm_mode, tt_FalconCausalLM, model_input, kv_cache_len, seq_len, batch, kv_len
    )

    profiler.start(f"model_run_for_inference")
    if llm_mode == "prefill":
        tt_outs = []
        # Embedding time is included in model run time for prefill
        tt_embeddings, tt_attention_mask = generate_embeddings(
            llm_mode, tt_FalconCausalLM, model_input, kv_cache_len, seq_len, batch, kv_len
        )
        for user_id in range(batch):
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_embeddings=tt_embeddings[user_id],
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
            input_embeddings=tt_embeddings,
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

    comment = f"kv_cache_len={kv_cache_len}_seq_len={seq_len}_num_layers={num_layers}_config=L1-bf16"
    cpu_time = profiler.get("hugging_face_reference_model")
    first_iter_time = profiler.get("first_model_run_with_compile")
    second_iter_time = profiler.get("model_run_for_inference")
    expected_compile_time = 30
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
        if is_wormhole_b0():  # only assert for pcc on wormhole until grayskull pcc is fixed
            assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "num_layers, pcc",
    ((32, 0.88),),
    ids=["layers_32"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-L1",))
class TestParametrized:
    @pytest.mark.parametrize(
        "llm_mode, batch, seq_len, kv_cache_len, expected_inference_time",
        (
            ("prefill", 1, 128, 0, 0.30),
            ("prefill", 1, 256, 0, 0.44),
            ("decode", 32, 1, 128, 0.27),
            ("decode", 32, 1, 1024, 0.35),
            ("decode", 32, 1, 2047, 0.48),
        ),
        ids=[
            "prefill_seq128",
            "prefill_seq256",
            "decode_batch32",
            "decode_batch32_1024",
            "decode_batch32_2047",
        ],
    )
    @skip_for_wormhole_b0()
    def test_perf_gs_bare_metal(
        use_program_cache,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        expected_inference_time,
        num_layers,
        pcc,
        request,
        model_config_str,
        model_location_generator,
        device,
    ):
        if is_e75(device) and batch == 32:
            pytest.skip("Falcon batch 32 is not supported on E75")

        model_config = get_model_config(model_config_str)
        tt_cache_path = get_tt_cache_path(model_version)

        disable_persistent_kernel_cache()
        disable_compilation_reports()

        tt_lib.profiler.set_profiler_location(f"falcon-7b_{request.node.callspec.id}")

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
            expected_inference_time,
        )

    @pytest.mark.parametrize("num_devices", (1, 2, 4))
    @pytest.mark.parametrize(
        "llm_mode, batch, seq_len, kv_cache_len, expected_inference_time",
        (
            ("prefill", 1, 128, 0, 0.4),
            ("prefill", 1, 256, 0, 0.6),
            ("decode", 32, 1, 128, 0.4),
            ("decode", 32, 1, 1024, 0.5),
            ("decode", 32, 1, 2047, 0.8),
        ),
        ids=[
            "prefill_seq128",
            "prefill_seq256",
            "decode_batch32",
            "decode_batch32_1024",
            "decode_batch32_2047",
        ],
    )
    @skip_for_grayskull()
    def test_perf_wh_bare_metal(
        use_program_cache,
        model_version,
        num_devices,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        expected_inference_time,
        num_layers,
        pcc,
        request,
        model_config_str,
        model_location_generator,
        all_devices,
    ):
        if num_devices > 1:
            pytest.skip(f"num_devices={num_devices} is not supported on CI yet")
        devices = get_devices_for_t3000(all_devices, num_devices)

        model_config = get_model_config(model_config_str)
        tt_cache_path = get_tt_cache_path(model_version)

        disable_persistent_kernel_cache()
        disable_compilation_reports()

        tt_lib.profiler.set_profiler_location(f"falcon-7b_{request.node.callspec.id}")

        run_test_FalconCausalLM_end_to_end(
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
            expected_inference_time,
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
    "num_layers, pcc",
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
    use_program_cache,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    expected_inference_time,
    num_layers,
    pcc,
    request,
    model_config_str,
    model_location_generator,
    device,
):
    if is_e75(device) and batch == 32:
        pytest.skip("Falcon batch 32 is not supported on E75")

    model_config = get_model_config(model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    tt_lib.profiler.set_profiler_location(f"falcon-7b_{request.node.callspec.id}")

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
        expected_inference_time,
    )
