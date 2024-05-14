# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import transformers
import pytest
from loguru import logger
import numpy as np
from sklearn.metrics import top_k_accuracy_score
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.ttnn_falcon7b.tt.common import create_custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.demos.ttnn_falcon7b.tt.falcon_causallm import TtFalconCausalLM

from models.demos.ttnn_falcon7b.tt.model_config import (
    get_model_config,
    get_tt_cache_path,
)

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.demos.ttnn_falcon7b.tt.common import (
    create_custom_preprocessor,
    create_kv_cache,
)

from models.utility_functions import (
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    disable_compilation_reports,
    is_e75,
    is_wormhole_b0,
    skip_for_wormhole_b0,
)
from models.perf.perf_utils import prep_perf_report
import ttnn


# TODO: Replace this with actual Falcon application-level tests
def run_test_FalconCausalLM_end_to_end(
    device,
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

    model_name = model_location_generator(model_version, model_subdir="Falcon")

    profiler.start("hugging_face_model_setup")

    configuration = transformers.FalconConfig.from_pretrained(model_version)
    configuration.num_hidden_layers = num_layers
    model = transformers.models.falcon.modeling_falcon.FalconForCausalLM.from_pretrained(
        model_version, config=configuration
    ).eval()

    profiler.end("hugging_face_model_setup")

    # Prepare input ------------------------------------------------------------------------
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    use_cache = True
    dtype = model_config["DEFAULT_DTYPE"]
    kv_len = seq_len if llm_mode == "prefill" else kv_cache_len + 1

    if True:
        model_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
    else:
        # batch identical sequences for debugging
        model_input = torch.stack([torch.arange(seq_len)] * batch).reshape(batch, seq_len)

    # Generate dummy kv_cache --------------------------------------------------------------
    if llm_mode == "prefill":
        past_key_values = None
        tt_layer_past = ()
        for i in range(num_layers):
            _, tt_current_layer_past = create_kv_cache(llm_mode, dtype, batch, kv_cache_len, configuration, device)
            tt_layer_past += (tt_current_layer_past,)
        attention_mask = None

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"
        attention_mask = torch.ones(batch, 1, seq_len, kv_len, dtype=int)
        attention_mask = attention_mask.triu(diagonal=1)

        past_key_values = ()
        tt_layer_past = ()
        for i in range(num_layers):
            current_layer_past, tt_current_layer_past = create_kv_cache(
                llm_mode, dtype, batch, kv_cache_len, configuration, device
            )
            past_key_values += (current_layer_past,)
            tt_layer_past += (tt_current_layer_past,)

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

    # Prepare output -----------------------------------------------------------------------
    profiler.start("hugging_face_reference_model")
    pytorch_out, pytorch_layer_present = model(
        input_ids=model_input,
        attention_mask=None,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=False,
    )
    profiler.end("hugging_face_reference_model")

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders

    profiler.start("TtFalcon_model_setup")

    def convert_to_ttnn(model, name):
        return not isinstance(model, torch.nn.Embedding)

    tt_cache_path = get_tt_cache_path(model_version)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=create_custom_preprocessor(model_config, tt_cache_path=tt_cache_path, device=device),
        convert_to_ttnn=convert_to_ttnn,
    )
    tt_FalconCausalLM = TtFalconCausalLM(
        device,
        num_layers,
        configuration,
        configuration.max_position_embeddings,
        model_config,
        parameters,
    )
    profiler.end("TtFalcon_model_setup")

    profiler.start("processing_of_input")
    # TODO: Generate embeddings and attention_mask on device
    if llm_mode == "prefill":
        model_inputs = torch.split(model_input, 1)
        tt_embeddings, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
        )
    elif llm_mode == "decode":
        tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )
    profiler.end("processing_of_input")

    # First run to fill compile cache ----------------------------------------------------
    logger.info(f"Running Falcon model once to fill caches -> disable profiler")
    profiler.disable()

    # Use force enable to only record this profiler call while others are disabled
    profiler.start("first_model_run_with_compile", force_enable=True)
    if llm_mode == "prefill":
        tt_outs = []
        model_inputs = torch.split(model_input, 1)
        tt_embeddings, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
        )
        for user_id in range(batch):
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_embeddings=tt_embeddings[user_id],
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask[user_id],
                user_id=user_id,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=True,
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
    ttnn.synchronize_device(device)
    profiler.end("first_model_run_with_compile", force_enable=True)
    del tt_out
    del tt_layer_past
    del tt_layer_present
    del tt_embeddings
    del tt_attention_mask

    # Second run for perf ----------------------------------------------------------------

    logger.info(f"Enable profiler and enable binary and compile cache")
    profiler.enable()
    enable_persistent_kernel_cache()
    if llm_mode == "prefill":
        tt_layer_past = ()
        for i in range(num_layers):
            _, tt_current_layer_past = create_kv_cache(llm_mode, dtype, batch, kv_cache_len, configuration, device)
            tt_layer_past += (tt_current_layer_past,)
        attention_mask = None

    elif llm_mode == "decode":
        tt_layer_past = ()
        for i in range(num_layers):
            current_layer_past, tt_current_layer_past = create_kv_cache(
                llm_mode, dtype, batch, kv_cache_len, configuration, device
            )
            tt_layer_past += (tt_current_layer_past,)

    if llm_mode == "prefill":
        model_inputs = torch.split(model_input, 1)
        tt_embeddings, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
        )
    elif llm_mode == "decode":
        tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )

    profiler.start(f"model_run_for_inference")
    if llm_mode == "prefill":
        tt_outs = []
        model_inputs = torch.split(model_input, 1)
        tt_embeddings, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
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
    ttnn.synchronize_device(device)
    profiler.end(f"model_run_for_inference")

    if llm_mode == "prefill":
        tt_out = torch.vstack([ttnn.to_torch(tt_out).squeeze(1) for tt_out in tt_outs])
    elif llm_mode == "decode":
        tt_out = ttnn.to_torch(tt_out).squeeze(1)
        tt_out = tt_out.transpose(0, 1)

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output: {output_pcc}")

    reference_logits = pytorch_out.view(batch, -1).float().detach().numpy()
    eval_logits = tt_out.view(batch, -1).float().detach().numpy()
    reference_top1 = np.argmax(reference_logits, axis=-1)
    top1_acc = top_k_accuracy_score(reference_top1, eval_logits, k=1, labels=np.arange(eval_logits.shape[-1]))
    top5_acc = top_k_accuracy_score(reference_top1, eval_logits, k=5, labels=np.arange(eval_logits.shape[-1]))
    logger.info(f"Top-1 Accuracy: {top1_acc}")
    logger.info(f"Top-5 Accuracy: {top5_acc}")

    for i in range(num_layers):
        tt_layer_pres = (
            ttnn.to_torch(tt_layer_present[i][0]),
            ttnn.to_torch(tt_layer_present[i][1]),
        )
        if llm_mode == "prefill":
            pytorch_layer_pres = pytorch_layer_present[i]
            tt_layer_pres = (
                tt_layer_pres[0][:, :, :kv_len, :],
                tt_layer_pres[1][:, :, :kv_len, :],
            )
        elif llm_mode == "decode":
            pytorch_layer_pres = (
                pytorch_layer_present[i][0][:, :, kv_cache_len, :],
                pytorch_layer_present[i][1][:, :, kv_cache_len, :],
            )
            tt_layer_pres = (
                tt_layer_pres[0][:, :, kv_cache_len, :],
                tt_layer_pres[1][:, :, kv_cache_len, :],
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
        model_name=f"ttnn_Falcon_{llm_mode}_{comment}",
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


@skip_for_wormhole_b0(reason_str="Does not run on single WH")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len, expected_inference_time",
    (
        ("prefill", 1, 128, 0, 0.285),
        ("prefill", 1, 256, 0, 0.4),
        ("decode", 32, 1, 128, 0.25),
        ("decode", 32, 1, 1024, 0.307),
        ("decode", 32, 1, 2047, 0.34),
    ),
    ids=[
        "prefill_seq128",
        "prefill_seq256",
        "decode_batch32",
        "decode_batch32_1024",
        "decode_batch32_2047",
    ],
)
@pytest.mark.parametrize(
    "num_layers, pcc",
    ((32, 0.86),),
    ids=["layers_32"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-L1",))
def test_perf_bare_metal(
    device,
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
):
    if is_e75(device) and batch == 32:
        pytest.skip("Falcon batch 32 is not supported on E75")

    model_config = get_model_config(model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    run_test_FalconCausalLM_end_to_end(
        device,
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
