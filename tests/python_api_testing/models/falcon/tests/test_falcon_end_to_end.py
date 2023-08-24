import torch
import pytest
from loguru import logger
from pathlib import Path

import tt_lib
from tests.python_api_testing.models.falcon.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from tests.python_api_testing.models.falcon.falcon_causallm import TtFalconCausalLM

# TODO: Remove this?
from tests.python_api_testing.models.falcon.falcon_common import (
    PytorchFalconCausalLM,
)

from tests.python_api_testing.models.falcon.model_config import (
    get_model_config,
    get_tt_cache_path,
)

from tests.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    disable_compilation_reports,
)


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
):
    model_name = model_location_generator(model_version, model_subdir="Falcon")

    profiler.start("hugging_face_model_setup")
    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()
    pytorch_FalconCausalLM = PytorchFalconCausalLM(
        hugging_face_reference_model, num_layers
    )
    profiler.end("hugging_face_model_setup")

    # Prepare input ------------------------------------------------------------------------
    torch.manual_seed(0)
    base_url = ""
    max_position_embeddings = 2048
    head_dim = configuration.hidden_size // configuration.n_head

    if 1:
        model_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
    else:
        # batch identical sequences for debugging
        model_input = torch.stack([torch.arange(seq_len)] * batch).reshape(
            batch, seq_len
        )

    # Generate dummy kv_cache --------------------------------------------------------------
    if llm_mode == "prefill":
        q_len, kv_len = seq_len, seq_len
        assert batch == 1, "For prefill, batch must be 1!"
        assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
        assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

        past_key_values = None
        tt_layer_past = None

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        k_cache = torch.rand(batch, 1, kv_cache_len, head_dim)
        v_cache = torch.rand(batch, 1, kv_cache_len, head_dim)
        past_key_values = ((k_cache, v_cache),)

        tt_k_cache = torch.zeros(batch, 1, max_position_embeddings, head_dim)
        tt_v_cache = torch.zeros(batch, 1, max_position_embeddings, head_dim)
        tt_k_cache[:, :, :kv_cache_len, :] = k_cache
        tt_v_cache[:, :, :kv_cache_len, :] = v_cache
        tt_k_cache = torch2tt_tensor(tt_k_cache, device)
        tt_v_cache = torch2tt_tensor(tt_v_cache, device)
        tt_layer_past = (tt_k_cache, tt_v_cache)

    else:
        raise NotImplementedError(
            f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode."
        )

    # Prepare output -----------------------------------------------------------------------
    profiler.start("hugging_face_reference_model")
    pytorch_out = pytorch_FalconCausalLM(
        input_ids=model_input, past_key_values=past_key_values
    )
    profiler.end("hugging_face_reference_model")

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders

    profiler.start("TtFalcon_model_setup")
    tt_FalconCausalLM = TtFalconCausalLM(
        device,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_position_embeddings,
        llm_mode,
        model_config,
        tt_cache_path,
    )
    profiler.end("TtFalcon_model_setup")

    profiler.start("processing_of_input")
    # TODO: Generate embeddings and attention_mask on device
    tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
        model_input, kv_cache_len
    )
    profiler.end("processing_of_input")

    # First run to fill compile cache ----------------------------------------------------
    logger.info(f"Running Falcon model once to fill caches -> disable profiler")
    profiler.disable()

    # Use force enable to only record this profiler call while others are disabled
    profiler.start("first_model_run_with_compile", force_enable=True)
    tt_out = tt_FalconCausalLM(
        input_embeddings=tt_embeddings,
        attention_mask=tt_attention_mask,
        layer_past=tt_layer_past,
        layer_past_len=kv_cache_len,
    )
    tt_lib.device.Synchronize()
    profiler.end("first_model_run_with_compile", force_enable=True)
    del tt_out

    # Second run for perf ----------------------------------------------------------------
    logger.info(f"Enable profiler and enable binary and compile cache")
    profiler.enable()
    enable_persistent_kernel_cache()
    tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
        model_input, kv_cache_len
    )
    profiler.start(f"model_run_for_inference")
    tt_out = tt_FalconCausalLM(
        input_embeddings=tt_embeddings,
        attention_mask=tt_attention_mask,
        layer_past=tt_layer_past,
        layer_past_len=kv_cache_len,
    )
    tt_lib.device.Synchronize()
    profiler.end(f"model_run_for_inference")
    tt_out = tt2torch_tensor(tt_out).squeeze(1)
    if llm_mode == "decode":
        tt_out = tt_out.transpose(0, 1)

    # check outputs ----------------------------------------------------------------------
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    profiler.print()

    if does_pass:
        logger.info("Falcon CausalLM Passed!")
    else:
        logger.warning("Falcon CausalLM Failed!")
        # TODO: Fix PCC for decode and uncomment this
        # assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 1, 128, 0),
        ("decode", 32, 1, 128),
        ("decode", 32, 1, 1024),
    ),
    ids=["prefill_seq128", "decode_batch32", "decode_batch32_1024"],
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
def test_FalconCausalLM_end_to_end_with_program_cache(
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
):
    model_config = get_model_config(model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    tt_lib.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/falcon-7b_{request.node.callspec.id}"
    )

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
    )
    tt_lib.device.CloseDevice(device)
