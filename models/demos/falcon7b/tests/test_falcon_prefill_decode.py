# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.falcon7b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from models.demos.falcon7b.tt.falcon_causallm import TtFalconCausalLM

from models.demos.falcon7b.tt.model_config import (
    get_model_config,
    get_tt_cache_path,
)

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class PytorchFalconCausalLM(torch.nn.Module):
    def __init__(self, hf_reference_model, num_layers):
        super().__init__()
        self.model = hf_reference_model
        self.model.transformer.h = self.model.transformer.h[:num_layers]

        # Disable dropout
        self.model.eval()

    def forward(self, input_ids, past_key_values, use_cache):
        result = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=False,
        )

        return result


def run_test_FalconCausalLM_inference(
    device,
    model_version,
    batch,
    seq_len,
    max_seq_len,
    num_layers,
    pcc,
    model_config,
    tt_cache_path,
    model_location_generator,
):
    model_name = model_location_generator(model_version, model_subdir="Falcon")

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input ------------------------------------------------------------------------
    torch.manual_seed(0)
    base_url = ""
    max_position_embeddings = max_seq_len
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    use_cache = True
    kv_cache_len = seq_len  # This will increment by one after each decode

    model_prefill_input = torch.arange(seq_len * 1).reshape(1, seq_len)  # Only generate input for one user
    model_decode_input = torch.ones(batch, 1, dtype=torch.int) * 11  # Batch 32 of start token

    # SETUP
    logger.info("Setting up KV-cache")
    # Generate dummy kv_cache --------------------------------------------------------------
    q_len, kv_len = seq_len, seq_len
    assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"

    tt_layer_past = ()
    k_cache = torch.zeros(batch, 1, max_position_embeddings, head_dim)
    v_cache = torch.zeros(batch, 1, max_position_embeddings, head_dim)
    for i in range(num_layers):
        tt_k_cache = torch2tt_tensor(k_cache, device)
        tt_v_cache = torch2tt_tensor(v_cache, device)
        tt_layer_past += ((tt_k_cache, tt_v_cache),)

    # CPU REFERENCE ------------------------------------------------------------------------
    logger.info("Setting up CPU model")
    pytorch_FalconCausalLM = PytorchFalconCausalLM(hugging_face_reference_model, num_layers)
    logger.info("Running CPU prefill")
    _, pytorch_layer_present = pytorch_FalconCausalLM(
        input_ids=model_prefill_input, past_key_values=None, use_cache=use_cache
    )
    # Generate past_key_values with batch 32 but only fill in cache for first user
    past_key_values = ()
    for i in range(num_layers):
        k_cache = torch.zeros(batch, 1, kv_cache_len, head_dim)
        v_cache = torch.zeros(batch, 1, kv_cache_len, head_dim)
        k_cache[0] = pytorch_layer_present[i][0]
        v_cache[0] = pytorch_layer_present[i][1]
        past_key_values += ((k_cache, v_cache),)
    logger.info("Running CPU decode")
    pytorch_out, pytorch_layer_present = pytorch_FalconCausalLM(
        input_ids=model_decode_input,
        past_key_values=past_key_values,
        use_cache=use_cache,
    )

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders
    logger.info("Setting up Falcon model")
    tt_FalconCausalLM = TtFalconCausalLM(
        device,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    )

    # TODO: Generate embeddings and attention_mask on device
    logger.info("Setting up inputs and attention masks")
    # NOTE: tt_decode_attention_mask is only valid for one decode right after prefill here
    # TODO: If we loop, we need to decouple generation for embeddings and just generate new attention_mask
    (
        tt_prefill_embeddings,
        tt_prefill_attention_mask,
    ) = tt_FalconCausalLM.model_preprocessing("prefill", model_prefill_input, 0, num_input_tokens=seq_len)
    (
        tt_decode_embeddings,
        tt_decode_attention_mask,
    ) = tt_FalconCausalLM.model_preprocessing("decode", model_decode_input, kv_cache_len, num_input_tokens=seq_len + 1)

    # PREFILL
    logger.info(f"Falcon prefill for seq_len {seq_len} and one user only")
    _, tt_layer_present = tt_FalconCausalLM(
        input_embeddings=tt_prefill_embeddings,
        llm_mode="prefill",
        attention_mask=tt_prefill_attention_mask,
        user_id=0,
        layer_past=tt_layer_past,
        layer_past_len=0,
        use_cache=use_cache,
    )

    # DECODE ONCE
    logger.info(f"Falcon decode once for {batch} users")
    tt_out, tt_layer_present = tt_FalconCausalLM(
        input_embeddings=tt_decode_embeddings,
        llm_mode="decode",
        attention_mask=tt_decode_attention_mask,
        layer_past=tt_layer_present,
        layer_past_len=kv_cache_len,
        use_cache=use_cache,
    )
    tt_out = tt2torch_tensor(tt_out).squeeze(1).transpose(0, 1)

    # check outputs for first user only --------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_out[0], tt_out[0], pcc)
    logger.info(f"Output: {output_pcc}")

    for i in range(num_layers):
        pytorch_layer_pres = pytorch_layer_present[i]
        tt_layer_pres = (
            tt2torch_tensor(tt_layer_present[i][0]),
            tt2torch_tensor(tt_layer_present[i][1]),
        )
        tt_layer_pres = (
            tt_layer_pres[0][:, :, : kv_cache_len + 1, :],
            tt_layer_pres[1][:, :, : kv_cache_len + 1, :],
        )

        does_pass2, output_pcc = comp_pcc(pytorch_layer_pres[0][0], tt_layer_pres[0][0], pcc)
        logger.info(f"K Cache Layer {i}: {output_pcc}")

        does_pass = does_pass and does_pass2

        does_pass2, output_pcc = comp_pcc(pytorch_layer_pres[1][0], tt_layer_pres[1][0], pcc)
        logger.info(f"V Cache Layer {i}: {output_pcc}")

        does_pass = does_pass and does_pass2

    if does_pass:
        logger.info("Falcon Prefill and Decode Passed!")
    else:
        logger.warning("Falcon Prefill and Decode Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-L1",))
def test_FalconCausalLM_inference(
    model_version,
    request,
    model_config_str,
    model_location_generator,
    device,
):
    model_config = get_model_config(model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)

    tt_lib.profiler.set_profiler_location(f"falcon-7b_{request.node.callspec.id}")

    batch = 32
    seq_len = 128
    max_seq_len = 2048
    num_layers = 32
    pcc = 0.87
    run_test_FalconCausalLM_inference(
        device,
        model_version,
        batch,
        seq_len,
        max_seq_len,
        num_layers,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
