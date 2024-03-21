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
from models.demos.falcon7b.tests.test_utils import get_rand_falcon_inputs, concat_device_out_layer_present
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, get_devices_for_t3000


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

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name)

    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input ------------------------------------------------------------------------
    torch.manual_seed(0)
    base_url = ""
    max_position_embeddings = 2048
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    use_cache = True

    if 1:
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

    # Prepare output -----------------------------------------------------------------------
    pytorch_FalconCausalLM = PytorchFalconCausalLM(hugging_face_reference_model, num_layers)
    pytorch_out, pytorch_layer_present = pytorch_FalconCausalLM(
        input_ids=model_input, past_key_values=past_key_values, use_cache=use_cache
    )

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders
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

    # TODO: Generate embeddings and attention_mask on device
    if llm_mode == "prefill":
        tt_outs = torch.zeros(global_batch, seq_len, configuration.vocab_size)  # Output tensor to overwrite
        tt_embeddings, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(
                    llm_mode, model_input[i::batch], kv_cache_len, num_input_tokens=seq_len
                )
                for i in range(batch)
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
            # Get outputs from all devices
            tt_outs[user_id::batch] = torch.concat([tt2torch_tensor(tt_out[i]).squeeze(1) for i in range(num_devices)])
        tt_out = tt_outs

    elif llm_mode == "decode":
        tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_embeddings=tt_embeddings,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        for i in range(num_devices):
            tt_out[i] = tt2torch_tensor(tt_out[i]).squeeze(1).transpose(0, 1)
        tt_out = torch.concat(tt_out)

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output: {output_pcc}")

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

    if does_pass:
        logger.info("Falcon CausalLM Passed!")
    else:
        logger.warning("Falcon CausalLM Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize("num_devices", (1, 2, 4))
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 2, 128, 0),
        ("decode", 32, 1, 128),
    ),
    ids=["prefill_seq128", "decode_batch32"],
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
def test_FalconCausalLM_inference(
    num_devices,
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
    all_devices,
):
    devices = get_devices_for_t3000(all_devices, num_devices)

    model_config = get_model_config(model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)

    tt_lib.profiler.set_profiler_location(f"falcon-7b_{request.node.callspec.id}")

    run_test_FalconCausalLM_inference(
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
    )
