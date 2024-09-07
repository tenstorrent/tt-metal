# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from models.demos.falcon7b_common.tt.falcon_model import TtFalconModel
from models.demos.falcon7b_common.tt.model_config import (
    get_model_config,
)
from models.demos.falcon7b_common.tests.test_utils import (
    get_rand_falcon_inputs,
    concat_device_out_layer_present,
    load_hf_model,
    get_num_devices,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import tt_tensors_to_torch_tensors


class PytorchFalconModel(torch.nn.Module):
    def __init__(self, hf_reference_model, num_layers):
        super().__init__()
        self.model = hf_reference_model.transformer
        self.model.h = self.model.h[:num_layers]
        self.model.eval()

    def forward(self, input_ids, past_key_values, use_cache):
        result = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=False,
        )

        return result


def run_test_FalconModel_inference(
    mesh_device,
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
    num_devices = get_num_devices(mesh_device)
    global_batch = batch * num_devices
    hugging_face_reference_model, state_dict = load_hf_model(model_location_generator, model_version)
    configuration = hugging_face_reference_model.config

    # Prepare input ------------------------------------------------------------------------
    torch.manual_seed(0)
    base_url = "transformer"
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
        mesh_device,
        global_batch,
        head_dim,
        max_position_embeddings,
        configuration,
        model_config,
        num_layers=num_layers,
        generate_attention_inputs=False,
    )

    # Prepare output -----------------------------------------------------------------------
    pytorch_FalconModel = PytorchFalconModel(hugging_face_reference_model, num_layers)
    pytorch_out, pytorch_layer_present = pytorch_FalconModel(
        input_ids=model_input, past_key_values=past_key_values, use_cache=use_cache
    )

    tt_FalconModel = TtFalconModel(
        mesh_device,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    )
    # TODO: Generate attention_mask on device
    if llm_mode == "prefill":
        tt_outs = torch.zeros(global_batch, seq_len, configuration.hidden_size)  # Output tensor to overwrite
        tt_input_ids, tt_attention_mask = zip(
            *[
                # Get input ids and attention_mask for each device
                tt_FalconModel.model_preprocessing(
                    llm_mode, model_input[i::batch], kv_cache_len, num_input_tokens=seq_len
                )
                for i in range(batch)
            ]
        )
        for user_id in range(batch):
            tt_out, tt_layer_present = tt_FalconModel(
                input_ids=tt_input_ids[user_id],
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask[user_id],
                user_id=user_id,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=use_cache,
            )
            # Get outputs from all devices
            tt_outs[user_id::batch] = tt_tensors_to_torch_tensors(tt_out, mesh_device, concat_dim=0).squeeze(1)
        tt_out = tt_outs

    elif llm_mode == "decode":
        tt_input_ids, tt_attention_mask = tt_FalconModel.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )
        tt_out, tt_layer_present = tt_FalconModel(
            input_ids=tt_input_ids,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        tt_out = tt_tensors_to_torch_tensors(tt_out, mesh_device, concat_dim=2).squeeze(1).transpose(0, 1)

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output: {output_pcc}")

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

        does_pass2, output_pcc = comp_pcc(pytorch_layer_pres[0], tt_layer_pres[0], pcc)
        logger.info(f"K Cache Layer {i}: {output_pcc}")

        does_pass = does_pass and does_pass2

        does_pass2, output_pcc = comp_pcc(pytorch_layer_pres[1], tt_layer_pres[1], pcc)
        logger.info(f"V Cache Layer {i}: {output_pcc}")

        does_pass = does_pass and does_pass2

    if does_pass:
        logger.info("Falcon Model Passed!")
    else:
        logger.warning("Falcon Model Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize("mesh_device", (1, 2, 4, (8, 4)), indirect=True, ids=["1chip", "2chip", "4chip", "32chipTG"])
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 1, 128, 0),
        ("decode", 32, 1, 128),
    ),
    ids=["prefill_seq128_batch1", "decode_batch32"],
)
@pytest.mark.parametrize(
    "num_layers, pcc",
    ((1, 0.98), (2, 0.98), (32, 0.98)),
    ids=["layers_1", "layers_2", "layers_32"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1", "BFLOAT16-L1_SHARDED"))
def test_FalconModel_inference(
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    pcc,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    mesh_device,
    enable_async_mode,
):
    if model_config_str == "BFLOAT16-L1_SHARDED" and llm_mode == "prefill":
        pytest.skip(f"prefill does not support L1_SHARDED")

    model_config = get_model_config(model_config_str, seq_len, batch)
    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )
    run_test_FalconModel_inference(
        mesh_device,
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
