# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

from models.demos.falcon7b_common.tt.falcon_decoder import TtFalconDecoderLayer
from models.demos.falcon7b_common.tt.model_config import get_model_config
from models.demos.falcon7b_common.tests.test_utils import (
    get_rand_falcon_inputs,
    concat_device_outputs,
    load_hf_model,
    get_num_devices,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)


class PytorchFalconDecoderModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.decoder = hf_reference_model.transformer.h[layer_num]

        # Disable dropout
        self.decoder.eval()

    def forward(self, x, alibi, attention_mask, layer_past, use_cache):
        result = self.decoder(
            hidden_states=x,
            alibi=alibi,
            attention_mask=attention_mask,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        return result


def run_test_FalconDecoder_inference(
    mesh_device,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    layer_num,
    pcc,
    model_config,
    tt_cache_path,
    model_location_generator,
):
    num_devices = get_num_devices(mesh_device)
    global_batch = batch * num_devices

    hugging_face_reference_model, state_dict = load_hf_model(model_location_generator, model_version)
    configuration = hugging_face_reference_model.config

    # Prepare input ========================================================================
    torch.manual_seed(0)
    base_url = "transformer.h"
    max_position_embeddings = 2048
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    use_cache = True
    user_id = 0

    # Generate input, attention_mask, and kv_cache --------------------------------------
    (
        decoder_input,
        attention_mask_bool,
        layer_past,
        tt_decoder_input,
        tt_attention_mask,
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
    )
    if layer_past is not None:
        layer_past = layer_past[0]
        layer_past = (layer_past[0].squeeze(1), layer_past[1].squeeze(1))
    tt_layer_past = tt_layer_past[0]

    # PyTorch output =======================================================================
    pytorch_FalconDecoder_model = PytorchFalconDecoderModel(hugging_face_reference_model, layer_num)
    pytorch_out, pytorch_layer_present = pytorch_FalconDecoder_model(
        x=decoder_input,
        alibi=None,
        attention_mask=attention_mask_bool,
        layer_past=layer_past,
        use_cache=use_cache,
    )

    # TT hardware execution =================================================================
    tt_FalconDecoder_model = TtFalconDecoderLayer(
        mesh_device,
        state_dict,
        base_url,
        layer_num,
        configuration,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    )

    tt_out, tt_layer_present = tt_FalconDecoder_model(
        hidden_states=tt_decoder_input,
        llm_mode=llm_mode,
        alibi=None,
        attention_mask=tt_attention_mask,
        user_id=user_id,
        layer_past=tt_layer_past,
        layer_past_len=kv_cache_len,
        use_cache=use_cache,
    )
    tt_out, tt_layer_present = concat_device_outputs(mesh_device, tt_out, llm_mode, tt_layer_present, kv_len)

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output: {output_pcc}")

    does_pass2, output_pcc = comp_pcc(pytorch_layer_present[0], tt_layer_present[0], pcc)
    logger.info(f"K Cache: {output_pcc}")

    does_pass = does_pass and does_pass2

    does_pass2, output_pcc = comp_pcc(pytorch_layer_present[1], tt_layer_present[1], pcc)
    logger.info(f"V Cache: {output_pcc}")

    does_pass = does_pass and does_pass2

    if does_pass:
        logger.info("Falcon Decoder output Passed!")
    else:
        logger.warning("Falcon Decoder output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize("mesh_device", (1, 2, 4, (8, 4)), indirect=True, ids=["1chip", "2chip", "4chip", "32chipTG"])
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 1, 128, 0),
        ("prefill", 1, 1024, 0),
        ("prefill", 1, 2048, 0),
        ("decode", 32, 1, 128),
    ),
    ids=["prefill_seq128", "prefill_seq1024", "prefill_seq2048", "decode_batch32"],
)
@pytest.mark.parametrize(
    "model_version, layer_num, pcc",
    (("tiiuae/falcon-7b-instruct", 0, 0.98),),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1", "BFLOAT16-L1_SHARDED"))
def test_FalconDecoder_inference(
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    layer_num,
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

    run_test_FalconDecoder_inference(
        mesh_device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        layer_num,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
