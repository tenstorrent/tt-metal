# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.demos.ttnn_falcon7b.tt.falcon_decoder import TtFalconDecoderLayer
from models.demos.ttnn_falcon7b.tt.model_config import get_model_config, get_tt_cache_path
from models.demos.ttnn_falcon7b.tt.common import create_custom_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
import transformers
from models.demos.ttnn_falcon7b.tt.common import (
    create_custom_preprocessor,
    create_attention_mask,
    create_kv_cache,
    create_attention_input,
    create_position_ids,
    strip_state_dict_prefix,
)

from loguru import logger
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor


PRETRAINED_MODEL_NAME = f"tiiuae/falcon-7b-instruct"


def get_model_prefix(layer_index: int = 0):
    return f"transformer.h.{layer_index}"


@pytest.fixture(scope="module")
def torch_model():
    hugging_face_reference_model = transformers.FalconForCausalLM.from_pretrained(
        PRETRAINED_MODEL_NAME, low_cpu_mem_usage=True
    ).eval()
    state_dict = hugging_face_reference_model.state_dict()
    mlp_state_dict = strip_state_dict_prefix(state_dict, get_model_prefix())

    configuration = transformers.FalconConfig.from_pretrained(PRETRAINED_MODEL_NAME)
    torch_model = transformers.models.falcon.modeling_falcon.FalconDecoderLayer(configuration).eval()
    torch_model.load_state_dict(mlp_state_dict)
    return torch_model


@pytest.mark.parametrize(
    "llm_mode, device_batch_size, seq_len, kv_cache_len",
    (
        ("prefill", 1, 128, 0),
        ("decode", 32, 1, 128),
    ),
    ids=["prefill_seq128", "decode_batch32"],
)
@pytest.mark.parametrize(
    "model_name, expected_pcc",
    (("tiiuae/falcon-7b-instruct", 0.98),),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
@pytest.mark.parametrize(
    "mesh_device",
    [
        2,
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "enable_async",
    [True, False],
)
def test_falcon_decoder(
    mesh_device,
    model_name,
    llm_mode,
    device_batch_size,
    seq_len,
    kv_cache_len,
    expected_pcc,
    model_config_str,
    torch_model,
    enable_async,
):
    for device in mesh_device.get_device_ids():
        mesh_device.get_device(device).enable_async(enable_async)

    torch.manual_seed(0)
    batch = device_batch_size * mesh_device.get_num_devices()
    if llm_mode == "decode":
        shard_dim = 2
    else:
        shard_dim = 0

    configuration = transformers.FalconConfig.from_pretrained(model_name)
    model_config = get_model_config(model_config_str)
    dtype = model_config["DEFAULT_DTYPE"]
    kv_len = seq_len if llm_mode == "prefill" else kv_cache_len + 1

    decoder_input, tt_decoder_input = create_attention_input(
        llm_mode,
        dtype,
        batch,
        seq_len,
        configuration.hidden_size,
        mesh_device,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=shard_dim),
    )
    position_ids = create_position_ids(llm_mode, kv_cache_len)
    attention_mask, tt_attention_mask = create_attention_mask(
        llm_mode,
        dtype,
        decoder_input,
        batch,
        seq_len,
        configuration.num_attention_heads,
        kv_cache_len,
        mesh_device,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=shard_dim),
    )
    layer_past, tt_layer_past = create_kv_cache(
        llm_mode,
        dtype,
        batch,
        kv_cache_len,
        configuration,
        mesh_device,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=0),
    )

    pytorch_out, pytorch_layer_present = torch_model(
        hidden_states=decoder_input,
        alibi=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        layer_past=layer_past,
        use_cache=True,
    )
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        device=mesh_device,
        custom_preprocessor=create_custom_preprocessor(
            model_config,
            tt_cache_path=get_tt_cache_path(f"{model_name}"),
            device=mesh_device,
            base_file_name=get_model_prefix(),
            weights_mesh_mapper=ReplicateTensorToMesh(mesh_device),
        ),
    )
    tt_FalconDecoder_model = TtFalconDecoderLayer(
        mesh_device,
        configuration,
        model_config,
        parameters,
    )

    tt_out, tt_layer_present = tt_FalconDecoder_model(
        hidden_states=tt_decoder_input,
        llm_mode=llm_mode,
        alibi=None,
        attention_mask=tt_attention_mask,
        user_id=0,
        layer_past=tt_layer_past,
        layer_past_len=kv_cache_len,
        use_cache=True,
    )
    tt_out = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=shard_dim)).squeeze(1)

    tt_layer_present = (
        ttnn.to_torch(tt_layer_present[0], mesh_composer=ConcatMeshToTensor(mesh_device, dim=0)).squeeze(1),
        ttnn.to_torch(tt_layer_present[1], mesh_composer=ConcatMeshToTensor(mesh_device, dim=0)).squeeze(1),
    )
    if llm_mode == "decode":
        tt_out = tt_out.transpose(0, 1)
    tt_layer_present = (
        tt_layer_present[0][:, :kv_len, :],
        tt_layer_present[1][:, :kv_len, :],
    )

    passed, pcc = assert_with_pcc(pytorch_out, tt_out.to(pytorch_out.dtype), expected_pcc)
    logger.success(f"Passed: pcc: {pcc}, expected: {expected_pcc}")
    assert_with_pcc(
        pytorch_layer_present[0].squeeze(1), tt_layer_present[0].to(pytorch_layer_present[0].dtype), expected_pcc
    )
    assert_with_pcc(
        pytorch_layer_present[1].squeeze(1), tt_layer_present[1].to(pytorch_layer_present[1].dtype), expected_pcc
    )

    for device in mesh_device.get_device_ids():
        mesh_device.get_device(device).enable_async(False)
