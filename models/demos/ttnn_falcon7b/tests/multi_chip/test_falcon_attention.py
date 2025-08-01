# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.ttnn_falcon7b.tt.common import (
    create_attention_input,
    create_attention_mask,
    create_custom_preprocessor,
    create_kv_cache,
    create_position_ids,
    strip_state_dict_prefix,
)
from models.demos.ttnn_falcon7b.tt.falcon_attention import TtFalconAttention
from models.demos.ttnn_falcon7b.tt.model_config import get_model_config, get_tt_cache_path
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh, ShardTensorToMesh

PRETRAINED_MODEL_NAME = f"tiiuae/falcon-7b-instruct"


def get_model_prefix(layer_index: int = 0):
    return f"transformer.h.{layer_index}.self_attention"


@pytest.fixture(scope="module")
def torch_model():
    hugging_face_reference_model = transformers.FalconForCausalLM.from_pretrained(
        PRETRAINED_MODEL_NAME, low_cpu_mem_usage=True
    ).eval()
    state_dict = hugging_face_reference_model.state_dict()
    filtered_state_dict = strip_state_dict_prefix(state_dict, get_model_prefix())

    configuration = transformers.FalconConfig.from_pretrained(PRETRAINED_MODEL_NAME)
    torch_model = transformers.models.falcon.modeling_falcon.FalconAttention(configuration, layer_idx=0).eval()
    torch_model.load_state_dict(filtered_state_dict)
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
    (("tiiuae/falcon-7b-instruct", 0.99),),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
@pytest.mark.parametrize(
    "mesh_device",
    [
        2,
    ],
    indirect=True,
)
def test_falcon_attention(
    mesh_device,
    model_name,
    llm_mode,
    device_batch_size,
    seq_len,
    kv_cache_len,
    expected_pcc,
    model_config_str,
    torch_model,
):
    torch.manual_seed(0)
    batch = device_batch_size * mesh_device.get_num_devices()
    if llm_mode == "decode":
        shard_dim = 2
        concat_dim = 1
    else:
        shard_dim = 0
        concat_dim = 0

    configuration = transformers.FalconConfig.from_pretrained(model_name)
    model_config = get_model_config(model_config_str)
    dtype = model_config["DEFAULT_DTYPE"]
    kv_len = seq_len if llm_mode == "prefill" else kv_cache_len + 1

    attention_input, tt_attention_input = create_attention_input(
        llm_mode,
        dtype,
        batch,
        seq_len,
        configuration.hidden_size,
        mesh_device,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=shard_dim),
    )
    position_ids = create_position_ids(llm_mode, seq_len, kv_cache_len)
    attention_mask, tt_attention_mask = create_attention_mask(
        llm_mode,
        dtype,
        attention_input,
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
    position_embeddings = torch_model.rotary_emb(attention_input, position_ids)

    pytorch_out, pytorch_layer_present = torch_model(
        attention_input,
        alibi=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        layer_past=layer_past,
        use_cache=True,
        position_embeddings=position_embeddings,
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
    tt_FalconAttention_model = TtFalconAttention(
        configuration.hidden_size,
        configuration.num_attention_heads,
        configuration.max_position_embeddings,
        model_config,
        parameters=parameters,
        core_grid=mesh_device.core_grid,
    )

    tt_out, tt_layer_present = tt_FalconAttention_model(
        tt_attention_input,
        alibi=None,
        attention_mask=tt_attention_mask,
        llm_mode=llm_mode,
        user_id=0,
        layer_past=tt_layer_past,
        layer_past_len=kv_cache_len,
        use_cache=True,
    )
    tt_out = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=concat_dim)).squeeze(1)

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
        pytorch_layer_present.key_cache[0].squeeze(1),
        tt_layer_present[0].to(pytorch_layer_present.key_cache[0].dtype),
        expected_pcc,
    )
    assert_with_pcc(
        pytorch_layer_present.value_cache[0].squeeze(1),
        tt_layer_present[1].to(pytorch_layer_present.value_cache[0].dtype),
        expected_pcc,
    )
