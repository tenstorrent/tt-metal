# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import transformers
import ttnn
from models.demos.ttnn_falcon7b.tt.falcon_model import TtFalconModel
from models.demos.ttnn_falcon7b.tt.model_config import (
    get_model_config,
    get_tt_cache_path,
)
from models.demos.ttnn_falcon7b.tt.common import create_custom_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.demos.ttnn_falcon7b.tt.common import (
    create_custom_preprocessor,
    create_kv_cache,
    strip_state_dict_prefix,
)

from loguru import logger
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor


PRETRAINED_MODEL_NAME = f"tiiuae/falcon-7b-instruct"


def get_model_prefix(layer_index: int = 0):
    return f"transformer"


@pytest.mark.parametrize(
    "llm_mode, device_batch_size, seq_len, kv_cache_len",
    (
        ("prefill", 1, 128, 0),
        ("decode", 32, 1, 128),
    ),
    ids=["prefill_seq128_batch32", "decode_batch32"],
)
@pytest.mark.parametrize(
    "num_layers, expected_pcc",
    (
        (1, 0.98),
        (2, 0.98),
        (32, 0.94),
    ),
    ids=[
        "layers_1",
        "layers_2",
        "layers_32",
    ],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
@pytest.mark.parametrize(
    "mesh_device",
    [
        2,
    ],
    indirect=True,
)
def test_falcon_model(
    mesh_device,
    use_program_cache,
    model_version,
    llm_mode,
    device_batch_size,
    seq_len,
    kv_cache_len,
    num_layers,
    expected_pcc,
    model_config_str,
):
    torch.manual_seed(0)
    batch = device_batch_size * mesh_device.get_num_devices()
    if llm_mode == "decode":
        shard_dim = 2
    else:
        shard_dim = 0

    causual_model = transformers.FalconForCausalLM.from_pretrained(PRETRAINED_MODEL_NAME, low_cpu_mem_usage=True).eval()
    state_dict = causual_model.state_dict()
    filtered_state_dict = strip_state_dict_prefix(state_dict, get_model_prefix())

    configuration = transformers.FalconConfig.from_pretrained(model_version)
    configuration.num_hidden_layers = num_layers
    torch_model = transformers.models.falcon.modeling_falcon.FalconModel.from_pretrained(
        model_version, config=configuration
    ).eval()

    torch_model.load_state_dict(filtered_state_dict, strict=False)
    model_config = get_model_config(model_config_str)
    dtype = model_config["DEFAULT_DTYPE"]
    kv_len = seq_len if llm_mode == "prefill" else kv_cache_len + 1

    model_input = torch.arange(seq_len * batch).reshape(batch, seq_len)

    if llm_mode == "prefill":
        past_key_values = None
        tt_layer_past = ()
        for i in range(num_layers):
            _, tt_current_layer_past = create_kv_cache(
                llm_mode,
                dtype,
                batch,
                kv_cache_len,
                configuration,
                mesh_device,
                mesh_mapper=ShardTensorToMesh(mesh_device, dim=0),
            )
            tt_layer_past += (tt_current_layer_past,)
        attention_mask = None

    elif llm_mode == "decode":
        past_key_values = ()
        tt_layer_past = ()
        for i in range(num_layers):
            current_layer_past, tt_current_layer_past = create_kv_cache(
                llm_mode,
                dtype,
                batch,
                kv_cache_len,
                configuration,
                mesh_device,
                mesh_mapper=ShardTensorToMesh(mesh_device, dim=0),
            )
            past_key_values += (current_layer_past,)
            tt_layer_past += (tt_current_layer_past,)

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

    pytorch_out, pytorch_layer_present = torch_model(
        input_ids=model_input,
        attention_mask=None,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=False,
    )

    def convert_to_ttnn(model, name):
        return not isinstance(model, torch.nn.Embedding)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        device=mesh_device,
        custom_preprocessor=create_custom_preprocessor(
            model_config,
            tt_cache_path=get_tt_cache_path(f"{model_version}"),
            device=mesh_device,
            base_file_name=get_model_prefix(),
            weights_mesh_mapper=ReplicateTensorToMesh(mesh_device),
        ),
        convert_to_ttnn=convert_to_ttnn,
    )
    tt_FalconModel = TtFalconModel(
        mesh_device,
        num_layers,
        configuration,
        configuration.max_position_embeddings,
        model_config,
        parameters,
    )
    # TODO: Generate embeddings and attention_mask on device
    if llm_mode == "prefill":
        tt_outs = []
        # model_inputs = torch.split(model_input, 1)
        tt_embeddings, tt_attention_mask = tt_FalconModel.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=seq_len
        )
        tt_out, tt_layer_present = tt_FalconModel(
            input_embeddings=tt_embeddings,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            user_id=0,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=True,
        )
        tt_out = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=shard_dim)).squeeze(1)

    elif llm_mode == "decode":
        tt_embeddings, tt_attention_mask = tt_FalconModel.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )
        tt_out, tt_layer_present = tt_FalconModel(
            input_embeddings=tt_embeddings,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=True,
        )
        tt_out = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=shard_dim)).squeeze(1)
        tt_out = tt_out.transpose(0, 1)

    assert_with_pcc(pytorch_out, tt_out.to(pytorch_out.dtype), expected_pcc)

    for i in range(num_layers):
        tt_layer_pres = (
            ttnn.to_torch(tt_layer_present[i][0], mesh_composer=ConcatMeshToTensor(mesh_device, dim=0)),
            ttnn.to_torch(tt_layer_present[i][1], mesh_composer=ConcatMeshToTensor(mesh_device, dim=0)),
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

        assert_with_pcc(pytorch_layer_pres[0], tt_layer_pres[0].to(pytorch_layer_pres[0].dtype), expected_pcc)
        assert_with_pcc(pytorch_layer_pres[1], tt_layer_pres[1].to(pytorch_layer_pres[1].dtype), expected_pcc)

    logger.info("Falcon Model Passed!")
