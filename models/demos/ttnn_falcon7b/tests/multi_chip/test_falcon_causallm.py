# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import transformers
import ttnn
from models.demos.ttnn_falcon7b.tt.falcon_causallm import TtFalconCausalLM
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
)

from loguru import logger
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor


PRETRAINED_MODEL_NAME = f"tiiuae/falcon-7b-instruct"


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
        (32, 0.60),
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
@pytest.mark.parametrize(
    "enable_async, num_loops",
    ((True, 20), (False, 1)),
)
def test_falcon_causal_lm(
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
    enable_async,
    num_loops,
):
    for device in mesh_device.get_device_ids():
        mesh_device.get_device(device).enable_async(enable_async)

    torch.manual_seed(0)
    batch = device_batch_size * mesh_device.get_num_devices()
    if llm_mode == "decode":
        shard_dim = 2
    else:
        shard_dim = 0

    configuration = transformers.FalconConfig.from_pretrained(model_version)
    configuration.num_hidden_layers = num_layers
    model = transformers.models.falcon.modeling_falcon.FalconForCausalLM.from_pretrained(
        model_version, config=configuration
    ).eval()
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

    pytorch_out, pytorch_layer_present = model(
        input_ids=model_input,
        attention_mask=None,  # when attention_mask is None, a causal mask is created under the hood
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=False,
    )

    def convert_to_ttnn(model, name):
        return not isinstance(model, torch.nn.Embedding)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=mesh_device,
        custom_preprocessor=create_custom_preprocessor(
            model_config,
            tt_cache_path=get_tt_cache_path(f"{model_version}"),
            device=mesh_device,
            weights_mesh_mapper=ReplicateTensorToMesh(mesh_device),
        ),
        convert_to_ttnn=convert_to_ttnn,
    )
    tt_FalconCausalLM = TtFalconCausalLM(
        mesh_device,
        num_layers,
        configuration,
        configuration.max_position_embeddings,
        model_config,
        parameters,
    )
    # TODO: Generate embeddings and attention_mask on device
    if llm_mode == "prefill":
        for loop in range(num_loops):
            tt_outs = []
            tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
                llm_mode, model_input, kv_cache_len, num_input_tokens=seq_len
            )
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_embeddings=tt_embeddings,
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask,
                user_id=0,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=True,
            )
            # Explicitly move tensor to host ... in async mode this is faster than calling from torch directly,
            # due to parallelization of tensor shards
            tt_out = ttnn.from_device(tt_out)
            tt_out = ttnn.to_torch(
                tt_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=shard_dim), device=mesh_device
            ).squeeze(1)

    elif llm_mode == "decode":
        for loop in range(num_loops):
            tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
                llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
            )
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_embeddings=tt_embeddings,
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=True,
            )
            tt_out = ttnn.from_device(tt_out)
            tt_out = ttnn.to_torch(
                tt_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=shard_dim), device=mesh_device
            ).squeeze(1)
            tt_out = tt_out.transpose(0, 1)

    passed, pcc = assert_with_pcc(pytorch_out, tt_out.to(pytorch_out.dtype), expected_pcc)
    logger.success(f"Passed: pcc: {pcc}, expected: {expected_pcc}")

    for i in range(num_layers):
        tt_layer_pres = (
            ttnn.to_torch(
                tt_layer_present[i][0], mesh_composer=ConcatMeshToTensor(mesh_device, dim=0), device=mesh_device
            ),
            ttnn.to_torch(
                tt_layer_present[i][1], mesh_composer=ConcatMeshToTensor(mesh_device, dim=0), device=mesh_device
            ),
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

        passed, pcc = assert_with_pcc(
            pytorch_layer_pres[0], tt_layer_pres[0].to(pytorch_layer_pres[0].dtype), expected_pcc
        )
        logger.success(f"Passed: pcc: {pcc}, expected: {expected_pcc}")
        passed, pcc = assert_with_pcc(
            pytorch_layer_pres[1], tt_layer_pres[1].to(pytorch_layer_pres[1].dtype), expected_pcc
        )
        logger.success(f"Passed: pcc: {pcc}, expected: {expected_pcc}")

    logger.info("Falcon CausalLM Passed!")

    for device in mesh_device.get_device_ids():
        mesh_device.get_device(device).enable_async(False)


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
        (32, 0.60),
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
    "enable_async, num_loops",
    ((True, 50), (False, 50)),
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 4829184}], indirect=True)
def test_t3k_falcon_causal_lm_with_trace(
    t3k_mesh_device,
    use_program_cache,
    model_version,
    llm_mode,
    device_batch_size,
    seq_len,
    kv_cache_len,
    num_layers,
    expected_pcc,
    model_config_str,
    enable_async,
    num_loops,
):
    for device in t3k_mesh_device.get_device_ids():
        t3k_mesh_device.get_device(device).enable_async(enable_async)
        t3k_mesh_device.get_device(device).enable_program_cache()

    torch.manual_seed(0)
    batch = device_batch_size * t3k_mesh_device.get_num_devices()
    if llm_mode == "decode":
        shard_dim = 2
    else:
        shard_dim = 0

    configuration = transformers.FalconConfig.from_pretrained(model_version)
    configuration.num_hidden_layers = num_layers
    model = transformers.models.falcon.modeling_falcon.FalconForCausalLM.from_pretrained(
        model_version, config=configuration
    ).eval()
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
                t3k_mesh_device,
                mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=0),
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
                t3k_mesh_device,
                mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=0),
            )
            past_key_values += (current_layer_past,)
            tt_layer_past += (tt_current_layer_past,)

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

    pytorch_out, pytorch_layer_present = model(
        input_ids=model_input,
        attention_mask=None,  # when attention_mask is None, a causal mask is created under the hood
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=False,
    )

    def convert_to_ttnn(model, name):
        return not isinstance(model, torch.nn.Embedding)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=t3k_mesh_device,
        custom_preprocessor=create_custom_preprocessor(
            model_config,
            tt_cache_path=get_tt_cache_path(f"{model_version}"),
            device=t3k_mesh_device,
            weights_mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
        ),
        convert_to_ttnn=convert_to_ttnn,
    )
    tt_FalconCausalLM = TtFalconCausalLM(
        t3k_mesh_device,
        num_layers,
        configuration,
        configuration.max_position_embeddings,
        model_config,
        parameters,
    )
    # Preallocate self-attn scalars on device, since its bad for perf to send this tensor repeateduly during runtime
    # and trace does not support writes to device
    if llm_mode == "prefill":
        scalar_shape = (1, 71, 128, 128)
    else:
        scalar_shape = (1, 71, 32, 160)

    for layer in tt_FalconCausalLM.layers:
        layer.self_attn.scalar = ttnn.from_torch(
            torch.full(scalar_shape, layer.self_attn.scalar),
            device=t3k_mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
        )
    # TODO: Generate embeddings and attention_mask on device
    tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
        llm_mode, model_input, kv_cache_len, num_input_tokens=seq_len
    )
    if llm_mode == "prefill":
        logger.info("Compiling Prefill Model")
        tt_FalconCausalLM(
            input_embeddings=tt_embeddings,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            user_id=0,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=True,
        )
        logger.info("Capture Prefill Trace")
        trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_embeddings=tt_embeddings,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            user_id=0,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=True,
        )
        ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Prefill Trace")

        for loop in range(num_loops):
            ttnn.execute_trace(t3k_mesh_device, trace_id, cq_id=0)
            # Explicitly move tensor to host ... in async mode this is faster than calling from torch directly,
            # due to parallelization of tensor shards
            tt_out_host = ttnn.from_device(tt_out)
            tt_out_host = ttnn.to_torch(
                tt_out_host, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=shard_dim), device=t3k_mesh_device
            ).squeeze(1)

    elif llm_mode == "decode":
        logger.info("Compiling Decode Model")
        tt_FalconCausalLM(
            input_embeddings=tt_embeddings,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=True,
        )
        logger.info("Capture Decode Trace")
        trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_embeddings=tt_embeddings,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=True,
        )
        ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Decode Trace")
        for loop in range(num_loops):
            ttnn.execute_trace(t3k_mesh_device, trace_id, cq_id=0)
            tt_out_host = ttnn.from_device(tt_out)
            tt_out_host = ttnn.to_torch(
                tt_out_host, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=shard_dim), device=t3k_mesh_device
            ).squeeze(1)
            tt_out_host = tt_out_host.transpose(0, 1)

    passed, pcc = assert_with_pcc(pytorch_out, tt_out_host.to(pytorch_out.dtype), expected_pcc)
    logger.success(f"Passed: pcc: {pcc}, expected: {expected_pcc}")

    for i in range(num_layers):
        tt_layer_pres = (
            ttnn.to_torch(
                tt_layer_present[i][0],
                mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0),
                device=t3k_mesh_device,
            ),
            ttnn.to_torch(
                tt_layer_present[i][1],
                mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0),
                device=t3k_mesh_device,
            ),
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

        passed, pcc = assert_with_pcc(
            pytorch_layer_pres[0], tt_layer_pres[0].to(pytorch_layer_pres[0].dtype), expected_pcc
        )
        logger.success(f"Passed: pcc: {pcc}, expected: {expected_pcc}")
        passed, pcc = assert_with_pcc(
            pytorch_layer_pres[1], tt_layer_pres[1].to(pytorch_layer_pres[1].dtype), expected_pcc
        )
        logger.success(f"Passed: pcc: {pcc}, expected: {expected_pcc}")

    logger.info("Falcon CausalLM Passed!")

    for device in t3k_mesh_device.get_device_ids():
        t3k_mesh_device.get_device(device).enable_async(False)
