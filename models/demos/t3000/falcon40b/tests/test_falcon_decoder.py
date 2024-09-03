# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor
from models.demos.t3000.falcon40b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from models.demos.t3000.falcon40b.tt.falcon_decoder import TtFalconDecoderLayer
from models.demos.t3000.falcon40b.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import skip_for_grayskull
from models.demos.t3000.falcon40b.tt.model_utils import generate_layernorm_persistent_tensors


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
    out_pcc,
    cache_pcc,
    token_pcc,
    model_config,
    tt_cache_path,
    model_location_generator,
):
    model_name = model_location_generator(model_version, model_subdir="Falcon")

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, num_hidden_layers=layer_num + 1
    )
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input ========================================================================
    torch.manual_seed(0)
    decoder_input = (torch.rand(batch, seq_len, configuration.hidden_size) * 2) - 1
    base_url = "transformer.h"
    max_position_embeddings = 2048
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    use_cache = True
    user_id = 0

    ln_output_tensors_dict = {"final_layernorm": dict(), "mlp_layernorm": dict(), "attn_layernorm": dict()}
    # Generate input, attention_mask, and kv_cache --------------------------------------
    # TODO: Generate attention_mask on device
    if llm_mode == "prefill":
        q_len, kv_len = seq_len, seq_len
        assert batch == 1, "For prefill, batch must be 1!"
        assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
        assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

        decoder_input = (torch.rand(batch, q_len, configuration.hidden_size) * 2) - 1
        attention_mask_bool = torch.ones(batch, 1, q_len, kv_len, dtype=bool).triu(diagonal=1)
        layer_past = None

        tt_decoder_input = ttnn.as_tensor(
            tensor=decoder_input.unsqueeze(1),
            dtype=model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=-1),
        )

        attention_mask_memconfig = model_config["ATTN_MASK_MEMCFG"]
        if attention_mask_memconfig.is_sharded():
            attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
            attn_mask_shard_shape[-1] = kv_len
            attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape

        tt_attention_mask = ttnn.as_tensor(
            tensor=attention_mask_bool,
            dtype=model_config["ATTN_MASK_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=attention_mask_memconfig,
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=1),
            preprocess=lambda x: (x * -1e5).expand(-1, mesh_device.get_num_devices(), -1, -1),
        )

        tt_k_cache_host = torch.zeros(batch, configuration.num_kv_heads, max_position_embeddings, head_dim)
        tt_v_cache_host = torch.zeros(batch, configuration.num_kv_heads, max_position_embeddings, head_dim)

        tt_k_cache = ttnn.as_tensor(
            tensor=tt_k_cache_host,
            dtype=model_config["KV_CACHE_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=model_config["KV_CACHE_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=1),
        )
        tt_v_cache = ttnn.as_tensor(
            tensor=tt_v_cache_host,
            dtype=model_config["KV_CACHE_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=model_config["KV_CACHE_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=1),
        )
        tt_layer_past = (tt_k_cache, tt_v_cache)

        if seq_len > model_config["layernorm_params"]["slice_size"]:
            generate_layernorm_persistent_tensors(
                seq_len,
                model_config["layernorm_params"]["slice_size"],
                ln_output_tensors_dict,
                mesh_device,
                configuration.hidden_size,
                model_config["LN_MLP_OUTPUT_DTYPE"],
            )

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        decoder_input = (torch.rand(batch, q_len, configuration.hidden_size) * 2) - 1
        attention_mask_bool = torch.zeros(batch, 1, q_len, kv_len, dtype=bool)
        k_cache = torch.rand(batch, configuration.num_kv_heads, kv_cache_len, head_dim)
        v_cache = torch.rand(batch, configuration.num_kv_heads, kv_cache_len, head_dim)
        layer_past = (
            torch.repeat_interleave(
                k_cache, configuration.num_attention_heads // configuration.num_kv_heads, 1
            ).flatten(0, 1),
            torch.repeat_interleave(
                v_cache, configuration.num_attention_heads // configuration.num_kv_heads, 1
            ).flatten(0, 1),
        )

        tt_decoder_input = ttnn.as_tensor(
            tensor=decoder_input,
            dtype=model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=-1),
            preprocess=lambda x: x.unsqueeze(1).transpose(0, 2),
        )

        kv_len_padded = (kv_len + 31) // 32 * 32
        attention_mask_bool_padded = torch.cat(
            (
                attention_mask_bool,
                torch.ones(batch, 1, q_len, kv_len_padded - kv_len, dtype=bool),
            ),
            dim=-1,
        )

        attention_mask_memconfig = model_config["ATTN_MASK_MEMCFG"]
        if attention_mask_memconfig.is_sharded():
            attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
            attn_mask_shard_shape[-1] = kv_len_padded
            attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape

        tt_attention_mask = ttnn.as_tensor(
            tensor=attention_mask_bool_padded,
            dtype=model_config["ATTN_MASK_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=attention_mask_memconfig,
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=1),
            preprocess=lambda x: (x.transpose(0, 2) * -1e5).expand(-1, configuration.num_attention_heads, -1, -1),
        )

        tt_k_cache_host = torch.zeros(batch, configuration.num_kv_heads, max_position_embeddings, head_dim)
        tt_v_cache_host = torch.zeros(batch, configuration.num_kv_heads, max_position_embeddings, head_dim)
        tt_k_cache_host[:, :, :kv_cache_len, :] = k_cache
        tt_v_cache_host[:, :, :kv_cache_len, :] = v_cache

        tt_k_cache = ttnn.as_tensor(
            tensor=tt_k_cache_host,
            dtype=model_config["KV_CACHE_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=model_config["KV_CACHE_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=1),
        )

        tt_v_cache = ttnn.as_tensor(
            tensor=tt_v_cache_host,
            dtype=model_config["KV_CACHE_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=model_config["KV_CACHE_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=1),
        )

        tt_layer_past = (tt_k_cache, tt_v_cache)

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

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
        None,
        ln_output_tensors_dict,
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

    tt_out_tensor = ttnn.to_torch(tt_out, device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))

    tt_layer_present = (
        ttnn.to_torch(tt_layer_present[0], device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=1)),
        ttnn.to_torch(tt_layer_present[1], device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=1)),
    )

    if llm_mode == "decode":
        tt_out = tt_out_tensor.transpose(0, 1)
    tt_layer_present = (
        torch.repeat_interleave(
            tt_layer_present[0][:, :, :kv_len, :], configuration.num_attention_heads // configuration.num_kv_heads, 1
        ).flatten(0, 1),
        torch.repeat_interleave(
            tt_layer_present[1][:, :, :kv_len, :], configuration.num_attention_heads // configuration.num_kv_heads, 1
        ).flatten(0, 1),
    )

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out_tensor, out_pcc)
    logger.info(f"Output: {output_pcc}")

    does_pass2, output_pcc = comp_pcc(pytorch_layer_present[0], tt_layer_present[0], cache_pcc)
    logger.info(f"K Cache: {output_pcc}")

    does_pass = does_pass and does_pass2

    does_pass2, output_pcc = comp_pcc(pytorch_layer_present[1], tt_layer_present[1], cache_pcc)
    logger.info(f"V Cache: {output_pcc}")

    does_pass = does_pass and does_pass2

    if llm_mode == "decode":
        does_pass2, output_pcc = comp_pcc(
            pytorch_layer_present[0][:, kv_len - 1 : kv_len, :],
            tt_layer_present[0][:, kv_len - 1 : kv_len, :],
            token_pcc,
        )
        logger.info(f"K Cache new token: {output_pcc}")

        does_pass = does_pass and does_pass2

        does_pass2, output_pcc = comp_pcc(
            pytorch_layer_present[1][:, kv_len - 1 : kv_len, :],
            tt_layer_present[1][:, kv_len - 1 : kv_len, :],
            token_pcc,
        )
        logger.info(f"V Cache new token: {output_pcc}")

        does_pass = does_pass and does_pass2

    if does_pass:
        logger.info("Falcon Decoder output Passed!")
    else:
        logger.warning("Falcon Decoder output Failed!")
        assert does_pass


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", (8,), ids=["8chips"])
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 1, 32, 0),
        ("prefill", 1, 128, 0),
        ("prefill", 1, 2048, 0),
        ("decode", 32, 1, 128),
    ),
    ids=[
        "prefill_seq32",
        "prefill_seq128",
        "prefill_seq2048",
        "decode_batch32",
    ],
)
@pytest.mark.parametrize(
    "layer_num",
    ((0),),
    ids=["layer_0"],
)
@pytest.mark.parametrize(
    "model_version",
    (("tiiuae/falcon-40b-instruct"),),
    ids=["falcon_40b"],
)
@pytest.mark.parametrize(
    "model_config_str, out_pcc, cache_pcc, token_pcc",
    [
        ("BFLOAT8_B-SHARDED", 0.99, 0.99, 0.99),
        ("BFLOAT16-SHARDED", 0.99, 0.99, 0.99),
        ("BFLOAT8_B-DRAM", 0.99, 0.99, 0.99),
        ("BFLOAT8_B-DRAM", 0.99, 0.99, 0.99),
    ],
    ids=["BFLOAT8_B-SHARDED", "BFLOAT16-SHARDED", "BFLOAT8_B-DRAM", "BFLOAT16-DRAM"],
)
def test_FalconDecoder_inference(
    num_devices,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    layer_num,
    out_pcc,
    cache_pcc,
    token_pcc,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    t3k_mesh_device,
    use_program_cache,
):
    if llm_mode == "prefill" and (model_config_str not in ["BFLOAT8_B-DRAM", "BFLOAT16-DRAM"] or num_devices != 8):
        pytest.skip("Prefill is only supported for DRAM memory config and 8 chips!")
    if llm_mode == "decode" and model_config_str not in ["BFLOAT8_B-SHARDED", "BFLOAT16-SHARDED"]:
        pytest.skip("Decode is only supported for SHARDED memory config!")

    input_shape = [batch, seq_len]
    model_config = get_model_config(model_config_str, llm_mode, input_shape, num_devices)
    devices = t3k_mesh_device.get_devices()
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    run_test_FalconDecoder_inference(
        t3k_mesh_device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        layer_num,
        out_pcc,
        cache_pcc,
        token_pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
