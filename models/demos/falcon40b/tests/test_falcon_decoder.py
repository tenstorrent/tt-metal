# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.falcon40b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from models.demos.falcon40b.tt.falcon_decoder import TtFalconDecoderLayer
from models.demos.falcon40b.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000


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
    devices,
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

        tt_decoder_input_host = torch.chunk(decoder_input.unsqueeze(1), len(devices), -1)
        tt_decoder_input = []
        for i in range(len(devices)):
            tt_decoder_input.append(
                torch2tt_tensor(
                    tt_decoder_input_host[i],
                    devices[i],
                    tt_layout=tt_lib.tensor.Layout.TILE,
                    tt_memory_config=model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"],
                    tt_dtype=model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
                )
            )

        attention_mask_bool_chunks = torch.chunk(
            (attention_mask_bool * -100000).expand(-1, configuration.num_attention_heads, -1, -1),
            len(devices),
            1,
        )
        tt_attention_mask = []
        attention_mask_memconfig = model_config["ATTN_MASK_MEMCFG"]
        if attention_mask_memconfig.is_sharded():
            attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
            attn_mask_shard_shape[-1] = kv_len
            attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape

        for i in range(len(devices)):
            tt_attention_mask.append(
                torch2tt_tensor(
                    attention_mask_bool_chunks[i],
                    devices[i],
                    tt_memory_config=attention_mask_memconfig,
                    tt_dtype=model_config["ATTN_MASK_DTYPE"],
                )
            )

        tt_k_cache_host = torch.zeros(batch, configuration.num_kv_heads, max_position_embeddings, head_dim)
        tt_v_cache_host = torch.zeros(batch, configuration.num_kv_heads, max_position_embeddings, head_dim)
        tt_k_cache_host = torch.chunk(tt_k_cache_host, len(devices), 1)
        tt_v_cache_host = torch.chunk(tt_v_cache_host, len(devices), 1)
        tt_k_cache = []
        tt_v_cache = []
        for j in range(len(devices)):
            tt_k_cache.append(
                torch2tt_tensor(
                    tt_k_cache_host[j],
                    devices[j],
                    tt_lib.tensor.Layout.TILE,
                    model_config["KV_CACHE_MEMCFG"],
                    model_config["KV_CACHE_DTYPE"],
                )
            )
            tt_v_cache.append(
                torch2tt_tensor(
                    tt_v_cache_host[j],
                    devices[j],
                    tt_lib.tensor.Layout.TILE,
                    model_config["KV_CACHE_MEMCFG"],
                    model_config["KV_CACHE_DTYPE"],
                )
            )
        tt_layer_past = (tt_k_cache, tt_v_cache)

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

        tt_decoder_input_host = torch.chunk(decoder_input.unsqueeze(1).transpose(0, 2), len(devices), -1)
        tt_decoder_input = []
        for i in range(len(devices)):
            tt_decoder_input.append(
                torch2tt_tensor(
                    tt_decoder_input_host[i],
                    devices[i],
                    tt_layout=tt_lib.tensor.Layout.TILE,
                    tt_memory_config=model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"],
                    tt_dtype=model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
                )
            )

        kv_len_padded = (kv_len + 31) // 32 * 32
        attention_mask_bool_padded = torch.cat(
            (
                attention_mask_bool,
                torch.ones(batch, 1, q_len, kv_len_padded - kv_len, dtype=bool),
            ),
            dim=-1,
        )
        attention_mask_bool_padded = torch.chunk(
            (attention_mask_bool_padded.transpose(0, 2) * -100000).expand(
                -1, configuration.num_attention_heads, -1, -1
            ),
            len(devices),
            1,
        )
        tt_attention_mask = []
        attention_mask_memconfig = model_config["ATTN_MASK_MEMCFG"]
        if attention_mask_memconfig.is_sharded():
            attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
            attn_mask_shard_shape[-1] = kv_len_padded
            attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape

        for i in range(len(devices)):
            tt_attention_mask.append(
                torch2tt_tensor(
                    attention_mask_bool_padded[i],
                    devices[i],
                    tt_memory_config=attention_mask_memconfig,
                    tt_dtype=model_config["ATTN_MASK_DTYPE"],
                )
            )
        tt_k_cache_host = torch.zeros(batch, configuration.num_kv_heads, max_position_embeddings, head_dim)
        tt_v_cache_host = torch.zeros(batch, configuration.num_kv_heads, max_position_embeddings, head_dim)
        tt_k_cache_host[:, :, :kv_cache_len, :] = k_cache
        tt_v_cache_host[:, :, :kv_cache_len, :] = v_cache
        tt_k_cache_host = torch.chunk(tt_k_cache_host, len(devices), 1)
        tt_v_cache_host = torch.chunk(tt_v_cache_host, len(devices), 1)
        tt_k_cache = []
        tt_v_cache = []
        for j in range(len(devices)):
            tt_k_cache.append(
                torch2tt_tensor(
                    tt_k_cache_host[j],
                    devices[j],
                    tt_lib.tensor.Layout.TILE,
                    model_config["KV_CACHE_MEMCFG"],
                    model_config["KV_CACHE_DTYPE"],
                )
            )
            tt_v_cache.append(
                torch2tt_tensor(
                    tt_v_cache_host[j],
                    devices[j],
                    tt_lib.tensor.Layout.TILE,
                    model_config["KV_CACHE_MEMCFG"],
                    model_config["KV_CACHE_DTYPE"],
                )
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
        devices,
        state_dict,
        base_url,
        layer_num,
        configuration,
        max_position_embeddings,
        model_config,
        tt_cache_path,
        None,
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

    tt_out = torch.cat([tt2torch_tensor(tt_o).squeeze(1) for tt_o in tt_out], -1)
    tt_layer_present = (
        torch.cat([tt2torch_tensor(tt_layer_p) for tt_layer_p in tt_layer_present[0]], 1),
        torch.cat([tt2torch_tensor(tt_layer_p) for tt_layer_p in tt_layer_present[1]], 1),
    )

    if llm_mode == "decode":
        tt_out = tt_out.transpose(0, 1)
    tt_layer_present = (
        torch.repeat_interleave(
            tt_layer_present[0][:, :, :kv_len, :], configuration.num_attention_heads // configuration.num_kv_heads, 1
        ).flatten(0, 1),
        torch.repeat_interleave(
            tt_layer_present[1][:, :, :kv_len, :], configuration.num_attention_heads // configuration.num_kv_heads, 1
        ).flatten(0, 1),
    )

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, out_pcc)
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
@pytest.mark.parametrize("num_devices", (4, 8), ids=["4chips", "8chips"])
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 1, 32, 0),
        ("prefill", 1, 64, 0),
        ("prefill", 1, 128, 0),
        ("decode", 32, 1, 128),
    ),
    ids=["prefill_seq32", "prefill_seq64", "prefill_seq128", "decode_batch32"],
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
    [("BFLOAT8_B-SHARDED", 0.99, 0.99, 0.99), ("BFLOAT16-SHARDED", 0.99, 0.99, 0.99)],
    ids=["BFLOAT8_B-SHARDED", "BFLOAT16-SHARDED"],
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
    all_devices,
    use_program_cache,
):
    if llm_mode == "prefill" and model_config_str == "BFLOAT16-SHARDED":
        pytest.skip("Prefill is only tested for BFLOAT8_B!")

    input_shape = [batch, seq_len]
    model_config = get_model_config(model_config_str, llm_mode, input_shape, num_devices)
    devices = get_devices_for_t3000(all_devices, num_devices)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    run_test_FalconDecoder_inference(
        devices,
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
